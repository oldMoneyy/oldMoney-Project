#!/usr/bin/env python3
"""
GPTQ-Enhanced NVFP4 Mixed-Precision Quantization for MiniCPM-SALA
=================================================================

Algorithm:
  Standard NVFP4 uses RTN (Round-To-Nearest).  This replaces RTN with
  GPTQ Hessian-based error compensation on the FP4-E2M1 non-uniform grid.

  FP4 E2M1 representable values: {0, 0.5, 1, 1.5, 2, 3, 4, 6} x {+, -}

Strategy:
  - Lightning-attn layers (24/32):  GPTQ-NVFP4 (Hessian + E2M1 grid)
  - MiniCPM4 layers      (8/32):   BF16 (untouched, preserves sparse attn)

Output: ModelOpt-compatible checkpoint -> SGLang ModelOptFp4Config.

"""

import os
import gc
import json
import math
import time
import glob
import shutil
import argparse
import functools
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np

os.environ["PYTORCH_ALLOC_CONF"] = (
    "expandable_segments:True,max_split_size_mb:64"
)

# ====================================================================
# NVFP4 Constants
# ====================================================================
FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
NVFP4_SCALE_FACTOR = FP4_E2M1_MAX * FP8_E4M3_MAX  # 2688.0
NVFP4_GROUP_SIZE = 16

# ====================================================================
# E2M1 Grid  (15 distinct values, sorted)
#
#   4-bit encoding = sign(1) | exponent(2) | mantissa(1)
#
#   +0.0->0b0000  +0.5->0b0001  +1.0->0b0010  +1.5->0b0011
#   +2.0->0b0100  +3.0->0b0101  +4.0->0b0110  +6.0->0b0111
#   -0.0->0b1000  -0.5->0b1001  -1.0->0b1010  -1.5->0b1011
#   -2.0->0b1100  -3.0->0b1101  -4.0->0b1110  -6.0->0b1111
# ====================================================================
_E2M1_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                          dtype=torch.float32)

# Sorted grid: [-6,-4,-3,-2,-1.5,-1,-0.5, 0, 0.5,1,1.5,2,3,4,6]
_SORTED_GRID = torch.cat([-_E2M1_POS.flip(0)[:-1], _E2M1_POS])  # 15 values

# Midpoints between consecutive grid values -> Voronoi boundaries
# for torch.bucketize.  14 boundaries for 15 bins.
_MIDPOINTS = (_SORTED_GRID[:-1] + _SORTED_GRID[1:]) / 2.0

# Grid-index (into _SORTED_GRID) -> 4-bit E2M1 hardware code
_GRID_IDX_TO_4BIT = np.array(
    [15, 14, 13, 12, 11, 10, 9, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint8
)

# Hessian accumulation chunk size (rows per GPU matmul).
# At K=16384: 8192 * 16384 * 4B = 512 MB GPU f32.
HESSIAN_CHUNK_ROWS = 8192


# ====================================================================
# S1  E2M1 Quantisation  (O(1) extra memory via bucketize)
# ====================================================================

def quantize_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Snap every element to nearest FP4-E2M1 grid value.

    Uses torch.bucketize on 14 precomputed midpoints.
    Memory: O(1) beyond input/output (14-element midpoint vector).
    Time:   O(N log 14) ~ O(4N) via binary search.
    """
    midpoints = _MIDPOINTS.to(x.device, dtype=x.dtype)
    grid = _SORTED_GRID.to(x.device, dtype=x.dtype)
    idx = torch.bucketize(x.clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX), midpoints)
    return grid[idx]


# ====================================================================
# S2  E2M1 Bitwise Packer  (zero re-quantisation)
# ====================================================================

def pack_e2m1_to_uint8(e2m1_vals: torch.Tensor) -> torch.Tensor:
    """Pack E2M1 float values -> uint8 (2 values per byte, low nibble first).

    Byte layout matches NVIDIA FP4-E2M1X2 with is_sf_swizzled_layout=False:
        byte[i] = (code[2i+1] << 4) | code[2i]

    Operates directly on GPTQ-optimised E2M1 values.
    No re-quantisation.  No official-packer fallback.  Pure bitwise.
    """
    N, K = e2m1_vals.shape
    assert K % 2 == 0, f"K={K} must be even for FP4 packing"

    midpoints = _MIDPOINTS.to(e2m1_vals.device, dtype=e2m1_vals.dtype)
    flat = e2m1_vals.reshape(-1).clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)

    # float -> grid index -> 4-bit code  (bucketize on GPU, LUT on CPU)
    grid_idx = torch.bucketize(flat, midpoints).cpu().numpy()
    codes = _GRID_IDX_TO_4BIT[grid_idx].reshape(N, K)

    # Pack adjacent pairs: low nibble = even col, high nibble = odd col
    lo = codes[:, 0::2]
    hi = codes[:, 1::2]
    packed = (hi.astype(np.uint16) << 4) | lo

    return torch.from_numpy(packed.astype(np.uint8))


# ====================================================================
# S3  GPTQ Core with FP4-E2M1 Grid
# ====================================================================

class GPTQ_FP4:
    """GPTQ quantiser using the FP4-E2M1 non-uniform grid.

    Design:
      1. Block scales pre-computed from W_orig (cloned before GPTQ
         touches anything).  Immune to error-propagation distortion.
      2. E2M1 snap via torch.bucketize: O(1) memory, O(N log 14) time.
      3. Hessian accumulation chunked at HESSIAN_CHUNK_ROWS to cap GPU mem.
    """

    def __init__(self, name: str, weight: torch.Tensor):
        self.name = name
        self.rows, self.columns = weight.shape
        self.H = torch.zeros(
            (self.columns, self.columns), device="cpu", dtype=torch.float64
        )
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor):
        """Accumulate Hessian  H += X^T X.

        Chunked along token dimension to prevent OOM.
        At K=16384: each 8192-row chunk uses ~512 MB GPU f32.
        The Hessian itself is K*K f64 on CPU (2 GB at K=16384).
        """
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        elif inp.dim() == 1:
            inp = inp.unsqueeze(0)

        n_tokens = inp.shape[0]
        self.nsamples += n_tokens

        for start in range(0, n_tokens, HESSIAN_CHUNK_ROWS):
            end = min(start + HESSIAN_CHUNK_ROWS, n_tokens)
            chunk = inp[start:end].float()   # [chunk, K] GPU f32
            h_chunk = chunk.T @ chunk        # [K, K] GPU f32
            self.H.add_(h_chunk.cpu().to(torch.float64))
            del chunk, h_chunk

    def quantize(
        self,
        weight: torch.Tensor,
        global_sf: torch.Tensor,
        damp_percent: float = 0.01,
        block_size: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run GPTQ with FP4-E2M1 grid.

        Returns:
            Q_deq        [N, K]       dequantised weights (next-layer calib)
            e2m1_values  [N, K]       E2M1 grid floats    (for direct packing)
            block_scales [N, K//16]   float32 block scales (for checkpoint)
        """
        dev = weight.device
        W = weight.float().clone()
        H = (self.H / self.nsamples).float().to(dev)
        del self.H

        rows, columns = W.shape
        gsf = global_sf.item()

        # -- Hessian inverse -------------------------------------------
        dead = torch.diag(H) == 0
        H[dead, dead] = 1.0
        W[:, dead] = 0.0

        damp = damp_percent * torch.diag(H).mean()
        H.diagonal().add_(damp)

        Hinv = self._make_hinv(H)
        del H

        # -- Pre-compute ALL block scales from W_orig ------------------
        #
        # Clone W NOW, before the GPTQ loop modifies it.  Every
        # group's scale reflects the TRUE, undistorted weight
        # distribution.
        #
        # Why not compute from error-propagated W_block?
        #   Error propagation within a GPTQ block shifts weights far
        #   from their original magnitude (esp. later columns in a
        #   128-wide block).  Computing block_scale from shifted values
        #   yields a scale "chasing" the distortion rather than
        #   representing the actual weight distribution.  Result:
        #   severe clipping on some groups, wasted dynamic range on
        #   others.
        #
        # GPTQ's error compensation naturally handles the small
        # mismatch between the "ideal" scale and actual quant error.
        #
        group_size = NVFP4_GROUP_SIZE
        num_groups = columns // group_size
        block_scales = torch.zeros(rows, num_groups, device=dev,
                                   dtype=torch.float32)

        W_orig = W.clone()
        for g in range(num_groups):
            gs = g * group_size
            ge = gs + group_size
            grp_amax = (W_orig[:, gs:ge].abs().max(dim=1).values * gsf
                        ).clamp(min=1e-12)
            bsf = (grp_amax / FP4_E2M1_MAX).clamp(
                min=torch.finfo(torch.float8_e4m3fn).tiny,
                max=FP8_E4M3_MAX,
            )
            block_scales[:, g] = bsf
        del W_orig

        # -- GPTQ block-wise loop --------------------------------------
        Q_deq = torch.zeros_like(W)
        e2m1_vals = torch.zeros_like(W)

        for blk_s in range(0, columns, block_size):
            blk_e = min(blk_s + block_size, columns)
            blen = blk_e - blk_s

            W_block = W[:, blk_s:blk_e].clone()
            Err_block = torch.zeros_like(W_block)
            Hinv_blk = Hinv[blk_s:blk_e, blk_s:blk_e]

            for j in range(blen):
                col = blk_s + j
                g = col // group_size
                bsf_col = block_scales[:, g]   # stable, pre-computed

                w = W_block[:, j]
                d = Hinv_blk[j, j]

                # Forward quantise -> snap to E2M1 -> dequantise
                w_scaled = w * gsf / bsf_col
                q_e2m1 = quantize_to_e2m1(w_scaled)
                w_deq = q_e2m1 * bsf_col / gsf

                e2m1_vals[:, col] = q_e2m1
                Q_deq[:, col] = w_deq

                # GPTQ error propagation
                err = (w - w_deq) / d
                Err_block[:, j] = err
                if j + 1 < blen:
                    W_block[:, j + 1:] -= (
                        err.unsqueeze(1) * Hinv_blk[j, j + 1:].unsqueeze(0)
                    )

            # Inter-block error propagation
            if blk_e < columns:
                W[:, blk_e:] -= Err_block @ Hinv[blk_s:blk_e, blk_e:]

        return Q_deq, e2m1_vals, block_scales

    def _make_hinv(self, H: torch.Tensor) -> torch.Tensor:
        """H -> upper Cholesky factor of H^{-1}."""
        for attempt in range(5):
            try:
                L = torch.linalg.cholesky(H)
                H_inv = torch.cholesky_inverse(L)
                del L
                return torch.linalg.cholesky(H_inv, upper=True)
            except RuntimeError:
                extra = (10 ** attempt) * 0.01 * torch.diag(H).mean()
                H.diagonal().add_(extra)
                print(f"    [{self.name}] Cholesky retry {attempt+1}, "
                      f"damp += {extra:.6f}")
        print(f"    [{self.name}] WARNING: Cholesky failed -> diagonal approx")
        return torch.linalg.cholesky(
            torch.diag(1.0 / torch.diag(H)), upper=True
        )


# ====================================================================
# S4  RTN Baseline  (quality comparison only)
# ====================================================================

def rtn_nvfp4_mse(W: torch.Tensor, gsf: float) -> float:
    """MSE of plain RTN FP4-E2M1 quantisation (no Hessian)."""
    N, K = W.shape
    gs = NVFP4_GROUP_SIZE
    Q = torch.zeros_like(W)
    for g in range(K // gs):
        s, e = g * gs, (g + 1) * gs
        grp = W[:, s:e]
        amax = (grp.abs().max(dim=1, keepdim=True).values * gsf
                ).clamp(min=1e-12)
        bsf = (amax / FP4_E2M1_MAX).clamp(
            min=torch.finfo(torch.float8_e4m3fn).tiny, max=FP8_E4M3_MAX
        )
        q_sc = quantize_to_e2m1(grp * gsf / bsf)
        Q[:, s:e] = q_sc * bsf / gsf
    return (Q - W.float()).pow(2).mean().item()


# ====================================================================
# S5  Cache Management
# ====================================================================

def nuke_caches():
    """Clear FLA / model caches + GPU memory."""
    try:
        from fla.utils import tensor_cache
        tensor_cache.clear()
    except Exception:
        pass
    for obj in gc.get_objects():
        if isinstance(obj, functools._lru_cache_wrapper):
            try:
                obj.cache_clear()
            except Exception:
                pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ====================================================================
# S6  VRAM-Backed Hidden-State Store (Optimized for 96GB GPU)
# ====================================================================

class HiddenStateStore:
    """Store each sample's hidden state directly in VRAM.
    Zero disk I/O. Maximum speed for high-VRAM GPUs.
    """

    def __init__(self, tmp_dir: str):
        # 保留 tmp_dir 参数以兼容后面的代码，但不再使用硬盘
        self._states: Dict[int, torch.Tensor] = {}

    def save(self, idx: int, tensor: torch.Tensor):
        # .detach().clone() 是核心，保留在显存且防止计算图内存泄漏
        self._states[idx] = tensor.detach().clone()

    def load(self, idx: int, device: torch.device) -> torch.Tensor:
        # 直接返回显存中的张量，速度极快
        return self._states[idx]

    def cleanup(self):
        self._states.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __len__(self):
        return len(self._states)


# ====================================================================
# S7  Calibration Data
# ====================================================================

def load_calibration_data(tokenizer, data_path, max_samples, max_len):
    samples = []
    with open(data_path) as f:
        for line in f:
            text = json.loads(line.strip())["question"]
            enc = tokenizer(text, max_length=max_len, truncation=True,
                            return_tensors="pt")
            samples.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
            })
            if len(samples) >= max_samples:
                break

    if len(samples) < max_samples:
        orig = samples.copy()
        while len(samples) < max_samples:
            samples.extend(orig)
        samples = samples[:max_samples]

    total_tokens = sum(s["input_ids"].shape[1] for s in samples)
    avg_len = total_tokens / len(samples)
    print(f"[INFO] {len(samples)} calibration samples, "
          f"{total_tokens:,} tokens (avg {avg_len:.0f}, max_len={max_len})")
    return samples


def find_all_linears(module: nn.Module) -> Dict[str, nn.Linear]:
    return {n: m for n, m in module.named_modules() if isinstance(m, nn.Linear)}


# ====================================================================
# S8  Main Pipeline
# ====================================================================

def quantize_model(args):
    device = torch.device("cuda:0")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"[INFO] GPU: {gpu} (SM{cap[0]}{cap[1]}, {vram:.0f} GB)")

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    tokenizer = AutoTokenizer.from_pretrained(args.input, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 2

    calib_data = load_calibration_data(
        tokenizer, args.calib_data, args.max_samples, args.max_len
    )

    print(f"[INFO] Loading model (CPU, bf16)...")
    config = AutoConfig.from_pretrained(args.input, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.input, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="cpu", config=config,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    model.config.use_cache = False

    num_layers = config.num_hidden_layers
    mixer_types = config.mixer_types

    quant_layers = [
        i for i, mt in enumerate(mixer_types)
        if mt in ("lightning", "lightning_attn", "lightning-attn")
    ]
    skip_layers = [i for i in range(num_layers) if i not in quant_layers]
    print(f"[INFO] GPTQ-NVFP4: {len(quant_layers)} layers | "
          f"BF16: {len(skip_layers)} layers")

    # -------- Phase 1: Embedding ---------------------------------
    print("\n[Phase 1] Embedding forward...")
    embed = model.model.embed_tokens.to(device)
    hs_store = HiddenStateStore(args.tmp_dir)
    all_masks: List[torch.Tensor] = []
    all_pos_ids: List[torch.Tensor] = []

    with torch.no_grad():
        for i, sample in enumerate(calib_data):
            ids = sample["input_ids"].to(device)
            mask = sample["attention_mask"].to(device)
            pos = mask.long().cumsum(-1) - 1
            pos.masked_fill_(mask == 0, 1)
            h = embed(ids) * config.scale_emb
            hs_store.save(i, h)
            all_masks.append(mask.cpu())
            all_pos_ids.append(pos.cpu())
            del h, ids
            if (i + 1) % 8 == 0:
                nuke_caches()

    embed = embed.cpu()
    nuke_caches()
    print(f"  {len(hs_store)} hidden states -> {args.tmp_dir}")

    # -------- Phase 2: Layer-by-layer ----------------------------
    quantized_tensors: Dict[str, torch.Tensor] = {}
    original_tensors: Dict[str, torch.Tensor] = {}
    original_tensors["model.embed_tokens.weight"] = (
        model.model.embed_tokens.weight.data.clone()
    )

    quality_log: List[Dict] = []

    for layer_idx in range(num_layers):
        t0 = time.time()
        mt = mixer_types[layer_idx]
        is_quant = layer_idx in quant_layers
        tag = "GPTQ-NVFP4" if is_quant else "BF16"

        print(f"\n{'='*70}")
        print(f"[Layer {layer_idx}/{num_layers-1}] {mt} -> {tag}")
        print(f"{'='*70}")

        layer = model.model.layers[layer_idx].to(device)
        layer.eval()

        if is_quant:
            _quantize_layer(
                layer, layer_idx, config, device, calib_data,
                hs_store, all_masks, all_pos_ids, args,
                quantized_tensors, original_tensors, quality_log,
            )
        else:
            print(f"  Keeping BF16 (minicpm4 sparse layer)")
            for name, param in layer.named_parameters():
                full_name = f"model.layers.{layer_idx}.{name}"
                original_tensors[full_name] = param.data.cpu().clone()

        # -- Propagate hidden states --
        print(f"  Propagating hidden states...")
        nuke_caches()
        with torch.no_grad():
            for i in range(len(calib_data)):
                inp = hs_store.load(i, device)
                mask = all_masks[i].to(device)
                pos = all_pos_ids[i].to(device)
                try:
                    out = layer(
                        inp, attention_mask=mask, position_ids=pos,
                        past_key_value=None, output_attentions=False,
                        use_cache=False,
                    )
                    hs_store.save(i, out[0])
                except Exception as e:
                    print(f"  [WARN] Sample {i}: {e}")
                del inp
                if (i + 1) % 4 == 0:
                    nuke_caches()

        layer = layer.cpu()
        model.model.layers[layer_idx] = layer
        nuke_caches()
        print(f"  Layer {layer_idx} done in {time.time() - t0:.1f}s")

    # -------- Phase 3: Final -------------------------------------
    print(f"\n[Phase 3] Final norm + lm_head...")
    original_tensors["model.norm.weight"] = (
        model.model.norm.weight.data.cpu().clone()
    )
    if not config.tie_word_embeddings:
        original_tensors["lm_head.weight"] = (
            model.lm_head.weight.data.cpu().clone()
        )

    # -------- Phase 4: Save --------------------------------------
    print(f"\n[Phase 4] Saving to {args.output}...")
    save_checkpoint(
        quantized_tensors, original_tensors, args, config,
        quant_layers, skip_layers, quality_log,
    )
    hs_store.cleanup()
    print("\n[DONE] GPTQ-enhanced NVFP4 quantisation complete!")


# ====================================================================
# S9  Per-Layer Quantisation
# ====================================================================

def _quantize_layer(
    layer, layer_idx, config, device, calib_data,
    hs_store, all_masks, all_pos_ids, args,
    quantized_tensors, original_tensors, quality_log,
):
    """GPTQ-NVFP4 quantise all linears in a single decoder layer."""

    linears = find_all_linears(layer)
    linear_names = sorted(linears.keys())
    print(f"  Linears: {linear_names}")

    quantizers: Dict[str, GPTQ_FP4] = {}
    act_amax: Dict[str, float] = defaultdict(float)

    for ln_name, ln_mod in linears.items():
        full_name = f"model.layers.{layer_idx}.{ln_name}"
        quantizers[ln_name] = GPTQ_FP4(full_name, ln_mod.weight.data)

    # -- Hooks: Hessian + activation amax --
    hooks = []
    for ln_name in linear_names:
        ln_mod = linears[ln_name]

        def make_hook(name):
            def hook_fn(module, inp, out):
                x = inp[0].data
                quantizers[name].add_batch(x)
                act_amax[name] = max(act_amax[name], x.abs().max().item())
            return hook_fn

        hooks.append(ln_mod.register_forward_hook(make_hook(ln_name)))

    # -- Collect Hessians --
    n_samples = len(calib_data)
    print(f"  Collecting Hessians ({n_samples} samples)...")
    nuke_caches()
    with torch.no_grad():
        for i in range(n_samples):
            inp = hs_store.load(i, device)
            mask = all_masks[i].to(device)
            pos = all_pos_ids[i].to(device)
            try:
                layer(
                    inp, attention_mask=mask, position_ids=pos,
                    past_key_value=None, output_attentions=False,
                    use_cache=False,
                )
            except Exception as e:
                print(f"  [WARN] Sample {i}: {e}")
            del inp
            if (i + 1) % 4 == 0:
                nuke_caches()

    for h in hooks:
        h.remove()

    # -- Quantise each linear --
    print(f"  Running GPTQ-FP4 quantisation...")
    for ln_name in linear_names:
        full_name = f"model.layers.{layer_idx}.{ln_name}"
        ln_mod = linears[ln_name]
        W = ln_mod.weight.data.to(device)
        N, K = W.shape

        w_amax = W.abs().max().float().clamp(min=1e-12)
        global_sf = torch.tensor(
            [NVFP4_SCALE_FACTOR / w_amax.item()],
            dtype=torch.float32, device=device,
        )
        w_scale_2_ckpt = torch.tensor(
            [w_amax.item() / NVFP4_SCALE_FACTOR], dtype=torch.float32
        )

        t1 = time.time()
        Q_deq, e2m1_vals, bscales = quantizers[ln_name].quantize(
            W, global_sf, damp_percent=args.damp, block_size=128,
        )
        dt = time.time() - t1

        # Quality metrics
        gptq_mse = (Q_deq - W.float()).pow(2).mean().item()
        gptq_rel = ((Q_deq - W.float()).norm() / W.float().norm()).item()
        rtn_mse = rtn_nvfp4_mse(W, global_sf.item())
        improvement = (
            (rtn_mse - gptq_mse) / rtn_mse * 100
            if rtn_mse > 1e-20 else 0.0
        )

        print(f"    {ln_name} [{N}x{K}] {dt:.1f}s")
        print(f"      GPTQ: MSE={gptq_mse:.3e}  RelErr={gptq_rel:.5f}")
        print(f"      RTN:  MSE={rtn_mse:.3e}")
        print(f"      Improvement: {improvement:+.1f}%")

        # Replace weight for next-layer calibration
        # dotv
        # import modelopt.torch.quantization as mtq
        
        # # 将 GPTQ 还原后的浮点权重赋给当前层
        # ln_mod.weight.data = Q_deq.to(ln_mod.weight.dtype).to(ln_mod.weight.device)
        
        # # 提取当前层计算出的最大激活值 (用于后续的 input_scale)
        # a_max = max(act_amax.get(ln_name, 1.0), 1e-12)

        # # 构造 ModelOpt 需要的量化配置字典 (强制走 NVFP4 + AWQ/GPTQ 兼容模式)
        # quant_config = {
        #     "quant_cfg": {
        #         "*weight_quantizer": {"num_bits": 4, "block_sizes": {-1: 16}, "enable": True},
        #         "*input_quantizer":  {"num_bits": 4, "block_sizes": {-1: 16}, "enable": True},
        #     },
        #     "algorithm": "awq" # 这里填awq是为了让modelopt接受静态的input_scale
        # }

        # # 让 modelopt 接管这个层，它会自动执行底层的 Swizzle 内存重排并转成 FP8 scales
        # # 我们传入 dummy 的前向函数，因为我们只需要它的打包功能，不需要它重量化
        # def dummy_forward(layer):
        #     pass

        # quantized_layer = mtq.quantize(ln_mod, quant_config, forward_loop=dummy_forward)

        # # 强制塞入我们计算好的最优激活值最大值
        # # 注意：modelopt 内部通常使用 amax 来推导 scale
        # for name, module in quantized_layer.named_modules():
        #     if hasattr(module, 'input_quantizer') and hasattr(module.input_quantizer, '_amax'):
        #          module.input_quantizer._amax.data = torch.tensor(a_max, dtype=torch.float32, device=device)

        # # 由于我们要导出完整的 checkpoint，我们不在这里单独存 state_dict
        # # 我们把量化后且经过 modelopt 包装的层，塞回到原来的模型里
        # # 下面的 original_tensors 收集环节，modelopt_export 会接管
        # setattr(layer, ln_name.split('.')[-1], quantized_layer)
        ln_mod.weight.data = Q_deq.to(ln_mod.weight.dtype).to(
            ln_mod.weight.device
        )

        # Direct bitwise packing (ZERO re-quantisation)
        packed_fp4 = pack_e2m1_to_uint8(e2m1_vals)

        block_sf_fp8 = bscales.clamp(
            min=torch.finfo(torch.float8_e4m3fn).tiny,
            max=FP8_E4M3_MAX,
        ).to(torch.float8_e4m3fn).cpu()

        a_max = max(act_amax.get(ln_name, 1.0), 1e-12)
        a_scale_ckpt = torch.tensor(
            [a_max / NVFP4_SCALE_FACTOR], dtype=torch.float32
        )

        assert packed_fp4.shape == (N, K // 2)
        assert block_sf_fp8.shape == (N, K // NVFP4_GROUP_SIZE)

        quantized_tensors[f"{full_name}.weight"] = packed_fp4.cpu()
        quantized_tensors[f"{full_name}.weight_scale"] = block_sf_fp8
        quantized_tensors[f"{full_name}.weight_scale_2"] = w_scale_2_ckpt
        quantized_tensors[f"{full_name}.input_scale"] = a_scale_ckpt

        quality_log.append({
            "layer": layer_idx, "linear": ln_name, "shape": f"{N}x{K}",
            "gptq_mse": gptq_mse, "gptq_rel_err": gptq_rel,
            "rtn_mse": rtn_mse, "improvement_pct": improvement,
        })

        del W, Q_deq, e2m1_vals, bscales
        gc.collect()

    # Save non-quantised params
    for name, param in layer.named_parameters():
        full_name = f"model.layers.{layer_idx}.{name}"
        is_qparam = any(
            name.startswith(ln + ".weight") or name.startswith(ln + ".bias")
            for ln in linear_names
        )
        if not is_qparam:
            original_tensors[full_name] = param.data.cpu().clone()


# ====================================================================
# S10  Checkpoint Saving
# ====================================================================

def save_checkpoint(
    quantized_tensors, original_tensors, args, config,
    quant_layers, skip_layers, quality_log,
):
    os.makedirs(args.output, exist_ok=True)

    state_dict = {}
    state_dict.update(quantized_tensors)
    state_dict.update(original_tensors)

    from safetensors.torch import save_file

    MAX_SHARD = 4 * 1024 * 1024 * 1024
    total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
    print(f"  Total: {total_size / 1024**3:.2f} GB")

    if total_size <= MAX_SHARD:
        save_file(state_dict, os.path.join(args.output, "model.safetensors"))
        print(f"  Saved model.safetensors")
    else:
        shards, current, cur_size, idx = {}, {}, 0, 1
        weight_map = {}
        for name in sorted(state_dict.keys()):
            t = state_dict[name]
            tsz = t.numel() * t.element_size()
            if cur_size + tsz > MAX_SHARD and current:
                sname = f"model-{idx:05d}-of-XXXXX.safetensors"
                shards[sname] = current
                current, cur_size, idx = {}, 0, idx + 1
            current[name] = t
            cur_size += tsz
        if current:
            shards[f"model-{idx:05d}-of-XXXXX.safetensors"] = current
        total_shards = len(shards)
        for old_name, data in list(shards.items()):
            new_name = old_name.replace("XXXXX", f"{total_shards:05d}")
            save_file(data, os.path.join(args.output, new_name))
            for tn in data:
                weight_map[tn] = new_name
            print(f"  Saved {new_name}")
        with open(os.path.join(args.output,
                               "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total_size},
                       "weight_map": weight_map}, f, indent=2)

    exclude = [f"model.layers.{i}.*" for i in skip_layers]

    with open(os.path.join(args.output, "hf_quant_config.json"), "w") as f:
        json.dump({"quantization": {
            "quant_algo": "NVFP4", "kv_cache_quant_algo": "auto",
            "group_size": NVFP4_GROUP_SIZE, "exclude_modules": exclude,
        }}, f, indent=2)

    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "generation_config.json"]:
        src = os.path.join(args.input, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output, fname))
    for fname in glob.glob(os.path.join(args.input, "*.py")):
        shutil.copy2(fname, os.path.join(args.output, os.path.basename(fname)))

    cfg_path = os.path.join(args.output, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["quantization_config"] = {
            "quant_method": "modelopt", "quant_algo": "NVFP4",
            "kv_cache_quant_algo": "auto", "group_size": NVFP4_GROUP_SIZE,
            "exclude_modules": exclude,
        }
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

    report_path = os.path.join(args.output, "gptq_quality_report.json")
    with open(report_path, "w") as f:
        json.dump(quality_log, f, indent=2)

    if quality_log:
        improvements = [q["improvement_pct"] for q in quality_log]
        gptq_mses = [q["gptq_mse"] for q in quality_log]
        rtn_mses = [q["rtn_mse"] for q in quality_log]
        print()
        print(f"  +----------------------------------------------+")
        print(f"  |  GPTQ-NVFP4 Quality Summary                  |")
        print(f"  +----------------------------------------------+")
        print(f"  |  Linears quantised:  {len(quality_log):<22d} |")
        print(f"  |  Avg GPTQ MSE:       {np.mean(gptq_mses):.4e}            |")
        print(f"  |  Avg RTN  MSE:       {np.mean(rtn_mses):.4e}            |")
        print(f"  |  Avg improvement:    {np.mean(improvements):+.1f}%                |")
        print(f"  |  Min improvement:    {min(improvements):+.1f}%                |")
        print(f"  |  Max improvement:    {max(improvements):+.1f}%                |")
        print(f"  +----------------------------------------------+")

    print(f"  All saved to {args.output}")


# ====================================================================
# S11  CLI
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GPTQ-Enhanced NVFP4 Quantisation for MiniCPM-SALA"
    )
    parser.add_argument("--input", required=True,
                        help="Original BF16 model path")
    parser.add_argument("--output", required=True,
                        help="Output checkpoint path")
    parser.add_argument("--calib-data", required=True,
                        help="JSONL calibration data ('question' field)")
    parser.add_argument("--max-samples", type=int, default=64,
                        help="Calibration samples (default: 64)")
    parser.add_argument("--max-len", type=int, default=8192,
                        help="Max sequence length (default: 8192). "
                             "At K=16384 the Hessian is 2 GB (f64). "
                             "Activation chunks capped at 8192 rows "
                             "per GPU matmul (~512 MB at K=16384).")
    parser.add_argument("--damp", type=float, default=0.01,
                        help="Hessian damping percent (default: 0.01)")
    parser.add_argument("--tmp-dir", type=str,
                        default="/tmp/nvfp4_gptq_hs",
                        help="Temp dir for hidden-state disk offload")

    args = parser.parse_args()

    print("=" * 70)
    print("  GPTQ-Enhanced NVFP4 Quantiser for MiniCPM-SALA")
    print("  -----------------------------------------------")
    print("  Algorithm : Hessian error compensation + FP4 E2M1 grid")
    print("  Packing   : Direct bitwise (zero re-quantisation)")
    print("  Scales    : Pre-computed from W_orig (no propagation lag)")
    print("  Hessian   : Chunked accumulation (8192 rows / GPU matmul)")
    print("=" * 70)
    print(f"  Input      : {args.input}")
    print(f"  Output     : {args.output}")
    print(f"  Samples    : {args.max_samples}")
    print(f"  Max len    : {args.max_len}")
    print(f"  Damp       : {args.damp}")
    print(f"  Tmp dir    : {args.tmp_dir}")

    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.input, trust_remote_code=True)
        nvfp4_n = sum(
            1 for mt in cfg.mixer_types
            if mt in ("lightning", "lightning_attn", "lightning-attn")
        )
        bf16_n = len(cfg.mixer_types) - nvfp4_n
        print(f"\n  Plan: {nvfp4_n} GPTQ-NVFP4 | {bf16_n} BF16")
        for i, mt in enumerate(cfg.mixer_types):
            t = "NVFP4" if mt in ("lightning", "lightning_attn",
                                   "lightning-attn") else "BF16 "
            print(f"    [{i:2d}] {mt:<20s} -> {t}")
    except Exception:
        pass

    print("=" * 70)
    t0 = time.time()
    quantize_model(args)
    elapsed = time.time() - t0
    print(f"\n[INFO] Total: {elapsed / 60:.1f} min ({elapsed / 3600:.1f} hr)")


if __name__ == "__main__":
    main()