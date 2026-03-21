#!/usr/bin/env python3
"""
GPTQ-Enhanced NVFP4 Mixed-Precision Quantization for MiniCPM-SALA (v2)
======================================================================

Fixes from MR-GPTQ (Egiazarian et al., ICLR 2026):
  1. MSE-optimized block scales (100-iteration grid search, Lp norm p=2.4)
  2. Fused global_scale across merged shards (qkv, gate_up) using min()
  3. FP8 round-trip for block scales during GPTQ (matches inference precision)

Strategy:
  - Lightning-attn layers (24/32):  GPTQ-NVFP4 (Hessian + E2M1 grid + MSE scales)
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

# E2M1 Grid (15 distinct values, sorted)
_E2M1_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
_SORTED_GRID = torch.cat([-_E2M1_POS.flip(0)[:-1], _E2M1_POS])  # 15 values
_MIDPOINTS = (_SORTED_GRID[:-1] + _SORTED_GRID[1:]) / 2.0

# Grid-index -> 4-bit E2M1 hardware code (for packing)
_GRID_IDX_TO_4BIT = np.array(
    [15, 14, 13, 12, 11, 10, 9, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint8
)

HESSIAN_CHUNK_ROWS = 8192


# ====================================================================
# S1  E2M1 Quantisation (O(1) extra memory via bucketize)
# ====================================================================

def quantize_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Snap every element to nearest FP4-E2M1 grid value."""
    midpoints = _MIDPOINTS.to(x.device, dtype=x.dtype)
    grid = _SORTED_GRID.to(x.device, dtype=x.dtype)
    idx = torch.bucketize(x.clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX), midpoints)
    return grid[idx]


# ====================================================================
# S2  E2M1 Bitwise Packer
# ====================================================================

def pack_e2m1_to_uint8(e2m1_vals: torch.Tensor) -> torch.Tensor:
    """Pack E2M1 float values -> uint8 (2 values per byte, low nibble first).
    Matches NVIDIA FP4-E2M1X2 layout.
    """
    N, K = e2m1_vals.shape
    assert K % 2 == 0, f"K={K} must be even for FP4 packing"
    midpoints = _MIDPOINTS.to(e2m1_vals.device, dtype=e2m1_vals.dtype)
    flat = e2m1_vals.reshape(-1).clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)
    grid_idx = torch.bucketize(flat, midpoints).cpu().numpy()
    codes = _GRID_IDX_TO_4BIT[grid_idx].reshape(N, K)
    lo = codes[:, 0::2]
    hi = codes[:, 1::2]
    packed = (hi.astype(np.uint16) << 4) | lo
    return torch.from_numpy(packed.astype(np.uint8))


# ====================================================================
# S3  MSE-Optimized Block Scale Search (from MR-GPTQ)
# ====================================================================

def compute_block_scales_mse(
    W: torch.Tensor,
    global_sf: float,
    group_size: int = NVFP4_GROUP_SIZE,
    search_iters: int = 100,
    max_shrink: float = 0.80,
    error_norm: float = 2.4,
) -> torch.Tensor:
    """Compute per-group block scales with MSE grid search.
    
    Instead of using absmax (which wastes dynamic range on outliers),
    we search over shrink factors [1.0, ..., 1-max_shrink] and pick
    the one that minimizes Lp reconstruction error per group.
    
    Then we quantize scales to FP8 E4M3 and back (round-trip),
    so GPTQ sees the actual inference-time scale precision.
    
    Args:
        W: weight tensor [N, K] in float32
        global_sf: global scale factor (2688 / max_amax_across_merged_shards)
        group_size: NVFP4 group size (16)
        search_iters: number of shrink factors to try
        max_shrink: maximum shrink from absmax (0.80 = try down to 20% of absmax)
        error_norm: Lp norm exponent for error measurement
    
    Returns:
        block_scales: [N, K//group_size] float32, already FP8-round-tripped
    """
    N, K = W.shape
    num_groups = K // group_size
    
    # Reshape to [N, num_groups, group_size]
    W_grouped = W.reshape(N, num_groups, group_size)
    
    # Initial absmax scales: [N, num_groups]
    init_scales = W_grouped.abs().amax(dim=-1) * global_sf / FP4_E2M1_MAX
    init_scales = init_scales.clamp(min=torch.finfo(torch.float8_e4m3fn).tiny)
    
    # Best tracking
    best_scales = init_scales.clone()
    best_error = torch.full((N, num_groups), float('inf'), device=W.device, dtype=W.dtype)
    
    for i in range(search_iters):
        shrink = 1.0 - i * max_shrink / search_iters
        candidate_scales = shrink * init_scales
        
        # FP8 round-trip on candidate scales (critical for matching inference)
        # NOTE: init_scales = group_amax * global_sf / FP4_MAX
        #   which is already in FP8-representable range [0, 448].
        #   Do NOT multiply by global_sf again (that was the v2.0 bug).
        candidate_scales_fp8 = candidate_scales.clamp(
            min=torch.finfo(torch.float8_e4m3fn).tiny,
            max=FP8_E4M3_MAX,
        ).to(torch.float8_e4m3fn).to(torch.float32)
        candidate_scales_fp8 = candidate_scales_fp8.clamp(min=1e-12)
        
        # Quantize and dequantize with candidate scales
        W_scaled = W_grouped * global_sf / candidate_scales_fp8.unsqueeze(-1)
        W_q = quantize_to_e2m1(W_scaled)
        W_deq = W_q * candidate_scales_fp8.unsqueeze(-1) / global_sf
        
        # Compute per-group Lp error
        error = (W_grouped - W_deq).abs().pow(error_norm).sum(dim=-1)
        
        # Update where improved
        improved = error < best_error
        best_error[improved] = error[improved]
        best_scales[improved] = candidate_scales_fp8[improved]
    
    return best_scales


# ====================================================================
# S4  GPTQ Core with FP4-E2M1 Grid + MSE Scales + FP8 Round-Trip
# ====================================================================

class GPTQ_FP4:
    """GPTQ quantiser using the FP4-E2M1 non-uniform grid.
    
    Key differences from v1:
      1. Block scales computed via MSE search (not absmax)
      2. Block scales FP8-round-tripped before GPTQ loop
      3. global_sf passed in (fused across merged shards)
    """

    def __init__(self, name: str, weight: torch.Tensor):
        self.name = name
        self.rows, self.columns = weight.shape
        self.H = torch.zeros(
            (self.columns, self.columns), device="cpu", dtype=torch.float64
        )
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor):
        """Accumulate Hessian H += X^T X (chunked)."""
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        elif inp.dim() == 1:
            inp = inp.unsqueeze(0)
        n_tokens = inp.shape[0]
        self.nsamples += n_tokens
        for start in range(0, n_tokens, HESSIAN_CHUNK_ROWS):
            end = min(start + HESSIAN_CHUNK_ROWS, n_tokens)
            chunk = inp[start:end].float()
            h_chunk = chunk.T @ chunk
            self.H.add_(h_chunk.cpu().to(torch.float64))
            del chunk, h_chunk

    def quantize(
        self,
        weight: torch.Tensor,
        global_sf: float,
        damp_percent: float = 0.01,
        block_size: int = 128,
        mse_iters: int = 100,
        mse_max_shrink: float = 0.80,
        mse_error_norm: float = 2.4,
        act_order: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run GPTQ with FP4-E2M1 grid, MSE-optimized scales, and ActOrder.

        ActOrder (activation reordering): quantize columns in order of
        decreasing Hessian diagonal (= decreasing activation magnitude).
        This ensures high-impact columns are quantized first with minimal
        accumulated error, while low-impact columns absorb the residual.
        After GPTQ, columns are restored to original order for checkpoint.

        Returns:
            Q_deq        [N, K]       dequantised weights (original col order)
            e2m1_values  [N, K]       E2M1 grid floats    (original col order)
            block_scales [N, K//16]   float32 block scales (FP8-round-tripped)
        """
        dev = weight.device
        W = weight.float().clone()
        H = (self.H / self.nsamples).float().to(dev)
        del self.H

        rows, columns = W.shape
        group_size = NVFP4_GROUP_SIZE
        num_groups = columns // group_size

        # -- Hessian prep -----------------------------------------------
        dead = torch.diag(H) == 0
        H[dead, dead] = 1.0
        W[:, dead] = 0.0
        damp = damp_percent * torch.diag(H).mean()
        H.diagonal().add_(damp)

        # -- ActOrder: sort columns by Hessian diagonal (descending) ----
        if act_order:
            perm = torch.argsort(torch.diag(H), descending=True)
            # group_idx[j] = which original group permuted column j belongs to
            group_idx = torch.arange(num_groups, device=dev).repeat_interleave(group_size)[perm]
        else:
            perm = torch.arange(columns, device=dev)
            group_idx = None
        perm_inv = torch.argsort(perm)

        # Permute H and W by columns
        H = H[perm][:, perm]
        W = W[:, perm]

        # -- Hessian inverse (on permuted H) ----------------------------
        Hinv = self._make_hinv(H)
        del H

        # -- MSE-optimized block scales (from ORIGINAL column order) ----
        # Scales are computed on un-permuted weights, since the checkpoint
        # stores them in original group order. During the GPTQ loop we
        # use group_idx to look up the correct scale for each permuted col.
        W_orig_order = W[:, perm_inv]
        block_scales = compute_block_scales_mse(
            W_orig_order, global_sf, group_size,
            search_iters=mse_iters,
            max_shrink=mse_max_shrink,
            error_norm=mse_error_norm,
        )
        del W_orig_order

        # -- GPTQ block-wise loop (on permuted columns) ----------------
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
                # ActOrder: look up original group for this permuted column
                if act_order:
                    g = group_idx[col].item()
                else:
                    g = col // group_size
                bsf_col = block_scales[:, g]  # MSE-optimized, FP8-round-tripped

                w = W_block[:, j]
                d = Hinv_blk[j, j]

                # Forward quantise -> snap to E2M1 -> dequantise
                w_scaled = w * global_sf / bsf_col
                q_e2m1 = quantize_to_e2m1(w_scaled)
                w_deq = q_e2m1 * bsf_col / global_sf

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

        # -- Invert permutation: restore original column order ----------
        Q_deq = Q_deq[:, perm_inv].contiguous()
        e2m1_vals = e2m1_vals[:, perm_inv].contiguous()

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
# S5  RTN Baseline MSE (for quality comparison)
# ====================================================================

def rtn_nvfp4_mse(W: torch.Tensor, gsf: float) -> float:
    """MSE of plain RTN FP4-E2M1 quantisation (absmax, no Hessian)."""
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
# S6  Cache Management
# ====================================================================

def nuke_caches():
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
# S7  VRAM-Backed Hidden-State Store
# ====================================================================

class HiddenStateStore:
    def __init__(self, tmp_dir: str):
        self._states: Dict[int, torch.Tensor] = {}

    def save(self, idx: int, tensor: torch.Tensor):
        self._states[idx] = tensor.detach().clone()

    def load(self, idx: int, device: torch.device) -> torch.Tensor:
        return self._states[idx]

    def cleanup(self):
        self._states.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __len__(self):
        return len(self._states)


# ====================================================================
# S8  Calibration Data
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
# S9  Main Pipeline
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
    print(f"  {len(hs_store)} hidden states cached")

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
# S10  Per-Layer Quantisation (with MR-GPTQ fixes)
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

    # ================================================================
    # FIX 2: Compute fused global_sf across merged shards using min()
    # ================================================================
    # For NVFP4: global_sf = 2688 / w_amax
    # Merged shards (qkv, gate_up) must share the SAME global_sf.
    # Use min(global_sf) = 2688 / max(w_amax) across merged shards.
    # This matches what SGLang does at load time: weight_scale_2.max()

    # Define merge groups
    merge_groups = {
        "qkv": [n for n in linear_names if any(
            n.endswith(s) for s in ["q_proj", "k_proj", "v_proj"]
        )],
        "gate_up": [n for n in linear_names if any(
            n.endswith(s) for s in ["gate_proj", "up_proj"]
        )],
    }
    # Standalone linears (o_proj, down_proj, z_proj, etc.)
    merged_names = set()
    for members in merge_groups.values():
        merged_names.update(members)
    standalone = [n for n in linear_names if n not in merged_names]

    # Compute per-linear w_amax
    per_linear_amax = {}
    for ln_name in linear_names:
        W = linears[ln_name].weight.data.to(device)
        per_linear_amax[ln_name] = W.abs().max().float().item()

    # Compute fused global_sf per group
    fused_global_sf: Dict[str, float] = {}

    for group_name, members in merge_groups.items():
        if not members:
            continue
        max_amax = max(per_linear_amax[m] for m in members)
        gsf = NVFP4_SCALE_FACTOR / max(max_amax, 1e-12)
        for m in members:
            fused_global_sf[m] = gsf
        print(f"    [{group_name}] fused global_sf={gsf:.4f} "
              f"(max_amax={max_amax:.6f} from {members})")

    for ln_name in standalone:
        amax = per_linear_amax[ln_name]
        fused_global_sf[ln_name] = NVFP4_SCALE_FACTOR / max(amax, 1e-12)

    # ================================================================
    # Quantise each linear with fused global_sf + MSE scales
    # ================================================================
    print(f"  Running GPTQ-FP4 quantisation (MSE scales + FP8 round-trip)...")
    for ln_name in linear_names:
        full_name = f"model.layers.{layer_idx}.{ln_name}"
        ln_mod = linears[ln_name]
        W = ln_mod.weight.data.to(device)
        N, K = W.shape

        global_sf = fused_global_sf[ln_name]
        w_amax_fused = NVFP4_SCALE_FACTOR / global_sf
        w_scale_2_ckpt = torch.tensor(
            [w_amax_fused / NVFP4_SCALE_FACTOR], dtype=torch.float32
        )

        t1 = time.time()
        Q_deq, e2m1_vals, bscales = quantizers[ln_name].quantize(
            W, global_sf,
            damp_percent=args.damp,
            block_size=128,
            mse_iters=args.mse_iters,
            mse_max_shrink=args.mse_max_shrink,
            mse_error_norm=args.mse_error_norm,
            act_order=not args.no_act_order,
        )
        dt = time.time() - t1

        # Quality metrics
        gptq_mse = (Q_deq - W.float()).pow(2).mean().item()
        gptq_rel = ((Q_deq - W.float()).norm() / W.float().norm()).item()
        rtn_mse = rtn_nvfp4_mse(W, global_sf)
        improvement = (
            (rtn_mse - gptq_mse) / rtn_mse * 100
            if rtn_mse > 1e-20 else 0.0
        )

        print(f"    {ln_name} [{N}x{K}] {dt:.1f}s")
        print(f"      GPTQ: MSE={gptq_mse:.3e}  RelErr={gptq_rel:.5f}")
        print(f"      RTN:  MSE={rtn_mse:.3e}")
        print(f"      Improvement: {improvement:+.1f}%")

        # Replace weight for next-layer calibration
        ln_mod.weight.data = Q_deq.to(ln_mod.weight.dtype).to(
            ln_mod.weight.device
        )

        # Direct bitwise packing (ZERO re-quantisation)
        packed_fp4 = pack_e2m1_to_uint8(e2m1_vals)

        # Block scales -> FP8 for checkpoint
        # bscales = group_amax * global_sf / FP4_MAX, already in [0, 448]
        # Do NOT multiply by global_sf again.
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

    # Save non-quantised params (layernorm, etc.)
    for name, param in layer.named_parameters():
        full_name = f"model.layers.{layer_idx}.{name}"
        is_qparam = any(
            name.startswith(ln + ".weight") or name.startswith(ln + ".bias")
            for ln in linear_names
        )
        if not is_qparam:
            original_tensors[full_name] = param.data.cpu().clone()


# ====================================================================
# S11  Checkpoint Saving
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
        print(f"  |  GPTQ-NVFP4 Quality Summary (v2 + MR-GPTQ)  |")
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
# S12  CLI
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GPTQ-Enhanced NVFP4 Quantisation for MiniCPM-SALA (v2)"
    )
    parser.add_argument("--input", required=True,
                        help="Original BF16 model path")
    parser.add_argument("--output", required=True,
                        help="Output checkpoint path")
    parser.add_argument("--calib-data", required=True,
                        help="JSONL calibration data ('question' field)")
    parser.add_argument("--max-samples", type=int, default=64,
                        help="Calibration samples (default: 64)")
    parser.add_argument("--max-len", type=int, default=65536,
                        help="Max sequence length for calibration. "
                             "Should match eval context length. "
                             "Your eval set uses ~60-74k tokens. "
                             "Use 131072 if memory allows.")
    parser.add_argument("--damp", type=float, default=0.01,
                        help="Hessian damping percent (default: 0.01)")
    parser.add_argument("--mse-iters", type=int, default=100,
                        help="MSE scale search iterations (default: 100)")
    parser.add_argument("--mse-max-shrink", type=float, default=0.80,
                        help="MSE max shrink factor (default: 0.80)")
    parser.add_argument("--mse-error-norm", type=float, default=2.4,
                        help="Lp error norm for MSE search (default: 2.4)")
    parser.add_argument("--tmp-dir", type=str,
                        default="/tmp/nvfp4_gptq_hs",
                        help="Temp dir for hidden-state offload")
    parser.add_argument("--no-act-order", action="store_true",
                        help="Disable ActOrder (activation reordering). "
                             "ActOrder is ON by default and typically "
                             "gives 1-1.5%% accuracy improvement.")

    args = parser.parse_args()

    print("=" * 70)
    print("  GPTQ-Enhanced NVFP4 Quantiser for MiniCPM-SALA (v2)")
    print("  ====================================================")
    print("  Fixes from MR-GPTQ (Egiazarian et al., ICLR 2026):")
    print("    1. MSE-optimized block scales (100-iter grid search)")
    print("    2. Fused global_sf across merged shards (min)")
    print("    3. FP8 round-trip for block scales during GPTQ")
    print("    4. ActOrder (quantize high-activation cols first)")
    print("=" * 70)
    print(f"  Input      : {args.input}")
    print(f"  Output     : {args.output}")
    print(f"  Samples    : {args.max_samples}")
    print(f"  Max len    : {args.max_len}")
    print(f"  Damp       : {args.damp}")
    print(f"  MSE iters  : {args.mse_iters}")
    print(f"  MSE shrink : {args.mse_max_shrink}")
    print(f"  MSE norm   : {args.mse_error_norm}")
    print(f"  ActOrder   : {'OFF' if args.no_act_order else 'ON'}")

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