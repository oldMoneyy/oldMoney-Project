#!/usr/bin/env python3
"""
Pure GPTQ W4A16 quantization for MiniCPM-SALA.

Bypasses GPTQModel entirely. Uses the model's own forward pass (trust_remote_code)
so all hybrid lightning-attn + minicpm4 logic is handled natively.

Output is in AutoGPTQ-compatible format (safetensors + quantize_config.json).
"""

import os
import gc
import sys
import json
import math
import time
import shutil
import struct
import glob
import argparse
import functools
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"


# ============== 修改后的配置块 ==============
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def seed_everything(seed=None):
    if seed is None:
        seed = random.randint(0, 99999) # 恢复抽卡模式
    
    print(f"[INFO] Using Random Seed: {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 关键修改：关闭强制确定性算子，允许 cuBLAS 使用稍微带点随机性但更优的内核
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)

# ==========================================


# ============================================================
# SECTION 1: GPTQ Algorithm (block-wise for performance)
# ============================================================

class GPTQ:
    """GPTQ quantizer for a single nn.Linear layer."""

    def __init__(self, name: str, weight: torch.Tensor):
        """
        Args:
            name: Layer name for logging
            weight: The weight tensor [out_features, in_features]
        """
        self.name = name
        self.rows, self.columns = weight.shape  # out_features, in_features
        # Accumulate on CPU in float64 for numerical stability
        self.H = torch.zeros(
            (self.columns, self.columns), device="cpu", dtype=torch.float64
        )
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor):
        """Accumulate Hessian H = X^T @ X from a batch of inputs.
        inp: [..., in_features]. GEMM runs on GPU, accumulated on CPU.
        """
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        elif inp.dim() == 1:
            inp = inp.unsqueeze(0)
        n = inp.shape[0]
        # GEMM on GPU (fast), then move to CPU and accumulate in float64 (stable)
        inp = inp.float()  # stays on whatever device it's on (GPU)
        h_update = inp.T @ inp  # GPU GEMM
        self.H.add_(h_update.cpu().to(torch.float64))
        self.nsamples += n

    def quantize(
        self,
        weight: torch.Tensor,
        bits: int = 4,
        group_size: int = 128,
        damp_percent: float = 0.01,
        block_size: int = 128,
        sym: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run block-wise GPTQ quantization.

        Args:
            sym: If True, use symmetric quantization (required for Marlin kernels).
                 scale = max(|w|) / (2^(bits-1) - 1), zero = 2^(bits-1)

        Returns:
            Q: dequantized weight [out_features, in_features] (same dtype as input)
            scales: [num_groups, out_features] float32
            zeros: [num_groups, out_features] float32  (integer zero points as float)
            int_weight: [out_features, in_features] int32 (0..2^bits-1)
        """
        dev = weight.device
        orig_dtype = weight.dtype
        W = weight.float().clone()
        H = (self.H / self.nsamples).float().to(dev)
        del self.H

        rows, columns = W.shape
        maxq = 2 ** bits - 1

        # Handle dead columns (zero Hessian diagonal)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1.0
        W[:, dead] = 0.0

        # Damping
        damp = damp_percent * torch.diag(H).mean()
        H.diagonal().add_(damp)

        # Cholesky decomposition: H = L L^T
        # Then H^{-1} = L^{-T} L^{-1}
        # Then factor H^{-1} = U^T U (upper Cholesky)
        for attempt in range(5):
            try:
                H_chol = torch.linalg.cholesky(H)
                break
            except RuntimeError:
                extra_damp = (10 ** attempt) * 0.01 * torch.diag(H).mean()
                H.diagonal().add_(extra_damp)
                print(f"    [{self.name}] Cholesky failed, adding damp={extra_damp:.6f} (attempt {attempt+1})")
        else:
            print(f"    [{self.name}] FATAL: Cholesky failed after 5 attempts. Using diagonal approximation.")
            H_inv = torch.diag(1.0 / torch.diag(H))
            Hinv = torch.linalg.cholesky(H_inv, upper=True)
            del H, H_inv
            return self._quantize_with_hinv(W, Hinv, bits, group_size, block_size, maxq, orig_dtype, sym)

        H_inv = torch.cholesky_inverse(H_chol)
        del H_chol, H

        # Upper Cholesky of H_inv
        Hinv = torch.linalg.cholesky(H_inv, upper=True)
        del H_inv

        return self._quantize_with_hinv(W, Hinv, bits, group_size, block_size, maxq, orig_dtype, sym)

    def _quantize_with_hinv(self, W, Hinv, bits, group_size, block_size, maxq, orig_dtype, sym=True):
        rows, columns = W.shape
        dev = W.device

        num_groups = (columns + group_size - 1) // group_size
        scales = torch.zeros(num_groups, rows, device=dev, dtype=torch.float32)
        zeros = torch.zeros(num_groups, rows, device=dev, dtype=torch.float32)
        int_weight = torch.zeros(rows, columns, device=dev, dtype=torch.int32)
        Q = torch.zeros_like(W)

        for block_start in range(0, columns, block_size):
            block_end = min(block_start + block_size, columns)
            blen = block_end - block_start

            W_block = W[:, block_start:block_end].clone()
            Err_block = torch.zeros_like(W_block)
            Hinv_block_diag = Hinv[block_start:block_end, block_start:block_end]

            for j in range(blen):
                col = block_start + j
                g = col // group_size

                # Compute scale/zero at group boundary
                if col % group_size == 0:
                    g_end = min(col + group_size, columns)
                    w_group = W[:, col:g_end]
                    if sym:
                        # Symmetric: scale = max(|w|) / (2^(bits-1) - 1)
                        # zero = 2^(bits-1) (midpoint of [0, maxq])
                        half_q = maxq // 2  # e.g. 7 for 4-bit
                        wmax_abs = w_group.abs().max(dim=1).values
                        tmp_scale = (wmax_abs / half_q).clamp(min=1e-10)
                        tmp_zero = torch.full_like(tmp_scale, half_q + 1)  # e.g. 8 for 4-bit
                    else:
                        # Asymmetric: scale = (max - min) / maxq
                        wmin = w_group.min(dim=1).values
                        wmax = w_group.max(dim=1).values
                        tmp_scale = ((wmax - wmin) / maxq).clamp(min=1e-10)
                        tmp_zero = (-wmin / tmp_scale).round().clamp(0, maxq)
                    scales[g] = tmp_scale
                    zeros[g] = tmp_zero

                w = W_block[:, j]
                d = Hinv_block_diag[j, j]

                q_int = (w / scales[g] + zeros[g]).round().clamp(0, maxq)
                q_deq = (q_int - zeros[g]) * scales[g]

                int_weight[:, col] = q_int.int()
                Q[:, col] = q_deq

                err = (w - q_deq) / d
                Err_block[:, j] = err

                # Error propagation within block
                if j + 1 < blen:
                    W_block[:, j + 1:] -= err.unsqueeze(1) * Hinv_block_diag[j, j + 1:].unsqueeze(0)

            # Propagate accumulated error to remaining columns (one big GEMM)
            if block_end < columns:
                W[:, block_end:] -= Err_block @ Hinv[block_start:block_end, block_end:]

        return Q.to(orig_dtype), scales, zeros, int_weight


# ============================================================
# SECTION 2: Packing into AutoGPTQ-compatible format
# ============================================================

def pack_int_weight(int_weight: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """
    Pack quantized weights into int32.
    int_weight: [out_features, in_features] with values 0..2^bits-1
    Returns: qweight [in_features // pack_num, out_features] int32
    """
    pack_num = 32 // bits
    # Transpose: [out, in] -> [in, out]
    iw = int_weight.T.contiguous().to(torch.int32)
    in_f, out_f = iw.shape
    assert in_f % pack_num == 0, f"in_features ({in_f}) must be divisible by {pack_num}"

    iw = iw.reshape(in_f // pack_num, pack_num, out_f)
    qweight = torch.zeros(in_f // pack_num, out_f, dtype=torch.int32)
    for k in range(pack_num):
        qweight |= (iw[:, k, :] & ((1 << bits) - 1)) << (k * bits)
    return qweight


def pack_zeros(zeros_int: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """
    Pack zero points into int32.
    zeros_int: [num_groups, out_features] with values 0..2^bits-1
    Returns: qzeros [num_groups, out_features // pack_num] int32
    """
    pack_num = 32 // bits
    num_groups, out_f = zeros_int.shape
    # Pad if out_features not divisible by pack_num
    if out_f % pack_num != 0:
        pad = pack_num - (out_f % pack_num)
        zeros_int = torch.cat([
            zeros_int,
            torch.zeros(num_groups, pad, dtype=zeros_int.dtype)
        ], dim=1)
        out_f = zeros_int.shape[1]

    z = zeros_int.to(torch.int32).reshape(num_groups, out_f // pack_num, pack_num)
    qzeros = torch.zeros(num_groups, out_f // pack_num, dtype=torch.int32)
    for k in range(pack_num):
        qzeros |= (z[:, :, k] & ((1 << bits) - 1)) << (k * bits)
    return qzeros


# ============================================================
# SECTION 3: Cache clearing utilities
# ============================================================

def nuke_all_caches():
    """Aggressively clear every FLA/model cache to prevent CUDA stale tensor issues."""
    # 1. FLA tensor_cache
    try:
        from fla.utils import tensor_cache
        tensor_cache.clear()
    except Exception:
        pass

    # 2. Any lru_cache decorated functions we can find
    for obj in gc.get_objects():
        if isinstance(obj, functools._lru_cache_wrapper):
            try:
                obj.cache_clear()
            except Exception:
                pass

    # 3. Python GC
    gc.collect()

    # 4. CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================
# SECTION 4: Calibration data
# ============================================================

def load_calibration_data(
    tokenizer, data_path: str, max_samples: int, max_len: int
) -> List[Dict[str, torch.Tensor]]:
    """Load, SHUFFLE, and tokenize calibration data."""
    samples = []
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Calibration data not found: {data_path}")

    # 1. 先读取文件里的所有行
    with open(data_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    
    # 2. 关键：根据当前传入的 Seed 对数据进行打乱！
    # 这样不同的 Seed 就会抽到不同的 64 道题，或者以不同的顺序输入
    random.shuffle(all_lines)

    # 3. 选取前 max_samples 条进行 Tokenize
    for line in all_lines:
        text = json.loads(line.strip())["question"]
        enc = tokenizer(
            text,
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        )
        samples.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        })
        if len(samples) >= max_samples:
            break

    # 如果数据量不够，循环填充
    if len(samples) < max_samples:
        print(f"[WARN] Only {len(samples)} samples found, need {max_samples}. Duplicating.")
        orig = samples.copy()
        while len(samples) < max_samples:
            samples.extend(orig)
        samples = samples[:max_samples]

    print(f"[INFO] Loaded {len(samples)} calibration samples (max_len={max_len})")
    return samples


# ============================================================
# SECTION 5: Main quantization pipeline
# ============================================================

def find_all_linears(module: nn.Module) -> Dict[str, nn.Linear]:
    """Find all nn.Linear submodules (non-recursive into sub-Linears)."""
    linears = {}
    for name, mod in module.named_modules():
        if isinstance(mod, nn.Linear):
            linears[name] = mod
    return linears


def resolve_layer_bits(args, layer_idx: int, mixer_type: str) -> int:
    """Determine quantization bits for a specific layer.
    
    Priority:
      1. --layer-bits "0:4,1:8,2:16,..." (explicit per-layer override)
      2. --minicpm4-bits / --lightning-bits (per-type default)
      3. --bits (global default)
    
    Returns 16 to skip quantization (keep original weights).
    """
    # Check explicit per-layer overrides first
    if args.layer_bits:
        for spec in args.layer_bits.split(","):
            spec = spec.strip()
            if ":" in spec:
                idx_str, bits_str = spec.split(":")
                # Support ranges like "0-7:8"
                if "-" in idx_str:
                    lo, hi = idx_str.split("-")
                    if int(lo) <= layer_idx <= int(hi):
                        return int(bits_str)
                elif int(idx_str) == layer_idx:
                    return int(bits_str)

    # Check per-type defaults
    if mixer_type == "minicpm4" and args.minicpm4_bits is not None:
        return args.minicpm4_bits
    if mixer_type in ["lightning", "lightning_attn", "lightning-attn"] and args.lightning_bits is not None:
        return args.lightning_bits

    # Global default
    return args.bits


def quantize_model(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading tokenizer from {args.input}...")

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    tokenizer = AutoTokenizer.from_pretrained(args.input, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 2

    # --- Load calibration data ---
    calib_data = load_calibration_data(
        tokenizer, args.calib_data, args.max_samples, args.max_len
    )

    # --- Load model on CPU ---
    print(f"[INFO] Loading model from {args.input} (CPU)...")
    config = AutoConfig.from_pretrained(args.input, trust_remote_code=True)

    # # dotv: every input will go to the dense attention branch
    # if hasattr(config, "sparse_config") and isinstance(config.sparse_config, dict):
    #     config.sparse_config["dense_len"] = 655360


    model = AutoModelForCausalLM.from_pretrained(
        args.input,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        config=config,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    model.config.use_cache = False

    num_layers = config.num_hidden_layers
    print(f"[INFO] Model has {num_layers} layers")

    # --- Phase 1: Run embedding to get initial hidden states ---
    print("[INFO] Phase 1: Computing initial hidden states (embedding)...")
    embed = model.model.embed_tokens.to(device)
    scale_emb = config.scale_emb

    # We store hidden_states, attention_mask, position_ids for each sample on CPU
    num_inps = [0]  # all_inps = []      # list of [1, seq_len, hidden_size] tensors on CPU  # dotv
    all_masks = []     # list of [1, seq_len] tensors on CPU
    all_pos_ids = []   # list of [1, seq_len] tensors on CPU

    all_inps = []

    with torch.no_grad():
        for i, sample in enumerate(calib_data):
            input_ids = sample["input_ids"].to(device)
            attention_mask = sample["attention_mask"].to(device)
            seq_len = input_ids.shape[1]
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

            hidden = embed(input_ids) * scale_emb
            all_inps.append(hidden)
            all_masks.append(attention_mask.cpu())
            all_pos_ids.append(position_ids.cpu())

    embed = embed.cpu()
    nuke_all_caches()
    tmp = all_inps[0]

    # --- Phase 2: Layer-by-layer quantization ---
    # Storage for packed quantized weights
    quantized_state = {}   # name -> {qweight, qzeros, scales, g_idx}
    original_state = {}    # name -> tensor (for non-quantized params)

    # Save embedding weight (not quantized)
    original_state["model.embed_tokens.weight"] = model.model.embed_tokens.weight.data.clone()

    for layer_idx in range(num_layers):
        t0 = time.time()
        mixer_type = config.mixer_types[layer_idx] if hasattr(config, "mixer_types") else "unknown"
        layer_bits = resolve_layer_bits(args, layer_idx, mixer_type)
        print(f"\n{'='*70}")
        print(f"[LAYER {layer_idx}/{num_layers-1}] type={mixer_type}  bits={layer_bits}")
        print(f"{'='*70}")

        layer = model.model.layers[layer_idx].to(device)
        layer.eval()

        # Find all Linears in this layer
        linears = find_all_linears(layer)
        linear_names = sorted(linears.keys())

        if layer_bits >= 16:
            # === SKIP QUANTIZATION: keep original weights ===
            print(f"  Keeping original weights (bits={layer_bits}, no quantization)")
            for name, param in layer.named_parameters():
                full_name = f"model.layers.{layer_idx}.{name}"
                original_state[full_name] = param.data.cpu().clone()

            # Still need to update hidden states through this layer
            print(f"  Updating hidden states for next layer...")
            nuke_all_caches()
            with torch.no_grad():
                for i in range(len(calib_data)):
                    inp = all_inps[i]
                    mask = all_masks[i].to(device)
                    pos = all_pos_ids[i].to(device)
                    out = layer(
                        inp,
                        attention_mask=mask,
                        position_ids=pos,
                        past_key_value=None,
                        output_attentions=False,
                        use_cache=False,
                    )
                    all_inps[i] = out[0]
                    if (i + 1) % 16 == 0:
                        nuke_all_caches()

            layer = layer.cpu()
            model.model.layers[layer_idx] = layer
            nuke_all_caches()
            elapsed = time.time() - t0
            print(f"  Layer {layer_idx} done in {elapsed:.1f}s (skipped)")
            continue

        # === QUANTIZE THIS LAYER ===
        print(f"  Linears to quantize: {linear_names}")

        # Create GPTQ quantizers
        quantizers: Dict[str, GPTQ] = {}
        for ln_name, ln_mod in linears.items():
            full_name = f"model.layers.{layer_idx}.{ln_name}"
            quantizers[ln_name] = GPTQ(full_name, ln_mod.weight.data)

        # Register hooks to capture inputs to each Linear
        hooks = []
        for ln_name in linear_names:
            ln_mod = linears[ln_name]
            qname = ln_name  # capture by value

            def make_hook(name):
                def hook_fn(module, inp, out):
                    quantizers[name].add_batch(inp[0].data)
                return hook_fn

            hooks.append(ln_mod.register_forward_hook(make_hook(qname)))

        # --- Forward pass: collect Hessians ---
        print(f"  Collecting Hessians ({len(calib_data)} samples)...")
        nuke_all_caches()

        with torch.no_grad():
            for i in range(len(calib_data)):
                inp = all_inps[i] 
                mask = all_masks[i].to(device)
                pos = all_pos_ids[i].to(device)
                try:
                    layer(
                        inp,
                        attention_mask=mask,
                        position_ids=pos,
                        past_key_value=None,
                        output_attentions=False,
                        use_cache=False,
                    )
                except Exception as e:
                    print(f"  [WARN] Sample {i} forward failed: {e}")
                    continue

                # Clear FLA caches after every sample to prevent accumulation
                if (i + 1) % 16 == 0:
                    nuke_all_caches()

        # Remove hooks
        for h in hooks:
            h.remove()

        # --- GPTQ quantization ---
        print(f"  Running GPTQ quantization (bits={layer_bits}, group_size={args.group_size})...")

        for ln_name in linear_names:
            full_name = f"model.layers.{layer_idx}.{ln_name}"
            ln_mod = linears[ln_name]
            W = ln_mod.weight.data.to(device)

            t1 = time.time()
            Q, sc, zp, iw = quantizers[ln_name].quantize(
                W,
                bits=layer_bits,
                group_size=args.group_size,
                damp_percent=args.damp,
                block_size=128,
                sym=args.sym,
            )
            t2 = time.time()

            # dotv
            mse = (Q.float() - W.float()).pow(2).mean().item()
            rel_err = (Q.float() - W.float()).norm() / W.float().norm()
            print(f"    {ln_name} MSE={mse:.6e} RelErr={rel_err:.4f}")
            ##################################################

            shape_str = f"{W.shape[0]}x{W.shape[1]}"
            print(f"    {ln_name} [{shape_str}] W{layer_bits} quantized in {t2-t1:.1f}s")

            # Replace weight with dequantized version for next-layer calibration
            ln_mod.weight.data = Q.to(ln_mod.weight.dtype).to(ln_mod.weight.device)

            # Pack and store
            qweight = pack_int_weight(iw.cpu(), layer_bits)
            scales_packed = sc.cpu().to(torch.float16)  # [groups, out_features]
            zeros_int = zp.cpu().round().int()
            qzeros = pack_zeros(zeros_int, layer_bits)
            g_idx = torch.arange(W.shape[1], dtype=torch.int32) // args.group_size

            quantized_state[full_name] = {
                "qweight": qweight,
                "qzeros": qzeros,
                "scales": scales_packed,
                "g_idx": g_idx,
                "bits": layer_bits,  # Track per-layer bits for save
            }

            del Q, sc, zp, iw, W
            gc.collect()

        # --- Re-run calibration with quantized weights to get updated hidden states ---
        print(f"  Updating hidden states for next layer...")
        nuke_all_caches()

        with torch.no_grad():
            for i in range(len(calib_data)):
                inp = all_inps[i] 
                mask = all_masks[i].to(device)
                pos = all_pos_ids[i].to(device)
                out = layer(
                    inp,
                    attention_mask=mask,
                    position_ids=pos,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )
                all_inps[i] = out[0]
                if (i + 1) % 16 == 0:
                    nuke_all_caches()


        # Save non-quantized params from this layer (layernorms, etc.)
        for name, param in layer.named_parameters():
            full_name = f"model.layers.{layer_idx}.{name}"
            # Check if this param belongs to a quantized Linear
            is_quantized = False
            for ln_name in linear_names:
                if name.startswith(ln_name + ".weight"):
                    is_quantized = True
                    break
            if not is_quantized:
                original_state[full_name] = param.data.cpu().clone()

        # Move layer back to CPU
        layer = layer.cpu()
        model.model.layers[layer_idx] = layer
        nuke_all_caches()

        elapsed = time.time() - t0
        print(f"  Layer {layer_idx} done in {elapsed:.1f}s")

    # --- Phase 3: Final norm + lm_head ---
    print(f"\n{'='*70}")
    print("[INFO] Phase 3: Processing final norm + lm_head...")
    print(f"{'='*70}")

    # Save norm weight
    original_state["model.norm.weight"] = model.model.norm.weight.data.cpu().clone()

    if args.quantize_lm_head:
        norm = model.model.norm.to(device)
        lm_head = model.lm_head.to(device)
        scale_width = config.hidden_size / config.dim_model_base

        # Create GPTQ for lm_head
        lm_quantizer = GPTQ("lm_head", lm_head.weight.data)

        print("  Collecting Hessian for lm_head...")
        with torch.no_grad():
            for i in range(len(calib_data)):
                inp = all_inps[i] 
                hidden = norm(inp)
                hidden = hidden / scale_width
                lm_quantizer.add_batch(hidden.reshape(-1, hidden.shape[-1]))
                if (i + 1) % 16 == 0:
                    nuke_all_caches()

        print("  Quantizing lm_head...")
        t1 = time.time()
        Q, sc, zp, iw = lm_quantizer.quantize(
            lm_head.weight.data.to(device),
            bits=args.bits,
            group_size=args.group_size,
            damp_percent=args.damp,
            block_size=128,
            sym=args.sym,
        )
        print(f"  lm_head [{lm_head.weight.shape[0]}x{lm_head.weight.shape[1]}] quantized in {time.time()-t1:.1f}s")

        qweight = pack_int_weight(iw.cpu(), args.bits)
        scales_packed = sc.cpu().to(torch.float16)
        zeros_int = zp.cpu().round().int()
        qzeros = pack_zeros(zeros_int, args.bits)
        g_idx = torch.arange(lm_head.weight.shape[1], dtype=torch.int32) // args.group_size

        quantized_state["lm_head"] = {
            "qweight": qweight,
            "qzeros": qzeros,
            "scales": scales_packed,
            "g_idx": g_idx,
        }
        norm.cpu()
        lm_head.cpu()
    else:
        # Save lm_head as-is
        original_state["lm_head.weight"] = model.lm_head.weight.data.cpu().clone()

    nuke_all_caches()

    # --- Phase 4: Save ---
    print(f"\n{'='*70}")
    print(f"[INFO] Phase 4: Saving quantized model to {args.output}")
    print(f"{'='*70}")
    save_quantized_model(quantized_state, original_state, args, config)
    # 删除 shutil.rmtree 这行，释放 GPU 内存即可
    all_inps.clear()
    torch.cuda.empty_cache()
    print("\n[INFO] Quantization complete!")


# ============================================================
# SECTION 6: Saving in AutoGPTQ-compatible format
# ============================================================

def save_quantized_model(
    quantized_state: Dict,
    original_state: Dict,
    args,
    config,
):
    """Save quantized model in AutoGPTQ-compatible safetensors format."""
    os.makedirs(args.output, exist_ok=True)

    # Build the complete state dict
    state_dict = {}

    # Add quantized layers
    for full_name, packed in quantized_state.items():
        for key in ["qweight", "qzeros", "scales", "g_idx"]:
            state_dict[f"{full_name}.{key}"] = packed[key]

    # Add non-quantized params
    for name, tensor in original_state.items():
        state_dict[name] = tensor

    # Try safetensors first, fall back to torch
    try:
        from safetensors.torch import save_file

        # Split into manageable chunks (~4GB each)
        MAX_SHARD_SIZE = 4 * 1024 * 1024 * 1024  # 4GB

        # Calculate total size
        total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
        print(f"  Total model size: {total_size / 1024**3:.2f} GB")

        if total_size <= MAX_SHARD_SIZE:
            # Single file
            save_file(state_dict, os.path.join(args.output, "model.safetensors"))
            index = None
        else:
            # Multiple files
            shards = {}
            current_shard = {}
            current_size = 0
            shard_idx = 1
            weight_map = {}

            for name, tensor in sorted(state_dict.items()):
                tensor_size = tensor.numel() * tensor.element_size()
                if current_size + tensor_size > MAX_SHARD_SIZE and current_shard:
                    shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
                    shards[shard_name] = current_shard
                    current_shard = {}
                    current_size = 0
                    shard_idx += 1

                current_shard[name] = tensor
                current_size += tensor_size

            if current_shard:
                shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
                shards[shard_name] = current_shard

            total_shards = len(shards)
            final_shards = {}
            for old_name, shard_data in shards.items():
                new_name = old_name.replace("XXXXX", f"{total_shards:05d}")
                final_shards[new_name] = shard_data
                for tensor_name in shard_data:
                    weight_map[tensor_name] = new_name

            for shard_name, shard_data in final_shards.items():
                save_file(shard_data, os.path.join(args.output, shard_name))
                print(f"  Saved {shard_name}")

            # Write index
            index = {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            }
            with open(os.path.join(args.output, "model.safetensors.index.json"), "w") as f:
                json.dump(index, f, indent=2)

        print(f"  Saved weights in safetensors format")

    except ImportError:
        print("  [WARN] safetensors not available, using torch.save")
        torch.save(state_dict, os.path.join(args.output, "pytorch_model.bin"))

    # Write quantize_config.json
    # Determine the dominant bits for the config
    all_bits = set()
    for packed in quantized_state.values():
        if "bits" in packed:
            all_bits.add(packed["bits"])
        else:
            all_bits.add(args.bits)
    primary_bits = min(all_bits) if all_bits else args.bits

    quant_config = {
        "bits": primary_bits,
        "group_size": args.group_size,
        "desc_act": False,
        "sym": args.sym,
        "damp_percent": args.damp,
        "true_sequential": False,
        "model_name_or_path": args.input,
        "model_file_base_name": "model",
        "quant_method": "gptq",
        "is_marlin_format": False,
        "checkpoint_format": "gptq",
    }

    # Record per-layer bit info if mixed
    if len(all_bits) > 1:
        layer_bits_map = {}
        if hasattr(config, "mixer_types"):
            for i, mt in enumerate(config.mixer_types):
                lb = resolve_layer_bits(args, i, mt)
                layer_bits_map[str(i)] = lb
        quant_config["mixed_precision"] = True
        quant_config["layer_bits"] = layer_bits_map
    with open(os.path.join(args.output, "quantize_config.json"), "w") as f:
        json.dump(quant_config, f, indent=2)

    # Copy config files from original model
    for fname in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ]:
        src = os.path.join(args.input, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output, fname))

    # Copy any Python model files (trust_remote_code)
    for fname in glob.glob(os.path.join(args.input, "*.py")):
        shutil.copy2(fname, os.path.join(args.output, os.path.basename(fname)))

    # Update config.json with quantization info
    config_path = os.path.join(args.output, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        cfg["quantization_config"] = {
            "bits": primary_bits,
            "group_size": args.group_size,
            "desc_act": False,
            "sym": args.sym,
            "quant_method": "gptq",
        }

        # dotv
        ###########################################################
        # Add per-layer bits for mixed precision
        if hasattr(config, 'mixer_types'):
            lb_map = {}
            for i, mt in enumerate(config.mixer_types):
                lb_map[str(i)] = resolve_layer_bits(args, i, mt)
            cfg["gptq_layer_bits"] = lb_map
        ###########################################################


        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

    print(f"  All files saved to {args.output}")


# ============================================================
# SECTION 7: CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Standalone GPTQ W4A16 quantization for MiniCPM-SALA"
    )
    parser.add_argument("--input", type=str, required=True, help="Original model path")
    parser.add_argument("--output", type=str, required=True, help="Output path for quantized model")
    parser.add_argument("--bits", type=int, default=4, help="Default quantization bits (default: 4)")
    parser.add_argument("--group-size", type=int, default=128, help="Group size (default: 128)")
    parser.add_argument("--damp", type=float, default=0.01, help="Damping percent (default: 0.01)")
    parser.add_argument(
        "--sym", action="store_true", default=True,
        help="Use symmetric quantization (default: True, required for gptq_marlin)"
    )
    parser.add_argument(
        "--no-sym", dest="sym", action="store_false",
        help="Use asymmetric quantization (not compatible with gptq_marlin)"
    )
    parser.add_argument(
        "--minicpm4-bits", type=int, default=None,
        help="Bits for minicpm4 (sparse attention) layers. Overrides --bits. Use 16 to skip."
    )
    parser.add_argument(
        "--lightning-bits", type=int, default=None,
        help="Bits for lightning-attn (linear attention) layers. Overrides --bits. Use 16 to skip."
    )
    parser.add_argument(
        "--layer-bits", type=str, default=None,
        help="Explicit per-layer bits. Format: '0:4,1:8,2-7:4,8:16'. Overrides all other bit settings."
    )
    parser.add_argument(
        "--calib-data", type=str,
        default="/opt/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl",
        help="Calibration data path (JSONL with 'question' field)"
    )
    parser.add_argument("--max-samples", type=int, default=128, help="Max calibration samples")
    parser.add_argument("--max-len", type=int, default=2048, help="Max sequence length")
    parser.add_argument(
        "--quantize-lm-head", action="store_true",
        help="Also quantize lm_head (default: keep fp16)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")

    args = parser.parse_args()
    seed_everything(args.seed)

    print("=" * 70)
    print("Standalone GPTQ Quantizer for MiniCPM-SALA (Mixed Precision)")
    print("=" * 70)
    print(f"  Input:           {args.input}")
    print(f"  Output:          {args.output}")
    print(f"  Default bits:    {args.bits}")
    print(f"  Symmetric:       {args.sym}")
    print(f"  minicpm4 bits:   {args.minicpm4_bits or '(use default)'}")
    print(f"  lightning bits:  {args.lightning_bits or '(use default)'}")
    print(f"  layer-bits:      {args.layer_bits or '(none)'}")
    print(f"  Group size:      {args.group_size}")
    print(f"  Damping:         {args.damp}")
    print(f"  Calib data:      {args.calib_data}")
    print(f"  Max samples:     {args.max_samples}")
    print(f"  Max length:      {args.max_len}")
    print(f"  Quant lm_head:   {args.quantize_lm_head}")

    # Print per-layer plan if config has mixer_types
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.input, trust_remote_code=True)
        if hasattr(cfg, "mixer_types"):
            print(f"\n  {'Layer':<8} {'Type':<20} {'Bits':<6}")
            print(f"  {'-'*34}")
            total_w4 = total_w8 = total_fp16 = 0
            for i, mt in enumerate(cfg.mixer_types):
                lb = resolve_layer_bits(args, i, mt)
                tag = f"W{lb}" if lb < 16 else "FP16"
                print(f"  {i:<8} {mt:<20} {tag:<6}")
                if lb == 4: total_w4 += 1
                elif lb == 8: total_w8 += 1
                else: total_fp16 += 1
            print(f"\n  Summary: {total_w4} W4 | {total_w8} W8 | {total_fp16} FP16")
    except Exception:
        pass

    print("=" * 70)

    t_start = time.time()
    quantize_model(args)
    t_total = time.time() - t_start
    print(f"\n[INFO] Total time: {t_total/60:.1f} minutes")


if __name__ == "__main__":
    main()
