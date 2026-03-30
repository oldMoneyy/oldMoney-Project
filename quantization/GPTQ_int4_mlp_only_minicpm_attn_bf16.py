#!/usr/bin/env python3
"""
GPTQ W4A16 quantization for MiniCPM-SALA: MiniCPM attention BF16, MLP+Lightning INT4.

Based on GPTQ_int4_flashinfer_dense_smoothing_gpu.py with selective quantization:
  - MiniCPM4 layers: attention linears (qkv_proj, o_proj, o_gate) kept in BF16
                     MLP linears (gate_up_proj, down_proj) quantized to INT4
  - Lightning layers: ALL linears (attention + MLP) quantized to INT4
  - Embedding, layernorms, lm_head: kept in original precision

Features inherited from parent script:
  1. Per-sample normalized Hessian accumulation
  2. SmoothQuant-style channel equalization (only on quantized linears)
  3. Deterministic calibration (seed=42, no shuffle)
  4. GPU-native Hessian (float64 for stability)
  5. Dense calibration mode (matches flashinfer serving)

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
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_everything(seed=42):
    print(f"[INFO] Using Deterministic Seed: {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)


# ============================================================
# SECTION 1: GPTQ Algorithm (block-wise for performance)
# ============================================================

class GPTQ:
    """GPTQ quantizer for a single nn.Linear layer."""

    def __init__(self, name: str, weight: torch.Tensor):
        self.name = name
        self.rows, self.columns = weight.shape
        self.H = torch.zeros(
            (self.columns, self.columns), device=weight.device, dtype=torch.float64
        )
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor):
        """Accumulate Hessian H = X^T @ X with per-sample normalization."""
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        elif inp.dim() == 1:
            inp = inp.unsqueeze(0)
        n_tokens = inp.shape[0]
        inp = inp.float()
        h_update = (inp.T @ inp) / max(n_tokens, 1)
        self.H.add_(h_update.to(torch.float64))
        self.nsamples += 1

    def apply_smooth_to_hessian(self, smooth: torch.Tensor):
        """Transform H after channel smoothing: H_new = diag(1/s) @ H @ diag(1/s)."""
        s = smooth.to(self.H.dtype).to(self.H.device)
        inv_s = 1.0 / s
        self.H = self.H * inv_s.unsqueeze(0) * inv_s.unsqueeze(1)

    def quantize(
        self,
        weight: torch.Tensor,
        bits: int = 4,
        group_size: int = 128,
        damp_percent: float = 0.01,
        block_size: int = 128,
        sym: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = weight.device
        orig_dtype = weight.dtype
        W = weight.float().clone()
        H = (self.H / self.nsamples).float().to(dev)
        del self.H

        rows, columns = W.shape
        maxq = 2 ** bits - 1

        dead = torch.diag(H) == 0
        H[dead, dead] = 1.0
        W[:, dead] = 0.0

        damp = damp_percent * torch.diag(H).mean()
        H.diagonal().add_(damp)

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

                if col % group_size == 0:
                    g_end = min(col + group_size, columns)
                    w_group = W[:, col:g_end]
                    if sym:
                        half_q = maxq // 2
                        wmax_abs = w_group.abs().max(dim=1).values
                        tmp_scale = (wmax_abs / half_q).clamp(min=1e-10)
                        tmp_zero = torch.full_like(tmp_scale, half_q + 1)
                    else:
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

                if j + 1 < blen:
                    W_block[:, j + 1:] -= err.unsqueeze(1) * Hinv_block_diag[j, j + 1:].unsqueeze(0)

            if block_end < columns:
                W[:, block_end:] -= Err_block @ Hinv[block_start:block_end, block_end:]

        return Q.to(orig_dtype), scales, zeros, int_weight


# ============================================================
# SECTION 2: Packing into AutoGPTQ-compatible format
# ============================================================

def pack_int_weight(int_weight: torch.Tensor, bits: int = 4) -> torch.Tensor:
    pack_num = 32 // bits
    iw = int_weight.T.contiguous().to(torch.int32)
    in_f, out_f = iw.shape
    assert in_f % pack_num == 0, f"in_features ({in_f}) must be divisible by {pack_num}"
    iw = iw.reshape(in_f // pack_num, pack_num, out_f)
    qweight = torch.zeros(in_f // pack_num, out_f, dtype=torch.int32)
    for k in range(pack_num):
        qweight |= (iw[:, k, :] & ((1 << bits) - 1)) << (k * bits)
    return qweight


def pack_zeros(zeros_int: torch.Tensor, bits: int = 4) -> torch.Tensor:
    pack_num = 32 // bits
    num_groups, out_f = zeros_int.shape
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


# ============================================================
# SECTION 3.5: Smoothing utilities
# ============================================================

def compute_smooth_factor(
    target_weights: List[torch.Tensor],
    H_diags: List[torch.Tensor],
    alpha: float = 0.5,
    clamp_min: float = 0.01,
    clamp_max: float = 100.0,
) -> torch.Tensor:
    K = target_weights[0].shape[1]
    device = target_weights[0].device
    H_diags = [hd.to(device) for hd in H_diags]
    avg_h_diag = torch.stack(H_diags).mean(dim=0)
    act_scale = avg_h_diag.sqrt().clamp(min=1e-8)
    weight_scale = torch.zeros(K, device=device, dtype=torch.float32)
    for W in target_weights:
        col_max = W.float().abs().amax(dim=0)
        weight_scale = torch.max(weight_scale, col_max)
    weight_scale = weight_scale.clamp(min=1e-8)
    smooth = act_scale.pow(alpha) / weight_scale.pow(1.0 - alpha)
    smooth = smooth / smooth.mean()
    return smooth.clamp(min=clamp_min, max=clamp_max)


def get_submodule_safe(module: nn.Module, path: str):
    parts = path.split(".")
    current = module
    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            return None
    return current


# ============================================================
# SECTION 3.6: Selective classification of linears
# ============================================================

# Attention-related linear suffixes
ATTN_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "o_gate", "z_proj")
# MLP-related linear suffixes
MLP_SUFFIXES = ("gate_proj", "up_proj", "down_proj")


def classify_linears(
    linear_names: List[str],
    mixer_type: str,
    quantize_minicpm4_attn: bool = False,
) -> Tuple[List[str], List[str]]:
    """Classify linears into quantize vs skip lists.

    Args:
        linear_names: all linear names in this layer
        mixer_type: "minicpm4" or "lightning"/"lightning_attn"/"lightning-attn"
        quantize_minicpm4_attn: if True, also quantize minicpm4 attention (False = BF16)

    Returns:
        (names_to_quantize, names_to_skip)
    """
    to_quantize = []
    to_skip = []

    for name in linear_names:
        is_attn = any(name.endswith(s) for s in ATTN_SUFFIXES)
        is_mlp = any(name.endswith(s) for s in MLP_SUFFIXES)

        if mixer_type == "minicpm4" and is_attn and not quantize_minicpm4_attn:
            # MiniCPM4 attention -> BF16 (skip)
            to_skip.append(name)
        else:
            # Everything else (MLP, lightning attn, lightning MLP) -> quantize
            to_quantize.append(name)

    return to_quantize, to_skip


def apply_smoothing_selective(
    layer: nn.Module,
    layer_idx: int,
    mixer_type: str,
    linears: Dict[str, nn.Linear],
    quantizers: Dict[str, "GPTQ"],
    alpha: float,
    names_to_quantize: List[str],
):
    """Apply SmoothQuant channel equalization only to linears being quantized.

    Smoothing groups (only applied if target linears are in names_to_quantize):
      Group 1: input_layernorm -> [q_proj, k_proj, v_proj] (+ z_proj/o_gate)
      Group 2: o_norm -> [o_proj] (lightning-attn layers only)
      Group 3: post_attention_layernorm -> [gate_proj, up_proj]
      Group 4: up_proj -> [down_proj] (output dim scaling)
    """
    # --- Group 1: input_layernorm -> attention projections ---
    # Only smooth if the attention linears are being quantized
    if mixer_type != "minicpm4":
        attn_targets = [n for n in names_to_quantize
                        if n.endswith(("q_proj", "k_proj", "v_proj", "z_proj"))]
    else:
        attn_targets = [n for n in names_to_quantize
                        if n.endswith(("q_proj", "k_proj", "v_proj", "o_gate"))]

    input_ln = get_submodule_safe(layer, "input_layernorm")
    if attn_targets and input_ln is not None and hasattr(input_ln, "weight"):
        tw = [linears[t].weight.data.float() for t in attn_targets]
        hd = [torch.diag(quantizers[t].H).float() / max(quantizers[t].nsamples, 1)
              for t in attn_targets]
        s = compute_smooth_factor(tw, hd, alpha=alpha)
        input_ln.weight.data.div_(s.to(input_ln.weight.dtype))
        for t in attn_targets:
            linears[t].weight.data.mul_(s.unsqueeze(0).to(linears[t].weight.dtype))
            quantizers[t].apply_smooth_to_hessian(s)
        print(f"  Smoothed [input_layernorm] -> {attn_targets} "
              f"| s: min={s.min():.4f} max={s.max():.4f} std={s.std():.4f}")

    # --- Group 2: o_norm -> o_proj (lightning-attn only) ---
    if mixer_type != "minicpm4":
        o_norm = get_submodule_safe(layer, "self_attn.o_norm")
        if o_norm is not None and "self_attn.o_proj" in quantizers and "self_attn.o_proj" in names_to_quantize:
            t = "self_attn.o_proj"
            W = linears[t].weight.data.float()
            h_diag = torch.diag(quantizers[t].H).float() / max(quantizers[t].nsamples, 1)
            s = compute_smooth_factor([W], [h_diag], alpha=alpha)
            o_norm.weight.data.div_(s.to(o_norm.weight.dtype))
            linears[t].weight.data.mul_(s.unsqueeze(0).to(linears[t].weight.dtype))
            quantizers[t].apply_smooth_to_hessian(s)
            print(f"  Smoothed [o_norm] -> [o_proj] "
                  f"| s: min={s.min():.4f} max={s.max():.4f} std={s.std():.4f}")

    # --- Group 3: post_attention_layernorm -> gate_proj, up_proj ---
    mlp_targets = [n for n in names_to_quantize if n.endswith(("gate_proj", "up_proj"))]
    post_ln = get_submodule_safe(layer, "post_attention_layernorm")
    if mlp_targets and post_ln is not None and hasattr(post_ln, "weight"):
        tw = [linears[t].weight.data.float() for t in mlp_targets]
        hd = [torch.diag(quantizers[t].H).float() / max(quantizers[t].nsamples, 1)
              for t in mlp_targets]
        s = compute_smooth_factor(tw, hd, alpha=alpha)
        post_ln.weight.data.div_(s.to(post_ln.weight.dtype))
        for t in mlp_targets:
            linears[t].weight.data.mul_(s.unsqueeze(0).to(linears[t].weight.dtype))
            quantizers[t].apply_smooth_to_hessian(s)
        print(f"  Smoothed [post_attn_layernorm] -> {mlp_targets} "
              f"| s: min={s.min():.4f} max={s.max():.4f} std={s.std():.4f}")

    # --- Group 4: up_proj -> down_proj (output dim scaling) ---
    up_name = "mlp.up_proj"
    down_name = "mlp.down_proj"
    if (up_name in linears and down_name in linears
            and down_name in quantizers and down_name in names_to_quantize):
        down_w = linears[down_name].weight.data.float()
        down_h = torch.diag(quantizers[down_name].H).float() / max(quantizers[down_name].nsamples, 1)
        s2 = compute_smooth_factor([down_w], [down_h], alpha=alpha)
        linears[up_name].weight.data.div_(
            s2.unsqueeze(1).to(linears[up_name].weight.dtype)
        )
        linears[down_name].weight.data.mul_(
            s2.unsqueeze(0).to(linears[down_name].weight.dtype)
        )
        quantizers[down_name].apply_smooth_to_hessian(s2)
        print(f"  Smoothed [up_proj] -> [down_proj] "
              f"| s: min={s2.min():.4f} max={s2.max():.4f} std={s2.std():.4f}")


# ============================================================
# SECTION 4: Calibration data
# ============================================================

def load_calibration_data(
    tokenizer, data_path: str, max_samples: int, max_len: int
) -> List[Dict[str, torch.Tensor]]:
    """Load and tokenize calibration data DETERMINISTICALLY (no shuffle)."""
    samples = []
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Calibration data not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            text = obj.get("question", obj.get("text", ""))
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

    if len(samples) < max_samples:
        print(f"[WARN] Only {len(samples)} samples found, need {max_samples}. Duplicating.")
        orig = samples.copy()
        while len(samples) < max_samples:
            samples.extend(orig)
        samples = samples[:max_samples]

    total_tokens = sum(s["input_ids"].shape[1] for s in samples)
    print(f"[INFO] Loaded {len(samples)} calibration samples, {total_tokens:,} tokens (max_len={max_len})")
    return samples


# ============================================================
# SECTION 5: Main quantization pipeline
# ============================================================

def find_all_linears(module: nn.Module) -> Dict[str, nn.Linear]:
    linears = {}
    for name, mod in module.named_modules():
        if isinstance(mod, nn.Linear):
            linears[name] = mod
    return linears


def quantize_model(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading tokenizer from {args.input}...")

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    tokenizer = AutoTokenizer.from_pretrained(args.input, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 2

    calib_data = load_calibration_data(
        tokenizer, args.calib_data, args.max_samples, args.max_len
    )

    print(f"[INFO] Loading model from {args.input} (CPU)...")
    config = AutoConfig.from_pretrained(args.input, trust_remote_code=True)

    # Dense attention mode for calibration
    if hasattr(config, "sparse_config") and isinstance(config.sparse_config, dict):
        config.sparse_config["dense_len"] = 655360

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

    # --- Phase 1: Embedding ---
    print("[INFO] Phase 1: Computing initial hidden states (embedding)...")
    embed = model.model.embed_tokens.to(device)
    scale_emb = config.scale_emb

    all_inps = []
    all_masks = []
    all_pos_ids = []

    with torch.no_grad():
        for i, sample in enumerate(calib_data):
            input_ids = sample["input_ids"].to(device)
            attention_mask = sample["attention_mask"].to(device)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            hidden = embed(input_ids) * scale_emb
            all_inps.append(hidden)
            all_masks.append(attention_mask.cpu())
            all_pos_ids.append(position_ids.cpu())

    embed = embed.cpu()
    nuke_all_caches()

    # --- Phase 2: Layer-by-layer selective quantization ---
    quantized_state = {}
    original_state = {}
    original_state["model.embed_tokens.weight"] = model.model.embed_tokens.weight.data.clone()

    # Track per-layer quantization decisions for config
    layer_quant_info = {}

    for layer_idx in range(num_layers):
        t0 = time.time()
        mixer_type = config.mixer_types[layer_idx] if hasattr(config, "mixer_types") else "minicpm4"
        print(f"\n{'='*70}")
        print(f"[LAYER {layer_idx}/{num_layers-1}] type={mixer_type}")
        print(f"{'='*70}")

        layer = model.model.layers[layer_idx].to(device)
        layer.eval()

        linears = find_all_linears(layer)
        all_linear_names = sorted(linears.keys())

        # Classify which linears to quantize vs skip
        names_to_quantize, names_to_skip = classify_linears(
            all_linear_names, mixer_type, quantize_minicpm4_attn=False
        )

        print(f"  All linears: {all_linear_names}")
        print(f"  -> Quantize (INT4): {names_to_quantize}")
        print(f"  -> Skip (BF16):     {names_to_skip}")

        layer_quant_info[layer_idx] = {
            "mixer_type": mixer_type,
            "quantized": names_to_quantize,
            "skipped_bf16": names_to_skip,
        }

        if not names_to_quantize:
            # Nothing to quantize in this layer - keep everything as-is
            print(f"  No linears to quantize, keeping entire layer in BF16")
            for name, param in layer.named_parameters():
                full_name = f"model.layers.{layer_idx}.{name}"
                original_state[full_name] = param.data.cpu().clone()

            # Still forward through to update hidden states
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
            print(f"  Layer {layer_idx} done in {elapsed:.1f}s (all BF16)")
            continue

        # === SELECTIVE QUANTIZATION ===

        # Create GPTQ quantizers ONLY for linears we will quantize
        quantizers: Dict[str, GPTQ] = {}
        for ln_name in names_to_quantize:
            ln_mod = linears[ln_name]
            full_name = f"model.layers.{layer_idx}.{ln_name}"
            quantizers[ln_name] = GPTQ(full_name, ln_mod.weight.data)

        # Register hooks only on linears to quantize
        hooks = []
        for ln_name in names_to_quantize:
            ln_mod = linears[ln_name]
            def make_hook(name):
                def hook_fn(module, inp, out):
                    quantizers[name].add_batch(inp[0].data)
                return hook_fn
            hooks.append(ln_mod.register_forward_hook(make_hook(ln_name)))

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
                if (i + 1) % 16 == 0:
                    nuke_all_caches()

        for h in hooks:
            h.remove()

        # --- Smoothing (only on quantized linears) ---
        print(f"  Applying SmoothQuant channel equalization (alpha={args.smooth_alpha})...")
        apply_smoothing_selective(
            layer=layer,
            layer_idx=layer_idx,
            mixer_type=mixer_type,
            linears=linears,
            quantizers=quantizers,
            alpha=args.smooth_alpha,
            names_to_quantize=names_to_quantize,
        )

        # --- GPTQ quantization (only on selected linears) ---
        print(f"  Running GPTQ quantization (bits={args.bits}, group_size={args.group_size})...")

        for ln_name in names_to_quantize:
            full_name = f"model.layers.{layer_idx}.{ln_name}"
            ln_mod = linears[ln_name]
            W = ln_mod.weight.data.to(device)

            t1 = time.time()
            Q, sc, zp, iw = quantizers[ln_name].quantize(
                W,
                bits=args.bits,
                group_size=args.group_size,
                damp_percent=args.damp,
                block_size=128,
                sym=args.sym,
            )
            t2 = time.time()

            mse = (Q.float() - W.float()).pow(2).mean().item()
            rel_err = (Q.float() - W.float()).norm() / W.float().norm()
            print(f"    {ln_name} MSE={mse:.6e} RelErr={rel_err:.4f}")

            shape_str = f"{W.shape[0]}x{W.shape[1]}"
            print(f"    {ln_name} [{shape_str}] W{args.bits} quantized in {t2-t1:.1f}s")

            ln_mod.weight.data = Q.to(ln_mod.weight.dtype).to(ln_mod.weight.device)

            qweight = pack_int_weight(iw.cpu(), args.bits)
            scales_packed = sc.cpu().to(torch.float16)
            zeros_int = zp.cpu().round().int()
            qzeros = pack_zeros(zeros_int, args.bits)
            g_idx = torch.arange(W.shape[1], dtype=torch.int32) // args.group_size

            quantized_state[full_name] = {
                "qweight": qweight,
                "qzeros": qzeros,
                "scales": scales_packed,
                "g_idx": g_idx,
                "bits": args.bits,
            }

            del Q, sc, zp, iw, W
            gc.collect()

        # --- Re-run calibration with quantized weights ---
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

        # Save non-quantized params (layernorms, BF16 attention weights, biases, etc.)
        for name, param in layer.named_parameters():
            full_name = f"model.layers.{layer_idx}.{name}"
            is_quantized = False
            for ln_name in names_to_quantize:
                if name.startswith(ln_name + ".weight"):
                    is_quantized = True
                    break
            if not is_quantized:
                original_state[full_name] = param.data.cpu().clone()

        layer = layer.cpu()
        model.model.layers[layer_idx] = layer
        nuke_all_caches()

        elapsed = time.time() - t0
        n_q = len(names_to_quantize)
        n_s = len(names_to_skip)
        print(f"  Layer {layer_idx} done in {elapsed:.1f}s ({n_q} INT4, {n_s} BF16)")

    # --- Phase 3: Final norm + lm_head ---
    print(f"\n{'='*70}")
    print("[INFO] Phase 3: Processing final norm + lm_head...")
    print(f"{'='*70}")

    original_state["model.norm.weight"] = model.model.norm.weight.data.cpu().clone()

    if args.quantize_lm_head:
        norm = model.model.norm.to(device)
        lm_head = model.lm_head.to(device)
        scale_width = config.hidden_size / config.dim_model_base

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
        original_state["lm_head.weight"] = model.lm_head.weight.data.cpu().clone()

    nuke_all_caches()

    # --- Phase 4: Save ---
    print(f"\n{'='*70}")
    print(f"[INFO] Phase 4: Saving quantized model to {args.output}")
    print(f"{'='*70}")
    save_quantized_model(quantized_state, original_state, args, config, layer_quant_info)
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
    layer_quant_info: Dict,
):
    os.makedirs(args.output, exist_ok=True)

    state_dict = {}
    for full_name, packed in quantized_state.items():
        for key in ["qweight", "qzeros", "scales", "g_idx"]:
            state_dict[f"{full_name}.{key}"] = packed[key]
    for name, tensor in original_state.items():
        state_dict[name] = tensor

    try:
        from safetensors.torch import save_file

        MAX_SHARD_SIZE = 4 * 1024 * 1024 * 1024

        total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
        print(f"  Total model size: {total_size / 1024**3:.2f} GB")

        if total_size <= MAX_SHARD_SIZE:
            save_file(state_dict, os.path.join(args.output, "model.safetensors"))
            index = None
        else:
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
    quant_config = {
        "bits": args.bits,
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
        "mixed_precision": True,
        "mixed_precision_strategy": "minicpm4_attn_bf16_mlp_lightning_int4",
    }

    # Record per-layer quantization details
    layer_bits_map = {}
    layer_details = {}
    for i in range(config.num_hidden_layers):
        info = layer_quant_info.get(i, {})
        mixer_type = info.get("mixer_type", "unknown")
        quantized = info.get("quantized", [])
        skipped = info.get("skipped_bf16", [])

        if quantized:
            layer_bits_map[str(i)] = args.bits
        else:
            layer_bits_map[str(i)] = 16

        layer_details[str(i)] = {
            "mixer_type": mixer_type,
            "quantized_linears": quantized,
            "bf16_linears": skipped,
        }

    quant_config["layer_bits"] = layer_bits_map
    quant_config["layer_details"] = layer_details

    with open(os.path.join(args.output, "quantize_config.json"), "w") as f:
        json.dump(quant_config, f, indent=2)

    # Copy config files
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

    for fname in glob.glob(os.path.join(args.input, "*.py")):
        shutil.copy2(fname, os.path.join(args.output, os.path.basename(fname)))

    # Update config.json with quantization info
    config_path = os.path.join(args.output, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)

        # Build exclude_modules list for attention linears in minicpm4 layers
        exclude_modules = []
        for i in range(config.num_hidden_layers):
            info = layer_quant_info.get(i, {})
            for skipped_name in info.get("skipped_bf16", []):
                exclude_modules.append(f"model.layers.{i}.{skipped_name}")

        cfg["quantization_config"] = {
            "bits": args.bits,
            "group_size": args.group_size,
            "desc_act": False,
            "sym": args.sym,
            "quant_method": "gptq",
            "exclude_modules": exclude_modules,
        }

        # Per-layer bits
        cfg["gptq_layer_bits"] = layer_bits_map

        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

    print(f"  All files saved to {args.output}")


# ============================================================
# SECTION 7: CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="GPTQ W4A16: MiniCPM attention BF16, MLP+Lightning INT4"
    )
    parser.add_argument("--input", type=str, required=True, help="Original model path")
    parser.add_argument("--output", type=str, required=True, help="Output path for quantized model")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits for MLP/lightning (default: 4)")
    parser.add_argument("--group-size", type=int, default=128, help="Group size (default: 128)")
    parser.add_argument("--damp", type=float, default=0.01, help="Damping percent (default: 0.01)")
    parser.add_argument(
        "--sym", action="store_true", default=True,
        help="Use symmetric quantization (default: True, required for gptq_marlin)"
    )
    parser.add_argument(
        "--no-sym", dest="sym", action="store_false",
        help="Use asymmetric quantization"
    )
    parser.add_argument(
        "--calib-data", type=str,
        default="/opt/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl",
        help="Calibration data path (JSONL with 'question' field)"
    )
    parser.add_argument("--max-samples", type=int, default=96, help="Max calibration samples")
    parser.add_argument("--max-len", type=int, default=131072, help="Max sequence length")
    parser.add_argument(
        "--smooth-alpha", type=float, default=0.5,
        help="SmoothQuant alpha (0=all weight, 1=all activation, default: 0.5)"
    )
    parser.add_argument(
        "--quantize-lm-head", action="store_true",
        help="Also quantize lm_head (default: keep fp16)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()
    seed_everything(args.seed)

    print("=" * 70)
    print("GPTQ Quantizer: MiniCPM Attn BF16 + MLP/Lightning INT4")
    print("=" * 70)
    print(f"  Input:           {args.input}")
    print(f"  Output:          {args.output}")
    print(f"  Bits (MLP/LA):   {args.bits}")
    print(f"  Symmetric:       {args.sym}")
    print(f"  Smooth alpha:    {args.smooth_alpha}")
    print(f"  Group size:      {args.group_size}")
    print(f"  Damping:         {args.damp}")
    print(f"  Calib data:      {args.calib_data}")
    print(f"  Max samples:     {args.max_samples}")
    print(f"  Max length:      {args.max_len}")
    print(f"  Quant lm_head:   {args.quantize_lm_head}")

    # Print per-layer plan
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.input, trust_remote_code=True)
        if hasattr(cfg, "mixer_types"):
            print(f"\n  {'Layer':<8} {'Type':<20} {'Attn':<10} {'MLP':<10}")
            print(f"  {'-'*48}")
            total_int4 = total_bf16_attn = 0
            for i, mt in enumerate(cfg.mixer_types):
                _, to_skip = classify_linears(
                    # approximate - just show the plan
                    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                     "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
                    mt, quantize_minicpm4_attn=False,
                )
                if to_skip:
                    attn_tag = "BF16"
                    total_bf16_attn += 1
                else:
                    attn_tag = f"W{args.bits}"
                    total_int4 += 1
                mlp_tag = f"W{args.bits}"
                print(f"  {i:<8} {mt:<20} {attn_tag:<10} {mlp_tag:<10}")
            print(f"\n  Summary: {total_bf16_attn} layers with BF16-attn | {total_int4} layers fully INT4")
    except Exception:
        pass

    print("=" * 70)

    t_start = time.time()
    quantize_model(args)
    t_total = time.time() - t_start
    print(f"\n[INFO] Total time: {t_total:.1f}s ({t_total/60:.1f}m)")


if __name__ == "__main__":
    main()
