#!/usr/bin/env python3
"""
GPTQ W4A16 quantization for MiniCPM-SALA: MiniCPM4 layers full BF16, Lightning layers INT4.

Quantization strategy:
  - MiniCPM4 layers (attn + MLP): entirely BF16 (no quantization)
  - Lightning layers: ALL linears (attn + MLP) quantized to INT4
  - Embedding, final norm, lm_head: BF16

Features:
  1. Per-sample normalized Hessian accumulation
  2. SmoothQuant-style channel equalization (on lightning layers only)
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
# SECTION 1: GPTQ Algorithm
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
# SECTION 3: Cache clearing
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


def apply_smoothing(
    layer: nn.Module,
    layer_idx: int,
    mixer_type: str,
    linears: Dict[str, nn.Linear],
    quantizers: Dict[str, "GPTQ"],
    alpha: float,
    names_to_quantize: List[str],
):
    """Apply SmoothQuant channel equalization to lightning layer linears.

    4 smoothing groups:
      Group 1: input_layernorm -> [q_proj, k_proj, v_proj, z_proj]
      Group 2: o_norm -> [o_proj]
      Group 3: post_attention_layernorm -> [gate_proj, up_proj]
      Group 4: up_proj -> [down_proj]
    """
    # --- Group 1: input_layernorm -> attention projections ---
    attn_targets = [n for n in names_to_quantize
                    if n.endswith(("q_proj", "k_proj", "v_proj", "z_proj"))]

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
    o_norm = get_submodule_safe(layer, "self_attn.o_norm")
    if o_norm is not None and "self_attn.o_proj" in quantizers:
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

    # --- Group 4: up_proj -> down_proj ---
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

LIGHTNING_TYPES = {"lightning", "lightning_attn", "lightning-attn"}


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

    # --- Phase 2: Layer-by-layer ---
    quantized_state = {}
    original_state = {}
    original_state["model.embed_tokens.weight"] = model.model.embed_tokens.weight.data.clone()

    layer_quant_info = {}

    for layer_idx in range(num_layers):
        t0 = time.time()
        mixer_type = config.mixer_types[layer_idx] if hasattr(config, "mixer_types") else "minicpm4"
        is_lightning = mixer_type in LIGHTNING_TYPES

        print(f"\n{'='*70}")
        print(f"[LAYER {layer_idx}/{num_layers-1}] type={mixer_type}  "
              f"-> {'INT4 (quantize)' if is_lightning else 'BF16 (skip)'}")
        print(f"{'='*70}")

        layer = model.model.layers[layer_idx].to(device)
        layer.eval()

        if not is_lightning:
            # ============================================
            # MiniCPM4 layer: keep ENTIRELY in BF16
            # ============================================
            print(f"  MiniCPM4 layer -> keeping all weights in BF16")
            for name, param in layer.named_parameters():
                full_name = f"model.layers.{layer_idx}.{name}"
                original_state[full_name] = param.data.cpu().clone()

            layer_quant_info[layer_idx] = {
                "mixer_type": mixer_type,
                "action": "bf16",
            }

            # Forward through to update hidden states
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
            print(f"  Layer {layer_idx} done in {elapsed:.1f}s (BF16 skip)")
            continue

        # ============================================
        # Lightning layer: quantize ALL linears to INT4
        # ============================================
        linears = find_all_linears(layer)
        linear_names = sorted(linears.keys())
        print(f"  Linears to quantize: {linear_names}")

        layer_quant_info[layer_idx] = {
            "mixer_type": mixer_type,
            "action": "int4",
            "quantized": linear_names,
        }

        # Create GPTQ quantizers
        quantizers: Dict[str, GPTQ] = {}
        for ln_name, ln_mod in linears.items():
            full_name = f"model.layers.{layer_idx}.{ln_name}"
            quantizers[ln_name] = GPTQ(full_name, ln_mod.weight.data)

        # Register hooks
        hooks = []
        for ln_name in linear_names:
            ln_mod = linears[ln_name]
            def make_hook(name):
                def hook_fn(module, inp, out):
                    quantizers[name].add_batch(inp[0].data)
                return hook_fn
            hooks.append(ln_mod.register_forward_hook(make_hook(ln_name)))

        # --- Collect Hessians ---
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

        # --- Smoothing ---
        print(f"  Applying SmoothQuant channel equalization (alpha={args.smooth_alpha})...")
        apply_smoothing(
            layer=layer,
            layer_idx=layer_idx,
            mixer_type=mixer_type,
            linears=linears,
            quantizers=quantizers,
            alpha=args.smooth_alpha,
            names_to_quantize=linear_names,
        )

        # --- GPTQ quantization ---
        print(f"  Running GPTQ quantization (bits={args.bits}, group_size={args.group_size})...")

        for ln_name in linear_names:
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

        # --- Update hidden states with quantized weights ---
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

        # Save non-quantized params (layernorms etc.)
        for name, param in layer.named_parameters():
            full_name = f"model.layers.{layer_idx}.{name}"
            is_quantized = False
            for ln_name in linear_names:
                if name.startswith(ln_name + ".weight"):
                    is_quantized = True
                    break
            if not is_quantized:
                original_state[full_name] = param.data.cpu().clone()

        layer = layer.cpu()
        model.model.layers[layer_idx] = layer
        nuke_all_caches()

        elapsed = time.time() - t0
        print(f"  Layer {layer_idx} done in {elapsed:.1f}s (INT4)")

    # --- Phase 3: Final norm + lm_head (BF16) ---
    print(f"\n{'='*70}")
    print("[INFO] Phase 3: Saving final norm + lm_head (BF16)...")
    print(f"{'='*70}")

    original_state["model.norm.weight"] = model.model.norm.weight.data.cpu().clone()
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
# SECTION 6: Saving
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

    # Build exclude_modules for minicpm4 layers (entire layer excluded)
    exclude_modules = []
    for i in range(config.num_hidden_layers):
        info = layer_quant_info.get(i, {})
        if info.get("action") == "bf16":
            exclude_modules.append(f"model.layers.{i}")

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
        "mixed_precision_strategy": "minicpm4_bf16_lightning_int4",
    }

    layer_bits_map = {}
    for i in range(config.num_hidden_layers):
        info = layer_quant_info.get(i, {})
        layer_bits_map[str(i)] = 16 if info.get("action") == "bf16" else args.bits
    quant_config["layer_bits"] = layer_bits_map

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

    # Update config.json
    config_path = os.path.join(args.output, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)

        cfg["quantization_config"] = {
            "bits": args.bits,
            "group_size": args.group_size,
            "desc_act": False,
            "sym": args.sym,
            "quant_method": "gptq",
            "exclude_modules": exclude_modules,
        }
        cfg["gptq_layer_bits"] = layer_bits_map

        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

    print(f"  All files saved to {args.output}")


# ============================================================
# SECTION 7: CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="GPTQ W4A16: MiniCPM4 layers BF16, Lightning layers INT4"
    )
    parser.add_argument("--input", type=str, required=True, help="Original model path")
    parser.add_argument("--output", type=str, required=True, help="Output path for quantized model")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits for lightning layers (default: 4)")
    parser.add_argument("--group-size", type=int, default=128, help="Group size (default: 128)")
    parser.add_argument("--damp", type=float, default=0.01, help="Damping percent (default: 0.01)")
    parser.add_argument(
        "--sym", action="store_true", default=True,
        help="Use symmetric quantization (default: True)"
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()
    seed_everything(args.seed)

    print("=" * 70)
    print("GPTQ Quantizer: MiniCPM4 BF16 + Lightning INT4")
    print("=" * 70)
    print(f"  Input:           {args.input}")
    print(f"  Output:          {args.output}")
    print(f"  Bits (lightning): {args.bits}")
    print(f"  Symmetric:       {args.sym}")
    print(f"  Smooth alpha:    {args.smooth_alpha}")
    print(f"  Group size:      {args.group_size}")
    print(f"  Damping:         {args.damp}")
    print(f"  Calib data:      {args.calib_data}")
    print(f"  Max samples:     {args.max_samples}")
    print(f"  Max length:      {args.max_len}")
    print(f"  Embedding:       BF16")
    print(f"  Final norm:      BF16")
    print(f"  lm_head:         BF16")

    # Print per-layer plan
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.input, trust_remote_code=True)
        if hasattr(cfg, "mixer_types"):
            print(f"\n  {'Layer':<8} {'Type':<20} {'Action':<12}")
            print(f"  {'-'*40}")
            n_bf16 = n_int4 = 0
            for i, mt in enumerate(cfg.mixer_types):
                if mt in LIGHTNING_TYPES:
                    tag = f"W{args.bits} (INT4)"
                    n_int4 += 1
                else:
                    tag = "BF16 (skip)"
                    n_bf16 += 1
                print(f"  {i:<8} {mt:<20} {tag:<12}")
            print(f"\n  Summary: {n_bf16} MiniCPM4 layers (BF16) | {n_int4} Lightning layers (INT4)")
    except Exception:
        pass

    print("=" * 70)

    t_start = time.time()
    quantize_model(args)
    t_total = time.time() - t_start
    print(f"\n[INFO] Total time: {t_total:.1f}s ({t_total/60:.1f}m)")


if __name__ == "__main__":
    main()
