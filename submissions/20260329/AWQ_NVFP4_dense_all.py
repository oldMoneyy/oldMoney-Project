#!/usr/bin/env python3
"""
Dense Flashinfer NVFP4 ALL-Params Quantization for MiniCPM-SALA
================================================================

ALL linear layers quantized to NVFP4. Only norms, lm_head, embeddings stay BF16.

Quantization plan:
  - ALL layers: ALL linears = NVFP4 (Q/K/V/O/Z/gate + MLP)
  - lm_head, embeddings, norms: BF16
  - Calibration: forced dense (sparse_config.dense_len = 655360)

This is the most aggressive quantization. Model size: ~5.5 GB.
Difference from AWQ_NVFP4_dense_flashinfer.py: minicpm4 attention also FP4.

Quantization plan (targeting flashinfer dense attention serving):
  - MiniCPM4 layers (8/32):  attn=BF16, MLP=NVFP4
  - Lightning-attn layers (24/32): ALL=NVFP4 (Q/K/V/Z/O + MLP)
  - lm_head, embeddings, norms: BF16

Why this works with flashinfer:
  Flashinfer runs ALL layers as standard dense softmax attention.
  No GLA recurrence → no systematic FP4 bias accumulation → no <unk> collapse.
  The GLA recurrence concern only applies with minicpm_flashinfer backend.

Calibration: forced dense via sparse_config["dense_len"] = 655360
  All inputs go through the dense attention branch during calibration,
  matching the flashinfer serving path.

Smoothing strategy:
  MiniCPM4 layers:    post_attn_layernorm → gate/up, up → down (MLP only)
  Lightning-attn layers: input_layernorm → Q/K/V/Z, post_attn_layernorm → gate/up,
                          up → down (ALL projections smoothed)

Model size: ~6.4 GB (vs 8.3 GB MLP-only, vs 19 GB full BF16)

Based on AWQ_NVFP4_mixed_bf16attn.py with these changes:
  1. Force dense calibration (sparse_config.dense_len = 655360)
  2. should_quantize_linear(): lightning-attn layers quantize ALL linears
  3. build_exclude_modules(): only exclude minicpm4 attention
  4. apply_layer_smoothing(): lightning-attn also smooths attention projections
  5. compute_fused_global_scales(): lightning-attn includes QKV fusion

Output: NVFP4 (E2M1 weights + FP8 E4M3 block scales + FP32 global/input scales)
Target: SGLang on NVIDIA Blackwell via flashinfer backend + ModelOptFp4Config
"""

import os
import gc
import json
import math
import time
import glob
import shutil
import argparse
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

# ====================================================================
# Section 1: NVFP4 Constants & E2M1 Grid
# ====================================================================

FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
NVFP4_SCALE_FACTOR = FP4_E2M1_MAX * FP8_E4M3_MAX  # 2688.0
NVFP4_GROUP_SIZE = 16

_E2M1_POS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)
_SORTED_GRID = torch.cat([-_E2M1_POS.flip(0)[:-1], _E2M1_POS])  # 15 values
_MIDPOINTS = (_SORTED_GRID[:-1] + _SORTED_GRID[1:]) / 2.0
_GRID_IDX_TO_4BIT = np.array(
    [15, 14, 13, 12, 11, 10, 9, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint8
)


# ====================================================================
# Section 2: E2M1 Quantization & Packing
# ====================================================================


def quantize_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    midpoints = _MIDPOINTS.to(x.device, dtype=x.dtype)
    grid = _SORTED_GRID.to(x.device, dtype=x.dtype)
    idx = torch.bucketize(x.clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX), midpoints)
    return grid[idx]


def pack_e2m1_to_uint8(e2m1_vals: torch.Tensor) -> torch.Tensor:
    N, K = e2m1_vals.shape
    midpoints = _MIDPOINTS.to(e2m1_vals.device, dtype=e2m1_vals.dtype)
    flat = e2m1_vals.reshape(-1).clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)
    grid_idx = torch.bucketize(flat, midpoints).cpu().numpy()
    codes = _GRID_IDX_TO_4BIT[grid_idx].reshape(N, K)
    packed = (codes[:, 1::2].astype(np.uint16) << 4) | codes[:, 0::2]
    return torch.from_numpy(packed.astype(np.uint8))


# ====================================================================
# Section 3: Observer (Per-Sample Normalized)
# ====================================================================


class ActAwareObserver:
    """
    Collects diagonal Hessian approximation.

    Each calibration sample contributes equally regardless of token count.
    Without this, a 128k-token sample dominates a 250-token MCQ sample
    by 500x, destroying the calibration balance.
    """

    def __init__(self, columns: int, device: torch.device):
        self.H_diag = torch.zeros(columns, device=device, dtype=torch.float64)
        self.act_amax = 0.0
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor):
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        elif inp.dim() == 1:
            inp = inp.unsqueeze(0)
        n_tokens = inp.shape[0]
        per_sample_h = (inp.float() ** 2).sum(dim=0) / max(n_tokens, 1)
        self.H_diag += per_sample_h
        self.nsamples += 1
        self.act_amax = max(self.act_amax, inp.abs().max().item())

    def get_h_diag(self) -> torch.Tensor:
        return (self.H_diag / max(self.nsamples, 1)).float()


# ====================================================================
# Section 4: Channel Smoothing
# ====================================================================


def compute_smooth_factor(
    target_weights: List[torch.Tensor],
    h_diags: List[torch.Tensor],
    alpha: float = 0.5,
    clamp_min: float = 0.01,
    clamp_max: float = 100.0,
) -> torch.Tensor:
    """
    Compute per-channel smooth factor for a norm -> [linears] group.
    s[j] = act_scale[j]^alpha / weight_scale[j]^(1-alpha)
    Normalized so mean=1, clamped to prevent numerical issues.
    """
    K = target_weights[0].shape[1]
    device = target_weights[0].device

    avg_h_diag = torch.stack(h_diags).mean(dim=0)
    act_scale = avg_h_diag.sqrt().clamp(min=1e-8)

    weight_scale = torch.zeros(K, device=device, dtype=torch.float32)
    for W in target_weights:
        col_max = W.float().abs().amax(dim=0)
        weight_scale = torch.max(weight_scale, col_max)
    weight_scale = weight_scale.clamp(min=1e-8)

    smooth = act_scale.pow(alpha) / weight_scale.pow(1.0 - alpha)
    smooth = smooth / smooth.mean()
    smooth = smooth.clamp(min=clamp_min, max=clamp_max)
    return smooth


def get_submodule_safe(module: nn.Module, path: str):
    parts = path.split(".")
    current = module
    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            return None
    return current


def apply_layer_smoothing(
    layer: nn.Module,
    layer_idx: int,
    mixer_type: str,
    linears: Dict[str, nn.Linear],
    observers: Dict[str, "ActAwareObserver"],
    alpha: float,
    names_to_quantize: List[str],
) -> Dict[str, torch.Tensor]:
    """
    Apply channel smoothing.

    ALL layers (ALL=FP4):
      Group 1: input_layernorm -> [q_proj, k_proj, v_proj] (+ z_proj for lightning)
      Group 2: o_norm -> [o_proj] (lightning only, minicpm4 has no o_norm)
      Group 3: post_attention_layernorm -> [gate_proj, up_proj]
      Group 4: up_proj -> [down_proj]
    """
    smooth_factors = {}

    # --- Smooth attention projections (ALL layer types) ---
    if True:
        attn_targets = [n for n in ["self_attn.q_proj", "self_attn.k_proj",
                                     "self_attn.v_proj", "self_attn.z_proj"]
                        if n in linears and n in names_to_quantize]
        input_ln = get_submodule_safe(layer, "input_layernorm")

        if attn_targets and input_ln is not None and hasattr(input_ln, "weight"):
            target_weights = [
                linears[t].weight.data.to(torch.float32) for t in attn_targets
            ]
            h_diags = [
                observers[t].get_h_diag().to(target_weights[0].device)
                for t in attn_targets
            ]
            s = compute_smooth_factor(target_weights, h_diags, alpha=alpha)
            input_ln.weight.data.div_(s.to(input_ln.weight.dtype))
            for t in attn_targets:
                linears[t].weight.data.mul_(
                    s.unsqueeze(0).to(linears[t].weight.dtype)
                )
                smooth_factors[t] = s
            print(
                f"    Smooth [input_layernorm] -> {attn_targets} "
                f"| s: min={s.min():.4f} max={s.max():.4f} std={s.std():.4f}"
            )

        # o_norm -> o_proj (lightning-attn only, o_proj input comes from o_norm)
        o_proj_name = "self_attn.o_proj"
        o_norm = get_submodule_safe(layer, "self_attn.o_norm")
        if (o_proj_name in linears and o_proj_name in names_to_quantize
                and o_norm is not None and hasattr(o_norm, "weight")
                and o_proj_name in observers):
            o_w = linears[o_proj_name].weight.data.to(torch.float32)
            o_h = observers[o_proj_name].get_h_diag().to(o_w.device)
            s_o = compute_smooth_factor([o_w], [o_h], alpha=alpha)
            o_norm.weight.data.div_(s_o.to(o_norm.weight.dtype))
            linears[o_proj_name].weight.data.mul_(
                s_o.unsqueeze(0).to(linears[o_proj_name].weight.dtype)
            )
            smooth_factors[o_proj_name] = s_o
            print(
                f"    Smooth [o_norm] -> [o_proj] "
                f"| s: min={s_o.min():.4f} max={s_o.max():.4f} std={s_o.std():.4f}"
            )

    # --- Group: post_attention_layernorm -> gate, up (ALL layers) ---
    mlp_targets = ["mlp.gate_proj", "mlp.up_proj"]
    mlp_present = [t for t in mlp_targets if t in linears and t in names_to_quantize]
    post_ln = get_submodule_safe(layer, "post_attention_layernorm")

    if mlp_present and post_ln is not None and hasattr(post_ln, "weight"):
        target_weights = [
            linears[t].weight.data.to(torch.float32) for t in mlp_present
        ]
        h_diags = [
            observers[t].get_h_diag().to(target_weights[0].device)
            for t in mlp_present
        ]
        s = compute_smooth_factor(target_weights, h_diags, alpha=alpha)
        post_ln.weight.data.div_(s.to(post_ln.weight.dtype))
        for t in mlp_present:
            linears[t].weight.data.mul_(
                s.unsqueeze(0).to(linears[t].weight.dtype)
            )
            smooth_factors[t] = s
        print(
            f"    Smooth [post_attn_layernorm] -> {mlp_present} "
            f"| s: min={s.min():.4f} max={s.max():.4f} std={s.std():.4f}"
        )

    # --- Group: up_proj -> down_proj (ALL layers) ---
    up_name = "mlp.up_proj"
    down_name = "mlp.down_proj"
    if (up_name in linears and down_name in linears
            and down_name in observers and down_name in names_to_quantize):
        down_w = linears[down_name].weight.data.to(torch.float32)
        down_h = observers[down_name].get_h_diag().to(down_w.device)

        s2 = compute_smooth_factor([down_w], [down_h], alpha=alpha)

        linears[up_name].weight.data.div_(
            s2.unsqueeze(1).to(linears[up_name].weight.dtype)
        )
        linears[down_name].weight.data.mul_(
            s2.unsqueeze(0).to(linears[down_name].weight.dtype)
        )
        smooth_factors[down_name] = s2
        print(
            f"    Smooth [up_proj] -> [down_proj] "
            f"| s: min={s2.min():.4f} max={s2.max():.4f} std={s2.std():.4f}"
        )

    return smooth_factors


# ====================================================================
# Section 5: AWQ Block Scale Search
# ====================================================================


def compute_awq_block_scales(
    W: torch.Tensor,
    global_sf: float,
    H_diag: torch.Tensor,
    group_size: int = NVFP4_GROUP_SIZE,
    search_iters: int = 80,
    max_shrink: float = 0.60,
    error_norm: float = 2.0,
) -> torch.Tensor:
    N, K = W.shape
    num_groups = K // group_size
    W_grouped = W.reshape(N, num_groups, group_size)

    act_weight = H_diag.reshape(1, num_groups, group_size).abs()
    act_weight = act_weight / (act_weight.max() + 1e-8)

    init_scales = W_grouped.abs().amax(dim=-1) * global_sf / FP4_E2M1_MAX
    init_scales = init_scales.clamp(min=torch.finfo(torch.float8_e4m3fn).tiny)

    best_scales = init_scales.clone()
    best_error = torch.full(
        (N, num_groups), float("inf"), device=W.device, dtype=W.dtype
    )

    for i in range(search_iters):
        shrink = 1.0 - i * max_shrink / search_iters
        candidate_scales = (shrink * init_scales).clamp(
            min=torch.finfo(torch.float8_e4m3fn).tiny, max=FP8_E4M3_MAX
        )
        candidate_scales = (
            candidate_scales.to(torch.float8_e4m3fn)
            .to(torch.float32)
            .clamp(min=1e-12)
        )
        W_scaled = W_grouped * global_sf / candidate_scales.unsqueeze(-1)
        W_q = quantize_to_e2m1(W_scaled)
        W_deq = W_q * candidate_scales.unsqueeze(-1) / global_sf
        error = (
            (W_grouped - W_deq).abs().pow(error_norm) * act_weight
        ).sum(dim=-1)
        improved = error < best_error
        best_error[improved] = error[improved]
        best_scales[improved] = candidate_scales[improved]

    return best_scales


# ====================================================================
# Section 6: Quantization Plan
# ====================================================================


def should_quantize_linear(
    layer_idx: int, linear_name: str, mixer_type: str
) -> bool:
    """ALL linears quantized to NVFP4. No exceptions."""
    return True


def build_exclude_modules(config) -> List[str]:
    """No exclusions — all linears quantized."""
    return []


def compute_fused_global_scales(
    linears: Dict[str, nn.Linear],
    names_to_quantize: List[str],
    mixer_type: str,
) -> Dict[str, float]:
    """
    Compute fused global scales for quantized linears.
    ALL layers get QKV fusion + gate_up fusion.
    """
    fuse_groups = {}

    # QKV fusion (all layer types — all attention quantized)
    qkv = [
        n for n in names_to_quantize
        if n.endswith(("q_proj", "k_proj", "v_proj"))
    ]
    if qkv:
        fuse_groups["qkv"] = qkv

    # gate_up fusion (all layer types)
    gate_up = [
        n for n in names_to_quantize if n.endswith(("gate_proj", "up_proj"))
    ]
    if gate_up:
        fuse_groups["gate_up"] = gate_up

    fused_members = set()
    result = {}

    for group_name, members in fuse_groups.items():
        max_amax = max(
            linears[m].weight.data.abs().max().item() for m in members
        )
        gsf = NVFP4_SCALE_FACTOR / max(max_amax, 1e-12)
        for m in members:
            result[m] = gsf
            fused_members.add(m)
        print(
            f"    [{group_name}] fused global_sf={gsf:.4f} "
            f"(max_amax={max_amax:.6f})"
        )

    for name in names_to_quantize:
        if name not in fused_members:
            amax = linears[name].weight.data.abs().max().item()
            result[name] = NVFP4_SCALE_FACTOR / max(amax, 1e-12)

    return result


# ====================================================================
# Section 7: Utilities
# ====================================================================


def nuke_caches():
    try:
        from fla.utils import tensor_cache
        tensor_cache.clear()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class HiddenStateStore:
    def __init__(self):
        self._states = {}

    def save(self, idx: int, tensor: torch.Tensor):
        self._states[idx] = tensor.detach().cpu()

    def load(self, idx: int, device: torch.device) -> torch.Tensor:
        return self._states[idx].to(device)

    def cleanup(self):
        self._states.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __len__(self):
        return len(self._states)


def load_calibration_data(tokenizer, data_path, max_samples, max_len):
    samples = []
    with open(data_path) as f:
        for line in f:
            obj = json.loads(line.strip())
            text = obj.get("question", obj.get("text", ""))
            enc = tokenizer(
                text, max_length=max_len, truncation=True, return_tensors="pt"
            )
            samples.append(
                {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                }
            )
            if len(samples) >= max_samples:
                break
    if len(samples) < max_samples:
        orig = samples.copy()
        while len(samples) < max_samples:
            samples.extend(orig)
        samples = samples[:max_samples]
    total_tokens = sum(s["input_ids"].shape[1] for s in samples)
    print(f"[INFO] {len(samples)} samples, {total_tokens:,} tokens")
    return samples


def find_all_linears(module: nn.Module) -> Dict[str, nn.Linear]:
    return {
        n: m for n, m in module.named_modules() if isinstance(m, nn.Linear)
    }


# ====================================================================
# Section 8: Per-Layer Pipeline
# ====================================================================


def collect_activations(
    layer, linears, linear_names, device, calib_data, hs_store,
    all_masks, all_pos_ids
):
    observers = {
        n: ActAwareObserver(m.weight.shape[1], device)
        for n, m in linears.items()
    }
    hooks = []
    for n in linear_names:
        hook = linears[n].register_forward_hook(
            lambda m, inp, out, name=n: observers[name].add_batch(inp[0].data)
        )
        hooks.append(hook)

    nuke_caches()
    with torch.no_grad():
        for i in range(len(calib_data)):
            inp = hs_store.load(i, device)
            try:
                layer(
                    inp,
                    attention_mask=all_masks[i].to(device),
                    position_ids=all_pos_ids[i].to(device),
                    use_cache=False,
                )
            except Exception as e:
                print(f"    [WARN] Sample {i} failed: {type(e).__name__}: {e}")
            del inp
            if (i + 1) % 4 == 0:
                nuke_caches()

    for h in hooks:
        h.remove()

    # Diagnostics
    for n, obs in observers.items():
        if obs.nsamples == 0:
            print(f"    [BUG] Observer '{n}' collected 0 samples!")
        h_diag = obs.get_h_diag()
        if torch.isnan(h_diag).any() or torch.isinf(h_diag).any():
            print(f"    [BUG] Observer '{n}' H_diag has NaN/Inf!")
        if obs.act_amax > 1e4:
            print(f"    [WARN] Observer '{n}' act_amax={obs.act_amax:.1f}")

    return observers


def quantize_single_linear(
    full_name, ln_mod, observer, global_sf, smooth_factor,
    args, quantized_tensors, device
):
    W = ln_mod.weight.data.to(device).float()
    N, K = W.shape

    H_diag = observer.get_h_diag().to(device)
    if smooth_factor is not None:
        s = smooth_factor.to(device)
        H_diag = H_diag / (s ** 2 + 1e-12)

    w_scale_2_ckpt = torch.tensor(
        [NVFP4_SCALE_FACTOR / global_sf / NVFP4_SCALE_FACTOR],
        dtype=torch.float32,
    )

    t1 = time.time()
    bscales = compute_awq_block_scales(
        W, global_sf, H_diag, NVFP4_GROUP_SIZE,
        search_iters=args.mse_iters,
        max_shrink=args.mse_max_shrink,
        error_norm=args.mse_error_norm,
    )

    num_groups = K // NVFP4_GROUP_SIZE
    e2m1_vals = torch.zeros_like(W)
    Q_deq = torch.zeros_like(W)

    for g in range(num_groups):
        gs = g * NVFP4_GROUP_SIZE
        ge = (g + 1) * NVFP4_GROUP_SIZE
        bsf = bscales[:, g]
        w_scaled = W[:, gs:ge] * global_sf / bsf.unsqueeze(1)
        q = quantize_to_e2m1(w_scaled)
        e2m1_vals[:, gs:ge] = q
        Q_deq[:, gs:ge] = q * bsf.unsqueeze(1) / global_sf

    dt = time.time() - t1
    mse_val = (Q_deq - W).pow(2).mean().item()
    rel_err = mse_val / (W.pow(2).mean().item() + 1e-12)
    print(
        f"    {full_name} [{N}x{K}] {dt:.1f}s | "
        f"MSE={mse_val:.3e} RelErr={rel_err:.3e}"
    )

    if rel_err > 0.05:
        print(f"    [BAD] RelErr > 5%!")
    elif rel_err > 0.02:
        print(f"    [WARN] RelErr > 2%")

    ln_mod.weight.data = Q_deq.to(ln_mod.weight.dtype)

    packed_fp4 = pack_e2m1_to_uint8(e2m1_vals)
    block_sf_fp8 = (
        bscales.clamp(
            min=torch.finfo(torch.float8_e4m3fn).tiny, max=FP8_E4M3_MAX
        )
        .to(torch.float8_e4m3fn)
        .cpu()
    )

    a_scale_ckpt = torch.tensor(
        [max(observer.act_amax, 1e-12) / NVFP4_SCALE_FACTOR],
        dtype=torch.float32,
    )

    quantized_tensors[f"{full_name}.weight"] = packed_fp4.cpu()
    quantized_tensors[f"{full_name}.weight_scale"] = block_sf_fp8
    quantized_tensors[f"{full_name}.weight_scale_2"] = w_scale_2_ckpt
    quantized_tensors[f"{full_name}.input_scale"] = a_scale_ckpt

    del W, Q_deq, e2m1_vals, bscales, H_diag
    gc.collect()
    return mse_val


def process_layer(
    layer, layer_idx, config, device, calib_data, hs_store,
    all_masks, all_pos_ids, args, quantized_tensors, original_tensors
):
    mixer_type = config.mixer_types[layer_idx]
    linears = find_all_linears(layer)
    linear_names = sorted(linears.keys())
    print(f"  Mixer: {mixer_type} | Linears: {linear_names}")

    # Phase 1: Collect activations
    print(f"  Phase 1: Collecting activations ({len(calib_data)} samples)...")
    observers = collect_activations(
        layer, linears, linear_names, device,
        calib_data, hs_store, all_masks, all_pos_ids,
    )

    # Phase 2: Quantization plan
    names_to_quantize = []
    names_bf16 = []
    for ln_name in linear_names:
        if should_quantize_linear(layer_idx, ln_name, mixer_type):
            names_to_quantize.append(ln_name)
        else:
            names_bf16.append(ln_name)

    if names_bf16:
        print(f"  BF16 (kept): {names_bf16}")
    if names_to_quantize:
        print(f"  NVFP4 (quantizing): {names_to_quantize}")

    # Phase 3: Channel smoothing
    print(f"  Phase 3: Channel smoothing (alpha={args.smooth_alpha})...")
    smooth_factors = apply_layer_smoothing(
        layer, layer_idx, mixer_type, linears, observers,
        alpha=args.smooth_alpha,
        names_to_quantize=names_to_quantize,
    )

    # Phase 4: Fused global scales
    fused_scales = {}
    if names_to_quantize:
        fused_scales = compute_fused_global_scales(
            linears, names_to_quantize, mixer_type
        )

    # Phase 5: Quantize
    layer_errors = {}
    if names_to_quantize:
        print(f"  Phase 5: AWQ MSE search ({args.mse_iters} iters)...")

    for ln_name in linear_names:
        full_name = f"model.layers.{layer_idx}.{ln_name}"
        if ln_name in names_bf16:
            original_tensors[f"{full_name}.weight"] = (
                linears[ln_name].weight.data.cpu().clone()
            )
            continue
        mse = quantize_single_linear(
            full_name=full_name,
            ln_mod=linears[ln_name],
            observer=observers[ln_name],
            global_sf=fused_scales[ln_name],
            smooth_factor=smooth_factors.get(ln_name),
            args=args,
            quantized_tensors=quantized_tensors,
            device=device,
        )
        layer_errors[ln_name] = mse

    # Phase 6: Save non-linear params (layernorms, norms)
    for name, param in layer.named_parameters():
        full_name = f"model.layers.{layer_idx}.{name}"
        is_linear_weight = any(
            name.startswith(ln + ".weight") or name.startswith(ln + ".bias")
            for ln in linear_names
        )
        if not is_linear_weight:
            original_tensors[full_name] = param.data.cpu().clone()

    return layer_errors


# ====================================================================
# Section 9: Checkpoint Saving
# ====================================================================


def save_checkpoint(
    quantized_tensors, original_tensors, args, config, exclude_modules
):
    os.makedirs(args.output, exist_ok=True)
    raw_state_dict = {**quantized_tensors, **original_tensors}

    # Force contiguous memory to prevent Safetensors crash
    state_dict = {k: v.contiguous() for k, v in raw_state_dict.items()}

    from safetensors.torch import save_file

    MAX_SHARD = 4 * 1024 * 1024 * 1024
    total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
    print(
        f"\n[Phase 4] Saving to {args.output} "
        f"(Total: {total_size / 1024**3:.2f} GB)..."
    )

    if total_size <= MAX_SHARD:
        save_file(state_dict, os.path.join(args.output, "model.safetensors"))
        weight_map = {n: "model.safetensors" for n in state_dict}
        print("  Saved model.safetensors")
    else:
        shards = {}
        current = {}
        cur_size = 0
        idx = 1
        weight_map = {}

        for name in sorted(state_dict.keys()):
            t = state_dict[name]
            tsz = t.numel() * t.element_size()
            if cur_size + tsz > MAX_SHARD and current:
                shard_name = f"model-{idx:05d}-of-XXXXX.safetensors"
                shards[shard_name] = current
                current = {}
                cur_size = 0
                idx += 1
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

        with open(
            os.path.join(args.output, "model.safetensors.index.json"), "w"
        ) as f:
            json.dump(
                {
                    "metadata": {"total_size": total_size},
                    "weight_map": weight_map,
                },
                f,
                indent=2,
            )

    # Copy model files
    for fname in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
    ]:
        src = os.path.join(args.input, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output, fname))

    for fname in glob.glob(os.path.join(args.input, "*.py")):
        shutil.copy2(fname, os.path.join(args.output, os.path.basename(fname)))

    for fname in glob.glob(os.path.join(args.input, "tokenizer.model")):
        shutil.copy2(fname, os.path.join(args.output, os.path.basename(fname)))

    # SGLang metadata
    exclude = sorted(set(exclude_modules))

    with open(os.path.join(args.output, "hf_quant_config.json"), "w") as f:
        json.dump(
            {
                "quantization": {
                    "quant_algo": "NVFP4",
                    "kv_cache_quant_algo": "auto",
                    "group_size": NVFP4_GROUP_SIZE,
                    "exclude_modules": exclude,
                }
            },
            f,
            indent=2,
        )

    cfg_path = os.path.join(args.output, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["quantization_config"] = {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "kv_cache_quant_algo": "auto",
            "group_size": NVFP4_GROUP_SIZE,
            "exclude_modules": exclude,
        }
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  Patched config.json with {len(exclude)} exclude_modules")

    print(f"  Excluded: {len(exclude)} modules (minicpm4 attention only)")


# ====================================================================
# Section 10: Main Engine
# ====================================================================


def quantize_model(args):
    # Deterministic seeds — ensure reproducible quantization across runs
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only=True: FA2 doesn't support deterministic mode, so this avoids
    # a hard crash while still making everything else deterministic
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device("cuda:0")

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    tokenizer = AutoTokenizer.from_pretrained(args.input, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 2

    calib_data = load_calibration_data(
        tokenizer, args.calib_data, args.max_samples, args.max_len
    )

    print("[INFO] Loading model (CPU, bf16)...")
    config = AutoConfig.from_pretrained(args.input, trust_remote_code=True)

    # CRITICAL: Force dense attention for calibration
    # All inputs go through the dense attention branch,
    # matching the flashinfer serving path (no sparse routing)
    if hasattr(config, "sparse_config") and isinstance(config.sparse_config, dict):
        config.sparse_config["dense_len"] = 655360
        print("[INFO] Forced dense attention (sparse_config.dense_len = 655360)")

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
    exclude_modules = build_exclude_modules(config)

    # Print quantization plan
    print(f"\n[INFO] Quantization Plan (ALL-Params NVFP4):")
    for i in range(num_layers):
        mt = config.mixer_types[i]
        print(f"  Layer {i:2d}: {mt} — ALL=FP4")
    print(f"  Summary: all {num_layers} layers fully FP4")
    print(f"  Exclude modules: {len(exclude_modules)} entries (minicpm4 attn only)")
    print(f"  lm_head: BF16")

    # Phase 1: Embedding
    embed = model.model.embed_tokens.to(device)
    hs_store = HiddenStateStore()
    all_masks = []
    all_pos_ids = []

    print(f"\n[Phase 1] Embedding forward ({len(calib_data)} samples)...")
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
            del ids, mask, pos, h
            if (i + 1) % 8 == 0:
                nuke_caches()

    embed = embed.cpu()
    nuke_caches()

    # Phase 2 & 3: Layer-by-layer
    quantized_tensors = {}
    original_tensors = {}
    all_layer_errors = {}

    original_tensors["model.embed_tokens.weight"] = (
        model.model.embed_tokens.weight.data.clone()
    )

    for layer_idx in range(num_layers):
        t0 = time.time()
        mixer_type = config.mixer_types[layer_idx]

        print(f"\n{'=' * 70}")
        tag = "ALL=FP4"
        print(f"[Layer {layer_idx}/{num_layers - 1}] {mixer_type} -> {tag}")
        print(f"{'=' * 70}")

        layer = model.model.layers[layer_idx].to(device)
        layer.eval()

        layer_errors = process_layer(
            layer, layer_idx, config, device,
            calib_data, hs_store, all_masks, all_pos_ids,
            args, quantized_tensors, original_tensors,
        )
        all_layer_errors[layer_idx] = layer_errors

        # Propagate hidden states
        print("  Propagating hidden states...")
        nan_count = 0
        with torch.no_grad():
            for i in range(len(calib_data)):
                inp = hs_store.load(i, device)
                out = layer(
                    inp,
                    attention_mask=all_masks[i].to(device),
                    position_ids=all_pos_ids[i].to(device),
                    use_cache=False,
                )
                hs = out[0]
                if torch.isnan(hs).any() or torch.isinf(hs).any():
                    nan_count += 1
                hs_store.save(i, hs)
                del inp, out, hs
                if (i + 1) % 4 == 0:
                    nuke_caches()

        if nan_count > 0:
            print(f"  [BUG] {nan_count}/{len(calib_data)} hidden states have NaN/Inf!")

        layer = layer.cpu()
        model.model.layers[layer_idx] = layer
        nuke_caches()
        print(f"  Layer {layer_idx} done in {time.time() - t0:.1f}s")

    # Save norm and lm_head (BF16)
    original_tensors["model.norm.weight"] = (
        model.model.norm.weight.data.cpu().clone()
    )
    if not config.tie_word_embeddings:
        original_tensors["lm_head.weight"] = (
            model.lm_head.weight.data.cpu().clone()
        )

    # Save checkpoint
    save_checkpoint(
        quantized_tensors, original_tensors, args, config, exclude_modules
    )

    # Error summary
    print(f"\n{'=' * 70}")
    print("QUANTIZATION ERROR SUMMARY")
    print(f"{'=' * 70}")
    all_mse = []
    for layer_idx in sorted(all_layer_errors.keys()):
        errors = all_layer_errors[layer_idx]
        if errors:
            avg_mse = sum(errors.values()) / len(errors)
            max_name = max(errors, key=errors.get)
            max_mse = errors[max_name]
            all_mse.extend(errors.values())
            mt = config.mixer_types[layer_idx]
            n_q = len(errors)
            print(
                f"  Layer {layer_idx:2d} ({mt}): {n_q} linears quantized, "
                f"avg_MSE={avg_mse:.3e} worst={max_name} ({max_mse:.3e})"
            )
    if all_mse:
        print(f"\n  Global avg MSE: {sum(all_mse) / len(all_mse):.3e}")
        print(f"  Global max MSE: {max(all_mse):.3e}")
        print(f"  Total quantized linears: {len(all_mse)}")

    hs_store.cleanup()
    print(f"\n[DONE] Dense Flashinfer NVFP4 quantization complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Dense Flashinfer NVFP4 Quantization for MiniCPM-SALA"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calib-data", required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=131072)
    parser.add_argument("--mse-iters", type=int, default=120)
    parser.add_argument("--mse-max-shrink", type=float, default=0.60)
    parser.add_argument("--mse-error-norm", type=float, default=2.0)
    parser.add_argument("--smooth-alpha", type=float, default=0.5)
    args = parser.parse_args()

    print("=" * 70)
    print("  Dense Flashinfer NVFP4 ALL-Params Quantizer for MiniCPM-SALA")
    print("  ALL linears = FP4 | Only norms/embed/lm_head = BF16")
    print("  Calibration: forced dense (no sparse routing)")
    print("=" * 70)
    print(f"  smooth_alpha : {args.smooth_alpha}")
    print(f"  mse_iters    : {args.mse_iters}")
    print(f"  mse_shrink   : {args.mse_max_shrink}")
    print(f"  mse_norm     : {args.mse_error_norm}")
    print(f"  max_samples  : {args.max_samples}")
    print(f"  max_len      : {args.max_len}")
    print()

    t0 = time.time()
    quantize_model(args)
    print(f"\n[INFO] Total time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
