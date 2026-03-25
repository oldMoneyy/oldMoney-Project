#!/usr/bin/env python3
"""
Activation-Aware NVFP4 Quantization for MiniCPM-SALA (Blackwell Ready)
======================================================================

Philosophy:
  1. No GPTQ error propagation (Preserves the fragile FP4 E2M1 grid).
  2. Pure AWQ approach: Use H_diag (Activation magnitude) to weight MSE search.
  3. Lightweight Observer: K floats instead of KxK matrix (No Cholesky OOM).
  4. SGLang/ModelOpt Strict Export: Full metadata, sharding, and exclude_modules.
  
Update: Now strictly processes ONLY the MLP layers (gate_proj, up_proj, down_proj) 
        within lightning layers to NVFP4. All other parameters (Attention, 
        non-lightning layers) are kept in BF16 and properly excluded in config.
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

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

# ====================================================================
# NVFP4 Constants
# ====================================================================
FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
NVFP4_SCALE_FACTOR = FP4_E2M1_MAX * FP8_E4M3_MAX  # 2688.0
NVFP4_GROUP_SIZE = 16

_E2M1_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
_SORTED_GRID = torch.cat([-_E2M1_POS.flip(0)[:-1], _E2M1_POS])
_MIDPOINTS = (_SORTED_GRID[:-1] + _SORTED_GRID[1:]) / 2.0
_GRID_IDX_TO_4BIT = np.array([15, 14, 13, 12, 11, 10, 9, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint8)

# ====================================================================
# S1 & S2  E2M1 Quantisation & Packer
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
# S3  Activation-Aware Block Scale Search (AWQ Core)
# ====================================================================

def compute_awq_block_scales(
    W: torch.Tensor,
    global_sf: float,
    H_diag: torch.Tensor,
    group_size: int = NVFP4_GROUP_SIZE,
    search_iters: int = 100,
    max_shrink: float = 0.80,
    error_norm: float = 2.4,
) -> torch.Tensor:
    N, K = W.shape
    num_groups = K // group_size
    W_grouped = W.reshape(N, num_groups, group_size)
    
    # Activation weighting: Use the sum of squared features as absolute weight, protecting large activations
    act_weight = H_diag.reshape(1, num_groups, group_size).abs()
    act_weight = act_weight / (act_weight.max() + 1e-6)
    
    init_scales = W_grouped.abs().amax(dim=-1) * global_sf / FP4_E2M1_MAX
    init_scales = init_scales.clamp(min=torch.finfo(torch.float8_e4m3fn).tiny)
    
    best_scales = init_scales.clone()
    best_error = torch.full((N, num_groups), float('inf'), device=W.device, dtype=W.dtype)
    
    for i in range(search_iters):
        shrink = 1.0 - i * max_shrink / search_iters
        candidate_scales = (shrink * init_scales).clamp(
            min=torch.finfo(torch.float8_e4m3fn).tiny, max=FP8_E4M3_MAX
        ).to(torch.float8_e4m3fn).to(torch.float32).clamp(min=1e-12)
        
        W_scaled = W_grouped * global_sf / candidate_scales.unsqueeze(-1)
        W_q = quantize_to_e2m1(W_scaled)
        W_deq = W_q * candidate_scales.unsqueeze(-1) / global_sf
        
        # Core: MSE search weighted by activation data
        error = ((W_grouped - W_deq).abs().pow(error_norm) * act_weight).sum(dim=-1) 
        
        improved = error < best_error
        best_error[improved] = error[improved]
        best_scales[improved] = candidate_scales[improved]
    
    return best_scales

# ====================================================================
# S4  Lightweight Observer (OOM-Safe)
# ====================================================================

class ActAwareObserver:
    def __init__(self, columns: int, device: torch.device):
        self.H_diag = torch.zeros(columns, device=device, dtype=torch.float64)
        self.act_amax = 0.0
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor):
        if inp.dim() == 3: inp = inp.reshape(-1, inp.shape[-1])
        elif inp.dim() == 1: inp = inp.unsqueeze(0)
        self.nsamples += inp.shape[0]
        self.H_diag += (inp.float() ** 2).sum(dim=0)
        self.act_amax = max(self.act_amax, inp.abs().max().item())

    def get_h_diag(self):
        return (self.H_diag / max(self.nsamples, 1)).float()

# ====================================================================
# S6-S8 Cache & Data Utils
# ====================================================================
def nuke_caches():
    try:
        from fla.utils import tensor_cache
        tensor_cache.clear()
    except: pass
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

class HiddenStateStore:
    def __init__(self, tmp_dir: str): self._states = {}
    def save(self, idx: int, tensor: torch.Tensor): self._states[idx] = tensor.detach().clone()
    def load(self, idx: int, device: torch.device): return self._states[idx]
    def cleanup(self):
        self._states.clear()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    def __len__(self): return len(self._states)

def load_calibration_data(tokenizer, data_path, max_samples, max_len):
    samples = []
    with open(data_path) as f:
        for line in f:
            enc = tokenizer(json.loads(line.strip())["question"], max_length=max_len, truncation=True, return_tensors="pt")
            samples.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
            if len(samples) >= max_samples: break
    if len(samples) < max_samples:
        orig = samples.copy()
        while len(samples) < max_samples: samples.extend(orig)
        samples = samples[:max_samples]
    total_tokens = sum(s["input_ids"].shape[1] for s in samples)
    print(f"[INFO] {len(samples)} samples, {total_tokens:,} tokens (max_len={max_len})")
    return samples

def find_all_linears(module: nn.Module):
    return {n: m for n, m in module.named_modules() if isinstance(m, nn.Linear)}

# ====================================================================
# S10  Per-Layer Quantisation (Strictly MLP only)
# ====================================================================

def _quantize_layer(layer, layer_idx, config, device, calib_data, hs_store, all_masks, all_pos_ids, args, quantized_tensors, original_tensors):
    all_linears = find_all_linears(layer)
    
    # Filter for MLP linear layers ONLY
    mlp_suffixes = ["gate_proj", "up_proj", "down_proj"]
    linears = {n: m for n, m in all_linears.items() if any(n.endswith(s) for s in mlp_suffixes)}
    linear_names = sorted(linears.keys())
    
    # Determine non-MLP linears to exclude from quantization explicitly (Attention Q/K/V/O etc)
    excluded_linears = [n for n in all_linears.keys() if n not in linear_names]
    exclude_patterns = [f"model.layers.{layer_idx}.{n}" for n in excluded_linears]

    print(f"  MLP Linears selected for NVFP4: {linear_names}")
    
    if not linear_names:
        print("  No MLP linears found, keeping layer entirely in BF16.")
        for name, param in layer.named_parameters():
            original_tensors[f"model.layers.{layer_idx}.{name}"] = param.data.cpu().clone()
        return exclude_patterns

    observers = {n: ActAwareObserver(m.weight.shape[1], device) for n, m in linears.items()}
    hooks = [linears[n].register_forward_hook(lambda m, i, o, name=n: observers[name].add_batch(i[0].data)) for n in linear_names]

    print(f"  Collecting Activations ({len(calib_data)} samples)...")
    nuke_caches()
    with torch.no_grad():
        for i in range(len(calib_data)):
            inp = hs_store.load(i, device)
            try: layer(inp, attention_mask=all_masks[i].to(device), position_ids=all_pos_ids[i].to(device), use_cache=False)
            except: pass
            del inp
            if (i + 1) % 4 == 0: nuke_caches()

    for h in hooks: h.remove()

    merge_groups = {
        "gate_up": [n for n in linear_names if any(n.endswith(s) for s in ["gate_proj", "up_proj"])],
    }
    merged_names = set(sum(merge_groups.values(), []))
    standalone = [n for n in linear_names if n not in merged_names]

    per_linear_amax = {n: linears[n].weight.data.abs().max().item() for n in linear_names}
    fused_global_sf = {}

    for group_name, members in merge_groups.items():
        if not members: continue
        max_amax = max(per_linear_amax[m] for m in members)
        gsf = NVFP4_SCALE_FACTOR / max(max_amax, 1e-12)
        for m in members: fused_global_sf[m] = gsf
        print(f"    [{group_name}] fused global_sf={gsf:.4f} (max_amax={max_amax:.6f})")

    for ln_name in standalone:
        fused_global_sf[ln_name] = NVFP4_SCALE_FACTOR / max(per_linear_amax[ln_name], 1e-12)

    print(f"  Running Act-Aware MSE-RTN (Protecting Outliers)...")
    for ln_name in linear_names:
        full_name = f"model.layers.{layer_idx}.{ln_name}"
        ln_mod = linears[ln_name]
        W = ln_mod.weight.data.to(device).float()
        N, K = W.shape

        global_sf = fused_global_sf[ln_name]
        w_scale_2_ckpt = torch.tensor([NVFP4_SCALE_FACTOR / global_sf / NVFP4_SCALE_FACTOR], dtype=torch.float32)

        t1 = time.time()
        H_diag = observers[ln_name].get_h_diag()
        
        bscales = compute_awq_block_scales(
            W, global_sf, H_diag, NVFP4_GROUP_SIZE,
            search_iters=args.mse_iters, max_shrink=args.mse_max_shrink, error_norm=args.mse_error_norm,
        )
        
        num_groups = K // NVFP4_GROUP_SIZE
        e2m1_vals = torch.zeros_like(W)
        Q_deq = torch.zeros_like(W)
        
        for g in range(num_groups):
            gs, ge = g * NVFP4_GROUP_SIZE, (g + 1) * NVFP4_GROUP_SIZE
            bsf = bscales[:, g]
            w_scaled = W[:, gs:ge] * global_sf / bsf.unsqueeze(1)
            q = quantize_to_e2m1(w_scaled)
            e2m1_vals[:, gs:ge] = q
            Q_deq[:, gs:ge] = q * bsf.unsqueeze(1) / global_sf

        dt = time.time() - t1
        mse_val = (Q_deq - W).pow(2).mean().item()
        print(f"    {ln_name} [{N}x{K}] {dt:.1f}s | Act-Aware MSE={mse_val:.3e}")

        ln_mod.weight.data = Q_deq.to(ln_mod.weight.dtype)
        packed_fp4 = pack_e2m1_to_uint8(e2m1_vals)
        block_sf_fp8 = bscales.clamp(min=torch.finfo(torch.float8_e4m3fn).tiny, max=FP8_E4M3_MAX).to(torch.float8_e4m3fn).cpu()
        a_scale_ckpt = torch.tensor([max(observers[ln_name].act_amax, 1e-12) / NVFP4_SCALE_FACTOR], dtype=torch.float32)

        quantized_tensors[f"{full_name}.weight"] = packed_fp4.cpu()
        quantized_tensors[f"{full_name}.weight_scale"] = block_sf_fp8
        quantized_tensors[f"{full_name}.weight_scale_2"] = w_scale_2_ckpt
        quantized_tensors[f"{full_name}.input_scale"] = a_scale_ckpt

        del W, Q_deq, e2m1_vals, bscales, H_diag
        gc.collect()

    for name, param in layer.named_parameters():
        full_name = f"model.layers.{layer_idx}.{name}"
        if not any(name.startswith(ln + ".weight") for ln in linear_names):
            original_tensors[full_name] = param.data.cpu().clone()

    return exclude_patterns

# ====================================================================
# S11 Checkpoint Saving (Restored Sharding & SGLang Config Patches)
# ====================================================================

def save_checkpoint(quantized_tensors, original_tensors, args, config, exclude_modules):
    os.makedirs(args.output, exist_ok=True)
    state_dict = {**quantized_tensors, **original_tensors}
    from safetensors.torch import save_file

    MAX_SHARD = 4 * 1024 * 1024 * 1024 # 4GB sharding
    total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
    print(f"\n[Phase 4] Saving to {args.output} (Total: {total_size / 1024**3:.2f} GB)...")

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
            for tn in data: weight_map[tn] = new_name
            print(f"  Saved {new_name}")
            
        with open(os.path.join(args.output, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)

    # Copy raw configs first
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"]:
        src = os.path.join(args.input, fname)
        if os.path.exists(src): shutil.copy2(src, os.path.join(args.output, fname))
    for fname in glob.glob(os.path.join(args.input, "*.py")):
        shutil.copy2(fname, os.path.join(args.output, os.path.basename(fname)))

    # --- SGLANG METADATA PATCHES (CRITICAL) ---

    # Patch 1: hf_quant_config.json
    with open(os.path.join(args.output, "hf_quant_config.json"), "w") as f:
        json.dump({"quantization": {
            "quant_algo": "NVFP4",
            "kv_cache_quant_algo": "auto",
            "group_size": NVFP4_GROUP_SIZE,
            "exclude_modules": exclude_modules
        }}, f, indent=2)

    # Patch 2: config.json quantization_config (For SGLang auto-detect)
    cfg_path = os.path.join(args.output, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f: cfg = json.load(f)
        cfg["quantization_config"] = {
            "quant_method": "modelopt", 
            "quant_algo": "NVFP4",
            "kv_cache_quant_algo": "auto", 
            "group_size": NVFP4_GROUP_SIZE,
            "exclude_modules": exclude_modules,
        }
        with open(cfg_path, "w") as f: json.dump(cfg, f, indent=2)
        print(f"  Patched config.json with quantization_config and {len(exclude_modules)} exclude_modules.")

# ====================================================================
# Main Engine
# ====================================================================

def quantize_model(args):
    device = torch.device("cuda:0")
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(args.input, trust_remote_code=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = 2

    calib_data = load_calibration_data(tokenizer, args.calib_data, args.max_samples, args.max_len)
    
    print(f"[INFO] Loading model (CPU, bf16)...")
    config = AutoConfig.from_pretrained(args.input, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.input, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="cpu", config=config, attn_implementation="flash_attention_2"
    )
    model.eval()
    model.config.use_cache = False

    num_layers = config.num_hidden_layers
    # Layers to keep in BF16 even if they are lightning-attn.
    # Layers 26-28 are the last lightning block before BF16 output layers (29-31).
    # Keeping them in BF16 preserves reasoning convergence at ~0.9GB extra cost.
    bf16_override = set()
    if args.bf16_late_layers:
        bf16_override = {26, 27, 28}
        print(f"[INFO] Keeping layers {sorted(bf16_override)} in BF16 (late-layer protection)")
        
    quant_layers = [i for i, mt in enumerate(config.mixer_types) if mt in ("lightning", "lightning_attn", "lightning-attn") and i not in bf16_override]
    
    embed = model.model.embed_tokens.to(device)
    hs_store = HiddenStateStore(args.tmp_dir)
    all_masks, all_pos_ids = [], []

    print("\n[Phase 1] Embedding forward...")
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
            if (i + 1) % 8 == 0: nuke_caches()

    embed = embed.cpu()
    nuke_caches()

    quantized_tensors, original_tensors = {}, {}
    original_tensors["model.embed_tokens.weight"] = model.model.embed_tokens.weight.data.clone()
    
    exclude_modules = []

    for layer_idx in range(num_layers):
        t0 = time.time()
        is_quant = layer_idx in quant_layers
        tag = "Act-Aware NVFP4 (Lightning MLP Only)" if is_quant else "BF16"
        
        print(f"\n{'='*70}\n[Layer {layer_idx}/{num_layers-1}] -> {tag}\n{'='*70}")
        layer = model.model.layers[layer_idx].to(device)
        layer.eval()

        if is_quant:
            # Captures exact sub-modules (like lightning attention matrices) excluded in quantization
            exc_patterns = _quantize_layer(layer, layer_idx, config, device, calib_data, hs_store, all_masks, all_pos_ids, args, quantized_tensors, original_tensors)
            exclude_modules.extend(exc_patterns)
        else:
            print("  Keeping BF16 (sparse/minicpm layer)")
            for name, param in layer.named_parameters():
                original_tensors[f"model.layers.{layer_idx}.{name}"] = param.data.cpu().clone()
            exclude_modules.append(f"model.layers.{layer_idx}.*")

        with torch.no_grad():
            for i in range(len(calib_data)):
                inp = hs_store.load(i, device)
                out = layer(inp, attention_mask=all_masks[i].to(device), position_ids=all_pos_ids[i].to(device), use_cache=False)
                hs_store.save(i, out[0])

        layer = layer.cpu()
        model.model.layers[layer_idx] = layer
        nuke_caches()
        print(f"  Layer {layer_idx} done in {time.time() - t0:.1f}s")

    original_tensors["model.norm.weight"] = model.model.norm.weight.data.cpu().clone()
    if not config.tie_word_embeddings:
        original_tensors["lm_head.weight"] = model.lm_head.weight.data.cpu().clone()

    save_checkpoint(quantized_tensors, original_tensors, args, config, exclude_modules)
    hs_store.cleanup()
    print("\n[DONE] Activation-Aware NVFP4 quantisation complete!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calib-data", required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=131072)
    parser.add_argument("--mse-iters", type=int, default=200)
    parser.add_argument("--mse-max-shrink", type=float, default=0.60)
    parser.add_argument("--mse-error-norm", type=float, default=2.0)
    parser.add_argument("--tmp-dir", type=str, default="/tmp/nvfp4_awq_hs")
    parser.add_argument("--bf16-late-layers", action="store_true",
                        help="Keep layers 26-28 in BF16 to protect reasoning convergence (~0.9GB extra)")
    args = parser.parse_args()

    print("=" * 70)
    print("  Activation-Aware NVFP4 Quantiser for MiniCPM-SALA (Lightning MLPs ONLY)")
    print("  (Perfectly matched for Blackwell FP8-Scales & Group=16)")
    print("=" * 70)
    
    t0 = time.time()
    quantize_model(args)
    print(f"\n[INFO] Total time: {(time.time() - t0)/60:.1f} min")

if __name__ == "__main__":
    main()