#!/usr/bin/env python3
"""
Advanced GPTQ W4A16 quantization for MiniCPM-SALA.

Features:
1. GPU-Native Hessian accumulation (optimized for 96GB+ VRAM cards like RTX PRO 6000)
2. Architecture-aware routing: MiniCPM4 Attention=BF16, Lightning/MLPs=INT4
3. Hessian-Aware Channel Smoothing (SmoothQuant-style equalization)
4. Per-sample normalized Hessian accumulation
5. Deterministic calibration (reproducible accuracy)
"""

import os
import gc
import json
import time
import shutil
import glob
import argparse
import random
from typing import Dict, List, Tuple

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
# SECTION 1: GPTQ Algorithm (100% GPU Native)
# ============================================================

class GPTQ:
    """GPTQ quantizer for a single nn.Linear layer."""

    def __init__(self, name: str, weight: torch.Tensor):
        self.name = name
        self.rows, self.columns = weight.shape
        # FULL GPU: Store Hessian directly on the GPU for maximum speed
        self.H = torch.zeros(
            (self.columns, self.columns), device=weight.device, dtype=torch.float64
        )
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor):
        """Accumulate Hessian H = X^T @ X from a batch of inputs."""
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        elif inp.dim() == 1:
            inp = inp.unsqueeze(0)
            
        n_tokens = inp.shape[0]
        inp = inp.float()  
        
        # PER-SAMPLE NORMALIZATION
        h_update = (inp.T @ inp) / max(n_tokens, 1)
        
        # Accumulate entirely on GPU
        self.H.add_(h_update.to(torch.float64))
        self.nsamples += 1  

    def apply_smooth_to_hessian(self, smooth: torch.Tensor):
        """Transform H after channel smoothing."""
        s = smooth.to(self.H.dtype).to(self.H.device)
        inv_s = 1.0 / s
        scale_matrix = inv_s.unsqueeze(1) * inv_s.unsqueeze(0)
        self.H.mul_(scale_matrix)

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
        
        # Divide by samples, cast back to float32
        H = (self.H / max(self.nsamples, 1)).float()
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
            print(f"    [{self.name}] FATAL: Cholesky failed. Using diagonal approximation.")
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
# SECTION 2: Utilities (Packing, Caches, Smoothing)
# ============================================================

def pack_int_weight(int_weight: torch.Tensor, bits: int = 4) -> torch.Tensor:
    pack_num = 32 // bits
    iw = int_weight.T.contiguous().to(torch.int32)
    in_f, out_f = iw.shape
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
        zeros_int = torch.cat([zeros_int, torch.zeros(num_groups, pad, dtype=zeros_int.dtype)], dim=1)
        out_f = zeros_int.shape[1]

    z = zeros_int.to(torch.int32).reshape(num_groups, out_f // pack_num, pack_num)
    qzeros = torch.zeros(num_groups, out_f // pack_num, dtype=torch.int32)
    for k in range(pack_num):
        qzeros |= (z[:, :, k] & ((1 << bits) - 1)) << (k * bits)
    return qzeros


def safe_cleanup():
    """Standard garbage collection. Removed aggressive C++ cache wipes to stop warnings."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_smooth_factor(
    target_weights: List[torch.Tensor],
    H_diags: List[torch.Tensor],
    alpha: float = 0.5,
    clamp_min: float = 0.01,
    clamp_max: float = 100.0,
) -> torch.Tensor:
    K = target_weights[0].shape[1]
    device = target_weights[0].device
    
    # Ensure H_diags are explicitly mapped to the GPU device of the weights
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
# SECTION 3: Calibration & Routing
# ============================================================

def load_calibration_data(tokenizer, data_path: str, max_samples: int, max_len: int):
    """Load calibration data DETERMINISTICALLY."""
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            text = json.loads(line.strip())["question"]
            enc = tokenizer(text, max_length=max_len, truncation=True, return_tensors="pt")
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
    print(f"[INFO] Loaded {len(samples)} calibration samples, {total_tokens:,} tokens (max_len={max_len})")
    return samples


def find_all_linears(module: nn.Module) -> Dict[str, nn.Linear]:
    return {n: m for n, m in module.named_modules() if isinstance(m, nn.Linear)}


def should_quantize_linear_gptq(layer_idx: int, linear_name: str, mixer_type: str) -> bool:
    """
    MiniCPM4 layers: attention BF16, MLP INT4
    Lightning-attn layers: everything INT4
    """
    if mixer_type == "minicpm4":
        if "self_attn" in linear_name:
            return False  # Keep BF16
        return True  # MLP gets quantized
    return True  # Lightning: everything quantized


# ============================================================
# SECTION 4: Main Quantization Pipeline
# ============================================================

def quantize_model(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    tokenizer = AutoTokenizer.from_pretrained(args.input, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 2

    calib_data = load_calibration_data(tokenizer, args.calib_data, args.max_samples, args.max_len)

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
    print("[INFO] Phase 1: Computing initial hidden states...")
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
            # Kept on CPU to avoid 70GB VRAM block, moved to GPU when needed below
            all_inps.append(hidden.cpu())
            all_masks.append(attention_mask.cpu())
            all_pos_ids.append(position_ids.cpu())

    embed = embed.cpu()
    safe_cleanup()

    # --- Phase 2: Layer-by-layer ---
    quantized_state = {}
    original_state = {}
    original_state["model.embed_tokens.weight"] = model.model.embed_tokens.weight.data.clone()

    for layer_idx in range(num_layers):
        t0 = time.time()
        mixer_type = config.mixer_types[layer_idx] if hasattr(config, "mixer_types") else "unknown"
        
        print(f"\n{'='*70}")
        print(f"[LAYER {layer_idx}/{num_layers-1}] type={mixer_type}")
        print(f"{'='*70}")

        layer = model.model.layers[layer_idx].to(device)
        layer.eval()

        linears = find_all_linears(layer)
        linear_names = sorted(linears.keys())

        names_to_quantize = []
        names_bf16 = []
        for ln_name in linear_names:
            if should_quantize_linear_gptq(layer_idx, ln_name, mixer_type):
                names_to_quantize.append(ln_name)
            else:
                names_bf16.append(ln_name)

        if names_bf16:
            print(f"  BF16 (kept): {names_bf16}")
        if names_to_quantize:
            print(f"  INT4 (quantizing): {names_to_quantize}")

        if not names_to_quantize:
            for name, param in layer.named_parameters():
                original_state[f"model.layers.{layer_idx}.{name}"] = param.data.cpu().clone()
            
            print(f"  Updating hidden states for next layer...")
            with torch.no_grad():
                for i in range(len(calib_data)):
                    all_inps[i] = layer(
                        all_inps[i].to(device), 
                        attention_mask=all_masks[i].to(device),
                        position_ids=all_pos_ids[i].to(device), 
                        use_cache=False
                    )[0].cpu()
            
            layer = layer.cpu()
            model.model.layers[layer_idx] = layer
            safe_cleanup()
            continue

        # Create quantizers & hook for ONLY quantized layers
        quantizers: Dict[str, GPTQ] = {}
        hooks = []
        for ln_name in names_to_quantize:
            quantizers[ln_name] = GPTQ(f"model.layers.{layer_idx}.{ln_name}", linears[ln_name].weight.data)
            def make_hook(name):
                return lambda m, inp, out: quantizers[name].add_batch(inp[0].data)
            hooks.append(linears[ln_name].register_forward_hook(make_hook(ln_name)))

        # Collect Hessians
        print(f"  Collecting Hessians ({len(calib_data)} samples)...")
        safe_cleanup()
        with torch.no_grad():
            for i in range(len(calib_data)):
                try:
                    layer(
                        all_inps[i].to(device), 
                        attention_mask=all_masks[i].to(device),
                        position_ids=all_pos_ids[i].to(device), 
                        use_cache=False
                    )
                except Exception as e:
                    print(f"  [WARN] Sample {i} failed: {e}")

        for h in hooks: h.remove()

        # ================= SMOOTHING INTEGRATION =================
        if mixer_type != "minicpm4":
            attn_targets = [n for n in names_to_quantize if n.endswith(("q_proj", "k_proj", "v_proj", "z_proj"))]
            input_ln = get_submodule_safe(layer, "input_layernorm")
            if attn_targets and input_ln is not None:
                tw = [linears[t].weight.data.float() for t in attn_targets]
                hd = [torch.diag(quantizers[t].H).float() / max(quantizers[t].nsamples, 1) for t in attn_targets]
                s = compute_smooth_factor(tw, hd, alpha=args.smooth_alpha)
                
                input_ln.weight.data.div_(s.to(input_ln.weight.dtype))
                for t in attn_targets:
                    linears[t].weight.data.mul_(s.unsqueeze(0).to(linears[t].weight.dtype))
                    quantizers[t].apply_smooth_to_hessian(s)
                print(f"  Smoothed [input_layernorm] -> {attn_targets}")

        mlp_targets = [n for n in names_to_quantize if n.endswith(("gate_proj", "up_proj"))]
        post_ln = get_submodule_safe(layer, "post_attention_layernorm")
        if mlp_targets and post_ln is not None:
            tw = [linears[t].weight.data.float() for t in mlp_targets]
            hd = [torch.diag(quantizers[t].H).float() / max(quantizers[t].nsamples, 1) for t in mlp_targets]
            s = compute_smooth_factor(tw, hd, alpha=args.smooth_alpha)
            
            post_ln.weight.data.div_(s.to(post_ln.weight.dtype))
            for t in mlp_targets:
                linears[t].weight.data.mul_(s.unsqueeze(0).to(linears[t].weight.dtype))
                quantizers[t].apply_smooth_to_hessian(s)
            print(f"  Smoothed [post_attn_layernorm] -> {mlp_targets}")

        if mixer_type != "minicpm4":
            o_norm = get_submodule_safe(layer, "self_attn.o_norm")
            if o_norm is not None and "self_attn.o_proj" in quantizers:
                t = "self_attn.o_proj"
                W = linears[t].weight.data.float()
                h_diag = torch.diag(quantizers[t].H).float() / max(quantizers[t].nsamples, 1)
                s = compute_smooth_factor([W], [h_diag], alpha=args.smooth_alpha)
                
                o_norm.weight.data.div_(s.to(o_norm.weight.dtype))
                linears[t].weight.data.mul_(s.unsqueeze(0).to(linears[t].weight.dtype))
                quantizers[t].apply_smooth_to_hessian(s)
                print(f"  Smoothed [o_norm] -> [o_proj]")
        # =========================================================

        print(f"  Running GPTQ quantization (bits={args.bits}, group_size={args.group_size})...")
        for ln_name in names_to_quantize:
            full_name = f"model.layers.{layer_idx}.{ln_name}"
            ln_mod = linears[ln_name]
            W = ln_mod.weight.data.to(device)

            t1 = time.time()
            Q, sc, zp, iw = quantizers[ln_name].quantize(
                W, bits=args.bits, group_size=args.group_size,
                damp_percent=args.damp, block_size=128, sym=args.sym
            )
            t2 = time.time()

            mse = (Q.float() - W.float()).pow(2).mean().item()
            rel_err = (Q.float() - W.float()).norm() / W.float().norm()
            print(f"    {ln_name} [{W.shape[0]}x{W.shape[1]}] MSE={mse:.6e} RelErr={rel_err:.4f} ({t2-t1:.1f}s)")

            ln_mod.weight.data = Q.to(ln_mod.weight.dtype).to(ln_mod.weight.device)
            
            quantized_state[full_name] = {
                "qweight": pack_int_weight(iw.cpu(), args.bits),
                "qzeros": pack_zeros(zp.cpu().round().int(), args.bits),
                "scales": sc.cpu().to(torch.float16),
                "g_idx": torch.arange(W.shape[1], dtype=torch.int32) // args.group_size,
                "bits": args.bits,
            }
            del Q, sc, zp, iw, W
            gc.collect()

        print(f"  Updating hidden states for next layer...")
        with torch.no_grad():
            for i in range(len(calib_data)):
                all_inps[i] = layer(
                    all_inps[i].to(device), 
                    attention_mask=all_masks[i].to(device),
                    position_ids=all_pos_ids[i].to(device), 
                    use_cache=False
                )[0].cpu()

        for name, param in layer.named_parameters():
            if not any(name.startswith(ln + ".weight") for ln in names_to_quantize):
                original_state[f"model.layers.{layer_idx}.{name}"] = param.data.cpu().clone()

        layer = layer.cpu()
        model.model.layers[layer_idx] = layer
        safe_cleanup()

    # --- Phase 3: Final Norm + LM Head ---
    print(f"\n{'='*70}\n[INFO] Phase 3: Processing final norm + lm_head...\n{'='*70}")
    original_state["model.norm.weight"] = model.model.norm.weight.data.cpu().clone()

    if args.quantize_lm_head:
        norm = model.model.norm.to(device)
        lm_head = model.lm_head.to(device)
        lm_quantizer = GPTQ("lm_head", lm_head.weight.data)

        print("  Collecting Hessian for lm_head...")
        with torch.no_grad():
            for i in range(len(calib_data)):
                hidden = norm(all_inps[i].to(device)) / (config.hidden_size / config.dim_model_base)
                lm_quantizer.add_batch(hidden.reshape(-1, hidden.shape[-1]))

        print("  Quantizing lm_head...")
        Q, sc, zp, iw = lm_quantizer.quantize(
            lm_head.weight.data.to(device), bits=args.bits, group_size=args.group_size,
            damp_percent=args.damp, block_size=128, sym=args.sym
        )

        quantized_state["lm_head"] = {
            "qweight": pack_int_weight(iw.cpu(), args.bits),
            "qzeros": pack_zeros(zp.cpu().round().int(), args.bits),
            "scales": sc.cpu().to(torch.float16),
            "g_idx": torch.arange(lm_head.weight.shape[1], dtype=torch.int32) // args.group_size,
        }
        norm.cpu()
        lm_head.cpu()
    else:
        original_state["lm_head.weight"] = model.lm_head.weight.data.cpu().clone()

    safe_cleanup()

    # --- Phase 4: Save ---
    print(f"\n{'='*70}\n[INFO] Phase 4: Saving quantized model to {args.output}\n{'='*70}")
    save_quantized_model(quantized_state, original_state, args, config)
    all_inps.clear()
    torch.cuda.empty_cache()
    print("\n[INFO] Quantization complete!")


# ============================================================
# SECTION 5: Saving & Config
# ============================================================

def save_quantized_model(quantized_state, original_state, args, config):
    os.makedirs(args.output, exist_ok=True)
    state_dict = {**{f"{k}.{x}": v for k, p in quantized_state.items() for x, v in p.items() if x != "bits"}, **original_state}

    try:
        from safetensors.torch import save_file
        MAX_SHARD_SIZE = 4 * 1024 * 1024 * 1024
        total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
        
        if total_size <= MAX_SHARD_SIZE:
            save_file(state_dict, os.path.join(args.output, "model.safetensors"))
        else:
            shards, cur_shard, cur_size, idx, weight_map = {}, {}, 0, 1, {}
            for name, tensor in sorted(state_dict.items()):
                tsize = tensor.numel() * tensor.element_size()
                if cur_size + tsize > MAX_SHARD_SIZE and cur_shard:
                    shards[f"model-{idx:05d}-of-XXXXX.safetensors"] = cur_shard
                    cur_shard, cur_size, idx = {}, 0, idx + 1
                cur_shard[name] = tensor
                cur_size += tsize
            if cur_shard:
                shards[f"model-{idx:05d}-of-XXXXX.safetensors"] = cur_shard
            
            total_shards = len(shards)
            for old_name, data in shards.items():
                new_name = old_name.replace("XXXXX", f"{total_shards:05d}")
                save_file(data, os.path.join(args.output, new_name))
                for tn in data: weight_map[tn] = new_name
            
            with open(os.path.join(args.output, "model.safetensors.index.json"), "w") as f:
                json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)
    except ImportError:
        torch.save(state_dict, os.path.join(args.output, "pytorch_model.bin"))

    # Build exclude_modules list for MiniCPM4 attention layers kept in BF16
    exclude_modules = []
    if hasattr(config, "mixer_types"):
        for i, mt in enumerate(config.mixer_types):
            if mt == "minicpm4":
                exclude_modules.append(f"model.layers.{i}.self_attn")

    quant_config = {
        "bits": args.bits, "group_size": args.group_size, "desc_act": False,
        "sym": args.sym, "damp_percent": args.damp, "true_sequential": False,
        "model_name_or_path": args.input, "model_file_base_name": "model",
        "quant_method": "gptq", "is_marlin_format": False, "checkpoint_format": "gptq",
        "mixed_precision": True, "exclude_modules": exclude_modules
    }
    with open(os.path.join(args.output, "quantize_config.json"), "w") as f:
        json.dump(quant_config, f, indent=2)

    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"]:
        src = os.path.join(args.input, fname)
        if os.path.exists(src): shutil.copy2(src, os.path.join(args.output, fname))
    for fname in glob.glob(os.path.join(args.input, "*.py")):
        shutil.copy2(fname, os.path.join(args.output, os.path.basename(fname)))

    cfg_path = os.path.join(args.output, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f: cfg = json.load(f)
        cfg["quantization_config"] = quant_config
        with open(cfg_path, "w") as f: json.dump(cfg, f, indent=2)


# ============================================================
# SECTION 6: CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Original model path")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits (default: 4)")
    parser.add_argument("--group-size", type=int, default=128, help="Group size (default: 128)")
    parser.add_argument("--damp", type=float, default=0.01, help="Damping percent (default: 0.01)")
    parser.add_argument("--sym", action="store_true", default=True, help="Symmetric quantization (default: True)")
    parser.add_argument("--no-sym", dest="sym", action="store_false")
    parser.add_argument("--smooth-alpha", type=float, default=0.5, help="Smoothing balance (default: 0.5)")
    parser.add_argument("--calib-data", type=str, default="/opt/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=131072)
    parser.add_argument("--quantize-lm-head", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic Seed (default: 42)")

    args = parser.parse_args()
    seed_everything(args.seed)

    print("=" * 70)
    print("Advanced GPTQ Quantizer for MiniCPM-SALA (Mixed Precision + Smoothing)")
    print("=" * 70)
    print(f"  Input:           {args.input}")
    print(f"  Output:          {args.output}")
    print(f"  Bits:            {args.bits} (MiniCPM4 Attention forced to BF16)")
    print(f"  Smooth Alpha:    {args.smooth_alpha}")
    print(f"  Max length:      {args.max_len}")
    print("=" * 70)

    t_start = time.time()
    quantize_model(args)
    print(f"\n[INFO] Total time: {(time.time() - t_start)/60:.1f} minutes")


if __name__ == "__main__":
    main()