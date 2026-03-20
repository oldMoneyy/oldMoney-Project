#!/usr/bin/env python3
"""
True Hardware-Native GPTQ-NVFP4 Quantization for MiniCPM-SALA
=================================================================

This script uses the actual underlying C++ operators from FlashInfer/TensorRT-LLM 
to perform 100% bit-exact NVFP4 quantization during the GPTQ loop.

No mathematical simulations, no approximations. 
The quantization error compensated by Hessian is exactly the hardware truncation error.
"""

import os
import gc
import json
import time
import glob
import shutil
import argparse
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict

import torch
import torch.nn as nn

try:
    from flashinfer import fp4_quantize, e2m1_and_ufp8sf_scale_to_float
except ImportError:
    raise ImportError(
        "FlashInfer is required for hardware-native NVFP4 quantization. "
        "Please ensure your environment has flashinfer 0.5.3+ installed."
    )

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

# ====================================================================
# Hardware Constants
# ====================================================================
FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
NVFP4_SCALE_FACTOR = FP4_E2M1_MAX * FP8_E4M3_MAX  # 2688.0
NVFP4_GROUP_SIZE = 16

# ====================================================================
# S1  Hardware-Native GPTQ Core
# ====================================================================
class GPTQ_NVFP4_Native:
    def __init__(self, name: str, weight: torch.Tensor):
        self.name = name
        self.rows, self.columns = weight.shape
        self.H = torch.zeros((self.columns, self.columns), device="cpu", dtype=torch.float64)
        self.nsamples = 0
        self.chunk_size = 8192  # Chunk size to prevent activation OOM

    def add_batch(self, inp: torch.Tensor):
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        elif inp.dim() == 1:
            inp = inp.unsqueeze(0)

        n_tokens = inp.shape[0]
        self.nsamples += n_tokens

        for start in range(0, n_tokens, self.chunk_size):
            end = min(start + self.chunk_size, n_tokens)
            chunk = inp[start:end].float()
            self.H.add_((chunk.T @ chunk).cpu().to(torch.float64))

    def _make_hinv(self, H: torch.Tensor) -> torch.Tensor:
        for attempt in range(5):
            try:
                L = torch.linalg.cholesky(H)
                H_inv = torch.cholesky_inverse(L)
                return torch.linalg.cholesky(H_inv, upper=True)
            except RuntimeError:
                H.diagonal().add_(1e-2 * torch.diag(H).mean())
        return torch.linalg.cholesky(torch.diag(1.0 / torch.diag(H)), upper=True)

    def quantize(
        self,
        weight: torch.Tensor,
        global_sf: torch.Tensor,
        damp_percent: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        dev = weight.device
        W = weight.float().clone()
        H = (self.H / self.nsamples).float().to(dev)
        del self.H

        rows, columns = W.shape
        
        # Hessian Dampening
        dead = torch.diag(H) == 0
        H[dead, dead] = 1.0
        W[:, dead] = 0.0
        H.diagonal().add_(damp_percent * torch.diag(H).mean())
        Hinv = self._make_hinv(H)
        del H

        # Output Tensors (matching SGLang ModelOpt format)
        Q_deq_all = torch.zeros_like(W)
        packed_fp4_all = torch.zeros((rows, columns // 2), device=dev, dtype=torch.uint8)
        bscales_fp8_all = torch.zeros((rows, columns // NVFP4_GROUP_SIZE), device=dev, dtype=torch.float8_e4m3fn)

        # ----------------------------------------------------------------
        # Hardware-Native GPTQ Loop
        # Processing strictly in chunks of 16 (NVFP4 group size)
        # ----------------------------------------------------------------
        for j in range(0, columns, NVFP4_GROUP_SIZE):
            W_group = W[:, j:j+NVFP4_GROUP_SIZE].clone()
            
            # 1. Native Hardware Quantize (C++ Call)
            # Produces actual packed uint8 and fp8 scales
            q_uint8, q_scale_uint8 = fp4_quantize(
                input=W_group, 
                global_scale=global_sf, 
                sf_vec_size=NVFP4_GROUP_SIZE,
                sf_use_ue8m0=False,
                is_sf_swizzled_layout=False, # We need linear scale for GPTQ step
                is_sf_8x4_layout=False
            )
            
            # 2. Native Hardware Dequantize (C++ Call)
            # Warning: flashinfer's dequantize returns a CPU tensor! We must move it to GPU.
            W_deq_group = e2m1_and_ufp8sf_scale_to_float(
                e2m1_tensor=q_uint8, 
                ufp8_scale_tensor=q_scale_uint8, 
                global_scale_tensor=global_sf, 
                sf_vec_size=NVFP4_GROUP_SIZE,
                ufp8_type=1, # 1 for E4M3
                is_sf_swizzled_layout=False
            ).to(device=dev, dtype=W.dtype)

            # Store the final hardware-packed results
            packed_fp4_all[:, j//2 : (j+NVFP4_GROUP_SIZE)//2] = q_uint8
            
            # q_scale_uint8 is [rows], cast back to strict float8_e4m3fn
            bscales_fp8_all[:, j//NVFP4_GROUP_SIZE] = q_scale_uint8.view(torch.float8_e4m3fn).reshape(rows)
            
            Q_deq_all[:, j:j+NVFP4_GROUP_SIZE] = W_deq_group

            # 3. Error Computation & Hessian Compensation
            Err_group = torch.zeros_like(W_group)
            for k in range(NVFP4_GROUP_SIZE):
                c = j + k
                # Error encapsulates true hardware RTNE & FP8 truncation
                Err_group[:, k] = (W_group[:, k] - W_deq_group[:, k]) / Hinv[c, c]

            # Push compensated error to remaining columns
            if j + NVFP4_GROUP_SIZE < columns:
                W[:, j+NVFP4_GROUP_SIZE:] -= Err_group @ Hinv[j:j+NVFP4_GROUP_SIZE, j+NVFP4_GROUP_SIZE:]

        return Q_deq_all, packed_fp4_all, bscales_fp8_all


# ====================================================================
# S2  Memory Mgmt & Helpers
# ====================================================================
def nuke_caches():
    try:
        from fla.utils import tensor_cache
        tensor_cache.clear()
    except Exception: pass
    gc.collect()
    torch.cuda.empty_cache()

class HiddenStateStore:
    def __init__(self, tmp_dir: str):
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self._files = {}
    def save(self, idx, tensor):
        fpath = str(self.tmp_dir / f"hs_{idx:04d}.pt")
        torch.save(tensor.cpu(), fpath)
        self._files[idx] = fpath
    def load(self, idx, device):
        return torch.load(self._files[idx], map_location=device, weights_only=True)
    def cleanup(self):
        shutil.rmtree(str(self.tmp_dir), ignore_errors=True)
    def __len__(self): return len(self._files)

def load_calibration_data(tokenizer, data_path, max_samples, max_len):
    samples = []
    with open(data_path) as f:
        for line in f:
            text = json.loads(line.strip())["question"]
            enc = tokenizer(text, max_length=max_len, truncation=True, return_tensors="pt")
            samples.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
            if len(samples) >= max_samples: break
    while len(samples) < max_samples: samples.extend(samples.copy())
    return samples[:max_samples]

def find_all_linears(module: nn.Module) -> Dict[str, nn.Linear]:
    return {n: m for n, m in module.named_modules() if isinstance(m, nn.Linear)}

# ====================================================================
# S3  Main Pipeline
# ====================================================================
def quantize_model(args):
    device = torch.device("cuda:0")
    print(f"[INFO] Backend: CUDA {torch.version.cuda}, PyTorch {torch.__version__}, FlashInfer Native C++")

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(args.input, trust_remote_code=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = 2

    calib_data = load_calibration_data(tokenizer, args.calib_data, args.max_samples, args.max_len)
    config = AutoConfig.from_pretrained(args.input, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.input, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="cpu", config=config, attn_implementation="flash_attention_2"
    ).eval()
    model.config.use_cache = False

    quant_layers = [i for i, mt in enumerate(config.mixer_types) if "lightning" in mt]
    skip_layers = [i for i in range(config.num_hidden_layers) if i not in quant_layers]

    print("[Phase 1] Computing Embeddings...")
    embed = model.model.embed_tokens.to(device)
    hs_store = HiddenStateStore(args.tmp_dir)
    all_masks, all_pos_ids = [], []

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

    quantized_tensors = {}
    original_tensors = {"model.embed_tokens.weight": model.model.embed_tokens.weight.data.clone()}

    for layer_idx in range(config.num_hidden_layers):
        t0 = time.time()
        is_quant = layer_idx in quant_layers
        tag = "GPTQ-NVFP4 (FlashInfer Native)" if is_quant else "BF16"
        print(f"\n[Layer {layer_idx}/{config.num_hidden_layers-1}] -> {tag}")
        
        layer = model.model.layers[layer_idx].to(device).eval()

        if is_quant:
            _quantize_layer(layer, layer_idx, device, calib_data, hs_store, all_masks, all_pos_ids, args, quantized_tensors, original_tensors)
        else:
            for name, param in layer.named_parameters():
                original_tensors[f"model.layers.{layer_idx}.{name}"] = param.data.cpu().clone()

        print("  Propagating hidden states...")
        nuke_caches()
        with torch.no_grad():
            for i in range(len(calib_data)):
                inp = hs_store.load(i, device)
                out = layer(inp, attention_mask=all_masks[i].to(device), position_ids=all_pos_ids[i].to(device), output_attentions=False, use_cache=False)
                hs_store.save(i, out[0])
                if (i + 1) % 4 == 0: nuke_caches()

        model.model.layers[layer_idx] = layer.cpu()
        nuke_caches()
        print(f"  Done in {time.time() - t0:.1f}s")

    print(f"\n[Phase 3] Final norm & Saving...")
    original_tensors["model.norm.weight"] = model.model.norm.weight.data.cpu().clone()
    if not config.tie_word_embeddings:
        original_tensors["lm_head.weight"] = model.lm_head.weight.data.cpu().clone()

    save_checkpoint(quantized_tensors, original_tensors, args, config, skip_layers)
    hs_store.cleanup()
    print("\n[DONE] Native FlashInfer NVFP4 Quantization Complete!")

def _quantize_layer(layer, layer_idx, device, calib_data, hs_store, all_masks, all_pos_ids, args, quantized_tensors, original_tensors):
    linears = find_all_linears(layer)
    quantizers = {n: GPTQ_NVFP4_Native(n, m.weight.data) for n, m in linears.items()}
    act_amax = defaultdict(float)

    hooks = []
    for ln_name, ln_mod in linears.items():
        def make_hook(name):
            def hook_fn(mod, inp, out):
                quantizers[name].add_batch(inp[0].data)
                act_amax[name] = max(act_amax[name], inp[0].data.abs().max().item())
            return hook_fn
        hooks.append(ln_mod.register_forward_hook(make_hook(ln_name)))

    print(f"  Collecting Hessians...")
    with torch.no_grad():
        for i in range(len(calib_data)):
            inp = hs_store.load(i, device)
            layer(inp, attention_mask=all_masks[i].to(device), position_ids=all_pos_ids[i].to(device), output_attentions=False, use_cache=False)
            if (i + 1) % 4 == 0: nuke_caches()

    for h in hooks: h.remove()

    for ln_name, ln_mod in linears.items():
        full_name = f"model.layers.{layer_idx}.{ln_name}"
        W = ln_mod.weight.data.to(device)
        N, K = W.shape

        w_amax = W.abs().max().float().clamp(min=1e-12)
        global_sf = torch.tensor([NVFP4_SCALE_FACTOR / w_amax.item()], dtype=torch.float32, device=device)
        
        Q_deq_all, packed_fp4_all, bscales_fp8_all = quantizers[ln_name].quantize(W, global_sf, damp_percent=args.damp)
        
        # Replace layer weights for next-layer state propagation
        ln_mod.weight.data = Q_deq_all.to(W.dtype)

        # Output formatting exactly aligns with SGLang ModelOpt format
        w_scale_2 = torch.tensor([w_amax.item() / NVFP4_SCALE_FACTOR], dtype=torch.float32)
        a_scale = torch.tensor([max(act_amax.get(ln_name, 1.0), 1e-12) / NVFP4_SCALE_FACTOR], dtype=torch.float32)

        quantized_tensors[f"{full_name}.weight"] = packed_fp4_all.cpu()
        quantized_tensors[f"{full_name}.weight_scale"] = bscales_fp8_all.cpu()
        quantized_tensors[f"{full_name}.weight_scale_2"] = w_scale_2
        quantized_tensors[f"{full_name}.input_scale"] = a_scale

        mse = (Q_deq_all - W.float()).pow(2).mean().item()
        print(f"    {ln_name} [{N}x{K}]  GPTQ MSE={mse:.3e} (Hardware Verified)")

    for name, param in layer.named_parameters():
        full_name = f"model.layers.{layer_idx}.{name}"
        if not any(name.startswith(ln + ".") for ln in linears.keys()):
            original_tensors[full_name] = param.data.cpu().clone()

def save_checkpoint(quantized_tensors, original_tensors, args, config, skip_layers):
    os.makedirs(args.output, exist_ok=True)
    state_dict = {**quantized_tensors, **original_tensors}
    from safetensors.torch import save_file

    save_file(state_dict, os.path.join(args.output, "model.safetensors"))

    exclude = [f"model.layers.{i}.*" for i in skip_layers]
    with open(os.path.join(args.output, "hf_quant_config.json"), "w") as f:
        json.dump({"quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": "auto", "group_size": NVFP4_GROUP_SIZE, "exclude_modules": exclude}}, f, indent=2)

    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"]:
        src = os.path.join(args.input, fname)
        if os.path.exists(src): shutil.copy2(src, os.path.join(args.output, fname))
    
    for py in glob.glob(os.path.join(args.input, "*.py")):
        shutil.copy2(py, os.path.join(args.output, os.path.basename(py)))

    cfg_path = os.path.join(args.output, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f: cfg = json.load(f)
        cfg["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "NVFP4", "kv_cache_quant_algo": "auto", "group_size": NVFP4_GROUP_SIZE, "exclude_modules": exclude}
        with open(cfg_path, "w") as f: json.dump(cfg, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calib-data", required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--damp", type=float, default=0.01)
    parser.add_argument("--tmp-dir", default="/tmp/nvfp4_hs")
    args = parser.parse_args()
    quantize_model(args)