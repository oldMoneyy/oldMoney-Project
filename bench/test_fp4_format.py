import torch
import json

# Match your model's config
H = 2304       # hidden_size (check your model's config.json)
I = 6400       # intermediate_size (check your model's config.json)

# Use whichever quantizer your GPU uses
from sglang.srt.utils.common import is_sm120_supported
if is_sm120_supported():
    from flashinfer import fp4_quantize
    print("Using flashinfer fp4_quantize (SM120+)")
else:
    from sgl_kernel import scaled_fp4_quant as fp4_quantize
    print("Using sgl_kernel scaled_fp4_quant")

# Also get silu_and_mul
from sgl_kernel import silu_and_mul

torch.manual_seed(42)
device = "cuda"

# Test 1: fp4_quantize format verification
T = 4  # small batch
x = torch.randn(T, H, dtype=torch.bfloat16, device=device)
scale_inv = torch.tensor(1.0 / 0.5, dtype=torch.float32, device=device)

x_fp4, x_sf = fp4_quantize(x, scale_inv)

print(f"\n=== fp4_quantize output ===")
print(f"input shape: {x.shape}, dtype: {x.dtype}")
print(f"x_fp4 shape: {x_fp4.shape}, dtype: {x_fp4.dtype}")
print(f"x_sf shape:  {x_sf.shape}, dtype: {x_sf.dtype}")
print(f"x_sf raw dtype (view as uint8): {x_sf.view(torch.uint8).shape}")
print(f"x_fp4 first 32 bytes: {x_fp4[0, :32].tolist()}")
print(f"x_sf first 16 values: {x_sf.view(torch.uint8)[0, :16].tolist()}")

# Test 2: silu_and_mul output shape
gate_up = torch.randn(T, 2 * I, dtype=torch.bfloat16, device=device)
out = torch.empty(T, I, dtype=torch.bfloat16, device=device)
silu_and_mul(gate_up, out)

print(f"\n=== silu_and_mul output ===")
print(f"input shape: {gate_up.shape}")
print(f"output shape: {out.shape}, dtype: {out.dtype}")

# Test 3: the full silu_and_mul -> fp4_quantize chain
out_fp4, out_sf = fp4_quantize(out, scale_inv)
print(f"\n=== silu_and_mul -> fp4_quantize chain ===")
print(f"out_fp4 shape: {out_fp4.shape}, dtype: {out_fp4.dtype}")
print(f"out_sf shape:  {out_sf.shape}, dtype: {out_sf.dtype}")

# Test 4: verify round-trip with a known pattern
# 16 elements = 1 scale block
known = torch.tensor([[1.0, -2.0, 0.5, 3.0, -1.5, 0.0, 4.0, -6.0,
                        2.0, -0.5, 1.5, -3.0, 0.25, -4.0, 5.0, -1.0]],
                      dtype=torch.bfloat16, device=device)
# Pad to H width
known_padded = torch.zeros(1, H, dtype=torch.bfloat16, device=device)
known_padded[0, :16] = known[0]

kfp4, ksf = fp4_quantize(known_padded, torch.tensor(1.0, dtype=torch.float32, device=device))
print(f"\n=== Known pattern round-trip ===")
print(f"Input first 16:  {known_padded[0, :16].tolist()}")
print(f"FP4 first 8 bytes (16 nibbles): {kfp4[0, :8].tolist()}")
print(f"Scale first value (uint8): {ksf.view(torch.uint8)[0, 0].item()}")

# Test 5: dump model config
try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained("/opt/model_nvfp4_dense_all", trust_remote_code=True)
    print(f"\n=== Model config ===")
    print(f"hidden_size: {cfg.hidden_size}")
    print(f"intermediate_size: {cfg.intermediate_size}")
    print(f"num_hidden_layers: {cfg.num_hidden_layers}")
    print(f"num_attention_heads: {cfg.num_attention_heads}")
    print(f"num_key_value_heads: {cfg.num_key_value_heads}")
    print(f"head_dim: {cfg.hidden_size // cfg.num_attention_heads}")
except Exception as e:
    print(f"Could not load config: {e}")
