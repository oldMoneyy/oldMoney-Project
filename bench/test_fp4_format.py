import torch

# Model config (from previous run)
H = 4096
I = 16384

from sglang.srt.utils.common import is_sm120_supported
if is_sm120_supported():
    from flashinfer import fp4_quantize
    print("Using flashinfer fp4_quantize (SM120+)")
else:
    from sgl_kernel import scaled_fp4_quant as fp4_quantize
    print("Using sgl_kernel scaled_fp4_quant")

from sgl_kernel import silu_and_mul

torch.manual_seed(42)
device = "cuda"

T = 32  # realistic small batch
scale_inv = torch.tensor(2.0, dtype=torch.float32, device=device)

# ============================================================
# Test 1: Baseline — separate silu_and_mul + fp4_quantize
# ============================================================
gate_up = torch.randn(T, 2 * I, dtype=torch.bfloat16, device=device)
gate_up_copy = gate_up.clone()

out_sep = torch.empty(T, I, dtype=torch.bfloat16, device=device)
silu_and_mul(gate_up, out_sep)
fp4_sep, sf_sep = fp4_quantize(out_sep, scale_inv)

print(f"\n=== Baseline: separate silu_and_mul + fp4_quantize ===")
print(f"out_sep shape: {out_sep.shape}")
print(f"fp4_sep shape: {fp4_sep.shape}, dtype: {fp4_sep.dtype}")
print(f"sf_sep shape:  {sf_sep.shape}, dtype: {sf_sep.dtype}")
print(f"fp4_sep first 16 bytes: {fp4_sep[0, :16].tolist()}")
print(f"sf_sep first 16 bytes:  {sf_sep.view(torch.uint8)[0, :16].tolist()}")

# ============================================================
# Test 2: Try flashinfer fused silu_and_mul + nvfp4 quantize
# ============================================================
try:
    from flashinfer import silu_and_mul_scaled_nvfp4_experts_quantize
    print("\n=== flashinfer silu_and_mul_scaled_nvfp4_experts_quantize available ===")

    # MoE API expects [num_experts, M, 2*N] input
    # Try with num_experts=1
    gate_up_3d = gate_up_copy.unsqueeze(0)  # [1, T, 2*I]
    masked_m = torch.tensor([T], dtype=torch.int32, device=device)

    fp4_fused, sf_fused = silu_and_mul_scaled_nvfp4_experts_quantize(
        gate_up_3d,
        masked_m,
        scale_inv.unsqueeze(0),  # must be 1D [num_experts]
    )

    print(f"fp4_fused shape: {fp4_fused.shape}, dtype: {fp4_fused.dtype}")
    print(f"sf_fused shape:  {sf_fused.shape}, dtype: {sf_fused.dtype}")
    print(f"fp4_fused first 16 bytes: {fp4_fused.view(torch.uint8)[0, :16].tolist()}")
    print(f"sf_fused first 16 bytes:  {sf_fused.view(torch.uint8)[0, :16].tolist()}")

    # Compare with baseline
    fp4_fused_2d = fp4_fused.squeeze(0) if fp4_fused.dim() == 3 else fp4_fused
    sf_fused_flat = sf_fused.view(torch.uint8)
    sf_sep_flat = sf_sep.view(torch.uint8)

    if fp4_fused_2d.shape == fp4_sep.shape:
        match = (fp4_fused_2d == fp4_sep).all().item()
        print(f"FP4 data match: {match}")
        if not match:
            diff_count = (fp4_fused_2d != fp4_sep).sum().item()
            print(f"  Differences: {diff_count} / {fp4_sep.numel()}")
    else:
        print(f"Shape mismatch: fused={fp4_fused_2d.shape} vs sep={fp4_sep.shape}")

    if sf_fused_flat.shape == sf_sep_flat.shape:
        match = (sf_fused_flat == sf_sep_flat).all().item()
        print(f"Scale match: {match}")
    else:
        print(f"Scale shape mismatch: fused={sf_fused_flat.shape} vs sep={sf_sep_flat.shape}")
        print(f"  (Different layout is expected — MoE uses per-expert layout)")

except ImportError as e:
    print(f"\nsilu_and_mul_scaled_nvfp4_experts_quantize not available: {e}")
except Exception as e:
    print(f"\nFused kernel failed: {type(e).__name__}: {e}")

# ============================================================
# Test 3: Try flashinfer scaled_fp4_grouped_quantize (dense)
# ============================================================
try:
    from flashinfer import scaled_fp4_grouped_quantize
    print("\n=== flashinfer scaled_fp4_grouped_quantize available ===")

    out_sep2 = torch.empty(T, I, dtype=torch.bfloat16, device=device)
    silu_and_mul(gate_up_copy, out_sep2)

    # Try as single group
    out_3d = out_sep2.unsqueeze(0)  # [1, T, I]
    masked_m = torch.tensor([T], dtype=torch.int32, device=device)

    fp4_grp, sf_grp = scaled_fp4_grouped_quantize(out_3d, masked_m, scale_inv.unsqueeze(0))
    print(f"fp4_grp shape: {fp4_grp.shape}, dtype: {fp4_grp.dtype}")
    print(f"sf_grp shape:  {sf_grp.shape}, dtype: {sf_grp.dtype}")

except ImportError as e:
    print(f"\nscaled_fp4_grouped_quantize not available: {e}")
except Exception as e:
    print(f"\nGrouped quantize failed: {type(e).__name__}: {e}")

# ============================================================
# Test 4: Correctness of fp4_quantize output
# ============================================================
print("\n=== Format verification ===")
known = torch.tensor([[1.0, -2.0, 0.5, 3.0, -1.5, 0.0, 4.0, -6.0,
                        2.0, -0.5, 1.5, -3.0, 0.0, -4.0, 5.0, -1.0] + [0.0]*16],
                      dtype=torch.bfloat16, device=device)
kfp4, ksf = fp4_quantize(known, torch.tensor(1.0, dtype=torch.float32, device=device))
print(f"Known input (32 elem): {known[0].tolist()}")
print(f"FP4 packed (16 bytes): {kfp4[0, :16].tolist()}")
print(f"Scales (2 blocks):     {ksf.view(torch.uint8)[0, :8].tolist()}")
