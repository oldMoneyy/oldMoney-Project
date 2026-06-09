#!/usr/bin/env python3
"""
Phase 0 FIX: Correct scale layout for FP4 Marlin.

From probing we know the kernel's FP4 access pattern:
  - K-groups are paired: (0,1), (2,3), ...
  - For n < N/2: kernel reads from the EVEN group's row
  - For n >= N/2: kernel reads from the ODD group's row
  - Even output groups read from source cols {0,1,4,5} (mod 8, period 4)
  - Odd output groups read from source cols {2,3,6,7} (mod 8, period 4)

The fix: interleave even/odd group scales into the correct column positions
within each row, so the kernel reads the right scale at each (g, n) output.
"""

import torch
torch.manual_seed(42)
DEVICE = "cuda"

from sglang.srt.layers.quantization.utils import get_scalar_types
_, scalar_types = get_scalar_types()
FP4 = scalar_types.float4_e2m1f
from sgl_kernel import gptq_marlin_gemm, gptq_marlin_repack
from sglang.srt.layers.quantization.marlin_utils import (
    marlin_permute_scales, marlin_make_workspace,
)

GROUP = 16
N, K = 128, 256
NG = K // GROUP  # 16
e2m1_pos = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=DEVICE)
midpoints = (e2m1_pos[:-1] + e2m1_pos[1:]) / 2.0

def nvfp4_proc_scales(s):
    s = s.to(torch.half)
    s = s.view(-1, 4)[:, [0, 2, 1, 3]].view(s.size(0), -1)
    s = s * (2 ** 7)
    s = torch.where(s < 2, torch.zeros_like(s), s)
    s = (s.view(torch.int16) << 1).view(torch.float8_e4m3fn)
    return s[:, 1::2].contiguous()

def nvfp4_proc_global(g):
    return g * (2.0 ** 7)

ws = marlin_make_workspace(DEVICE)
pe = torch.empty(0, dtype=torch.int32, device=DEVICE)

EVEN_COLS = {0, 1, 4, 5}  # within each 8-col block
ODD_COLS  = {2, 3, 6, 7}

def interleave_scales_for_fp4(true_scales, N):
    """
    Rearrange scale tensor [NG, N] so the FP4 Marlin kernel reads
    the correct scale at each (group, column) output position.

    true_scales[g, n] = the scale we want applied to K-group g, output column n.
    (Typically uniform across n for standard per-group quantization.)

    Returns: scale_pre[NG, N] ready for nvfp4_proc_scales + marlin_permute_scales.
    """
    NG_local = true_scales.shape[0]
    scale_pre = torch.zeros_like(true_scales)
    N_half = N // 2

    for pair in range(NG_local // 2):
        g_even = 2 * pair
        g_odd = 2 * pair + 1

        for n in range(N):
            col_in_block = n % 8

            if n < N_half:
                # Kernel reads from row g_even for ALL outputs with n < N/2
                if col_in_block in EVEN_COLS:
                    scale_pre[g_even, n] = true_scales[g_even, n]
                else:  # col_in_block in ODD_COLS
                    scale_pre[g_even, n] = true_scales[g_odd, n]
                # Row g_odd, n < N/2: dead (not read), fill with g_odd's scale
                scale_pre[g_odd, n] = true_scales[g_odd, n]
            else:
                # Kernel reads from row g_odd for ALL outputs with n >= N/2
                # Column mapping FLIPS: even output groups read ODD cols,
                # odd output groups read EVEN cols
                if col_in_block in EVEN_COLS:
                    scale_pre[g_odd, n] = true_scales[g_odd, n]   # odd output group
                else:
                    scale_pre[g_odd, n] = true_scales[g_even, n]  # even output group
                # Row g_even, n >= N/2: dead, fill with g_even's scale
                scale_pre[g_even, n] = true_scales[g_even, n]

    return scale_pre


# ═══════════════════════════════════════════════════════════
# TEST 1: Verify the fix with the linearly increasing scale probe
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 1: Verify fix with per-group probe (linearly increasing scales)")
print("=" * 70)

scales_test = torch.zeros(NG, N, device=DEVICE)
for g in range(NG):
    scales_test[g, :] = 0.25 * (g + 1)

# --- Original (wrong) ---
gl = scales_test.max()
bn_orig = (scales_test / gl).to(torch.float8_e4m3fn)
bs_orig = nvfp4_proc_scales(bn_orig)
bs_orig = marlin_permute_scales(bs_orig.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)
gt = nvfp4_proc_global(gl).to(torch.half).reshape(1, 1).to(DEVICE)

# --- Fixed ---
scales_interleaved = interleave_scales_for_fp4(scales_test, N)
bn_fix = (scales_interleaved / gl).to(torch.float8_e4m3fn)
bs_fix = nvfp4_proc_scales(bn_fix)
bs_fix = marlin_permute_scales(bs_fix.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)

# Probe each group
print(f"\n  Per-group probe (code=4, val=2.0, one-hot activation):")
print(f"  {'g':>3} | {'orig Y':>8} | {'fix Y':>8} | {'expected':>8} | orig | fix")
print(f"  {'-'*60}")

for g in range(NG):
    codes = torch.zeros(K, N, dtype=torch.int32, device=DEVICE)
    codes[g * GROUP, :] = 4

    cNK = codes.T.contiguous()
    gg = cNK.reshape(N, K // 8, 8)
    pk = torch.zeros(N, K // 8, dtype=torch.int32, device=DEVICE)
    for i in range(8):
        pk |= (gg[:, :, i] & 0xF) << (i * 4)
    qw = gptq_marlin_repack(pk.T.contiguous(), pe, K, N, num_bits=4)

    X = torch.zeros(1, K, device=DEVICE, dtype=torch.half)
    X[0, g * GROUP] = 1.0

    Y_orig = gptq_marlin_gemm(
        X, None, qw, bs_orig, gt, None, None, None, ws, FP4,
        1, N, K, True, False, True, False
    ).float()

    Y_fix = gptq_marlin_gemm(
        X, None, qw, bs_fix, gt, None, None, None, ws, FP4,
        1, N, K, True, False, True, False
    ).float()

    expected = 2.0 * 0.25 * (g + 1)
    n_check = 0  # check column 0
    o_ok = "OK" if abs(Y_orig[0, n_check].item() - expected) < 0.01 else "WRONG"
    f_ok = "OK" if abs(Y_fix[0, n_check].item() - expected) < 0.01 else "WRONG"
    print(f"  {g:3d} | {Y_orig[0,n_check]:8.4f} | {Y_fix[0,n_check]:8.4f} | {expected:8.4f} | {o_ok:5s} | {f_ok}")

    # Also check n=64 (second half)
    if g < 4:
        n_check2 = 64
        o2 = Y_orig[0, n_check2].item()
        f2 = Y_fix[0, n_check2].item()
        f2_ok = "OK" if abs(f2 - expected) < 0.01 else "WRONG"
        print(f"      (n=64: orig={o2:.4f}, fix={f2:.4f}, exp={expected:.4f}, fix={f2_ok})")


# ═══════════════════════════════════════════════════════════
# TEST 2: Full smoke test — random weights, compare kernel vs reference
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("TEST 2: Full smoke test (random weights)")
print("=" * 70)

torch.manual_seed(42)
M = 16
W = torch.randn(K, N, device=DEVICE, dtype=torch.float32)
X = torch.randn(M, K, device=DEVICE, dtype=torch.half)

# Per-group scales
Wg = W.reshape(NG, GROUP, N)
true_scales = Wg.abs().amax(dim=1) / 6.0
true_scales = true_scales.clamp(min=1e-10)

# Quantize
W_sc = W / true_scales.repeat_interleave(GROUP, dim=0)
ai = torch.bucketize(W_sc.abs().reshape(-1), midpoints).reshape(K, N)
sb = (W_sc < 0).int()
codes = (ai.int() | (sb << 3)) & 0xF
av = e2m1_pos[ai.long()]
sm = torch.where(sb.bool(), -1.0, 1.0)

# Pack weights
cNK = codes.T.contiguous()
gg = cNK.reshape(N, K // 8, 8)
pk = torch.zeros(N, K // 8, dtype=torch.int32, device=DEVICE)
for i in range(8):
    pk |= (gg[:, :, i] & 0xF) << (i * 4)
qw = gptq_marlin_repack(pk.T.contiguous(), pe, K, N, num_bits=4)

# Global scale
gl = true_scales.max()

# --- Original pipeline ---
bn_orig = (true_scales / gl).to(torch.float8_e4m3fn)
bs_orig = nvfp4_proc_scales(bn_orig)
bs_orig = marlin_permute_scales(bs_orig.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)
gt = nvfp4_proc_global(gl).to(torch.half).reshape(1, 1).to(DEVICE)

# --- Fixed pipeline ---
scales_interleaved = interleave_scales_for_fp4(true_scales, N)
bn_fix = (scales_interleaved / gl).to(torch.float8_e4m3fn)
bs_fix = nvfp4_proc_scales(bn_fix)
bs_fix = marlin_permute_scales(bs_fix.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)

# Reference (using FP8-rounded scales)
kern_scales = bn_orig.to(torch.float32) * gl.item()
W_deq = av * sm * kern_scales.repeat_interleave(GROUP, dim=0)
Y_ref = X.float() @ W_deq

# For fixed pipeline, compute the reference with the FP8-rounded interleaved scales
# decoded back to the TRUE group assignment
kern_scales_fix_raw = bn_fix.to(torch.float32) * gl.item()
# The interleaved tensor has mixed scales — we need the ORIGINAL true_scales
# rounded through FP8 for the reference
true_scales_fp8 = true_scales.to(torch.float8_e4m3fn).to(torch.float32)
# Wait - the interleaved tensor was normalized by gl then FP8'd, so:
interleaved_fp8 = (scales_interleaved / gl).to(torch.float8_e4m3fn).to(torch.float32) * gl.item()
# But the kernel will decode each weight using the interleaved scale at the mapped position
# which should equal true_scales[g] after the fix. So the reference should use true_scales_fp8_norm
true_scales_norm_fp8 = (true_scales / gl).to(torch.float8_e4m3fn).to(torch.float32) * gl.item()
W_deq_fix_ref = av * sm * true_scales_norm_fp8.repeat_interleave(GROUP, dim=0)
Y_ref_fix = X.float() @ W_deq_fix_ref

Y_orig = gptq_marlin_gemm(
    X, None, qw, bs_orig, gt, None, None, None, ws, FP4,
    M, N, K, True, False, True, False
).float()

Y_fix = gptq_marlin_gemm(
    X, None, qw, bs_fix, gt, None, None, None, ws, FP4,
    M, N, K, True, False, True, False
).float()

err_orig_f32 = (Y_orig - Y_ref).norm() / Y_ref.norm()
err_fix_f32 = (Y_fix - Y_ref).norm() / Y_ref.norm()
err_fix_fp8 = (Y_fix - Y_ref_fix).norm() / Y_ref_fix.norm()

print(f"  Original: kernel vs ref_f32  = {err_orig_f32:.4f} ({err_orig_f32*100:.1f}%)")
print(f"  Fixed:    kernel vs ref_f32  = {err_fix_f32:.4f} ({err_fix_f32*100:.1f}%)")
print(f"  Fixed:    kernel vs ref_fp8  = {err_fix_fp8:.4f} ({err_fix_fp8*100:.1f}%)")
print(f"  ||Y_ref||={Y_ref.norm():.1f}, ||Y_orig||={Y_orig.norm():.1f}, ||Y_fix||={Y_fix.norm():.1f}")

# Element-wise check
print(f"\n  Element-wise (row 0, cols 0-3 and 64-67):")
for n in list(range(4)) + list(range(64, 68)):
    print(f"    [{n:3d}] orig={Y_orig[0,n]:8.4f} fix={Y_fix[0,n]:8.4f} "
          f"ref={Y_ref[0,n]:8.4f} ref_fp8={Y_ref_fix[0,n]:8.4f}")


# ═══════════════════════════════════════════════════════════
# TEST 3: Uniform scale (sanity — should still work)
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("TEST 3: Uniform scale = 0.25 (sanity check)")
print("=" * 70)

su = torch.full((NG, N), 0.25, device=DEVICE)
su_interleaved = interleave_scales_for_fp4(su, N)
print(f"  Interleaved == original for uniform scale: {torch.allclose(su, su_interleaved)}")

codes_u, av_u, sm_u = ai, sb, sm  # reuse
W_deq_u = av * sm * 0.25
Y_ref_u = X.float() @ W_deq_u

gl_u = su.max()
bn_u = (su_interleaved / gl_u).to(torch.float8_e4m3fn)
bs_u = nvfp4_proc_scales(bn_u)
bs_u = marlin_permute_scales(bs_u.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)
gt_u = nvfp4_proc_global(gl_u).to(torch.half).reshape(1, 1).to(DEVICE)

Y_u = gptq_marlin_gemm(
    X, None, qw, bs_u, gt_u, None, None, None, ws, FP4,
    M, N, K, True, False, True, False
).float()

err_u = (Y_u - Y_ref_u).norm() / Y_ref_u.norm()
print(f"  Uniform: rel L2 = {err_u:.6f} ({err_u*100:.3f}%)")

print("\nDONE")
