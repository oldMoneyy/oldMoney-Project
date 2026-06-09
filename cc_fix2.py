#!/usr/bin/env python3
"""
Phase 0 FIX v2: Correct scale layout for FP4 Marlin.

Root cause chain:
  1. nvfp4_proc_scales: column-swap [0,2,1,3] + take 1::2 → keeps cols {2,3,6,7,...}
     Output [NG, N/2], then reshape to [NG/2, N] (pairs consecutive groups)
  2. marlin_permute_scales: 8×8 transpose within 64-element blocks
  3. Kernel FP4 access: s_sh[s_sh_rd * 2 + warp_row % 2]
     Even warp_row reads even byte positions → even proc 8-blocks
     Odd warp_row reads odd byte positions → odd proc 8-blocks

  Proc 8-block index b maps to input columns where n//16 = b.
  Even warp_row (even K-group) always reads from even proc 8-blocks.
  Odd warp_row (odd K-group) reads from odd proc 8-blocks.

Fix: interleave even/odd group scales in 16-column blocks so the
8×8 transpose routes them to the correct perm byte positions.
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
NG = K // GROUP
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


def interleave_scales_for_fp4(true_scales):
    """
    Rearrange [NG, N] scale tensor for correct FP4 Marlin kernel access.

    For each group pair (g_even=2r, g_odd=2r+1):
      - Input columns where (n // 16) % 2 == 0: use g_even's scale
      - Input columns where (n // 16) % 2 == 1: use g_odd's scale
    Both rows of the pair get the same interleaved pattern.
    """
    NG_local, N_local = true_scales.shape
    scale_pre = torch.empty_like(true_scales)

    col_idx = torch.arange(N_local, device=true_scales.device)
    even_block = ((col_idx // 16) % 2 == 0)

    for p in range(NG_local // 2):
        ge, go = 2 * p, 2 * p + 1
        row = torch.where(even_block, true_scales[ge], true_scales[go])
        scale_pre[ge] = row
        scale_pre[go] = row

    return scale_pre


# ═══════════════════════════════════════════════════════════
# TEST 1: Per-group probe — verify each group reads correct scale
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 1: Per-group probe (linearly increasing scales)")
print("  scale[g] = 0.25*(g+1), one-hot activation per group")
print("=" * 70)

scales_test = torch.zeros(NG, N, device=DEVICE)
for g in range(NG):
    scales_test[g, :] = 0.25 * (g + 1)

gl = scales_test.max()

# --- Original (wrong) pipeline ---
bn_orig = (scales_test / gl).to(torch.float8_e4m3fn)
bs_orig = nvfp4_proc_scales(bn_orig)
bs_orig = marlin_permute_scales(bs_orig.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)
gt = nvfp4_proc_global(gl).to(torch.half).reshape(1, 1).to(DEVICE)

# --- Fixed pipeline ---
scales_interleaved = interleave_scales_for_fp4(scales_test)
bn_fix = (scales_interleaved / gl).to(torch.float8_e4m3fn)
bs_fix = nvfp4_proc_scales(bn_fix)
bs_fix = marlin_permute_scales(bs_fix.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)

print(f"\n  {'g':>3} | {'n=0 orig':>9} {'n=0 fix':>9} {'expected':>9} | "
      f"{'n=64 orig':>10} {'n=64 fix':>10} {'expected':>9} | orig | fix")
print(f"  {'-'*85}")

orig_ok_count = 0
fix_ok_count = 0
total_checks = 0

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
    for n_check in [0, 64]:
        yo = Y_orig[0, n_check].item()
        yf = Y_fix[0, n_check].item()
        total_checks += 1
        if abs(yo - expected) < 0.05:
            orig_ok_count += 1
        if abs(yf - expected) < 0.05:
            fix_ok_count += 1

    yo0 = Y_orig[0, 0].item()
    yf0 = Y_fix[0, 0].item()
    yo64 = Y_orig[0, 64].item()
    yf64 = Y_fix[0, 64].item()
    o_ok0 = "OK" if abs(yo0 - expected) < 0.05 else "X"
    f_ok0 = "OK" if abs(yf0 - expected) < 0.05 else "X"
    o_ok64 = "OK" if abs(yo64 - expected) < 0.05 else "X"
    f_ok64 = "OK" if abs(yf64 - expected) < 0.05 else "X"
    print(f"  {g:3d} | {yo0:9.4f} {yf0:9.4f} {expected:9.4f} | "
          f"{yo64:10.4f} {yf64:10.4f} {expected:9.4f} | {o_ok0}/{o_ok64} | {f_ok0}/{f_ok64}")

print(f"\n  Original: {orig_ok_count}/{total_checks} correct")
print(f"  Fixed:    {fix_ok_count}/{total_checks} correct")


# ═══════════════════════════════════════════════════════════
# TEST 2: Full smoke test — random weights, compare kernel vs reference
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("TEST 2: Full smoke test (random weights, M=16)")
print("=" * 70)

torch.manual_seed(42)
M = 16
W = torch.randn(K, N, device=DEVICE, dtype=torch.float32)
X = torch.randn(M, K, device=DEVICE, dtype=torch.half)

Wg = W.reshape(NG, GROUP, N)
true_scales = Wg.abs().amax(dim=1) / 6.0
true_scales = true_scales.clamp(min=1e-10)

W_sc = W / true_scales.repeat_interleave(GROUP, dim=0)
ai = torch.bucketize(W_sc.abs().reshape(-1), midpoints).reshape(K, N)
sb = (W_sc < 0).int()
codes = (ai.int() | (sb << 3)) & 0xF
av = e2m1_pos[ai.long()]
sm = torch.where(sb.bool(), -1.0, 1.0)

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
scales_interleaved = interleave_scales_for_fp4(true_scales)
bn_fix = (scales_interleaved / gl).to(torch.float8_e4m3fn)
bs_fix = nvfp4_proc_scales(bn_fix)
bs_fix = marlin_permute_scales(bs_fix.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)

# Reference: FP8-rounded true scales (what the kernel SHOULD compute)
true_scales_fp8 = (true_scales / gl).to(torch.float8_e4m3fn).to(torch.float32) * gl.item()
W_deq = av * sm * true_scales_fp8.repeat_interleave(GROUP, dim=0)
Y_ref = X.float() @ W_deq

# Also compute f32 reference (without FP8 rounding)
W_deq_f32 = av * sm * true_scales.repeat_interleave(GROUP, dim=0)
Y_ref_f32 = X.float() @ W_deq_f32

Y_orig = gptq_marlin_gemm(
    X, None, qw, bs_orig, gt, None, None, None, ws, FP4,
    M, N, K, True, False, True, False
).float()

Y_fix = gptq_marlin_gemm(
    X, None, qw, bs_fix, gt, None, None, None, ws, FP4,
    M, N, K, True, False, True, False
).float()

err_orig_fp8 = (Y_orig - Y_ref).norm() / Y_ref.norm()
err_fix_fp8 = (Y_fix - Y_ref).norm() / Y_ref.norm()
err_orig_f32 = (Y_orig - Y_ref_f32).norm() / Y_ref_f32.norm()
err_fix_f32 = (Y_fix - Y_ref_f32).norm() / Y_ref_f32.norm()
err_fp8_only = (Y_ref - Y_ref_f32).norm() / Y_ref_f32.norm()

print(f"  FP8 scale rounding error:              {err_fp8_only:.4f} ({err_fp8_only*100:.2f}%)")
print(f"  Original: kernel vs ref (fp8 scales)  = {err_orig_fp8:.4f} ({err_orig_fp8*100:.1f}%)")
print(f"  Fixed:    kernel vs ref (fp8 scales)  = {err_fix_fp8:.4f} ({err_fix_fp8*100:.2f}%)")
print(f"  Original: kernel vs ref (f32 scales)  = {err_orig_f32:.4f} ({err_orig_f32*100:.1f}%)")
print(f"  Fixed:    kernel vs ref (f32 scales)  = {err_fix_f32:.4f} ({err_fix_f32*100:.2f}%)")

if err_fix_fp8 < 0.05:
    print(f"  >>> PASS: error < 5% <<<")
else:
    print(f"  >>> FAIL: error >= 5% <<<")

# Element-wise spot check
print(f"\n  Element-wise (row 0):")
print(f"  {'n':>4} | {'orig':>9} {'fix':>9} {'ref':>9} {'ref_f32':>9}")
print(f"  {'-'*48}")
for n in [0, 1, 32, 33, 64, 65, 96, 97]:
    print(f"  {n:4d} | {Y_orig[0,n]:9.4f} {Y_fix[0,n]:9.4f} "
          f"{Y_ref[0,n]:9.4f} {Y_ref_f32[0,n]:9.4f}")


# ═══════════════════════════════════════════════════════════
# TEST 3: Uniform scale (sanity — interleaving should be identity)
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("TEST 3: Uniform scale = 0.25 (sanity check)")
print("=" * 70)

su = torch.full((NG, N), 0.25, device=DEVICE)
su_interleaved = interleave_scales_for_fp4(su)
print(f"  Interleaved == original for uniform: {torch.allclose(su, su_interleaved)}")

gl_u = su.max()
bn_u = (su_interleaved / gl_u).to(torch.float8_e4m3fn)
bs_u = nvfp4_proc_scales(bn_u)
bs_u = marlin_permute_scales(bs_u.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)
gt_u = nvfp4_proc_global(gl_u).to(torch.half).reshape(1, 1).to(DEVICE)

W_deq_u = av * sm * 0.25
Y_ref_u = X.float() @ W_deq_u

Y_u = gptq_marlin_gemm(
    X, None, qw, bs_u, gt_u, None, None, None, ws, FP4,
    M, N, K, True, False, True, False
).float()

err_u = (Y_u - Y_ref_u).norm() / Y_ref_u.norm()
print(f"  Uniform: rel L2 = {err_u:.6f} ({err_u*100:.3f}%)")


# ═══════════════════════════════════════════════════════════
# TEST 4: Larger dimensions
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("TEST 4: Larger dimensions (N=256, K=512, M=32)")
print("=" * 70)

torch.manual_seed(123)
N4, K4, M4 = 256, 512, 32
NG4 = K4 // GROUP

W4 = torch.randn(K4, N4, device=DEVICE, dtype=torch.float32)
X4 = torch.randn(M4, K4, device=DEVICE, dtype=torch.half)

Wg4 = W4.reshape(NG4, GROUP, N4)
ts4 = Wg4.abs().amax(dim=1) / 6.0
ts4 = ts4.clamp(min=1e-10)

W_sc4 = W4 / ts4.repeat_interleave(GROUP, dim=0)
ai4 = torch.bucketize(W_sc4.abs().reshape(-1), midpoints).reshape(K4, N4)
sb4 = (W_sc4 < 0).int()
codes4 = (ai4.int() | (sb4 << 3)) & 0xF
av4 = e2m1_pos[ai4.long()]
sm4 = torch.where(sb4.bool(), -1.0, 1.0)

pe4 = torch.empty(0, dtype=torch.int32, device=DEVICE)
cNK4 = codes4.T.contiguous()
gg4 = cNK4.reshape(N4, K4 // 8, 8)
pk4 = torch.zeros(N4, K4 // 8, dtype=torch.int32, device=DEVICE)
for i in range(8):
    pk4 |= (gg4[:, :, i] & 0xF) << (i * 4)
qw4 = gptq_marlin_repack(pk4.T.contiguous(), pe4, K4, N4, num_bits=4)

gl4 = ts4.max()

# Original
bn4_orig = (ts4 / gl4).to(torch.float8_e4m3fn)
bs4_orig = nvfp4_proc_scales(bn4_orig)
bs4_orig = marlin_permute_scales(bs4_orig.reshape(-1, N4), size_k=K4, size_n=N4, group_size=GROUP)
gt4 = nvfp4_proc_global(gl4).to(torch.half).reshape(1, 1).to(DEVICE)

# Fixed
ts4_il = interleave_scales_for_fp4(ts4)
bn4_fix = (ts4_il / gl4).to(torch.float8_e4m3fn)
bs4_fix = nvfp4_proc_scales(bn4_fix)
bs4_fix = marlin_permute_scales(bs4_fix.reshape(-1, N4), size_k=K4, size_n=N4, group_size=GROUP)

ts4_fp8 = (ts4 / gl4).to(torch.float8_e4m3fn).to(torch.float32) * gl4.item()
W_deq4 = av4 * sm4 * ts4_fp8.repeat_interleave(GROUP, dim=0)
Y_ref4 = X4.float() @ W_deq4

Y4_orig = gptq_marlin_gemm(
    X4, None, qw4, bs4_orig, gt4, None, None, None, ws, FP4,
    M4, N4, K4, True, False, True, False
).float()
Y4_fix = gptq_marlin_gemm(
    X4, None, qw4, bs4_fix, gt4, None, None, None, ws, FP4,
    M4, N4, K4, True, False, True, False
).float()

e4_orig = (Y4_orig - Y_ref4).norm() / Y_ref4.norm()
e4_fix = (Y4_fix - Y_ref4).norm() / Y_ref4.norm()
print(f"  Original: rel L2 = {e4_orig:.4f} ({e4_orig*100:.1f}%)")
print(f"  Fixed:    rel L2 = {e4_fix:.4f} ({e4_fix*100:.2f}%)")
if e4_fix < 0.05:
    print(f"  >>> PASS <<<")
else:
    print(f"  >>> FAIL <<<")


# ═══════════════════════════════════════════════════════════
# TEST 5: Power-of-2 scales (exact in FP8)
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("TEST 5: Power-of-2 scales (exact in FP8, isolates layout error)")
print("=" * 70)

torch.manual_seed(77)
W5 = torch.randn(K, N, device=DEVICE, dtype=torch.float32)
X5 = torch.randn(M, K, device=DEVICE, dtype=torch.half)
Wg5 = W5.reshape(NG, GROUP, N)
ts5 = 2.0 ** torch.round(torch.log2(Wg5.abs().amax(dim=1) / 6.0))
ts5 = ts5.clamp(min=2**-7)

W_sc5 = W5 / ts5.repeat_interleave(GROUP, dim=0)
ai5 = torch.bucketize(W_sc5.abs().reshape(-1), midpoints).reshape(K, N)
sb5 = (W_sc5 < 0).int()
codes5 = (ai5.int() | (sb5 << 3)) & 0xF
av5 = e2m1_pos[ai5.long()]
sm5 = torch.where(sb5.bool(), -1.0, 1.0)

cNK5 = codes5.T.contiguous()
gg5 = cNK5.reshape(N, K // 8, 8)
pk5 = torch.zeros(N, K // 8, dtype=torch.int32, device=DEVICE)
for i in range(8):
    pk5 |= (gg5[:, :, i] & 0xF) << (i * 4)
qw5 = gptq_marlin_repack(pk5.T.contiguous(), pe, K, N, num_bits=4)

gl5 = ts5.max()
# P2 scales survive FP8 exactly
ts5_fp8 = (ts5 / gl5).to(torch.float8_e4m3fn).to(torch.float32) * gl5.item()
p2_err = (ts5 - ts5_fp8).abs().max().item()
print(f"  FP8 roundtrip max error: {p2_err:.6e} (should be ~0)")

# Original
bn5_orig = (ts5 / gl5).to(torch.float8_e4m3fn)
bs5_orig = nvfp4_proc_scales(bn5_orig)
bs5_orig = marlin_permute_scales(bs5_orig.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)
gt5 = nvfp4_proc_global(gl5).to(torch.half).reshape(1, 1).to(DEVICE)

# Fixed
ts5_il = interleave_scales_for_fp4(ts5)
bn5_fix = (ts5_il / gl5).to(torch.float8_e4m3fn)
bs5_fix = nvfp4_proc_scales(bn5_fix)
bs5_fix = marlin_permute_scales(bs5_fix.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)

W_deq5 = av5 * sm5 * ts5.repeat_interleave(GROUP, dim=0)
Y_ref5 = X5.float() @ W_deq5

Y5_orig = gptq_marlin_gemm(
    X5, None, qw5, bs5_orig, gt5, None, None, None, ws, FP4,
    M, N, K, True, False, True, False
).float()
Y5_fix = gptq_marlin_gemm(
    X5, None, qw5, bs5_fix, gt5, None, None, None, ws, FP4,
    M, N, K, True, False, True, False
).float()

e5_orig = (Y5_orig - Y_ref5).norm() / Y_ref5.norm()
e5_fix = (Y5_fix - Y_ref5).norm() / Y_ref5.norm()
print(f"  Original: rel L2 = {e5_orig:.4f} ({e5_orig*100:.1f}%)")
print(f"  Fixed:    rel L2 = {e5_fix:.4f} ({e5_fix*100:.2f}%)")
if e5_fix < 0.05:
    print(f"  >>> PASS <<<")
else:
    print(f"  >>> FAIL <<<")


print("\n\nDONE")
