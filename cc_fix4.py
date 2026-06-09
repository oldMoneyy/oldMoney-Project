#!/usr/bin/env python3
"""
Phase 0 FIX v4: Isolate whether the 26% error is from weights or scales.

Tests:
1. ALL weights = code 4 (val 2.0), non-uniform per-32 scales → isolates scale path
2. Non-uniform weights, uniform scale → isolates weight packing
3. Compare gptq_marlin_repack output for INT4 vs FP4 to see if repack differs
4. Test if Marlin FP4 expects group_size=32 in marlin_permute_scales
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
BLOCK32 = 32
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

def pack_and_repack(codes, K_local, N_local):
    cNK = codes.T.contiguous()
    gg = cNK.reshape(N_local, K_local // 8, 8)
    pk = torch.zeros(N_local, K_local // 8, dtype=torch.int32, device=DEVICE)
    for i in range(8):
        pk |= (gg[:, :, i] & 0xF) << (i * 4)
    return gptq_marlin_repack(pk.T.contiguous(), pe, K_local, N_local, num_bits=4)

def make_scales_for_kernel(scales, K_local, N_local):
    gl = scales.max()
    bn = (scales / gl).to(torch.float8_e4m3fn)
    bs = nvfp4_proc_scales(bn)
    bs = marlin_permute_scales(bs.reshape(-1, N_local), size_k=K_local, size_n=N_local, group_size=GROUP)
    gt = nvfp4_proc_global(gl).to(torch.half).reshape(1, 1).to(DEVICE)
    return bs, gt, gl


# ═══════════════════════════════════════════════════════════
# TEST 1: ALL weights = code 4 (val 2.0), non-uniform per-32 scales
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 1: Uniform weights (code=4), non-uniform per-32 scales")
print("  Isolates scale path — if this passes, scales are correct")
print("=" * 70)

M = 16
torch.manual_seed(42)
X = torch.randn(M, K, device=DEVICE, dtype=torch.half)

# All codes = 4 (value 2.0)
codes_uniform = torch.full((K, N), 4, dtype=torch.int32, device=DEVICE)
qw_uniform = pack_and_repack(codes_uniform, K, N)

# Per-32 scales (different per pair)
scales_32 = torch.zeros(NG, N, device=DEVICE)
for pair in range(NG // 2):
    val = 0.5 * (pair + 1)
    scales_32[2*pair, :] = val
    scales_32[2*pair+1, :] = val

bs1, gt1, gl1 = make_scales_for_kernel(scales_32, K, N)

# Reference
scales_fp8 = (scales_32 / gl1).to(torch.float8_e4m3fn).to(torch.float32) * gl1.item()
W_deq1 = 2.0 * scales_fp8.repeat_interleave(GROUP, dim=0)  # code 4 → val 2.0
Y_ref1 = X.float() @ W_deq1

Y1 = gptq_marlin_gemm(
    X, None, qw_uniform, bs1, gt1, None, None, None, ws, FP4,
    M, N, K, True, False, True, False
).float()

err1 = (Y1 - Y_ref1).norm() / Y_ref1.norm()
print(f"  Error: {err1:.6f} ({err1*100:.3f}%)")
if err1 < 0.05:
    print("  >>> PASS: Scale path is correct <<<")
else:
    print("  >>> FAIL: Scale path has issues <<<")

# Spot check
print(f"  Spot check row 0:")
for n in [0, 32, 64, 96]:
    print(f"    n={n:3d}: kernel={Y1[0,n]:10.4f} ref={Y_ref1[0,n]:10.4f} diff={abs(Y1[0,n]-Y_ref1[0,n]):8.4f}")


# ═══════════════════════════════════════════════════════════
# TEST 2: Non-uniform weights, uniform scale
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("TEST 2: Non-uniform weights, uniform scale = 1.0")
print("  Isolates weight packing — if this passes, weights are correct")
print("=" * 70)

torch.manual_seed(42)
W = torch.randn(K, N, device=DEVICE, dtype=torch.float32)

# Uniform scale
scales_uni = torch.ones(NG, N, device=DEVICE)

# Quantize with uniform scale
W_sc = W / scales_uni.repeat_interleave(GROUP, dim=0)
ai = torch.bucketize(W_sc.abs().reshape(-1), midpoints).reshape(K, N)
sb = (W_sc < 0).int()
codes = (ai.int() | (sb << 3)) & 0xF
av = e2m1_pos[ai.long()]
sm = torch.where(sb.bool(), -1.0, 1.0)

qw2 = pack_and_repack(codes, K, N)
bs2, gt2, gl2 = make_scales_for_kernel(scales_uni, K, N)

W_deq2 = av * sm * 1.0
Y_ref2 = X.float() @ W_deq2

Y2 = gptq_marlin_gemm(
    X, None, qw2, bs2, gt2, None, None, None, ws, FP4,
    M, N, K, True, False, True, False
).float()

err2 = (Y2 - Y_ref2).norm() / Y_ref2.norm()
print(f"  Error: {err2:.6f} ({err2*100:.3f}%)")
if err2 < 0.05:
    print("  >>> PASS: Weight packing is correct <<<")
else:
    print("  >>> FAIL: Weight packing has issues <<<")


# ═══════════════════════════════════════════════════════════
# TEST 3: Per-pair-uniform weights, non-uniform per-32 scales
#   Each pair has the SAME weight value but different scales
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("TEST 3: Per-pair weights + non-uniform per-32 scales")
print("  All weights in pair p have code = (p % 7 + 1)")
print("=" * 70)

codes_pair = torch.zeros(K, N, dtype=torch.int32, device=DEVICE)
for pair in range(NG // 2):
    code_val = (pair % 7) + 1  # codes 1-7
    codes_pair[pair*32:(pair+1)*32, :] = code_val

qw3 = pack_and_repack(codes_pair, K, N)

# Same per-32 scales as TEST 1
bs3, gt3, gl3 = make_scales_for_kernel(scales_32, K, N)
scales_fp8_3 = (scales_32 / gl3).to(torch.float8_e4m3fn).to(torch.float32) * gl3.item()

# Reference: for each pair, weight value = e2m1_pos[code] (unsigned, all positive)
W_deq3 = torch.zeros(K, N, device=DEVICE)
for pair in range(NG // 2):
    code_val = (pair % 7) + 1
    val = e2m1_pos[code_val].item()
    scale_pair = scales_fp8_3[2*pair, :].unsqueeze(0)  # [1, N]
    W_deq3[pair*32:(pair+1)*32, :] = val * scale_pair.expand(32, N)

Y_ref3 = X.float() @ W_deq3

Y3 = gptq_marlin_gemm(
    X, None, qw3, bs3, gt3, None, None, None, ws, FP4,
    M, N, K, True, False, True, False
).float()

err3 = (Y3 - Y_ref3).norm() / Y_ref3.norm()
print(f"  Error: {err3:.6f} ({err3*100:.3f}%)")
if err3 < 0.05:
    print("  >>> PASS <<<")
else:
    print("  >>> FAIL <<<")


# ═══════════════════════════════════════════════════════════
# TEST 4: Try marlin_permute_scales with group_size=32
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("TEST 4: Use group_size=32 for BOTH quantization AND marlin_permute_scales")
print("=" * 70)

torch.manual_seed(42)
W4 = torch.randn(K, N, device=DEVICE, dtype=torch.float32)
X4 = torch.randn(M, K, device=DEVICE, dtype=torch.half)

# Per-32 scales
Wg32 = W4.reshape(K // BLOCK32, BLOCK32, N)
ts32 = Wg32.abs().amax(dim=1) / 6.0
ts32 = ts32.clamp(min=1e-10)  # [8, N]

# Quantize with per-32 scales (duplicate to per-16 for weight quantization)
ts32_exp = ts32.repeat_interleave(2, dim=0)  # [16, N]
W_sc4 = W4 / ts32_exp.repeat_interleave(GROUP, dim=0)
ai4 = torch.bucketize(W_sc4.abs().reshape(-1), midpoints).reshape(K, N)
sb4 = (W_sc4 < 0).int()
codes4 = (ai4.int() | (sb4 << 3)) & 0xF
av4 = e2m1_pos[ai4.long()]
sm4 = torch.where(sb4.bool(), -1.0, 1.0)

qw4 = pack_and_repack(codes4, K, N)

# Prepare scales with group_size=32 for marlin_permute_scales
# Scale tensor shape: [K//32, N] = [8, N]
gl4 = ts32.max()
bn4 = (ts32 / gl4).to(torch.float8_e4m3fn)

# Process scales: first nvfp4_proc_scales on the [8, N] tensor
bs4 = nvfp4_proc_scales(bn4)
print(f"  Scale shapes: ts32={ts32.shape}, bn4={bn4.shape}, bs4={bs4.shape}")

# Now use group_size=32 for permutation
bs4 = marlin_permute_scales(bs4.reshape(-1, N), size_k=K, size_n=N, group_size=32)
gt4 = nvfp4_proc_global(gl4).to(torch.half).reshape(1, 1).to(DEVICE)
print(f"  Permuted scale shape: {bs4.shape}")

# Reference
ts32_fp8 = (ts32 / gl4).to(torch.float8_e4m3fn).to(torch.float32) * gl4.item()
ts32_exp_fp8 = ts32_fp8.repeat_interleave(2, dim=0)
W_deq4 = av4 * sm4 * ts32_exp_fp8.repeat_interleave(GROUP, dim=0)
Y_ref4 = X4.float() @ W_deq4

Y4 = gptq_marlin_gemm(
    X4, None, qw4, bs4, gt4, None, None, None, ws, FP4,
    M, N, K, True, False, True, False
).float()

err4 = (Y4 - Y_ref4).norm() / Y_ref4.norm()
print(f"  Error: {err4:.6f} ({err4*100:.3f}%)")
if err4 < 0.05:
    print("  >>> PASS <<<")
else:
    print("  >>> FAIL <<<")


# ═══════════════════════════════════════════════════════════
# TEST 5: Random weights, per-32 scales,
#   but DON'T use nvfp4_proc_scales — use raw FP8 values directly
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("TEST 5: Try DIFFERENT global scale formulations")
print("  A: global_scale = max(scales)")
print("  B: global_scale = max(W) / 6.0")
print("=" * 70)

torch.manual_seed(42)
W5 = torch.randn(K, N, device=DEVICE, dtype=torch.float32)
X5 = torch.randn(M, K, device=DEVICE, dtype=torch.half)

# Per-16 scales
Wg5 = W5.reshape(NG, GROUP, N)
ts5 = Wg5.abs().amax(dim=1) / 6.0
ts5 = ts5.clamp(min=1e-10)

# Duplicate to per-32
ts5_p32 = ts5.clone()
for pair in range(NG // 2):
    pair_max = torch.maximum(ts5[2*pair], ts5[2*pair+1])
    ts5_p32[2*pair] = pair_max
    ts5_p32[2*pair+1] = pair_max

W_sc5 = W5 / ts5_p32.repeat_interleave(GROUP, dim=0)
ai5 = torch.bucketize(W_sc5.abs().reshape(-1), midpoints).reshape(K, N)
sb5 = (W_sc5 < 0).int()
codes5 = (ai5.int() | (sb5 << 3)) & 0xF
av5 = e2m1_pos[ai5.long()]
sm5 = torch.where(sb5.bool(), -1.0, 1.0)

qw5 = pack_and_repack(codes5, K, N)

# Version A: global = max(block_scales)
gl5a = ts5_p32.max()
bn5a = (ts5_p32 / gl5a).to(torch.float8_e4m3fn)
bs5a = nvfp4_proc_scales(bn5a)
bs5a = marlin_permute_scales(bs5a.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)
gt5a = nvfp4_proc_global(gl5a).to(torch.half).reshape(1, 1).to(DEVICE)

ts5a_fp8 = (ts5_p32 / gl5a).to(torch.float8_e4m3fn).to(torch.float32) * gl5a.item()
W_deq5a = av5 * sm5 * ts5a_fp8.repeat_interleave(GROUP, dim=0)
Y_ref5a = X5.float() @ W_deq5a

Y5a = gptq_marlin_gemm(
    X5, None, qw5, bs5a, gt5a, None, None, None, ws, FP4,
    M, N, K, True, False, True, False
).float()

err5a = (Y5a - Y_ref5a).norm() / Y_ref5a.norm()
print(f"  Version A (gl=max(scales)): error = {err5a:.6f} ({err5a*100:.3f}%)")

# Version B: global = max(|W|) / 6.0
gl5b = W5.abs().max() / 6.0
bn5b = (ts5_p32 / gl5b).to(torch.float8_e4m3fn)
bs5b = nvfp4_proc_scales(bn5b)
bs5b = marlin_permute_scales(bs5b.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)
gt5b = nvfp4_proc_global(gl5b).to(torch.half).reshape(1, 1).to(DEVICE)

ts5b_fp8 = (ts5_p32 / gl5b).to(torch.float8_e4m3fn).to(torch.float32) * gl5b.item()
W_deq5b = av5 * sm5 * ts5b_fp8.repeat_interleave(GROUP, dim=0)
Y_ref5b = X5.float() @ W_deq5b

Y5b = gptq_marlin_gemm(
    X5, None, qw5, bs5b, gt5b, None, None, None, ws, FP4,
    M, N, K, True, False, True, False
).float()

err5b = (Y5b - Y_ref5b).norm() / Y_ref5b.norm()
print(f"  Version B (gl=max(W)/6): error = {err5b:.6f} ({err5b*100:.3f}%)")


# ═══════════════════════════════════════════════════════════
# TEST 6: Dump what the kernel ACTUALLY computes for specific elements
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("TEST 6: Element-wise diagnostic — find where kernel diverges")
print("=" * 70)

# Use the TEST 1 setup (uniform codes, per-32 scales)
# With X = identity-like (one-hot per row), we can see each group's contribution
print(f"  Row 0 comparison (uniform codes, per-32 scales):")
print(f"  {'n':>4} | {'kernel':>10} {'ref':>10} {'ratio':>8}")
print(f"  {'-'*40}")
for n in range(0, N, 8):
    k_val = Y1[0, n].item()
    r_val = Y_ref1[0, n].item()
    ratio = k_val / r_val if abs(r_val) > 1e-6 else float('inf')
    print(f"  {n:4d} | {k_val:10.4f} {r_val:10.4f} {ratio:8.4f}")


# ═══════════════════════════════════════════════════════════
# TEST 7: Single group active, all codes = 4 in that group
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("TEST 7: Full group activation (all 16 k in group, all code=4)")
print("  Tests if intra-group weight packing is correct")
print("=" * 70)

# Use per-32 scales
bs7, gt7, gl7 = make_scales_for_kernel(scales_32, K, N)
scales_fp8_7 = (scales_32 / gl7).to(torch.float8_e4m3fn).to(torch.float32) * gl7.item()

print(f"  {'g':>3} | {'n=0':>10} {'exp':>10} {'ok':>4} | {'n=64':>10} {'exp':>10} {'ok':>4}")
print(f"  {'-'*60}")

for g in range(min(NG, 8)):
    # All 16 k positions in group g have code=4
    codes_g = torch.zeros(K, N, dtype=torch.int32, device=DEVICE)
    codes_g[g*GROUP:(g+1)*GROUP, :] = 4

    qw_g = pack_and_repack(codes_g, K, N)

    # X = all ones in group g's k range
    X_g = torch.zeros(1, K, device=DEVICE, dtype=torch.half)
    X_g[0, g*GROUP:(g+1)*GROUP] = 1.0

    Y_g = gptq_marlin_gemm(
        X_g, None, qw_g, bs7, gt7, None, None, None, ws, FP4,
        1, N, K, True, False, True, False
    ).float()

    # Expected: 16 * 2.0 * scale_pair = 32 * scale_pair
    pair = g // 2
    expected = 16 * 2.0 * scales_fp8_7[2*pair, 0].item()
    y0 = Y_g[0, 0].item()
    y64 = Y_g[0, 64].item()
    ok0 = "OK" if abs(y0 - expected) / abs(expected) < 0.05 else "X"
    ok64 = "OK" if abs(y64 - expected) / abs(expected) < 0.05 else "X"
    print(f"  {g:3d} | {y0:10.4f} {expected:10.4f} {ok0:>4} | {y64:10.4f} {expected:10.4f} {ok64:>4}")


# ═══════════════════════════════════════════════════════════
# TEST 8: Two groups active, check for cross-group interference
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("TEST 8: Two groups active at once — check additivity")
print("=" * 70)

# Groups 0 and 2 active simultaneously, code=4, X = all 1s in those groups
codes_8 = torch.zeros(K, N, dtype=torch.int32, device=DEVICE)
codes_8[0:16, :] = 4   # group 0
codes_8[32:48, :] = 4  # group 2
qw_8 = pack_and_repack(codes_8, K, N)

X_8 = torch.zeros(1, K, device=DEVICE, dtype=torch.half)
X_8[0, 0:16] = 1.0   # group 0
X_8[0, 32:48] = 1.0  # group 2

Y_8 = gptq_marlin_gemm(
    X_8, None, qw_8, bs7, gt7, None, None, None, ws, FP4,
    1, N, K, True, False, True, False
).float()

# Expected: contribution from g0 + g2
# g0 in pair 0: 16 * 2.0 * scale_pair0
# g2 in pair 1: 16 * 2.0 * scale_pair1
exp_g0 = 16 * 2.0 * scales_fp8_7[0, 0].item()
exp_g2 = 16 * 2.0 * scales_fp8_7[2, 0].item()
exp_total = exp_g0 + exp_g2

print(f"  Expected: g0 contrib={exp_g0:.4f} + g2 contrib={exp_g2:.4f} = {exp_total:.4f}")
print(f"  Kernel n=0: {Y_8[0, 0].item():.4f}")
print(f"  Kernel n=64: {Y_8[0, 64].item():.4f}")
err8 = abs(Y_8[0, 0].item() - exp_total) / abs(exp_total)
print(f"  Relative error at n=0: {err8:.6f} ({err8*100:.3f}%)")
if err8 < 0.05:
    print("  >>> PASS: Additivity holds <<<")
else:
    print("  >>> FAIL: Cross-group interference <<<")


print("\n\nDONE")
