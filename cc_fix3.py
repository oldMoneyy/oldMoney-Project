#!/usr/bin/env python3
"""
Phase 0 FIX v3: Diagnose whether FP4 Marlin supports per-16 or per-32 group scales.

Hypothesis: the kernel uses ONE scale per PAIR of 16-groups (effective group_size=32).
The warp_row%2 in s_sh[s_sh_rd*2 + warp_row%2] distinguishes M-rows or N-columns,
NOT K-groups. Both K-groups in a pair read the SAME FP8 byte.

Tests:
1. Dump perm tensor bytes to verify interleaving
2. Group_size=32 test (shared scale per pair) — should give ~0% error
3. Full pipeline with group_size=32 quantization
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


# ═══════════════════════════════════════════════════════════
# DIAG 1: Dump perm tensor bytes to see if interleaving works
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("DIAG 1: Verify perm tensor values after interleaving")
print("=" * 70)

scales_test = torch.zeros(NG, N, device=DEVICE)
for g in range(NG):
    scales_test[g, :] = 0.25 * (g + 1)

gl = scales_test.max()

# Original
bn_orig = (scales_test / gl).to(torch.float8_e4m3fn)
bs_orig = nvfp4_proc_scales(bn_orig)
bs_orig_pre = bs_orig.reshape(-1, N).clone()
bs_orig = marlin_permute_scales(bs_orig.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)

# Interleaved (from fix v2)
col_idx = torch.arange(N, device=DEVICE)
even_block = ((col_idx // 16) % 2 == 0)
scales_il = torch.empty_like(scales_test)
for p in range(NG // 2):
    ge, go = 2*p, 2*p+1
    row = torch.where(even_block, scales_test[ge], scales_test[go])
    scales_il[ge] = row
    scales_il[go] = row

bn_il = (scales_il / gl).to(torch.float8_e4m3fn)
bs_il = nvfp4_proc_scales(bn_il)
bs_il_pre = bs_il.reshape(-1, N).clone()
bs_il = marlin_permute_scales(bs_il.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)

# Print FP8 tensor shapes
print(f"\n  nvfp4_proc_scales output shape: {bs_orig_pre.shape}")
print(f"  After permute shape: {bs_orig.shape}")

# Dump first row's raw bytes for original vs interleaved
print(f"\n  Original perm row 0, cols 0-7 (as float):")
print(f"    {[f'{v:.4f}' for v in bs_orig[0, :8].to(torch.float32).tolist()]}")
print(f"  Interleaved perm row 0, cols 0-7:")
print(f"    {[f'{v:.4f}' for v in bs_il[0, :8].to(torch.float32).tolist()]}")

# Check if alternating values differ in interleaved version
print(f"\n  Interleaved perm row 0:")
print(f"    Even cols (0,2,4,6): {[f'{v:.4f}' for v in bs_il[0, 0:8:2].to(torch.float32).tolist()]}")
print(f"    Odd cols  (1,3,5,7): {[f'{v:.4f}' for v in bs_il[0, 1:8:2].to(torch.float32).tolist()]}")
print(f"    Even == Odd: {torch.allclose(bs_il[0, 0:8:2].to(torch.float32), bs_il[0, 1:8:2].to(torch.float32))}")

# Check raw byte representation
bs_orig_bytes = bs_orig.view(torch.uint8)
bs_il_bytes = bs_il.view(torch.uint8)
print(f"\n  Original perm row 0, first 8 bytes (hex):")
print(f"    {[f'0x{b:02X}' for b in bs_orig_bytes[0, :8].tolist()]}")
print(f"  Interleaved perm row 0, first 8 bytes (hex):")
print(f"    {[f'0x{b:02X}' for b in bs_il_bytes[0, :8].tolist()]}")

# Also check if rows 0 and 1 differ
print(f"\n  Original: row 0 == row 1: {torch.equal(bs_orig[0].view(torch.uint8), bs_orig[1].view(torch.uint8))}")
print(f"  Interleaved: row 0 == row 1: {torch.equal(bs_il[0].view(torch.uint8), bs_il[1].view(torch.uint8))}")


# ═══════════════════════════════════════════════════════════
# TEST 1: Group_size=32 — shared scale per pair
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("TEST 1: Group_size=32 (shared scale per pair of groups)")
print("=" * 70)

torch.manual_seed(42)
M = 16
W = torch.randn(K, N, device=DEVICE, dtype=torch.float32)
X = torch.randn(M, K, device=DEVICE, dtype=torch.half)

# Per-32 scales (shared within each pair)
BLOCK32 = 32
Wg32 = W.reshape(K // BLOCK32, BLOCK32, N)
scales32 = Wg32.abs().amax(dim=1) / 6.0  # [8, N]
scales32 = scales32.clamp(min=1e-10)

# Expand to [16, N] by duplicating
scales_expanded = scales32.repeat_interleave(2, dim=0)  # [16, N]

# Quantize using per-16-group code assignment but per-32 scales
W_sc = W / scales_expanded.repeat_interleave(GROUP, dim=0)
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

# Pipeline with per-32 scales (duplicated)
gl32 = scales_expanded.max()
bn32 = (scales_expanded / gl32).to(torch.float8_e4m3fn)
bs32 = nvfp4_proc_scales(bn32)
bs32 = marlin_permute_scales(bs32.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)
gt32 = nvfp4_proc_global(gl32).to(torch.half).reshape(1, 1).to(DEVICE)

# Reference: FP8-rounded per-32 scales
scales32_fp8 = (scales_expanded / gl32).to(torch.float8_e4m3fn).to(torch.float32) * gl32.item()
W_deq32 = av * sm * scales32_fp8.repeat_interleave(GROUP, dim=0)
Y_ref32 = X.float() @ W_deq32

# Also compute f32 reference
W_deq32_f32 = av * sm * scales_expanded.repeat_interleave(GROUP, dim=0)
Y_ref32_f32 = X.float() @ W_deq32_f32

Y32 = gptq_marlin_gemm(
    X, None, qw, bs32, gt32, None, None, None, ws, FP4,
    M, N, K, True, False, True, False
).float()

err32_fp8 = (Y32 - Y_ref32).norm() / Y_ref32.norm()
err32_f32 = (Y32 - Y_ref32_f32).norm() / Y_ref32_f32.norm()
print(f"  Group_size=32: kernel vs ref (fp8)  = {err32_fp8:.6f} ({err32_fp8*100:.3f}%)")
print(f"  Group_size=32: kernel vs ref (f32)  = {err32_f32:.6f} ({err32_f32*100:.3f}%)")

if err32_fp8 < 0.05:
    print(f"  >>> PASS: error < 5% — confirms kernel uses per-pair scales <<<")
else:
    print(f"  >>> FAIL <<<")


# ═══════════════════════════════════════════════════════════
# TEST 2: Per-group probe with group_size=32
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("TEST 2: Per-group probe with group_size=32 scales")
print("=" * 70)

# Scale per pair: pair 0 → 0.5, pair 1 → 1.0, etc.
scales_probe = torch.zeros(NG, N, device=DEVICE)
for pair in range(NG // 2):
    scales_probe[2*pair, :] = 0.5 * (pair + 1)
    scales_probe[2*pair+1, :] = 0.5 * (pair + 1)  # SAME as even

gl_p = scales_probe.max()
bn_p = (scales_probe / gl_p).to(torch.float8_e4m3fn)
bs_p = nvfp4_proc_scales(bn_p)
bs_p = marlin_permute_scales(bs_p.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)
gt_p = nvfp4_proc_global(gl_p).to(torch.half).reshape(1, 1).to(DEVICE)

print(f"\n  {'g':>3} | {'n=0':>8} {'exp':>8} {'ok':>4} | {'n=64':>8} {'exp':>8} {'ok':>4}")
print(f"  {'-'*55}")

all_ok = True
for g in range(NG):
    codes_g = torch.zeros(K, N, dtype=torch.int32, device=DEVICE)
    codes_g[g * GROUP, :] = 4

    cNK_g = codes_g.T.contiguous()
    gg_g = cNK_g.reshape(N, K // 8, 8)
    pk_g = torch.zeros(N, K // 8, dtype=torch.int32, device=DEVICE)
    for i in range(8):
        pk_g |= (gg_g[:, :, i] & 0xF) << (i * 4)
    qw_g = gptq_marlin_repack(pk_g.T.contiguous(), pe, K, N, num_bits=4)

    X_g = torch.zeros(1, K, device=DEVICE, dtype=torch.half)
    X_g[0, g * GROUP] = 1.0

    Y_g = gptq_marlin_gemm(
        X_g, None, qw_g, bs_p, gt_p, None, None, None, ws, FP4,
        1, N, K, True, False, True, False
    ).float()

    pair = g // 2
    expected = 2.0 * 0.5 * (pair + 1)
    y0 = Y_g[0, 0].item()
    y64 = Y_g[0, 64].item()
    ok0 = "OK" if abs(y0 - expected) < 0.1 else "X"
    ok64 = "OK" if abs(y64 - expected) < 0.1 else "X"
    if ok0 == "X" or ok64 == "X":
        all_ok = False
    print(f"  {g:3d} | {y0:8.4f} {expected:8.4f} {ok0:>4} | {y64:8.4f} {expected:8.4f} {ok64:>4}")

print(f"\n  All correct: {all_ok}")


# ═══════════════════════════════════════════════════════════
# TEST 3: Compare group_size=16 (wrong) vs 32 (correct) on random data
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("TEST 3: group_size=16 vs 32 on random weights")
print("=" * 70)

torch.manual_seed(42)
W = torch.randn(K, N, device=DEVICE, dtype=torch.float32)
X = torch.randn(M, K, device=DEVICE, dtype=torch.half)

# --- group_size=16 (original, wrong) ---
Wg16 = W.reshape(NG, GROUP, N)
scales16 = Wg16.abs().amax(dim=1) / 6.0
scales16 = scales16.clamp(min=1e-10)

W_sc16 = W / scales16.repeat_interleave(GROUP, dim=0)
ai16 = torch.bucketize(W_sc16.abs().reshape(-1), midpoints).reshape(K, N)
sb16 = (W_sc16 < 0).int()
codes16 = (ai16.int() | (sb16 << 3)) & 0xF
av16 = e2m1_pos[ai16.long()]
sm16 = torch.where(sb16.bool(), -1.0, 1.0)

cNK16 = codes16.T.contiguous()
gg16 = cNK16.reshape(N, K // 8, 8)
pk16 = torch.zeros(N, K // 8, dtype=torch.int32, device=DEVICE)
for i in range(8):
    pk16 |= (gg16[:, :, i] & 0xF) << (i * 4)
qw16 = gptq_marlin_repack(pk16.T.contiguous(), pe, K, N, num_bits=4)

gl16 = scales16.max()
bn16 = (scales16 / gl16).to(torch.float8_e4m3fn)
bs16 = nvfp4_proc_scales(bn16)
bs16 = marlin_permute_scales(bs16.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)
gt16 = nvfp4_proc_global(gl16).to(torch.half).reshape(1, 1).to(DEVICE)

scales16_fp8 = (scales16 / gl16).to(torch.float8_e4m3fn).to(torch.float32) * gl16.item()
W_deq16 = av16 * sm16 * scales16_fp8.repeat_interleave(GROUP, dim=0)
Y_ref16 = X.float() @ W_deq16

Y16 = gptq_marlin_gemm(
    X, None, qw16, bs16, gt16, None, None, None, ws, FP4,
    M, N, K, True, False, True, False
).float()

err16 = (Y16 - Y_ref16).norm() / Y_ref16.norm()

# --- group_size=32 (correct) ---
Wg32b = W.reshape(K // BLOCK32, BLOCK32, N)
scales32b = Wg32b.abs().amax(dim=1) / 6.0
scales32b = scales32b.clamp(min=1e-10)
scales_exp32 = scales32b.repeat_interleave(2, dim=0)

W_sc32 = W / scales_exp32.repeat_interleave(GROUP, dim=0)
ai32 = torch.bucketize(W_sc32.abs().reshape(-1), midpoints).reshape(K, N)
sb32b = (W_sc32 < 0).int()
codes32b = (ai32.int() | (sb32b << 3)) & 0xF
av32b = e2m1_pos[ai32.long()]
sm32b = torch.where(sb32b.bool(), -1.0, 1.0)

cNK32 = codes32b.T.contiguous()
gg32 = cNK32.reshape(N, K // 8, 8)
pk32b = torch.zeros(N, K // 8, dtype=torch.int32, device=DEVICE)
for i in range(8):
    pk32b |= (gg32[:, :, i] & 0xF) << (i * 4)
qw32b = gptq_marlin_repack(pk32b.T.contiguous(), pe, K, N, num_bits=4)

gl32b = scales_exp32.max()
bn32b = (scales_exp32 / gl32b).to(torch.float8_e4m3fn)
bs32b = nvfp4_proc_scales(bn32b)
bs32b = marlin_permute_scales(bs32b.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)
gt32b = nvfp4_proc_global(gl32b).to(torch.half).reshape(1, 1).to(DEVICE)

scales32_fp8b = (scales_exp32 / gl32b).to(torch.float8_e4m3fn).to(torch.float32) * gl32b.item()
W_deq32b = av32b * sm32b * scales32_fp8b.repeat_interleave(GROUP, dim=0)
Y_ref32b = X.float() @ W_deq32b

Y32b = gptq_marlin_gemm(
    X, None, qw32b, bs32b, gt32b, None, None, None, ws, FP4,
    M, N, K, True, False, True, False
).float()

err32b = (Y32b - Y_ref32b).norm() / Y_ref32b.norm()

print(f"  group_size=16: kernel vs ref = {err16:.4f} ({err16*100:.1f}%)")
print(f"  group_size=32: kernel vs ref = {err32b:.6f} ({err32b*100:.3f}%)")


# ═══════════════════════════════════════════════════════════
# TEST 4: Larger dimensions with group_size=32
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("TEST 4: Larger dimensions (N=256, K=512, M=32) with group_size=32")
print("=" * 70)

torch.manual_seed(123)
N4, K4, M4 = 256, 512, 32
NG4 = K4 // GROUP

W4 = torch.randn(K4, N4, device=DEVICE, dtype=torch.float32)
X4 = torch.randn(M4, K4, device=DEVICE, dtype=torch.half)

Wg4 = W4.reshape(K4 // BLOCK32, BLOCK32, N4)
ts4 = Wg4.abs().amax(dim=1) / 6.0
ts4 = ts4.clamp(min=1e-10)
ts4_exp = ts4.repeat_interleave(2, dim=0)

W_sc4 = W4 / ts4_exp.repeat_interleave(GROUP, dim=0)
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

gl4 = ts4_exp.max()
bn4 = (ts4_exp / gl4).to(torch.float8_e4m3fn)
bs4 = nvfp4_proc_scales(bn4)
bs4 = marlin_permute_scales(bs4.reshape(-1, N4), size_k=K4, size_n=N4, group_size=GROUP)
gt4 = nvfp4_proc_global(gl4).to(torch.half).reshape(1, 1).to(DEVICE)

ts4_fp8 = (ts4_exp / gl4).to(torch.float8_e4m3fn).to(torch.float32) * gl4.item()
W_deq4 = av4 * sm4 * ts4_fp8.repeat_interleave(GROUP, dim=0)
Y_ref4 = X4.float() @ W_deq4

Y4 = gptq_marlin_gemm(
    X4, None, qw4, bs4, gt4, None, None, None, ws, FP4,
    M4, N4, K4, True, False, True, False
).float()

err4 = (Y4 - Y_ref4).norm() / Y_ref4.norm()
print(f"  group_size=32: rel L2 = {err4:.6f} ({err4*100:.3f}%)")
if err4 < 0.05:
    print(f"  >>> PASS <<<")
else:
    print(f"  >>> FAIL <<<")


# ═══════════════════════════════════════════════════════════
# TEST 5: Power-of-2 scales with group_size=32
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("TEST 5: Power-of-2 scales with group_size=32")
print("=" * 70)

torch.manual_seed(77)
W5 = torch.randn(K, N, device=DEVICE, dtype=torch.float32)
X5 = torch.randn(M, K, device=DEVICE, dtype=torch.half)

Wg5 = W5.reshape(K // BLOCK32, BLOCK32, N)
ts5 = 2.0 ** torch.round(torch.log2(Wg5.abs().amax(dim=1) / 6.0))
ts5 = ts5.clamp(min=2**-7)
ts5_exp = ts5.repeat_interleave(2, dim=0)

W_sc5 = W5 / ts5_exp.repeat_interleave(GROUP, dim=0)
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

gl5 = ts5_exp.max()
bn5 = (ts5_exp / gl5).to(torch.float8_e4m3fn)
bs5 = nvfp4_proc_scales(bn5)
bs5 = marlin_permute_scales(bs5.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)
gt5 = nvfp4_proc_global(gl5).to(torch.half).reshape(1, 1).to(DEVICE)

ts5_fp8 = (ts5_exp / gl5).to(torch.float8_e4m3fn).to(torch.float32) * gl5.item()
W_deq5 = av5 * sm5 * ts5_fp8.repeat_interleave(GROUP, dim=0)
Y_ref5 = X5.float() @ W_deq5

Y5 = gptq_marlin_gemm(
    X5, None, qw5, bs5, gt5, None, None, None, ws, FP4,
    M, N, K, True, False, True, False
).float()

err5 = (Y5 - Y_ref5).norm() / Y_ref5.norm()
print(f"  Power-of-2, group_size=32: rel L2 = {err5:.6f} ({err5*100:.3f}%)")
if err5 < 0.05:
    print(f"  >>> PASS <<<")
else:
    print(f"  >>> FAIL <<<")


print("\n\nDONE")
