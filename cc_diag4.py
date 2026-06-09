#!/usr/bin/env python3
"""
Phase 0 - Scale mapping probe across BOTH K groups AND N columns.

From Test D we know: for n=0, the kernel reads scale[g//2] not scale[g].
The kernel formula is s_sh[s_sh_rd * 2 + warp_row % 2].
warp_row depends on N position. So different N columns may read different
scale slots, and odd-group scales may appear at different N positions.

This test builds the complete (group, column) -> actual_scale map.
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
e2m1_pos = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=DEVICE)

def nvfp4_proc_scales(s):
    s = s.to(torch.half)
    s = s.view(-1, 4)[:, [0, 2, 1, 3]].view(s.size(0), -1)
    s = s * (2 ** 7)
    s = torch.where(s < 2, torch.zeros_like(s), s)
    s = (s.view(torch.int16) << 1).view(torch.float8_e4m3fn)
    return s[:, 1::2].contiguous()

def nvfp4_proc_global(g):
    return g * (2.0 ** 7)

N, K = 128, 256
NG = K // GROUP

# ═══════════════════════════════════════════════════════════
# Build weight tensor: code=4 (value 2.0) at EVERY (g, n) position
# but only one non-zero weight per group-column pair
# Scale[g, n] = 0.25 * (g + 1) so we can identify which scale is applied
# ═══════════════════════════════════════════════════════════

scales = torch.zeros(NG, N, device=DEVICE)
for g in range(NG):
    scales[g, :] = 0.25 * (g + 1)  # distinct per group: 0.25, 0.5, ..., 4.0

# Prepare scale tensor for kernel
gl = scales.max()  # 4.0
bn = (scales / gl).to(torch.float8_e4m3fn)
bs = nvfp4_proc_scales(bn)
bs = marlin_permute_scales(bs.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)
gt = nvfp4_proc_global(gl).to(torch.half).reshape(1, 1).to(DEVICE)

# ═══════════════════════════════════════════════════════════
# PROBE 1: For each group g, place code=4 at k=g*16, and probe
#           which scale the kernel applies at EACH output column n
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("PROBE 1: Per-group, per-column scale mapping")
print("  Weight: code=4 (val=2.0) at k=g*16 for target column only")
print("  Scale: 0.25*(g+1) per group, uniform across columns")
print("  Expected: Y = 2.0 * scale[g] = 0.5*(g+1)")
print("=" * 70)

# We'll probe a subset of (g, n) pairs
probe_groups = list(range(min(8, NG)))  # first 8 groups
probe_cols = [0, 1, 2, 3, 4, 8, 16, 32, 64, 127]  # sample N positions

ws = marlin_make_workspace(DEVICE)
pe = torch.empty(0, dtype=torch.int32, device=DEVICE)

print(f"\n{'g':>3} | ", end="")
for n in probe_cols:
    print(f"{'n='+str(n):>8}", end=" ")
print(f" | expected")
print("-" * (6 + 9 * len(probe_cols) + 12))

for g in probe_groups:
    # Place code=4 at position (k=g*16, all n columns)
    codes = torch.zeros(K, N, dtype=torch.int32, device=DEVICE)
    for n in range(N):
        codes[g * GROUP, n] = 4  # val=2.0 at first K position of group g

    # Pack weights
    cNK = codes.T.contiguous()
    gg = cNK.reshape(N, K // 8, 8)
    pk = torch.zeros(N, K // 8, dtype=torch.int32, device=DEVICE)
    for i in range(8):
        pk |= (gg[:, :, i] & 0xF) << (i * 4)
    qw = gptq_marlin_repack(pk.T.contiguous(), pe, K, N, num_bits=4)

    # Activation: one-hot at k=g*16
    X = torch.zeros(1, K, device=DEVICE, dtype=torch.half)
    X[0, g * GROUP] = 1.0

    Y = gptq_marlin_gemm(
        X, None, qw, bs, gt,
        None, None, None, ws, FP4,
        1, N, K, is_k_full=True,
        use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False
    ).float()

    expected = 2.0 * 0.25 * (g + 1)  # code_val * scale
    print(f"{g:3d} | ", end="")
    for n in probe_cols:
        actual_scale = Y[0, n].item() / 2.0  # Y = code_val * actual_scale
        target_scale = 0.25 * (g + 1)
        marker = " " if abs(actual_scale - target_scale) < 0.01 else "*"
        print(f"{actual_scale:7.4f}{marker}", end=" ")
    print(f" | {expected:.3f} (s={0.25*(g+1):.3f})")

# ═══════════════════════════════════════════════════════════
# PROBE 2: Which group's scale does each (g, n) actually use?
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'=' * 70}")
print("PROBE 2: Scale source identification")
print("  For each (g, n), identify which group's scale was applied")
print("  Format: actual_group (expected_group)")
print("=" * 70)

print(f"\n{'g':>3} | ", end="")
for n in probe_cols:
    print(f"{'n='+str(n):>8}", end=" ")
print()
print("-" * (6 + 9 * len(probe_cols)))

for g in probe_groups:
    codes = torch.zeros(K, N, dtype=torch.int32, device=DEVICE)
    for n in range(N):
        codes[g * GROUP, n] = 4

    cNK = codes.T.contiguous()
    gg = cNK.reshape(N, K // 8, 8)
    pk = torch.zeros(N, K // 8, dtype=torch.int32, device=DEVICE)
    for i in range(8):
        pk |= (gg[:, :, i] & 0xF) << (i * 4)
    qw = gptq_marlin_repack(pk.T.contiguous(), pe, K, N, num_bits=4)

    X = torch.zeros(1, K, device=DEVICE, dtype=torch.half)
    X[0, g * GROUP] = 1.0

    Y = gptq_marlin_gemm(
        X, None, qw, bs, gt,
        None, None, None, ws, FP4,
        1, N, K, is_k_full=True,
        use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False
    ).float()

    print(f"{g:3d} | ", end="")
    for n in probe_cols:
        actual_scale = Y[0, n].item() / 2.0
        # Which group has this scale? scale[g'] = 0.25*(g'+1) -> g' = scale/0.25 - 1
        actual_group = round(actual_scale / 0.25 - 1) if actual_scale > 0.01 else -1
        marker = " " if actual_group == g else "!"
        print(f"  {actual_group:2d}({g}){marker}", end=" ")
    print()

# ═══════════════════════════════════════════════════════════
# PROBE 3: Check the column (N) dimension mapping too
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'=' * 70}")
print("PROBE 3: N-column scale mapping")
print("  Use group 0, but set DIFFERENT scales per N column")
print("  Scale[0, n] = 0.25 * (n % 8 + 1)")
print("  Check if the kernel applies the correct per-column scale")
print("=" * 70)

scales_ncol = torch.zeros(NG, N, device=DEVICE)
for n in range(N):
    scales_ncol[:, n] = 0.25 * (n % 8 + 1)

gl_n = scales_ncol.max()
bn_n = (scales_ncol / gl_n).to(torch.float8_e4m3fn)
bs_n = nvfp4_proc_scales(bn_n)
bs_n = marlin_permute_scales(bs_n.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)
gt_n = nvfp4_proc_global(gl_n).to(torch.half).reshape(1, 1).to(DEVICE)

# Place code=4 at k=0 (group 0), all columns
codes_n = torch.zeros(K, N, dtype=torch.int32, device=DEVICE)
codes_n[0, :] = 4

cNK_n = codes_n.T.contiguous()
gg_n = cNK_n.reshape(N, K // 8, 8)
pk_n = torch.zeros(N, K // 8, dtype=torch.int32, device=DEVICE)
for i in range(8):
    pk_n |= (gg_n[:, :, i] & 0xF) << (i * 4)
qw_n = gptq_marlin_repack(pk_n.T.contiguous(), pe, K, N, num_bits=4)

X_n = torch.zeros(1, K, device=DEVICE, dtype=torch.half)
X_n[0, 0] = 1.0

Y_n = gptq_marlin_gemm(
    X_n, None, qw_n, bs_n, gt_n,
    None, None, None, ws, FP4,
    1, N, K, is_k_full=True,
    use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False
).float()

print(f"\n  Column mapping (first 32 columns):")
print(f"  {'n':>3} | {'target_s':>8} | {'actual_s':>8} | {'src_n%8':>6} | status")
print(f"  {'-'*45}")
for n in range(32):
    target_scale = 0.25 * (n % 8 + 1)
    actual_scale = Y_n[0, n].item() / 2.0
    src = round(actual_scale / 0.25 - 1) if actual_scale > 0.01 else -1
    ok = "OK" if abs(actual_scale - target_scale) < 0.01 else f"got col%8={src}"
    print(f"  {n:3d} | {target_scale:8.4f} | {actual_scale:8.4f} | {n%8:6d} | {ok}")

print("\nDONE")
