#!/usr/bin/env python3
"""
Phase 0 - Build COMPLETE scale mapping and construct the FP4 fix.

Strategy:
1. For each of 16 K-groups, probe ALL 128 N columns to build full mapping
2. From the mapping, derive the correct scale tensor layout
3. Test it — if error drops to ~0%, Phase 0 is solved
"""

import torch
torch.manual_seed(42)
DEVICE = "cuda"

from sglang.srt.layers.quantization.utils import get_scalar_types
_, scalar_types = get_scalar_types()
FP4 = scalar_types.float4_e2m1f
from sgl_kernel import gptq_marlin_gemm, gptq_marlin_repack
from sglang.srt.layers.quantization.marlin_utils import (
    marlin_permute_scales, marlin_make_workspace, get_scale_perms,
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

# ═══════════════════════════════════════════════════════════
# STEP 1: Build unique-per-group scales, probe all N columns
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: Build complete (group, column) -> actual_group mapping")
print("=" * 70)

# Scale[g, :] = 0.25*(g+1), uniform across columns
# Expected Y[0, n] = 2.0 * scale[g] when probing group g
scales = torch.zeros(NG, N, device=DEVICE)
for g in range(NG):
    scales[g, :] = 0.25 * (g + 1)

gl = scales.max()
bn = (scales / gl).to(torch.float8_e4m3fn)
bs = nvfp4_proc_scales(bn)
bs = marlin_permute_scales(bs.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)
gt = nvfp4_proc_global(gl).to(torch.half).reshape(1, 1).to(DEVICE)

# mapping[g, n] = which group's scale the kernel actually applied
mapping = torch.zeros(NG, N, dtype=torch.int32)

for g in range(NG):
    codes = torch.zeros(K, N, dtype=torch.int32, device=DEVICE)
    codes[g * GROUP, :] = 4  # code=4 → value 2.0 at first K of group g, ALL columns

    cNK = codes.T.contiguous()
    gg = cNK.reshape(N, K // 8, 8)
    pk = torch.zeros(N, K // 8, dtype=torch.int32, device=DEVICE)
    for i in range(8):
        pk |= (gg[:, :, i] & 0xF) << (i * 4)
    qw = gptq_marlin_repack(pk.T.contiguous(), pe, K, N, num_bits=4)

    X = torch.zeros(1, K, device=DEVICE, dtype=torch.half)
    X[0, g * GROUP] = 1.0

    Y = gptq_marlin_gemm(
        X, None, qw, bs, gt, None, None, None, ws, FP4,
        1, N, K, True, False, True, False
    ).float()

    # Y[0, n] = 2.0 * actual_scale → actual_scale = Y / 2.0
    # actual_group = round(actual_scale / 0.25 - 1)
    for n in range(N):
        actual_scale = Y[0, n].item() / 2.0
        actual_group = round(actual_scale / 0.25 - 1) if actual_scale > 0.01 else -1
        mapping[g, n] = actual_group

# Print the mapping
print("\nGroup mapping (target_group → actual_group) for n=0..7 and n=64..71:")
print(f"{'g':>3} | {'n=0..7':^24} | {'n=64..71':^24}")
print("-" * 60)
for g in range(NG):
    first = [mapping[g, n].item() for n in range(8)]
    second = [mapping[g, n].item() for n in range(64, 72)]
    f_str = " ".join(f"{v:2d}" for v in first)
    s_str = " ".join(f"{v:2d}" for v in second)
    print(f"{g:3d} | {f_str} | {s_str}")

# Check if there's a simple pattern
print("\n\nPattern analysis:")
# For each output (g, n), what's the offset?
offsets = mapping - torch.arange(NG).unsqueeze(1)
print(f"  Offset (actual - target) unique values: {sorted(set(offsets.reshape(-1).tolist()))}")

# N-boundary
for n in range(N):
    if mapping[0, n].item() != mapping[0, 0].item():
        print(f"  Group 0 changes at n={n}: {mapping[0, 0].item()} -> {mapping[0, n].item()}")
        break

for n in range(N):
    if mapping[1, n].item() != mapping[1, 0].item():
        print(f"  Group 1 changes at n={n}: {mapping[1, 0].item()} -> {mapping[1, n].item()}")
        break

# ═══════════════════════════════════════════════════════════
# STEP 2: Build N-column mapping with per-column distinct scales
# ═══════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("STEP 2: N-column mapping (per-column distinct scales)")
print("=" * 70)

# Use distinct scales per column for group 0 AND group 1
# Scale values: use power-of-2 fractions so they survive FP8
# scale[g, n] = 2^(-n%8) for g=0 (gives 1, 0.5, 0.25, 0.125, 0.0625, ...)
# No, FP8 E4M3 has limited range. Let me use: scale_val = (n%8 + 1) * 0.125
# which gives 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0

scales2 = torch.zeros(NG, N, device=DEVICE)
for n in range(N):
    for g in range(NG):
        scales2[g, n] = (n % 8 + 1) * 0.125  # different per n%8, same for all g

gl2 = scales2.max()
bn2 = (scales2 / gl2).to(torch.float8_e4m3fn)
bs2 = nvfp4_proc_scales(bn2)
bs2 = marlin_permute_scales(bs2.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)
gt2 = nvfp4_proc_global(gl2).to(torch.half).reshape(1, 1).to(DEVICE)

# Probe group 0
codes2 = torch.zeros(K, N, dtype=torch.int32, device=DEVICE)
codes2[0, :] = 4
cNK2 = codes2.T.contiguous()
gg2 = cNK2.reshape(N, K // 8, 8)
pk2 = torch.zeros(N, K // 8, dtype=torch.int32, device=DEVICE)
for i in range(8):
    pk2 |= (gg2[:, :, i] & 0xF) << (i * 4)
qw2 = gptq_marlin_repack(pk2.T.contiguous(), pe, K, N, num_bits=4)

X2 = torch.zeros(1, K, device=DEVICE, dtype=torch.half)
X2[0, 0] = 1.0

Y2 = gptq_marlin_gemm(
    X2, None, qw2, bs2, gt2, None, None, None, ws, FP4,
    1, N, K, True, False, True, False
).float()

# Build column mapping
col_mapping = []
print(f"\n  N-column mapping for group 0 (first 64 columns):")
print(f"  {'out_n':>5} | {'target%8':>8} | {'actual_s':>8} | {'src%8':>5}")
print(f"  {'-'*40}")
for n in range(64):
    actual_scale = Y2[0, n].item() / 2.0
    src_col_mod8 = round(actual_scale / 0.125 - 1) if actual_scale > 0.01 else -1
    target_mod8 = n % 8
    col_mapping.append(src_col_mod8)
    if n < 32:
        print(f"  {n:5d} | {target_mod8:8d} | {actual_scale:8.4f} | {src_col_mod8:5d}")

# Find the period
period_found = False
for period in [2, 4, 8, 16, 32]:
    if all(col_mapping[i] == col_mapping[i % period] for i in range(min(64, len(col_mapping)))):
        print(f"\n  Column mapping has period {period}")
        print(f"  Pattern: {col_mapping[:period]}")
        period_found = True
        break
if not period_found:
    print(f"  No simple period found in first 64 columns")
    print(f"  First 16: {col_mapping[:16]}")

# Also probe group 1 (odd) for comparison
codes3 = torch.zeros(K, N, dtype=torch.int32, device=DEVICE)
codes3[GROUP, :] = 4  # group 1
cNK3 = codes3.T.contiguous()
gg3 = cNK3.reshape(N, K // 8, 8)
pk3 = torch.zeros(N, K // 8, dtype=torch.int32, device=DEVICE)
for i in range(8):
    pk3 |= (gg3[:, :, i] & 0xF) << (i * 4)
qw3 = gptq_marlin_repack(pk3.T.contiguous(), pe, K, N, num_bits=4)
X3 = torch.zeros(1, K, device=DEVICE, dtype=torch.half)
X3[0, GROUP] = 1.0
Y3 = gptq_marlin_gemm(
    X3, None, qw3, bs2, gt2, None, None, None, ws, FP4,
    1, N, K, True, False, True, False
).float()

col_mapping_g1 = []
print(f"\n  N-column mapping for group 1 (first 32 columns):")
print(f"  {'out_n':>5} | {'target%8':>8} | {'actual_s':>8} | {'src%8':>5}")
print(f"  {'-'*40}")
for n in range(32):
    actual_scale = Y3[0, n].item() / 2.0
    src_col_mod8 = round(actual_scale / 0.125 - 1) if actual_scale > 0.01 else -1
    col_mapping_g1.append(src_col_mod8)
    print(f"  {n:5d} | {n%8:8d} | {actual_scale:8.4f} | {src_col_mod8:5d}")

# ═══════════════════════════════════════════════════════════
# STEP 3: Analyze the inverse permutation needed
# ═══════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("STEP 3: Construct the correct scale permutation for FP4")
print("=" * 70)

# The current scale_perm (for INT4):
scale_perm, _ = get_scale_perms()
print(f"  INT4 scale_perm (64 elements):")
for row in range(8):
    print(f"    row {row}: {scale_perm[row*8:(row+1)*8]}")

# The kernel ACTUALLY reads from these positions (per the probe).
# We need to construct fp4_scale_perm such that:
#   scale_tensor_permuted[actual_position] = scale_tensor_original[desired_position]
# where actual_position is determined by the kernel's FP4 access pattern.

# From Probes 1&2: group mapping
# From Probes 2&3: column mapping within groups of 8
# The COMBINED mapping gives us the full permutation within each 64-element block.

# Build the full mapping for one tile (8 K-groups × 8 N-columns)
# For group g (0..7) and output column n (0..7):
#   kernel reads from: (mapped_group, mapped_col)
#   where mapped_group = depends on Probe 2 pattern
#   and mapped_col = depends on Probe 3 pattern

# From Probe 2 (for n < N/2):
#   g even → reads g's scale (correct)
#   g odd → reads (g-1)'s scale (even partner)
# So mapped_group(g, n<N/2) = g - (g%2)  = even partner

# From Probe 3 (for group 0, even, n < N/2):
#   output n%8 → source col%8: [0, 1, 4, 5, 0, 1, 4, 5]

# We need: what column does group 1 (odd, n<N/2) read?
# That's what col_mapping_g1 shows.

print(f"\n  Group 0 (even) col pattern: {col_mapping[:8]}")
print(f"  Group 1 (odd)  col pattern: {col_mapping_g1[:8]}")

# ═══════════════════════════════════════════════════════════
# STEP 4: Try to fix the scale layout
# ═══════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("STEP 4: Test corrected scale layout")
print("=" * 70)

# Based on the mapping, construct a scale tensor where the kernel
# reads the correct values.

# Build inverse: we want scale_permuted[actual_pos] = correct_scale[desired_pos]
# actual_pos is where the kernel reads from
# desired_pos is the (g, n) we want for that output

# For output (g, n): kernel reads scale at (mapped_g, mapped_n)
# We want scale_at(mapped_g, mapped_n) = true_scale(g, n)
# Since true_scale(g, n) = scale[g, n] and it's the same for all n in a group,
# we need scale_at(mapped_g, mapped_n) = scale[g]

# This means we need to PRE-PLACE the scales in the positions where
# the kernel will find them for each (g, n) output.

# Use random scales for the actual test
torch.manual_seed(42)
W = torch.randn(K, N, device=DEVICE, dtype=torch.float32)
X = torch.randn(16, K, device=DEVICE, dtype=torch.half)

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

# --- Method 1: Original (wrong) pipeline ---
gl_orig = true_scales.max()
bn_orig = (true_scales / gl_orig).to(torch.float8_e4m3fn)
bs_orig = nvfp4_proc_scales(bn_orig)
bs_orig = marlin_permute_scales(bs_orig.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)
gt_orig = nvfp4_proc_global(gl_orig).to(torch.half).reshape(1, 1).to(DEVICE)

# Reference (using FP8-rounded scales)
kern_scales_orig = bn_orig.to(torch.float32) * gl_orig.item()
W_deq = av * sm * kern_scales_orig.repeat_interleave(GROUP, dim=0)
Y_ref = X.float() @ W_deq

Y_orig = gptq_marlin_gemm(
    X, None, qw, bs_orig, gt_orig, None, None, None, ws, FP4,
    16, N, K, True, False, True, False
).float()

err_orig = (Y_orig - Y_ref).norm() / Y_ref.norm()
print(f"  Original pipeline: rel L2 = {err_orig:.4f} ({err_orig*100:.1f}%)")

# --- Method 2: Rearrange scales using the probed mapping ---
# For each output (g, n), the kernel reads from (mapping[g,n], col_map(n))
# We need to place scale[g] at position (mapping[g,n], col_map(n))
# Since multiple (g,n) outputs may map to the same source position,
# this only works if the mapping is a bijection (1-to-1).

# First, let's check if the mapping IS a bijection
# Build the full source -> target map for first 64 columns (n<64)
src_positions = set()
for g in range(NG):
    for n in range(N):
        src_g = mapping[g, n].item()
        src_positions.add((src_g, n))

print(f"  Unique source positions: {len(src_positions)} (should be {NG * N} = {NG*N})")

# --- Method 3: Direct scale rearrangement ---
# Construct scale tensor where scale_rearranged[mapped_g, n] = true_scale[g]
# This is the INVERSE of the kernel's mapping
scale_rearranged = torch.zeros(NG, N, device=DEVICE)
collision_count = 0
for g in range(NG):
    for n in range(N):
        src_g = mapping[g, n].item()
        # At position (src_g, n), we want true_scale[g]
        # But the kernel at output (g, n) reads from (src_g, n)
        # So we need scale_tensor[src_g, n] = true_scale[g, n]
        if scale_rearranged[src_g, n] != 0 and abs(scale_rearranged[src_g, n] - true_scales[g, n]) > 1e-6:
            collision_count += 1
        scale_rearranged[src_g, n] = true_scales[g, n]

print(f"  Collisions in rearrangement: {collision_count}")

# Now process the rearranged scales through the pipeline
gl_fix = scale_rearranged.max()
bn_fix = (scale_rearranged / gl_fix).to(torch.float8_e4m3fn)
bs_fix = nvfp4_proc_scales(bn_fix)
bs_fix = marlin_permute_scales(bs_fix.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)
gt_fix = nvfp4_proc_global(gl_fix).to(torch.half).reshape(1, 1).to(DEVICE)

# Reference using the rearranged-then-FP8'd scales as decoded by kernel
kern_scales_fix = bn_fix.to(torch.float32) * gl_fix.item()
# Rebuild ref with the CORRECTED scale assignment
W_deq_fix = av * sm * true_scales.repeat_interleave(GROUP, dim=0)
Y_ref_fix = X.float() @ W_deq_fix

Y_fix = gptq_marlin_gemm(
    X, None, qw, bs_fix, gt_fix, None, None, None, ws, FP4,
    16, N, K, True, False, True, False
).float()

err_fix = (Y_fix - Y_ref_fix).norm() / Y_ref_fix.norm()
print(f"  Fixed pipeline:   rel L2 = {err_fix:.4f} ({err_fix*100:.1f}%)")
print(f"  Improvement: {err_orig/err_fix:.1f}x")

# If there are collisions, explain
if collision_count > 0:
    print(f"\n  WARNING: {collision_count} collisions means the mapping is not 1-to-1!")
    print(f"  Multiple (g,n) outputs want different scales at the same source position.")
    print(f"  This means simple rearrangement can't fix it — need a different approach.")

print("\nDONE")
