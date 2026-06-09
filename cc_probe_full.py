#!/usr/bin/env python3
"""
Empirically determine the COMPLETE pre-pipeline → kernel-reads mapping.

Strategy: set exactly ONE scale position to a non-zero value (all others zero).
The kernel output reveals which (g_out, n_out) positions read from that source.

We probe each pre-pipeline position (g, n) and record which outputs use it.
This gives us the EXACT mapping needed to construct the correct scale layout.
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

# Prepare weights: ALL codes = 4 (value 2.0) at EVERY position
# This means Y[m, n] = sum_k X[m,k] * 2.0 * scale(group(k), n)
codes_all = torch.full((K, N), 4, dtype=torch.int32, device=DEVICE)
cNK = codes_all.T.contiguous()
gg = cNK.reshape(N, K // 8, 8)
pk = torch.zeros(N, K // 8, dtype=torch.int32, device=DEVICE)
for i in range(8):
    pk |= (gg[:, :, i] & 0xF) << (i * 4)
qw = gptq_marlin_repack(pk.T.contiguous(), pe, K, N, num_bits=4)

# For each source (g_src, n_src), set ONLY that position to 1.0,
# all others to a tiny non-zero value (to avoid the clamp-to-zero issue).
# Then the kernel output tells us where that scale appears.

# Actually simpler: set ALL scales to baseline=0.5, then set ONE position to 1.0.
# The positions where kernel output is higher reveal the mapping.

# BUT: each output element sums over ALL K groups. So changing one group's scale
# affects the output proportionally to that group's contribution.
# With all codes = 4 (val 2.0) and uniform activation X = ones:
#   Y[0, n] = sum_g 16 * 2.0 * scale[g, n] (16 elements per group)

# With baseline=0.5 and one position (g*, n*) = 1.0:
#   Y[0, n*] = 16 * 2.0 * (0.5 * (NG-1) + 1.0) = 32 * (7.5 + 1) = 32 * 8.5 = 272
#   vs baseline Y[0, n] = 32 * 0.5 * NG = 32 * 8 = 256

# The difference is small. Better approach: use one-hot activation per group,
# one-hot scale per column.

# BEST approach: For each source column n_src (in pre-pipeline):
#   Set scale[:, n_src] = 1.0 and scale[:, other] = 0.5
#   Use all-ones activation, but only care about the N-column output pattern
#   The output columns where the sum increases reveal which output n_out
#   maps to this source n_src.

# Even better: For each source (g_src, n_src):
#   Use one-hot activation at k = g_src * 16
#   Set scale = baseline everywhere, scale[g_src, n_src] = probe_val
#   Y[0, n_out] = 2.0 * scale_seen_at(g_src, n_out)
#   Columns where Y differs from baseline reveal the mapping.

# This requires NG * N probes = 16 * 128 = 2048. Too many.
# But we can batch: for each g_src, probe all n_src at once with unique values.

# For each g_src: set scale[g_src, n] = (n % 16) * 0.0625 + 0.0625
# (unique per n%16, repeating every 16 columns)
# This gives 16 unique values (0.0625 to 1.0 in steps of 0.0625).
# But after FP8 rounding, some might collide. Let's use power-of-2 fractions.

# Actually, let's just use values 0.125 * (n % 8 + 1) as before,
# but also vary by n // 8 using a different scale per 8-col block.

# Simplest: for each g_src, set scale[g_src, :] to a unique COLUMN-DEPENDENT
# pattern, with other groups at baseline. Then probe with one-hot activation.

print("=" * 70)
print("PROBE: Complete pre-pipeline position → kernel output mapping")
print("  For each source row g_src, varying column scales")
print("=" * 70)

# We'll probe 4 source rows (g_src = 0, 1, 2, 3) to understand the pattern.
# For each, we set that row's scales to unique-per-column values.

baseline = 0.5  # baseline scale for all non-probed positions
gl = torch.tensor(1.0, device=DEVICE)  # global scale = 1.0 for simplicity
gt = nvfp4_proc_global(gl).to(torch.half).reshape(1, 1).to(DEVICE)

for g_src in range(4):
    print(f"\n--- Source row g={g_src} ---")

    # Set scales: all rows = baseline, row g_src has unique per-column values
    scales = torch.full((NG, N), baseline, device=DEVICE)

    # Use 8 distinct values per 8-column block, repeating
    for n in range(N):
        scales[g_src, n] = (n % 8 + 1) * 0.0625  # 0.0625 to 0.5

    bn = (scales / 1.0).to(torch.float8_e4m3fn)  # gl=1.0
    bs = nvfp4_proc_scales(bn)
    bs = marlin_permute_scales(bs.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)

    # Probe with one-hot activation at k = g_src * 16
    X = torch.zeros(1, K, device=DEVICE, dtype=torch.half)
    X[0, g_src * GROUP] = 1.0

    Y = gptq_marlin_gemm(
        X, None, qw, bs, gt, None, None, None, ws, FP4,
        1, N, K, True, False, True, False
    ).float()

    # Y[0, n] = 2.0 * actual_scale_at(g_src, n)
    # Baseline output = 2.0 * 0.5 = 1.0
    # Probed output = 2.0 * scales[g_src, n_src_mapped_to_n]

    # For output columns 0-15 and 64-79:
    print(f"  Output cols 0-15 (n < N/2):")
    for n in range(16):
        actual_scale = Y[0, n].item() / 2.0
        src_n_mod8 = round(actual_scale / 0.0625 - 1)
        is_baseline = abs(actual_scale - baseline) < 0.01
        if is_baseline:
            print(f"    n={n:3d}: scale={actual_scale:.4f} → BASELINE (not from g={g_src})")
        else:
            print(f"    n={n:3d}: scale={actual_scale:.4f} → src_n%8={src_n_mod8}")

    print(f"  Output cols 64-79 (n >= N/2):")
    for n in range(64, 80):
        actual_scale = Y[0, n].item() / 2.0
        src_n_mod8 = round(actual_scale / 0.0625 - 1)
        is_baseline = abs(actual_scale - baseline) < 0.01
        if is_baseline:
            print(f"    n={n:3d}: scale={actual_scale:.4f} → BASELINE (not from g={g_src})")
        else:
            print(f"    n={n:3d}: scale={actual_scale:.4f} → src_n%8={src_n_mod8}")

print("\n\n" + "=" * 70)
print("PROBE 2: Which g_src contributes to which (g_out, n_out)?")
print("  Set scale[g_src, :] = 1.0, others = 0.25")
print("  Probe each g_out with one-hot activation")
print("=" * 70)

for g_src in range(4):
    scales2 = torch.full((NG, N), 0.25, device=DEVICE)
    scales2[g_src, :] = 1.0

    bn2 = (scales2 / 1.0).to(torch.float8_e4m3fn)
    bs2 = nvfp4_proc_scales(bn2)
    bs2 = marlin_permute_scales(bs2.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)

    print(f"\n  Source g_src={g_src}: scale=1.0 (others=0.25)")

    for g_out in range(4):
        X2 = torch.zeros(1, K, device=DEVICE, dtype=torch.half)
        X2[0, g_out * GROUP] = 1.0

        Y2 = gptq_marlin_gemm(
            X2, None, qw, bs2, gt, None, None, None, ws, FP4,
            1, N, K, True, False, True, False
        ).float()

        # Y = 2.0 * actual_scale. If from g_src: 2.0. If from other: 0.5.
        val_n0 = Y2[0, 0].item() / 2.0
        val_n64 = Y2[0, 64].item() / 2.0
        src_n0 = "g_src" if abs(val_n0 - 1.0) < 0.1 else "other"
        src_n64 = "g_src" if abs(val_n64 - 1.0) < 0.1 else "other"
        print(f"    g_out={g_out}: n=0 scale={val_n0:.3f}({src_n0}), "
              f"n=64 scale={val_n64:.3f}({src_n64})")

print("\nDONE")
