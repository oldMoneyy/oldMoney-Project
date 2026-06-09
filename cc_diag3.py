#!/usr/bin/env python3
"""
Phase 0 ROOT CAUSE diagnostic.
Fix: marlin_make_workspace(DEVICE) not (N, DEVICE)
"""

import torch, inspect
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
midpoints = (e2m1_pos[:-1] + e2m1_pos[1:]) / 2.0

# ═══ STEP 0: Print marlin_permute_scales + get_scale_perms ═══
print("=" * 70)
print("STEP 0: marlin_permute_scales source")
print("=" * 70)
print(inspect.getsource(marlin_permute_scales))

from sglang.srt.layers.quantization.marlin_utils import get_scale_perms
print("\nget_scale_perms source:")
print(inspect.getsource(get_scale_perms))

# ═══ Helpers ═══
def nvfp4_proc_scales(s):
    s = s.to(torch.half)
    s = s.view(-1, 4)[:, [0, 2, 1, 3]].view(s.size(0), -1)
    s = s * (2 ** 7)
    s = torch.where(s < 2, torch.zeros_like(s), s)
    s = (s.view(torch.int16) << 1).view(torch.float8_e4m3fn)
    return s[:, 1::2].contiguous()

def nvfp4_proc_global(g):
    return g * (2.0 ** 7)

def quantize(W, scales):
    K, N = W.shape
    W_sc = W / scales.repeat_interleave(GROUP, dim=0)
    ai = torch.bucketize(W_sc.abs().reshape(-1), midpoints).reshape(K, N)
    sb = (W_sc < 0).int()
    codes = (ai.int() | (sb << 3)) & 0xF
    return codes, e2m1_pos[ai.long()], torch.where(sb.bool(), -1.0, 1.0)

def pack_weights(codes, K, N):
    cNK = codes.T.contiguous()
    g = cNK.reshape(N, K // 8, 8)
    pk = torch.zeros(N, K // 8, dtype=torch.int32, device=DEVICE)
    for i in range(8):
        pk |= (g[:, :, i] & 0xF) << (i * 4)
    pe = torch.empty(0, dtype=torch.int32, device=DEVICE)
    return gptq_marlin_repack(pk.T.contiguous(), pe, K, N, num_bits=4), pe

def run_kernel(X, qw, bs, gs, pe, M, N, K, fp32_reduce=False):
    ws = marlin_make_workspace(DEVICE)
    return gptq_marlin_gemm(
        X, None, qw, bs, gs,
        None, None, None, ws, FP4,
        M, N, K, is_k_full=True,
        use_atomic_add=False, use_fp32_reduce=fp32_reduce, is_zp_float=False
    )

M, N, K = 16, 128, 256
NG = K // GROUP

W = torch.randn(K, N, device=DEVICE, dtype=torch.float32)
X = torch.randn(M, K, device=DEVICE, dtype=torch.half)

# ═══ TEST A: Baseline ═══
print("\n" + "=" * 70)
print("TEST A: Baseline (random f32 scales)")
print("=" * 70)

Wg = W.reshape(NG, GROUP, N)
scales_f32 = Wg.abs().amax(dim=1) / 6.0
scales_f32 = scales_f32.clamp(min=1e-10)

codes, av, sm = quantize(W, scales_f32)
qw, pe = pack_weights(codes, K, N)

gl = scales_f32.max()
bn = (scales_f32 / gl).to(torch.float8_e4m3fn)
bs = nvfp4_proc_scales(bn)
bs = marlin_permute_scales(bs.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)
gt = nvfp4_proc_global(gl).to(torch.half).reshape(1,1).to(DEVICE)

W_deq_f32 = av * sm * scales_f32.repeat_interleave(GROUP, dim=0)
Y_ref_f32 = X.float() @ W_deq_f32

kern_scales = bn.to(torch.float32) * gl.item()
W_deq_k = av * sm * kern_scales.repeat_interleave(GROUP, dim=0)
Y_ref_k = X.float() @ W_deq_k

Yk16 = run_kernel(X, qw, bs, gt, pe, M, N, K, fp32_reduce=False).float()
Yk32 = run_kernel(X, qw, bs, gt, pe, M, N, K, fp32_reduce=True).float()

e_f32_16 = (Yk16 - Y_ref_f32).norm() / Y_ref_f32.norm()
e_f32_32 = (Yk32 - Y_ref_f32).norm() / Y_ref_f32.norm()
e_k_16 = (Yk16 - Y_ref_k).norm() / Y_ref_k.norm()
e_k_32 = (Yk32 - Y_ref_k).norm() / Y_ref_k.norm()
e_16v32 = (Yk16 - Yk32).norm() / Yk32.norm()

sfp8err = (scales_f32 - kern_scales).abs().div(scales_f32.abs())
print(f"  Scale FP8 quant err: mean={sfp8err.mean():.4f} max={sfp8err.max():.4f}")
print(f"  ref_f32 vs ref_kern: {(Y_ref_f32 - Y_ref_k).norm() / Y_ref_f32.norm():.4f}")
print(f"  Kernel(fp16) vs ref_f32:  {e_f32_16:.4f} ({e_f32_16*100:.1f}%)")
print(f"  Kernel(fp32) vs ref_f32:  {e_f32_32:.4f} ({e_f32_32*100:.1f}%)")
print(f"  Kernel(fp16) vs ref_kern: {e_k_16:.4f} ({e_k_16*100:.1f}%)")
print(f"  Kernel(fp32) vs ref_kern: {e_k_32:.4f} ({e_k_32*100:.1f}%)")
print(f"  Kernel fp16 vs fp32:      {e_16v32:.4f} ({e_16v32*100:.1f}%)")

# ═══ TEST B: Power-of-2 scales ═══
print("\n" + "=" * 70)
print("TEST B: Power-of-2 scales (exact in FP8)")
print("=" * 70)

torch.manual_seed(77)
W2 = torch.randn(K, N, device=DEVICE, dtype=torch.float32)
X2 = torch.randn(M, K, device=DEVICE, dtype=torch.half)
Wg2 = W2.reshape(NG, GROUP, N)
scales_p2 = 2.0 ** torch.round(torch.log2(Wg2.abs().amax(dim=1) / 6.0))
scales_p2 = scales_p2.clamp(min=2**-7)

codes2, av2, sm2 = quantize(W2, scales_p2)
qw2, pe2 = pack_weights(codes2, K, N)

gl2 = scales_p2.max()
bn2 = (scales_p2 / gl2).to(torch.float8_e4m3fn)
kern_s2 = bn2.to(torch.float32) * gl2.item()
s2e = (scales_p2 - kern_s2).abs().div(scales_p2.abs())
print(f"  Scale norm+FP8 err: mean={s2e.mean():.6f} max={s2e.max():.6f}")

bs2 = nvfp4_proc_scales(bn2)
bs2 = marlin_permute_scales(bs2.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)
gt2 = nvfp4_proc_global(gl2).to(torch.half).reshape(1,1).to(DEVICE)

Y_ref2 = X2.float() @ (av2 * sm2 * scales_p2.repeat_interleave(GROUP, dim=0))
Y_ref2k = X2.float() @ (av2 * sm2 * kern_s2.repeat_interleave(GROUP, dim=0))

Yk2_32 = run_kernel(X2, qw2, bs2, gt2, pe2, M, N, K, fp32_reduce=True).float()
e2_f32 = (Yk2_32 - Y_ref2).norm() / Y_ref2.norm()
e2_k = (Yk2_32 - Y_ref2k).norm() / Y_ref2k.norm()
print(f"  Kernel(fp32) vs ref_f32:  {e2_f32:.4f} ({e2_f32*100:.1f}%)")
print(f"  Kernel(fp32) vs ref_kern: {e2_k:.4f} ({e2_k*100:.1f}%)")

# ═══ TEST C: Uniform scale (sanity) ═══
print("\n" + "=" * 70)
print("TEST C: Uniform scale = 0.25 (sanity)")
print("=" * 70)

torch.manual_seed(42)
W3 = torch.randn(K, N, device=DEVICE, dtype=torch.float32)
X3 = torch.randn(M, K, device=DEVICE, dtype=torch.half)
su = torch.full((NG, N), 0.25, device=DEVICE)
codes3, av3, sm3 = quantize(W3, su)
qw3, pe3 = pack_weights(codes3, K, N)
gl3 = su.max()
bn3 = (su / gl3).to(torch.float8_e4m3fn)
bs3 = nvfp4_proc_scales(bn3)
bs3 = marlin_permute_scales(bs3.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)
gt3 = nvfp4_proc_global(gl3).to(torch.half).reshape(1,1).to(DEVICE)
Y_ref3 = X3.float() @ (av3 * sm3 * 0.25)
Yk3 = run_kernel(X3, qw3, bs3, gt3, pe3, M, N, K, fp32_reduce=True).float()
e3 = (Yk3 - Y_ref3).norm() / Y_ref3.norm()
print(f"  Uniform: rel L2 = {e3:.6f} ({e3*100:.3f}%)")

# ═══ TEST D: Per-group scale probe ═══
print("\n" + "=" * 70)
print("TEST D: Per-group scale probe (linearly increasing scales)")
print("=" * 70)

torch.manual_seed(42)
X_f = torch.ones(1, K, device=DEVICE, dtype=torch.half)

codes_g = torch.zeros(K, N, dtype=torch.int32, device=DEVICE)
scales_g = torch.zeros(NG, N, device=DEVICE)
for g in range(NG):
    codes_g[g * GROUP, 0] = 4  # code=4 → value 2.0
    scales_g[g, :] = 0.25 * (g + 1)  # 0.25, 0.5, ..., 4.0

qw_g, pe_g = pack_weights(codes_g, K, N)
gl_g = scales_g.max()
bn_g = (scales_g / gl_g).to(torch.float8_e4m3fn)
bs_g = nvfp4_proc_scales(bn_g)
bs_g = marlin_permute_scales(bs_g.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)
gt_g = nvfp4_proc_global(gl_g).to(torch.half).reshape(1,1).to(DEVICE)

expected_total = sum(2.0 * 0.25 * (g+1) for g in range(NG))
Yk_g = run_kernel(X_f, qw_g, bs_g, gt_g, pe_g, 1, N, K, fp32_reduce=True).float()
print(f"  Total Y[0,0] = {Yk_g[0,0]:.4f}  (expected: {expected_total:.1f})")

# Probe each group individually
print(f"\n  Per-group decode (activation=one-hot at k=g*16):")
for g in range(NG):
    X_1g = torch.zeros(1, K, device=DEVICE, dtype=torch.half)
    X_1g[0, g * GROUP] = 1.0
    Yk_1g = run_kernel(X_1g, qw_g, bs_g, gt_g, pe_g, 1, N, K, fp32_reduce=True).float()
    exp = 2.0 * 0.25 * (g + 1)
    actual_scale = Yk_1g[0, 0].item() / 2.0
    target_scale = 0.25 * (g + 1)
    ratio = actual_scale / target_scale if target_scale > 0 else 0
    ok = "OK" if abs(ratio - 1.0) < 0.05 else f"WRONG (ratio={ratio:.3f})"
    print(f"    g={g:2d}: Y={Yk_1g[0,0]:8.4f}  exp={exp:6.3f}  "
          f"actual_s={actual_scale:6.4f}  target_s={target_scale:5.3f}  {ok}")

# ═══ TEST E: Two-scale alternating ═══
print("\n" + "=" * 70)
print("TEST E: Two alternating scales (0.5 vs 0.25 by group)")
print("=" * 70)

torch.manual_seed(42)
X_e = torch.randn(M, K, device=DEVICE, dtype=torch.half)
scales_two = torch.zeros(NG, N, device=DEVICE)
for g in range(NG):
    scales_two[g, :] = 0.5 if g % 2 == 0 else 0.25

codes_e = torch.full((K, N), 2, dtype=torch.int32, device=DEVICE)
av_e = torch.ones(K, N, device=DEVICE)
sm_e = torch.ones(K, N, device=DEVICE)
qw_e, pe_e = pack_weights(codes_e, K, N)

gl_e = scales_two.max()
bn_e = (scales_two / gl_e).to(torch.float8_e4m3fn)
bs_e = nvfp4_proc_scales(bn_e)
bs_e = marlin_permute_scales(bs_e.reshape(-1, N), size_k=K, size_n=N, group_size=GROUP)
gt_e = nvfp4_proc_global(gl_e).to(torch.half).reshape(1,1).to(DEVICE)

W_deq_e = av_e * sm_e * scales_two.repeat_interleave(GROUP, dim=0)
Y_ref_e = X_e.float() @ W_deq_e

Yk_e = run_kernel(X_e, qw_e, bs_e, gt_e, pe_e, M, N, K, fp32_reduce=True).float()
ee = (Yk_e - Y_ref_e).norm() / Y_ref_e.norm()
print(f"  Kernel vs correct ref: {ee:.4f} ({ee*100:.1f}%)")

ratios = Yk_e / Y_ref_e
print(f"  Ratio kernel/ref: mean={ratios.mean():.4f} std={ratios.std():.4f}")
print(f"  Sample ratios [0,:8]: {[f'{r:.3f}' for r in ratios[0,:8].tolist()]}")

print("\nDONE")
