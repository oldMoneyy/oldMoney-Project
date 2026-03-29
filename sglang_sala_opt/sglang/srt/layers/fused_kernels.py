"""Fused Triton kernels for MiniCPM acceleration.

These kernels fuse adjacent memory-bound operations to eliminate
intermediate DRAM round-trips without changing numerical results.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_scale_add_rmsnorm_kernel(
    X_ptr,          # [T, H] - input (scaled attn/mlp output), overwritten with norm output
    Residual_ptr,   # [T, H] - residual, overwritten with x*scale + residual
    Weight_ptr,     # [H]    - RMSNorm weight
    scale,          # float  - residual scale factor
    eps,            # float  - RMSNorm epsilon
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused: residual = x * scale + residual; x = rmsnorm(residual)

    Replaces separate `x *= scale` + `fused_add_rmsnorm(x, residual)`.
    Saves one full [T, H] memory round-trip per call.
    """
    row = tl.program_id(0)
    X_row = X_ptr + row * H
    R_row = Residual_ptr + row * H

    # Accumulate variance in FP32 for numerical stability
    variance = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Pass 1: compute residual_new = x * scale + residual, accumulate variance
    for off in range(0, H, BLOCK_H):
        cols = off + tl.arange(0, BLOCK_H)
        mask = cols < H

        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R_row + cols, mask=mask, other=0.0).to(tl.float32)

        # Fused: residual_new = x * scale + residual
        r_new = x * scale + r
        tl.store(R_row + cols, r_new.to(tl.bfloat16), mask=mask)

        variance += r_new * r_new

    # Compute rsqrt(mean(x^2) + eps)
    var_sum = tl.sum(variance)
    rrms = tl.math.rsqrt(var_sum / H + eps)

    # Pass 2: normalize and write output
    for off in range(0, H, BLOCK_H):
        cols = off + tl.arange(0, BLOCK_H)
        mask = cols < H

        r_new = tl.load(R_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(Weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        out = r_new * rrms * w
        tl.store(X_row + cols, out.to(tl.bfloat16), mask=mask)


def fused_scale_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: float,
    eps: float,
) -> None:
    """In-place fused: residual = x * scale + residual; x = rmsnorm(residual).

    Args:
        x: [T, H] bf16 - overwritten with norm output
        residual: [T, H] bf16 - overwritten with updated residual
        weight: [H] - RMSNorm weight parameter
        scale: scalar - residual scale factor (scale_depth / sqrt(num_layers))
        eps: RMSNorm epsilon
    """
    T, H = x.shape
    BLOCK_H = triton.next_power_of_2(min(H, 4096))

    _fused_scale_add_rmsnorm_kernel[(T,)](
        x, residual, weight,
        scale, eps,
        H=H,
        BLOCK_H=BLOCK_H,
    )
