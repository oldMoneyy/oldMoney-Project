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


@triton.jit
def _fused_sigmoid_mul_kernel(
    O_ptr,          # [T, D] - attention output, overwritten with o * sigmoid(z)
    Z_ptr,          # [T, D] - gate values
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused: o = o * sigmoid(z) in a single pass."""
    row = tl.program_id(0)
    O_row = O_ptr + row * D
    Z_row = Z_ptr + row * D

    for off in range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D

        o = tl.load(O_row + cols, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(Z_row + cols, mask=mask, other=0.0).to(tl.float32)

        out = o * tl.sigmoid(z)
        tl.store(O_row + cols, out.to(tl.bfloat16), mask=mask)


def fused_sigmoid_mul(o: torch.Tensor, z: torch.Tensor) -> None:
    """In-place fused: o = o * sigmoid(z).

    Args:
        o: [T, D] bf16 - overwritten with result
        z: [T, D] bf16 - gate values
    """
    T, D = o.shape
    BLOCK_D = triton.next_power_of_2(min(D, 4096))
    _fused_sigmoid_mul_kernel[(T,)](o, z, D=D, BLOCK_D=BLOCK_D)


@triton.jit
def _rmsnorm_kernel(
    X_ptr,          # [N, head_dim]
    W_ptr,          # [head_dim]
    eps,
    stride_row,
    head_dim: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Single-row RMSNorm kernel, reused for both q and k."""
    row = tl.program_id(0)
    Row_ptr = X_ptr + row * stride_row

    variance = tl.zeros([BLOCK_H], dtype=tl.float32)
    for off in range(0, head_dim, BLOCK_H):
        cols = off + tl.arange(0, BLOCK_H)
        mask = cols < head_dim
        x = tl.load(Row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        variance += x * x

    var_sum = tl.sum(variance)
    rrms = tl.math.rsqrt(var_sum / head_dim + eps)

    for off in range(0, head_dim, BLOCK_H):
        cols = off + tl.arange(0, BLOCK_H)
        mask = cols < head_dim
        x = tl.load(Row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = x * rrms * w
        tl.store(Row_ptr + cols, out.to(tl.bfloat16), mask=mask)


def fused_qk_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
) -> None:
    """In-place fused RMSNorm for q and k tensors.

    Launches both q and k norms on the same CUDA stream back-to-back
    with zero CPU overhead between them (no Python loop, no cat/copy).

    Args:
        q: [N_q, head_dim] bf16 - overwritten with rmsnorm(q)
        k: [N_k, head_dim] bf16 - overwritten with rmsnorm(k)
        q_weight: [head_dim] - q RMSNorm weight
        k_weight: [head_dim] - k RMSNorm weight
        eps: RMSNorm epsilon
    """
    N_q, head_dim = q.shape
    N_k = k.shape[0]
    BLOCK_H = triton.next_power_of_2(min(head_dim, 4096))

    _rmsnorm_kernel[(N_q,)](
        q, q_weight, eps,
        q.stride(0),
        head_dim=head_dim, BLOCK_H=BLOCK_H,
    )
    _rmsnorm_kernel[(N_k,)](
        k, k_weight, eps,
        k.stride(0),
        head_dim=head_dim, BLOCK_H=BLOCK_H,
    )


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


@triton.jit
def _fused_rmsnorm_sigmoid_mul_kernel(
    O_ptr,          # [T, D] - input, overwritten with rmsnorm(o) * sigmoid(z)
    Z_ptr,          # [T, D] - gate values
    W_ptr,          # [D]    - RMSNorm weight
    eps,            # float  - RMSNorm epsilon
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused: o = rmsnorm(o, weight) * sigmoid(z) in a single 2-pass kernel.

    Replaces separate `o_norm(o)` + `fused_sigmoid_mul(o, z)`.
    Saves one full [T, D] memory round-trip per call.
    """
    row = tl.program_id(0)
    O_row = O_ptr + row * D
    Z_row = Z_ptr + row * D

    # Pass 1: accumulate variance in FP32
    variance = tl.zeros([BLOCK_D], dtype=tl.float32)
    for off in range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D
        o = tl.load(O_row + cols, mask=mask, other=0.0).to(tl.float32)
        variance += o * o

    var_sum = tl.sum(variance)
    rrms = tl.math.rsqrt(var_sum / D + eps)

    # Pass 2: normalize, gate with sigmoid, and write
    for off in range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D

        o = tl.load(O_row + cols, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(Z_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        out = (o * rrms * w) * tl.sigmoid(z)
        tl.store(O_row + cols, out.to(tl.bfloat16), mask=mask)


def fused_rmsnorm_sigmoid_mul(
    o: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    """In-place fused: o = rmsnorm(o, weight) * sigmoid(z).

    Fuses RMSNorm + sigmoid gating into a single kernel, eliminating
    the intermediate normalized tensor write/read.

    Args:
        o: [T, D] bf16 - overwritten with result
        z: [T, D] bf16 - gate values
        weight: [D] - RMSNorm weight
        eps: RMSNorm epsilon
    """
    T, D = o.shape
    BLOCK_D = triton.next_power_of_2(min(D, 4096))

    _fused_rmsnorm_sigmoid_mul_kernel[(T,)](
        o, z, weight,
        eps,
        D=D,
        BLOCK_D=BLOCK_D,
    )
