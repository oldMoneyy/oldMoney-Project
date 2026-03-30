"""Fused Triton kernels for MiniCPM acceleration.

These kernels fuse adjacent memory-bound operations to eliminate
intermediate DRAM round-trips without changing numerical results.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Autotune configurations
# ---------------------------------------------------------------------------

_scale_add_rmsnorm_configs = [
    triton.Config(kwargs={"BLOCK_H": 1024}, num_warps=4, num_stages=1),
    triton.Config(kwargs={"BLOCK_H": 1024}, num_warps=8, num_stages=1),
    triton.Config(kwargs={"BLOCK_H": 2048}, num_warps=8, num_stages=1),
    triton.Config(kwargs={"BLOCK_H": 2048}, num_warps=16, num_stages=1),
    triton.Config(kwargs={"BLOCK_H": 4096}, num_warps=8, num_stages=1),
    triton.Config(kwargs={"BLOCK_H": 4096}, num_warps=16, num_stages=1),
    triton.Config(kwargs={"BLOCK_H": 4096}, num_warps=8, num_stages=4),
    triton.Config(kwargs={"BLOCK_H": 4096}, num_warps=16, num_stages=4),
    triton.Config(kwargs={"BLOCK_H": 4096}, num_warps=32, num_stages=1),
    triton.Config(kwargs={"BLOCK_H": 4096}, num_warps=32, num_stages=4),
]

_sigmoid_mul_configs = [
    triton.Config(kwargs={"BLOCK_D": 1024}, num_warps=4, num_stages=1),
    triton.Config(kwargs={"BLOCK_D": 1024}, num_warps=8, num_stages=1),
    triton.Config(kwargs={"BLOCK_D": 2048}, num_warps=8, num_stages=1),
    triton.Config(kwargs={"BLOCK_D": 2048}, num_warps=16, num_stages=1),
    triton.Config(kwargs={"BLOCK_D": 4096}, num_warps=8, num_stages=1),
    triton.Config(kwargs={"BLOCK_D": 4096}, num_warps=16, num_stages=1),
    triton.Config(kwargs={"BLOCK_D": 4096}, num_warps=8, num_stages=4),
    triton.Config(kwargs={"BLOCK_D": 4096}, num_warps=16, num_stages=4),
]

_rmsnorm_configs = [
    triton.Config(kwargs={"BLOCK_H": 64}, num_warps=2, num_stages=1),
    triton.Config(kwargs={"BLOCK_H": 128}, num_warps=4, num_stages=1),
    triton.Config(kwargs={"BLOCK_H": 128}, num_warps=8, num_stages=1),
    triton.Config(kwargs={"BLOCK_H": 256}, num_warps=8, num_stages=1),
    triton.Config(kwargs={"BLOCK_H": 256}, num_warps=16, num_stages=1),
]

_rmsnorm_sigmoid_mul_configs = [
    triton.Config(kwargs={"BLOCK_D": 1024}, num_warps=4, num_stages=1),
    triton.Config(kwargs={"BLOCK_D": 1024}, num_warps=8, num_stages=1),
    triton.Config(kwargs={"BLOCK_D": 2048}, num_warps=8, num_stages=1),
    triton.Config(kwargs={"BLOCK_D": 2048}, num_warps=16, num_stages=1),
    triton.Config(kwargs={"BLOCK_D": 4096}, num_warps=8, num_stages=1),
    triton.Config(kwargs={"BLOCK_D": 4096}, num_warps=16, num_stages=1),
    triton.Config(kwargs={"BLOCK_D": 4096}, num_warps=8, num_stages=4),
    triton.Config(kwargs={"BLOCK_D": 4096}, num_warps=16, num_stages=4),
]


# ---------------------------------------------------------------------------
# Fused scale + add + RMSNorm
# ---------------------------------------------------------------------------

@triton.autotune(configs=_scale_add_rmsnorm_configs, key=["H"])
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
    row = tl.program_id(0)
    X_row = X_ptr + row * H
    R_row = Residual_ptr + row * H

    variance = tl.zeros([BLOCK_H], dtype=tl.float32)

    for off in range(0, H, BLOCK_H):
        cols = off + tl.arange(0, BLOCK_H)
        mask = cols < H

        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R_row + cols, mask=mask, other=0.0).to(tl.float32)

        r_new = x * scale + r
        tl.store(R_row + cols, r_new.to(tl.bfloat16), mask=mask)

        variance += r_new * r_new

    var_sum = tl.sum(variance)
    rrms = tl.math.rsqrt(var_sum / H + eps)

    for off in range(0, H, BLOCK_H):
        cols = off + tl.arange(0, BLOCK_H)
        mask = cols < H

        r_new = tl.load(R_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(Weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        out = r_new * rrms * w
        tl.store(X_row + cols, out.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# Fused sigmoid * mul
# ---------------------------------------------------------------------------

@triton.autotune(configs=_sigmoid_mul_configs, key=["D"])
@triton.jit
def _fused_sigmoid_mul_kernel(
    O_ptr,          # [T, D] - attention output, overwritten with o * sigmoid(z)
    Z_ptr,          # [T, D] - gate values
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
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
    T, D = o.shape
    _fused_sigmoid_mul_kernel[(T,)](o, z, D=D)


# ---------------------------------------------------------------------------
# RMSNorm (for q/k norm)
# ---------------------------------------------------------------------------

@triton.autotune(configs=_rmsnorm_configs, key=["head_dim"])
@triton.jit
def _rmsnorm_kernel(
    X_ptr,          # [N, head_dim]
    W_ptr,          # [head_dim]
    eps,
    stride_row,
    head_dim: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
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
    N_q, head_dim = q.shape
    N_k = k.shape[0]

    _rmsnorm_kernel[(N_q,)](
        q, q_weight, eps,
        q.stride(0),
        head_dim=head_dim,
    )
    _rmsnorm_kernel[(N_k,)](
        k, k_weight, eps,
        k.stride(0),
        head_dim=head_dim,
    )


def fused_scale_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: float,
    eps: float,
) -> None:
    T, H = x.shape

    _fused_scale_add_rmsnorm_kernel[(T,)](
        x, residual, weight,
        scale, eps,
        H=H,
    )


# ---------------------------------------------------------------------------
# Fused RMSNorm + sigmoid * mul
# ---------------------------------------------------------------------------

@triton.autotune(configs=_rmsnorm_sigmoid_mul_configs, key=["D"])
@triton.jit
def _fused_rmsnorm_sigmoid_mul_kernel(
    O_ptr,          # [T, D] - input, overwritten with rmsnorm(o) * sigmoid(z)
    Z_ptr,          # [T, D] - gate values
    W_ptr,          # [D]    - RMSNorm weight
    eps,            # float  - RMSNorm epsilon
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    O_row = O_ptr + row * D
    Z_row = Z_ptr + row * D

    variance = tl.zeros([BLOCK_D], dtype=tl.float32)
    for off in range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D
        o = tl.load(O_row + cols, mask=mask, other=0.0).to(tl.float32)
        variance += o * o

    var_sum = tl.sum(variance)
    rrms = tl.math.rsqrt(var_sum / D + eps)

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
    T, D = o.shape

    _fused_rmsnorm_sigmoid_mul_kernel[(T,)](
        o, z, weight,
        eps,
        D=D,
    )
