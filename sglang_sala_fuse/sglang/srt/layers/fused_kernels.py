"""Fused Triton kernels for MiniCPM-SALA inference optimization."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_scaled_add_rmsnorm_kernel(
    X_ptr,          # attn/mlp output (gets scaled and added)
    Residual_ptr,   # residual (updated in-place: residual + x * scale)
    Out_ptr,        # normed output
    Weight_ptr,     # RMSNorm weight
    scale,          # muP scale factor (float)
    eps,            # RMSNorm epsilon
    hidden_size,    # hidden dimension
    BLOCK_SIZE: tl.constexpr,
):
    """Fuses: residual_new = residual + x * scale; out = rmsnorm(residual_new)

    This eliminates 2 extra memory passes per invocation vs doing them separately.
    """
    row_idx = tl.program_id(0)
    row_offset = row_idx * hidden_size

    # Load x and residual in one pass
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size

    x = tl.load(X_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(Residual_ptr + row_offset + offsets, mask=mask, other=0.0).to(
        tl.float32
    )

    # Scaled add: residual_new = residual + x * scale
    residual_new = residual + x * scale

    # Store updated residual
    tl.store(
        Residual_ptr + row_offset + offsets,
        residual_new.to(tl.bfloat16),
        mask=mask,
    )

    # RMSNorm: out = residual_new / rms * weight
    variance = tl.sum(residual_new * residual_new, axis=0) / hidden_size
    inv_rms = tl.rsqrt(variance + eps)

    weight = tl.load(Weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    normed = residual_new * inv_rms * weight

    tl.store(Out_ptr + row_offset + offsets, normed.to(tl.bfloat16), mask=mask)


@triton.jit
def _fused_rmsnorm_kernel(
    X_ptr,          # input (standalone, no residual add)
    Out_ptr,        # normed output
    Weight_ptr,     # RMSNorm weight
    eps,            # RMSNorm epsilon
    hidden_size,    # hidden dimension
    BLOCK_SIZE: tl.constexpr,
):
    """Standalone RMSNorm for the first layer's input_layernorm (no preceding scaled add)."""
    row_idx = tl.program_id(0)
    row_offset = row_idx * hidden_size

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size

    x = tl.load(X_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)

    variance = tl.sum(x * x, axis=0) / hidden_size
    inv_rms = tl.rsqrt(variance + eps)

    weight = tl.load(Weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    normed = x * inv_rms * weight

    tl.store(Out_ptr + row_offset + offsets, normed.to(tl.bfloat16), mask=mask)


def _next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def fused_scaled_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused scaled-add + RMSNorm.

    Computes:
        residual = residual + x * scale   (in-place)
        out = rmsnorm(residual, weight, eps)

    Returns: (out, residual)  — residual is modified in-place
    """
    assert x.shape == residual.shape
    num_tokens = x.shape[0]
    hidden_size = x.shape[-1]

    out = torch.empty_like(x)
    BLOCK_SIZE = _next_power_of_2(hidden_size)

    _fused_scaled_add_rmsnorm_kernel[(num_tokens,)](
        x,
        residual,
        out,
        weight,
        scale,
        eps,
        hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out, residual
