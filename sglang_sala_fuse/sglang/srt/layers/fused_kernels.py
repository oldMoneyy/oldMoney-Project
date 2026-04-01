"""Fused Triton kernels for MiniCPM-SALA inference.

Provides fused_scaled_add_rmsnorm which combines:
    residual = residual + x * scale
    out = rmsnorm(residual, weight, eps)
into a single GPU kernel pass, eliminating ~2 redundant reads/writes of the
hidden-state tensor per invocation (fires 64x per forward = 32 layers x 2).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_scaled_add_rmsnorm_kernel(
    X_ptr,
    Residual_ptr,
    Out_ptr,
    Weight_ptr,
    scale,
    eps,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Fuses: residual = residual + x * scale; out = rmsnorm(residual, w, eps)

    One program per row.  Entire hidden dimension fits in BLOCK_SIZE.
    All arithmetic in fp32; stores auto-cast to the tensor's dtype.
    """
    row = tl.program_id(0)
    base = row * hidden_size
    off = tl.arange(0, BLOCK_SIZE)
    mask = off < hidden_size

    # --- load x and residual ---
    x = tl.load(X_ptr + base + off, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(Residual_ptr + base + off, mask=mask, other=0.0).to(tl.float32)

    # --- scaled residual add ---
    r_new = r + x * scale

    # --- write updated residual (auto-casts to tensor dtype) ---
    tl.store(Residual_ptr + base + off, r_new, mask=mask)

    # --- RMSNorm ---
    var = tl.sum(r_new * r_new, axis=0) / hidden_size
    rrms = tl.rsqrt(var + eps)
    w = tl.load(Weight_ptr + off, mask=mask, other=1.0).to(tl.float32)
    out = r_new * rrms * w

    tl.store(Out_ptr + base + off, out, mask=mask)


def _next_pow2(n: int) -> int:
    """Smallest power of 2 >= n."""
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
) -> tuple:
    """Fused scaled-add + RMSNorm.

    Computes in one kernel pass:
        residual = residual + x * scale   (in-place)
        out      = rmsnorm(residual, weight, eps)

    Args:
        x:        [num_tokens, hidden_size]  sublayer output (attn or mlp)
        residual: [num_tokens, hidden_size]  running residual (modified in-place)
        weight:   [hidden_size]              RMSNorm weight
        scale:    muP scale factor  (scale_depth / sqrt(num_hidden_layers))
        eps:      RMSNorm epsilon

    Returns:
        (out, residual) — residual is the same tensor, modified in-place.
    """
    num_tokens, hidden_size = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = _next_pow2(hidden_size)

    _fused_scaled_add_rmsnorm_kernel[(num_tokens,)](
        x, residual, out, weight,
        scale, eps, hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=max(4, min(16, BLOCK_SIZE // 256)),
    )

    return out, residual
