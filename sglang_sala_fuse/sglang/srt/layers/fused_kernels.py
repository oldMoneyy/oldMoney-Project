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

    PRECISION CONTRACT (must match the unfused PyTorch path exactly):
      1. Scaled-add computed in fp32 (PyTorch's opmath for bf16).
      2. Result rounded to bf16 and stored as residual.
      3. RMSNorm operates on the bf16-rounded value (converted back to fp32),
         NOT the full-precision fp32 value from step 1.  This matches the
         original code where rmsnorm() receives a bf16 tensor and upcasts
         to fp32 internally.
    """
    row = tl.program_id(0)
    base = row.to(tl.int64) * hidden_size  # int64 prevents overflow for large token counts
    off = tl.arange(0, BLOCK_SIZE)
    mask = off < hidden_size

    # --- load x and residual in fp32 ---
    x = tl.load(X_ptr + base + off, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(Residual_ptr + base + off, mask=mask, other=0.0).to(tl.float32)

    # --- scaled residual add (fp32, matching PyTorch opmath) ---
    r_new = r + x * tl.cast(scale, tl.float32)

    # --- round to bf16: CRITICAL for numerical equivalence ---
    # The original code stores the sum as bf16, then rmsnorm loads bf16.
    # Without this round-trip, RMSNorm sees slightly different fp32 values
    # and the error cascades through 32 layers via attention softmax.
    r_rounded = r_new.to(tl.bfloat16)
    tl.store(Residual_ptr + base + off, r_rounded, mask=mask)

    # --- RMSNorm on bf16-rounded value (upcast to fp32 for precision) ---
    r_f32 = r_rounded.to(tl.float32)
    var = tl.sum(r_f32 * r_f32, axis=0) / hidden_size
    rrms = tl.rsqrt(var + tl.cast(eps, tl.float32))
    w = tl.load(Weight_ptr + off, mask=mask, other=1.0).to(tl.float32)
    out = r_f32 * rrms * w

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
