"""Fused Triton kernel for MiniCPM-SALA scaled residual add.

Fuses `result = residual + x * scale` (two PyTorch kernels: mul + add)
into a single kernel, eliminating the intermediate tensor and one kernel launch.

Does NOT touch RMSNorm — that stays with sgl_kernel.rmsnorm for bit-exact results.

PRECISION CONTRACT:
  PyTorch computes `residual + hidden_states * scale` as two ops:
    1. temp = bf16(fp32(x) * fp32(scale))      — mul with bf16 rounding
    2. result = bf16(fp32(residual) + fp32(temp)) — add with bf16 rounding
  This kernel reproduces the SAME two-rounding behavior in one pass:
    load fp32(x), fp32(r) → multiply → round to bf16 → widen to fp32 → add → round to bf16 → store
  Same IEEE 754 RNE hardware instruction for both roundings → bit-for-bit identical.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_scaled_add_kernel(
    X_ptr,
    Residual_ptr,
    Out_ptr,
    scale,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    """result = residual + x * scale, matching PyTorch's two-rounding bf16 path."""
    row = tl.program_id(0)
    base = row.to(tl.int64) * hidden_size
    off = tl.arange(0, BLOCK_SIZE)
    mask = off < hidden_size

    x = tl.load(X_ptr + base + off, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(Residual_ptr + base + off, mask=mask, other=0.0).to(tl.float32)

    # Step 1: x * scale → round to bf16 (matches PyTorch's mul kernel output)
    scaled = (x * tl.cast(scale, tl.float32)).to(tl.bfloat16).to(tl.float32)

    # Step 2: residual + scaled → round to bf16 (matches PyTorch's add kernel output)
    result = (r + scaled).to(tl.bfloat16)

    tl.store(Out_ptr + base + off, result, mask=mask)


def _next_pow2(n: int) -> int:
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def fused_scaled_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Fused: result = residual + x * scale.

    Replaces two PyTorch kernel launches (mul + add) with one Triton kernel.
    Bit-for-bit identical to the unfused path (same two-rounding behavior).

    Args:
        x:        [num_tokens, hidden_size]  sublayer output (attn or mlp)
        residual: [num_tokens, hidden_size]  running residual
        scale:    muP scale factor

    Returns:
        New tensor = residual + x * scale
    """
    num_tokens, hidden_size = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = _next_pow2(hidden_size)

    _fused_scaled_add_kernel[(num_tokens,)](
        x, residual, out,
        scale, hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=max(4, min(16, BLOCK_SIZE // 256)),
    )

    return out
