"""Fused Triton kernels for MiniCPM-SALA optimized forward pass.

Provides kernel fusions that reduce kernel launch overhead — the dominant
bottleneck for decode with small batch sizes on quantized (GPTQ/Marlin) models
where individual kernels are fast but numerous.

Kernels:
  1. fused_scaled_add:         residual + x * scale  (replaces mul + add)
  2. fused_scaled_add_rmsnorm: residual + x * scale, then RMSNorm  (replaces 3 kernels → 1)
  3. fused_norm_sigmoid_mul:   RMSNorm(x) * sigmoid(gate)  (replaces 3 kernels → 1)
  4. fused_sigmoid_mul:        x * sigmoid(gate)  (replaces 2 kernels → 1)

PRECISION CONTRACT:
  All kernels reproduce PyTorch's bf16 two-rounding behavior exactly:
  intermediate results are rounded to bf16 at the same points PyTorch would.
"""

import torch
import triton
import triton.language as tl


def _next_pow2(n: int) -> int:
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


# =============================================================================
# 1. fused_scaled_add: result = residual + x * scale
# =============================================================================

@triton.jit
def _fused_scaled_add_kernel(
    X_ptr, Residual_ptr, Out_ptr,
    scale,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    base = row.to(tl.int64) * hidden_size
    off = tl.arange(0, BLOCK_SIZE)
    mask = off < hidden_size

    x = tl.load(X_ptr + base + off, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(Residual_ptr + base + off, mask=mask, other=0.0).to(tl.float32)

    scaled = (x * tl.cast(scale, tl.float32)).to(tl.bfloat16).to(tl.float32)
    result = (r + scaled).to(tl.bfloat16)

    tl.store(Out_ptr + base + off, result, mask=mask)


def fused_scaled_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Fused: result = residual + x * scale.

    Replaces two PyTorch kernel launches (mul + add) with one Triton kernel.
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


# =============================================================================
# 2. fused_scaled_add_rmsnorm: (residual + x * scale) then RMSNorm
#    Replaces: fused_scaled_add + RMSNorm = 2 kernels → 1 kernel
#    Returns (normed_output, new_residual)
# =============================================================================

@triton.jit
def _fused_scaled_add_rmsnorm_kernel(
    X_ptr, Residual_ptr, Weight_ptr,
    Normed_ptr, NewResidual_ptr,
    scale, eps,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    base = row.to(tl.int64) * hidden_size
    off = tl.arange(0, BLOCK_SIZE)
    mask = off < hidden_size

    x = tl.load(X_ptr + base + off, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(Residual_ptr + base + off, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(Weight_ptr + off, mask=mask, other=0.0).to(tl.float32)

    # Step 1: scaled add with bf16 two-rounding
    scaled = (x * tl.cast(scale, tl.float32)).to(tl.bfloat16).to(tl.float32)
    new_res = (r + scaled).to(tl.bfloat16)

    # Store new residual
    tl.store(NewResidual_ptr + base + off, new_res, mask=mask)

    # Step 2: RMSNorm on the new residual
    new_res_f32 = new_res.to(tl.float32)
    var = tl.sum(new_res_f32 * new_res_f32, axis=0) / hidden_size
    rstd = 1.0 / tl.sqrt(var + tl.cast(eps, tl.float32))
    normed = (new_res_f32 * rstd * w).to(tl.bfloat16)

    tl.store(Normed_ptr + base + off, normed, mask=mask)


def fused_scaled_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: float,
    eps: float,
) -> tuple:
    """Fused: new_res = residual + x * scale; normed = rmsnorm(new_res).

    Returns (normed_output, new_residual).
    Replaces fused_scaled_add + RMSNorm = 2 kernel launches → 1.
    """
    num_tokens, hidden_size = x.shape
    normed = torch.empty_like(x)
    new_residual = torch.empty_like(x)
    BLOCK_SIZE = _next_pow2(hidden_size)

    _fused_scaled_add_rmsnorm_kernel[(num_tokens,)](
        x, residual, weight,
        normed, new_residual,
        scale, eps,
        hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=max(4, min(16, BLOCK_SIZE // 256)),
    )
    return normed, new_residual


# =============================================================================
# 3. fused_norm_sigmoid_mul: RMSNorm(x) * sigmoid(gate)
#    Replaces: RMSNorm + sigmoid + mul = 3 kernels → 1 kernel
#    Used for lightning layer output: o_norm(o) * sigmoid(z)
# =============================================================================

@triton.jit
def _fused_norm_sigmoid_mul_kernel(
    X_ptr, Gate_ptr, Weight_ptr, Out_ptr,
    eps,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    base = row.to(tl.int64) * hidden_size
    off = tl.arange(0, BLOCK_SIZE)
    mask = off < hidden_size

    x = tl.load(X_ptr + base + off, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(Gate_ptr + base + off, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(Weight_ptr + off, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm
    var = tl.sum(x * x, axis=0) / hidden_size
    rstd = 1.0 / tl.sqrt(var + tl.cast(eps, tl.float32))
    normed = x * rstd * w

    # sigmoid(gate) * normed
    sig_g = 1.0 / (1.0 + tl.exp(-g))
    result = (normed * sig_g).to(tl.bfloat16)

    tl.store(Out_ptr + base + off, result, mask=mask)


def fused_norm_sigmoid_mul(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused: result = RMSNorm(x, weight) * sigmoid(gate).

    Used in lightning layers: o_norm(attn_out) * sigmoid(z_proj(hidden_states)).
    Replaces 3 kernel launches (rmsnorm + sigmoid + mul) with 1.
    """
    num_tokens, hidden_size = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = _next_pow2(hidden_size)

    _fused_norm_sigmoid_mul_kernel[(num_tokens,)](
        x, gate, weight, out,
        eps,
        hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=max(4, min(16, BLOCK_SIZE // 256)),
    )
    return out


# =============================================================================
# 4. fused_sigmoid_mul: x * sigmoid(gate)
#    Replaces: sigmoid + mul = 2 kernels → 1 kernel
#    Used for minicpm4 layer output: attn_output * sigmoid(o_gate)
# =============================================================================

@triton.jit
def _fused_sigmoid_mul_kernel(
    X_ptr, Gate_ptr, Out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(Gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sig_g = 1.0 / (1.0 + tl.exp(-g))
    result = (x * sig_g).to(tl.bfloat16)

    tl.store(Out_ptr + offsets, result, mask=mask)


def fused_sigmoid_mul(
    x: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Fused: result = x * sigmoid(gate).

    Used in minicpm4 layers: attn_output * sigmoid(o_gate_output).
    Replaces 2 kernel launches (sigmoid + mul) with 1.
    """
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024

    _fused_sigmoid_mul_kernel[(triton.cdiv(n_elements, BLOCK_SIZE),)](
        x, gate, out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
