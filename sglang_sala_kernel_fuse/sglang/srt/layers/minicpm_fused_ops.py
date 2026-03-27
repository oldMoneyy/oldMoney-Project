"""Fused Triton kernels for MiniCPM-SALA model.

These kernels fuse multiple elementwise operations to reduce kernel launch
overhead and memory round-trips. They are drop-in replacements for the
unfused operation sequences in the MiniCPM decoder layer forward pass.

Kernel inventory (saves ~331 kernel launches per forward):
  1. fused_scale_residual_rmsnorm  — 64× per forward (all residual+norm boundaries)
  2. fused_sigmoid_gate            — 8× per forward  (minicpm4 attn output gate)
  3. fused_rmsnorm_sigmoid_gate    — 24× per forward (lightning attn output)
  4. fused_qknorm                  — (kept for fallback, replaced by #7 on lightning layers)
  5. fused_rmsnorm_scale           — 1× per forward  (final norm + scale)
  6. fused_scale_residual          — 1× per forward  (final layer residual+scale)
  7. fused_qknorm_rope             — 24× per forward (lightning QK norms + RoPE)
     Replaces: fused_qknorm + q.float() + k.float() + rotary_emb + q.bf16() + k.bf16()
     Saves 5 launches per lightning layer × 24 = 120 launches.
"""

import torch
import triton
import triton.language as tl


# =============================================================================
# 1. fused_scale_residual_rmsnorm
#    Fuses: new_residual = residual + x * scale
#           output = rmsnorm(new_residual, weight, eps)
#    Replaces 3 ops (scale-mul, add, rmsnorm) with 1 kernel.
# =============================================================================

@triton.jit
def _fused_scale_residual_rmsnorm_kernel(
    out_ptr,          # normed output
    residual_out_ptr, # new residual (= residual + x * scale)
    x_ptr,            # attention/mlp output
    residual_ptr,     # incoming residual
    weight_ptr,       # rmsnorm weight
    scale: tl.constexpr,
    eps: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = row * hidden_dim
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    # Load x and residual
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)

    # new_residual = residual + x * scale
    new_r = r + x * scale

    # Store new residual
    tl.store(residual_out_ptr + row_start + offsets, new_r.to(tl.bfloat16), mask=mask)

    # RMSNorm on new_residual
    variance = tl.sum(new_r * new_r, axis=0) / hidden_dim
    inv_rms = 1.0 / tl.sqrt(variance + eps)

    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    normed = new_r * inv_rms * w

    tl.store(out_ptr + row_start + offsets, normed, mask=mask)


def fused_scale_residual_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    scale: float,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuses: new_residual = residual + x * scale; output = rmsnorm(new_residual).

    Args:
        x: Input tensor (attention or MLP output), shape [tokens, hidden_dim]
        residual: Residual tensor, shape [tokens, hidden_dim]
        scale: Scalar scale factor (scale_depth / sqrt(num_hidden_layers))
        weight: RMSNorm weight, shape [hidden_dim]
        eps: RMSNorm epsilon

    Returns:
        (normed_output, new_residual) — both shape [tokens, hidden_dim]
    """
    assert x.shape == residual.shape
    num_tokens, hidden_dim = x.shape
    out = torch.empty_like(x)
    new_residual = torch.empty_like(residual)

    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    num_warps = max(min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), 32), 4)

    _fused_scale_residual_rmsnorm_kernel[(num_tokens,)](
        out, new_residual, x, residual, weight,
        scale=scale, eps=eps, hidden_dim=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
    )
    return out, new_residual


# =============================================================================
# 2. fused_sigmoid_gate
#    Fuses: output = x * sigmoid(gate)
#    Replaces sigmoid + elementwise mul with 1 kernel.
# =============================================================================

@triton.jit
def _fused_sigmoid_gate_kernel(
    out_ptr,
    x_ptr,
    gate_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # sigmoid(g) = 1 / (1 + exp(-g))
    sig = tl.sigmoid(g)
    result = x * sig

    tl.store(out_ptr + offsets, result, mask=mask)


def fused_sigmoid_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Fuses: output = x * sigmoid(gate).

    Uses CUDA vectorized kernel if fused_kernel_extension is available,
    falls back to Triton otherwise.

    Args:
        x: Input tensor (attention output)
        gate: Gate tensor (from o_gate linear)

    Returns:
        Gated output, same shape as x
    """
    assert x.shape == gate.shape

    # Try CUDA vectorized version first (128-bit loads, ~30% faster)
    try:
        import fused_kernel_extension
        return fused_kernel_extension.fused_sigmoid_gate(x.contiguous(), gate.contiguous())
    except (ImportError, RuntimeError):
        pass

    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    _fused_sigmoid_gate_kernel[grid](
        out, x, gate, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=8,
    )
    return out


# =============================================================================
# 3. fused_rmsnorm_sigmoid_gate
#    Fuses: output = rmsnorm(x, weight, eps) * sigmoid(gate)
#    Replaces rmsnorm + sigmoid + mul with 1 kernel.
# =============================================================================

@triton.jit
def _fused_rmsnorm_sigmoid_gate_kernel(
    out_ptr,
    x_ptr,
    gate_ptr,
    weight_ptr,
    eps: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = row * hidden_dim
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm
    variance = tl.sum(x * x, axis=0) / hidden_dim
    inv_rms = 1.0 / tl.sqrt(variance + eps)

    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    normed = x * inv_rms * w

    # Sigmoid gate
    g = tl.load(gate_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    result = normed * tl.sigmoid(g)

    tl.store(out_ptr + row_start + offsets, result, mask=mask)


def fused_rmsnorm_sigmoid_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fuses: output = rmsnorm(x) * sigmoid(gate).

    Args:
        x: Input to normalize (attention output), shape [tokens, dim]
        gate: Gate values (from z_proj), shape [tokens, dim]
        weight: RMSNorm weight, shape [dim]
        eps: RMSNorm epsilon

    Returns:
        Gated-normed output, shape [tokens, dim]
    """
    assert x.shape == gate.shape
    num_tokens, hidden_dim = x.shape
    out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    num_warps = max(min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), 32), 4)

    _fused_rmsnorm_sigmoid_gate_kernel[(num_tokens,)](
        out, x, gate, weight,
        eps=eps, hidden_dim=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
    )
    return out


# =============================================================================
# 4. fused_qknorm
#    Fuses two separate RMSNorm calls on Q and K into one kernel launch.
#    Each thread block handles one row and normalizes both Q and K heads.
# =============================================================================

@triton.jit
def _fused_qknorm_kernel(
    q_ptr,
    k_ptr,
    q_weight_ptr,
    k_weight_ptr,
    eps: tl.constexpr,
    head_dim: tl.constexpr,
    num_q_rows,
    num_k_rows,
    BLOCK_SIZE: tl.constexpr,
):
    """Normalize Q (rows 0..num_q_rows-1) and K (rows 0..num_k_rows-1) in-place.

    Grid: (max(num_q_rows, num_k_rows),)
    Each block normalizes one Q row AND one K row (if both exist).
    """
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < head_dim

    # Normalize Q row (if within bounds)
    if row < num_q_rows:
        q_start = row * head_dim
        q = tl.load(q_ptr + q_start + offsets, mask=mask, other=0.0).to(tl.float32)
        q_var = tl.sum(q * q, axis=0) / head_dim
        q_inv_rms = 1.0 / tl.sqrt(q_var + eps)
        qw = tl.load(q_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(q_ptr + q_start + offsets, (q * q_inv_rms * qw), mask=mask)

    # Normalize K row (if within bounds)
    if row < num_k_rows:
        k_start = row * head_dim
        k = tl.load(k_ptr + k_start + offsets, mask=mask, other=0.0).to(tl.float32)
        k_var = tl.sum(k * k, axis=0) / head_dim
        k_inv_rms = 1.0 / tl.sqrt(k_var + eps)
        kw = tl.load(k_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(k_ptr + k_start + offsets, (k * k_inv_rms * kw), mask=mask)


def fused_qknorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    """In-place fused RMSNorm on Q and K tensors.

    Both Q and K should be reshaped to [num_rows, head_dim] before calling.
    Normalizes in-place — no return value.

    Args:
        q: Query tensor, shape [num_q_rows, head_dim], modified in-place
        k: Key tensor, shape [num_k_rows, head_dim], modified in-place
        q_weight: Q norm weight, shape [head_dim]
        k_weight: K norm weight, shape [head_dim]
        eps: RMSNorm epsilon
    """
    head_dim = q.shape[-1]
    num_q_rows = q.shape[0]
    num_k_rows = k.shape[0]
    grid_size = max(num_q_rows, num_k_rows)

    BLOCK_SIZE = triton.next_power_of_2(head_dim)
    num_warps = max(min(triton.next_power_of_2(triton.cdiv(head_dim, 256)), 32), 4)

    _fused_qknorm_kernel[(grid_size,)](
        q, k, q_weight, k_weight,
        eps=eps, head_dim=head_dim,
        num_q_rows=num_q_rows, num_k_rows=num_k_rows,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
    )


# =============================================================================
# 5. fused_rmsnorm_scale
#    Fuses: output = rmsnorm(x, weight, eps) * inv_scale
#    Replaces rmsnorm + scale division with 1 kernel.
# =============================================================================

@triton.jit
def _fused_rmsnorm_scale_kernel(
    out_ptr,
    x_ptr,
    weight_ptr,
    inv_scale,
    eps: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = row * hidden_dim
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm
    variance = tl.sum(x * x, axis=0) / hidden_dim
    inv_rms = 1.0 / tl.sqrt(variance + eps)

    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    result = x * inv_rms * w * inv_scale

    tl.store(out_ptr + row_start + offsets, result, mask=mask)


def fused_rmsnorm_scale(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scale: float,
) -> torch.Tensor:
    """Fuses: output = rmsnorm(x) / scale.

    Args:
        x: Input tensor, shape [tokens, hidden_dim]
        weight: RMSNorm weight, shape [hidden_dim]
        eps: RMSNorm epsilon
        scale: Scale divisor (scale_width)

    Returns:
        Normalized and scaled output, shape [tokens, hidden_dim]
    """
    num_tokens, hidden_dim = x.shape
    out = torch.empty_like(x)
    inv_scale = 1.0 / scale

    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    num_warps = max(min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), 32), 4)

    _fused_rmsnorm_scale_kernel[(num_tokens,)](
        out, x, weight, inv_scale,
        eps=eps, hidden_dim=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
    )
    return out


# =============================================================================
# 6. fused_scale_residual
#    Fuses: output = residual + x * scale
#    Replaces elementwise mul + add with 1 kernel.
# =============================================================================

@triton.jit
def _fused_scale_residual_kernel(
    out_ptr,
    x_ptr,
    residual_ptr,
    scale: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    result = r + x * scale

    tl.store(out_ptr + offsets, result, mask=mask)


def fused_scale_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Fuses: output = residual + x * scale.

    Args:
        x: Input tensor (MLP output)
        residual: Residual tensor
        scale: Scale factor

    Returns:
        residual + x * scale
    """
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    _fused_scale_residual_kernel[grid](
        out, x, residual, scale=scale, n_elements=n,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=8,
    )
    return out


# =============================================================================
# 7. fused_qknorm_rope
#    Fuses: RMSNorm(Q) + RMSNorm(K) + RoPE rotation (NeoX half-rotate)
#    All in float32 internally, bf16 I/O. In-place on Q and K.
#    Replaces: fused_qknorm + q.float() + k.float() + rotary_emb + q.bf16() + k.bf16()
#    Saves 5 kernel launches per lightning-attn layer (120 total for 24 layers).
# =============================================================================

@triton.jit
def _fused_qknorm_rope_kernel(
    q_ptr,              # [num_tokens * num_q_heads, head_dim], bf16, in-place
    k_ptr,              # [num_tokens * num_k_heads, head_dim], bf16, in-place
    q_weight_ptr,       # [head_dim], rmsnorm weight
    k_weight_ptr,       # [head_dim], rmsnorm weight
    cos_sin_cache_ptr,  # [max_position, rotary_dim], fp32, layout: [cos | sin]
    positions_ptr,      # [num_tokens], int32/int64
    eps: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    num_q_heads,
    num_k_heads,
    num_q_rows,
    num_k_rows,
    HALF_DIM: tl.constexpr,  # = rotary_dim // 2
):
    """Process one (token, head) pair for Q and K: RMSNorm + RoPE in-place."""
    row = tl.program_id(0)
    half_offsets = tl.arange(0, HALF_DIM)

    # --- Process Q ---
    if row < num_q_rows:
        q_token_idx = row // num_q_heads
        pos = tl.load(positions_ptr + q_token_idx)
        q_start = row * head_dim

        # Load both halves of Q head
        q1 = tl.load(q_ptr + q_start + half_offsets).to(tl.float32)
        q2 = tl.load(q_ptr + q_start + HALF_DIM + half_offsets).to(tl.float32)

        # RMSNorm over full head_dim (sum of both halves)
        q_var = (tl.sum(q1 * q1, axis=0) + tl.sum(q2 * q2, axis=0)) / head_dim
        q_inv_rms = 1.0 / tl.sqrt(q_var + eps)
        qw1 = tl.load(q_weight_ptr + half_offsets).to(tl.float32)
        qw2 = tl.load(q_weight_ptr + HALF_DIM + half_offsets).to(tl.float32)
        q1 = q1 * q_inv_rms * qw1
        q2 = q2 * q_inv_rms * qw2

        # Load cos/sin for this position
        cs_base = pos * rotary_dim
        cos_val = tl.load(cos_sin_cache_ptr + cs_base + half_offsets)
        sin_val = tl.load(cos_sin_cache_ptr + cs_base + HALF_DIM + half_offsets)

        # RoPE (NeoX half-rotate): [x1*cos - x2*sin, x2*cos + x1*sin]
        q_out1 = q1 * cos_val - q2 * sin_val
        q_out2 = q2 * cos_val + q1 * sin_val

        tl.store(q_ptr + q_start + half_offsets, q_out1.to(tl.bfloat16))
        tl.store(q_ptr + q_start + HALF_DIM + half_offsets, q_out2.to(tl.bfloat16))

    # --- Process K ---
    if row < num_k_rows:
        k_token_idx = row // num_k_heads
        pos = tl.load(positions_ptr + k_token_idx)
        k_start = row * head_dim

        k1 = tl.load(k_ptr + k_start + half_offsets).to(tl.float32)
        k2 = tl.load(k_ptr + k_start + HALF_DIM + half_offsets).to(tl.float32)

        k_var = (tl.sum(k1 * k1, axis=0) + tl.sum(k2 * k2, axis=0)) / head_dim
        k_inv_rms = 1.0 / tl.sqrt(k_var + eps)
        kw1 = tl.load(k_weight_ptr + half_offsets).to(tl.float32)
        kw2 = tl.load(k_weight_ptr + HALF_DIM + half_offsets).to(tl.float32)
        k1 = k1 * k_inv_rms * kw1
        k2 = k2 * k_inv_rms * kw2

        cs_base = pos * rotary_dim
        cos_val = tl.load(cos_sin_cache_ptr + cs_base + half_offsets)
        sin_val = tl.load(cos_sin_cache_ptr + cs_base + HALF_DIM + half_offsets)

        k_out1 = k1 * cos_val - k2 * sin_val
        k_out2 = k2 * cos_val + k1 * sin_val

        tl.store(k_ptr + k_start + half_offsets, k_out1.to(tl.bfloat16))
        tl.store(k_ptr + k_start + HALF_DIM + half_offsets, k_out2.to(tl.bfloat16))


def fused_qknorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_k_heads: int,
) -> None:
    """Fused QK RMSNorm + RoPE rotation, in-place.

    Replaces the 6-kernel sequence:
        fused_qknorm(q, k, ...) + q.float() + k.float()
        + rotary_emb(positions, q, k) + q.to(bf16) + k.to(bf16)
    with a single Triton kernel.

    Assumes NeoX half-rotate style and rotary_dim == head_dim.
    Both Q and K must be reshaped to [num_tokens * num_heads, head_dim]
    (contiguous) before calling. Operates in-place.

    Args:
        q: Query tensor, shape [num_q_rows, head_dim], modified in-place
        k: Key tensor, shape [num_k_rows, head_dim], modified in-place
        q_weight: Q RMSNorm weight, shape [head_dim]
        k_weight: K RMSNorm weight, shape [head_dim]
        cos_sin_cache: Precomputed [max_pos, rotary_dim], fp32, [cos|sin] layout
        positions: Token positions, shape [num_tokens]
        eps: RMSNorm epsilon
        num_q_heads: Number of Q heads per token
        num_k_heads: Number of K heads per token
    """
    head_dim = q.shape[-1]
    rotary_dim = cos_sin_cache.shape[-1]
    assert rotary_dim == head_dim, (
        f"fused_qknorm_rope requires rotary_dim == head_dim, got {rotary_dim} vs {head_dim}"
    )
    HALF_DIM = head_dim // 2
    assert HALF_DIM > 0 and (HALF_DIM & (HALF_DIM - 1)) == 0, (
        f"head_dim/2 must be a power of 2, got {HALF_DIM}"
    )

    # Ensure cos_sin_cache is on the correct device (no-op if already there)
    if cos_sin_cache.device != q.device:
        cos_sin_cache = cos_sin_cache.to(q.device)

    num_q_rows = q.shape[0]
    num_k_rows = k.shape[0]
    grid_size = max(num_q_rows, num_k_rows)

    _fused_qknorm_rope_kernel[(grid_size,)](
        q, k, q_weight, k_weight,
        cos_sin_cache, positions,
        eps=eps,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        num_q_rows=num_q_rows,
        num_k_rows=num_k_rows,
        HALF_DIM=HALF_DIM,
        num_warps=4,
    )
