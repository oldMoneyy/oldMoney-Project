"""Fused FP4->FP8 dequantization with selective token support."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fp4_to_bf16_kernel(
    packed_ptr,
    scale_ptr,
    out_ptr,
    total_packed_bytes,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_packed_bytes

    p = tl.load(packed_ptr + offs, mask=mask, other=0)

    scale_idx = offs // 8
    sc = tl.load(scale_ptr + scale_idx, mask=mask, other=127).to(tl.float32)
    sf = tl.exp2(sc - 127.0)

    low = (p & 0x0F).to(tl.int32)
    high = ((p >> 4) & 0x0F).to(tl.int32)

    ls = (low >> 3) & 1
    lm = low & 7
    hs = (high >> 3) & 1
    hm = high & 7

    lv = (
        (lm == 1).to(tl.float32) * 0.5 +
        (lm == 2).to(tl.float32) * 1.0 +
        (lm == 3).to(tl.float32) * 1.5 +
        (lm == 4).to(tl.float32) * 2.0 +
        (lm == 5).to(tl.float32) * 3.0 +
        (lm == 6).to(tl.float32) * 4.0 +
        (lm == 7).to(tl.float32) * 6.0
    )
    hv = (
        (hm == 1).to(tl.float32) * 0.5 +
        (hm == 2).to(tl.float32) * 1.0 +
        (hm == 3).to(tl.float32) * 1.5 +
        (hm == 4).to(tl.float32) * 2.0 +
        (hm == 5).to(tl.float32) * 3.0 +
        (hm == 6).to(tl.float32) * 4.0 +
        (hm == 7).to(tl.float32) * 6.0
    )

    lv = tl.where(ls > 0, -lv, lv) * sf
    hv = tl.where(hs > 0, -hv, hv) * sf

    tl.store(out_ptr + offs * 2, lv.to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + offs * 2 + 1, hv.to(tl.bfloat16), mask=mask)


@triton.jit
def _fp4_to_bf16_indexed_kernel(
    packed_ptr,     # [pool_size, H, half_D] uint8 flat
    scale_ptr,      # [pool_size, S] uint8 flat
    out_ptr,        # [pool_size, H, D] bf16 flat (scratch)
    indices_ptr,    # [N] int64 token indices
    N,              # number of active tokens
    row_packed_bytes,  # H * half_D (packed bytes per token)
    row_scale_bytes,   # S (scale bytes per token)
    row_out_elems,     # H * D (bf16 elements per token)
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,   # packed bytes per row chunk
):
    """Dequant only tokens at indices[0..N-1] from FP4 -> BF16, scatter into scratch."""
    pid_n = tl.program_id(0)  # which token
    pid_r = tl.program_id(1)  # which chunk within the token row

    n_idx = pid_n
    if n_idx >= N:
        return

    token_loc = tl.load(indices_ptr + n_idx)

    # Byte offsets within this token's row
    r_offs = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    r_mask = r_offs < row_packed_bytes

    # Load packed bytes for this token
    packed_offset = token_loc * row_packed_bytes + r_offs
    p = tl.load(packed_ptr + packed_offset, mask=r_mask, other=0)

    # Load scale (each 8 packed bytes share one scale)
    # Global scale index = token_loc * row_scale_bytes + r_offs // 8
    scale_offset = token_loc * row_scale_bytes + r_offs // 8
    sc = tl.load(scale_ptr + scale_offset, mask=r_mask, other=127).to(tl.float32)
    sf = tl.exp2(sc - 127.0)

    low = (p & 0x0F).to(tl.int32)
    high = ((p >> 4) & 0x0F).to(tl.int32)
    ls = (low >> 3) & 1
    lm = low & 7
    hs = (high >> 3) & 1
    hm = high & 7

    lv = (
        (lm == 1).to(tl.float32) * 0.5 +
        (lm == 2).to(tl.float32) * 1.0 +
        (lm == 3).to(tl.float32) * 1.5 +
        (lm == 4).to(tl.float32) * 2.0 +
        (lm == 5).to(tl.float32) * 3.0 +
        (lm == 6).to(tl.float32) * 4.0 +
        (lm == 7).to(tl.float32) * 6.0
    )
    hv = (
        (hm == 1).to(tl.float32) * 0.5 +
        (hm == 2).to(tl.float32) * 1.0 +
        (hm == 3).to(tl.float32) * 1.5 +
        (hm == 4).to(tl.float32) * 2.0 +
        (hm == 5).to(tl.float32) * 3.0 +
        (hm == 6).to(tl.float32) * 4.0 +
        (hm == 7).to(tl.float32) * 6.0
    )

    lv = tl.where(ls > 0, -lv, lv) * sf
    hv = tl.where(hs > 0, -hv, hv) * sf

    # Write to scratch at same token position (scatter)
    out_base = token_loc * row_out_elems + pid_r * BLOCK_R * 2
    even_offs = tl.arange(0, BLOCK_R) * 2
    odd_offs = even_offs + 1
    tl.store(out_ptr + out_base + even_offs, lv.to(tl.bfloat16), mask=r_mask)
    tl.store(out_ptr + out_base + odd_offs, hv.to(tl.bfloat16), mask=r_mask)


_bf16_scratch = {}

def _get_bf16_scratch(shape, device):
    key = (shape, device)
    if key not in _bf16_scratch:
        _bf16_scratch[key] = torch.empty(shape, dtype=torch.bfloat16, device=device)
    return _bf16_scratch[key]


def fp4_to_fp8_indexed(packed, scale, out, indices):
    """Dequant only tokens at `indices` from FP4 -> BF16 -> FP8 into scratch.

    Args:
        packed: [pool_size, H, half_D] uint8
        scale:  [pool_size, S] uint8
        out:    [pool_size, H, D] uint8 (scratch, FP8)
        indices: [N] int64 active token indices
    """
    N = indices.numel()
    if N == 0:
        return

    pool_size, H, half_D = packed.shape
    D = half_D * 2
    S = scale.shape[1]
    row_packed_bytes = H * half_D
    row_scale_bytes = S
    row_out_elems = H * D

    # BF16 scratch (same shape as out)
    bf16_buf = _get_bf16_scratch((pool_size, H, D), packed.device)

    BLOCK_R = 128  # packed bytes per chunk
    grid_r = (row_packed_bytes + BLOCK_R - 1) // BLOCK_R

    _fp4_to_bf16_indexed_kernel[(N, grid_r)](
        packed.reshape(-1),
        scale.reshape(-1),
        bf16_buf.reshape(-1),
        indices,
        N,
        row_packed_bytes,
        row_scale_bytes,
        row_out_elems,
        BLOCK_N=1,
        BLOCK_R=BLOCK_R,
    )

    # Cast only the indexed rows from bf16 -> fp8
    out[indices] = bf16_buf[indices].to(torch.float8_e4m3fn).view(torch.uint8)


def fp4_to_fp8_triton(packed, scale, out):
    """Full-buffer dequant FP4 -> FP8 (for prefill or when indices unknown)."""
    total_packed = packed.numel()
    if total_packed == 0:
        return

    total_out = total_packed * 2
    bf16_buf = _get_bf16_scratch(total_out, packed.device)

    BLOCK_SIZE = 1024
    grid = ((total_packed + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _fp4_to_bf16_kernel[grid](
        packed.reshape(-1),
        scale.reshape(-1),
        bf16_buf,
        total_packed,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    out.reshape(-1).copy_(
        bf16_buf[:total_out].to(torch.float8_e4m3fn).view(torch.uint8)
    )
