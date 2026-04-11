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


# Cached scratch buffers (keyed by size)
_bf16_compact_scratch = {}

def _get_compact_scratch(n_tokens, H, D, device):
    key = (n_tokens, H, D, str(device))
    if key not in _bf16_compact_scratch or _bf16_compact_scratch[key].shape[0] < n_tokens:
        _bf16_compact_scratch[key] = torch.empty(n_tokens, H, D, dtype=torch.bfloat16, device=device)
    return _bf16_compact_scratch[key][:n_tokens]


def _dequant_compact(packed_compact, scale_compact):
    """Run triton kernel on compact [N, H, half_D] -> [N, H, D] bf16."""
    total_packed = packed_compact.numel()
    if total_packed == 0:
        return torch.empty(0, dtype=torch.bfloat16, device=packed_compact.device)

    N, H, half_D = packed_compact.shape
    D = half_D * 2
    out_bf16 = _get_compact_scratch(N, H, D, packed_compact.device)

    BLOCK_SIZE = 1024
    grid = ((total_packed + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _fp4_to_bf16_kernel[grid](
        packed_compact.reshape(-1),
        scale_compact.reshape(-1),
        out_bf16.reshape(-1),
        total_packed,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out_bf16


def fp4_to_fp8_indexed(packed, scale, out, indices):
    """Dequant only tokens at `indices`: gather -> dequant -> fp8 cast -> scatter.

    Args:
        packed: [pool_size, H, half_D] uint8
        scale:  [pool_size, S] uint8
        out:    [pool_size, H, D] uint8 (scratch, FP8)
        indices: [N] int64 active token indices
    """
    N = indices.numel()
    if N == 0:
        return

    # Gather active rows (compact, contiguous)
    packed_compact = packed[indices]   # [N, H, half_D]
    scale_compact = scale[indices]     # [N, S]

    # Dequant compact block: FP4 -> BF16
    bf16_compact = _dequant_compact(packed_compact, scale_compact)

    # Cast BF16 -> FP8 and scatter back
    fp8_compact = bf16_compact.to(torch.float8_e4m3fn).view(torch.uint8)
    out[indices] = fp8_compact


def fp4_to_fp8_triton(packed, scale, out):
    """Full-buffer dequant FP4 -> FP8."""
    total_packed = packed.numel()
    if total_packed == 0:
        return

    T, H, half_D = packed.shape
    D = half_D * 2
    bf16_buf = _get_compact_scratch(T, H, D, packed.device)

    BLOCK_SIZE = 1024
    grid = ((total_packed + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _fp4_to_bf16_kernel[grid](
        packed.reshape(-1),
        scale.reshape(-1),
        bf16_buf.reshape(-1),
        total_packed,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    out.reshape(-1).copy_(
        bf16_buf.reshape(-1).to(torch.float8_e4m3fn).view(torch.uint8)
    )
