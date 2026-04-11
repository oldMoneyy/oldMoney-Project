"""Fused FP4->FP8 with fixed-size scratch buffer, chunked processing."""

import torch
import triton
import triton.language as tl

# Fixed scratch: 256K tokens max, ~32MB for H=2,D=128
_SCRATCH_MAX_TOKENS = 262144
_bf16_scratch = None
_bf16_scratch_device = None


def _get_fixed_scratch(H, D, device):
    global _bf16_scratch, _bf16_scratch_device
    if _bf16_scratch is None or _bf16_scratch_device != str(device):
        _bf16_scratch = torch.empty(_SCRATCH_MAX_TOKENS, H, D, dtype=torch.bfloat16, device=device)
        _bf16_scratch_device = str(device)
    return _bf16_scratch


@triton.jit
def _fp4_to_bf16_kernel(
    packed_ptr, scale_ptr, out_ptr,
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
    ls = (low >> 3) & 1;  lm = low & 7
    hs = (high >> 3) & 1; hm = high & 7

    lv = ((lm==1).to(tl.float32)*0.5 + (lm==2).to(tl.float32)*1.0 +
          (lm==3).to(tl.float32)*1.5 + (lm==4).to(tl.float32)*2.0 +
          (lm==5).to(tl.float32)*3.0 + (lm==6).to(tl.float32)*4.0 +
          (lm==7).to(tl.float32)*6.0)
    hv = ((hm==1).to(tl.float32)*0.5 + (hm==2).to(tl.float32)*1.0 +
          (hm==3).to(tl.float32)*1.5 + (hm==4).to(tl.float32)*2.0 +
          (hm==5).to(tl.float32)*3.0 + (hm==6).to(tl.float32)*4.0 +
          (hm==7).to(tl.float32)*6.0)

    lv = tl.where(ls > 0, -lv, lv) * sf
    hv = tl.where(hs > 0, -hv, hv) * sf

    tl.store(out_ptr + offs * 2, lv.to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + offs * 2 + 1, hv.to(tl.bfloat16), mask=mask)


def _run_kernel_chunk(packed_flat, scale_flat, bf16_flat, n_packed):
    BLOCK_SIZE = 1024
    grid = ((n_packed + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _fp4_to_bf16_kernel[grid](packed_flat, scale_flat, bf16_flat, n_packed, BLOCK_SIZE=BLOCK_SIZE)


def fp4_to_fp8_triton(packed, scale, out):
    """Full-buffer FP4->FP8 using fixed scratch, chunked to avoid OOM."""
    T, H, half_D = packed.shape
    D = half_D * 2
    if T == 0:
        return

    scratch = _get_fixed_scratch(H, D, packed.device)
    chunk_tokens = _SCRATCH_MAX_TOKENS
    packed_per_token = H * half_D
    scale_per_token = scale.shape[1]

    for start in range(0, T, chunk_tokens):
        end = min(start + chunk_tokens, T)
        ct = end - start

        p_chunk = packed[start:end].reshape(-1)
        s_chunk = scale[start:end].reshape(-1)
        bf16_chunk = scratch[:ct].reshape(-1)

        n_packed = ct * packed_per_token
        # Chunk the triton kernel for int32 safety
        MAX_PACKED = 500_000_000
        for ks in range(0, n_packed, MAX_PACKED):
            ke = min(ks + MAX_PACKED, n_packed)
            klen = ke - ks
            _run_kernel_chunk(
                p_chunk[ks:ke],
                s_chunk[ks // 8 : ke // 8],
                bf16_chunk[ks * 2 : ke * 2],
                klen,
            )

        out[start:end] = scratch[:ct].to(torch.float8_e4m3fn).view(torch.uint8)


def fp4_to_fp8_indexed(packed, scale, out, indices):
    """Dequant only tokens at indices."""
    N = indices.numel()
    if N == 0:
        return

    H = packed.shape[1]
    half_D = packed.shape[2]
    D = half_D * 2

    scratch = _get_fixed_scratch(H, D, packed.device)
    chunk_tokens = _SCRATCH_MAX_TOKENS

    for start in range(0, N, chunk_tokens):
        end = min(start + chunk_tokens, N)
        ct = end - start
        idx_chunk = indices[start:end]

        p_compact = packed[idx_chunk].reshape(-1)
        s_compact = scale[idx_chunk].reshape(-1)
        bf16_chunk = scratch[:ct].reshape(-1)

        n_packed = p_compact.numel()
        MAX_PACKED = 500_000_000
        for ks in range(0, n_packed, MAX_PACKED):
            ke = min(ks + MAX_PACKED, n_packed)
            klen = ke - ks
            _run_kernel_chunk(
                p_compact[ks:ke],
                s_compact[ks // 8 : ke // 8],
                bf16_chunk[ks * 2 : ke * 2],
                klen,
            )

        out[idx_chunk] = scratch[:ct].to(torch.float8_e4m3fn).view(torch.uint8)
