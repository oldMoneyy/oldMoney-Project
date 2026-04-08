import torch
import triton
import triton.language as tl

_FLASHMLA_CREATE_KV_BLOCK_SIZE = 4096
FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON = tl.constexpr(_FLASHMLA_CREATE_KV_BLOCK_SIZE)


@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        # index into req_to_token_ptr needs to be int64
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        mask = offset < kv_end - kv_start
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + offset,
            mask=mask,
        )
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)


@triton.jit
def build_sparse_kv_indices_kernel(
    # Inputs
    req_to_token_ptr,           # (max_reqs, max_seq_len) int32
    req_pool_indices_ptr,       # (bs,) int32
    seq_lens_ptr,               # (bs,) int32
    block_pool_ptr,             # (max_reqs, max_blocks_per_req) int32
    n_blocks_ptr,               # (max_reqs,) int32
    # Outputs
    kv_indices_out_ptr,         # (total_buf_size,) int32
    kv_indptr_ptr,              # (bs+1,) int32 — INPUT: cumulative offsets
    # Strides
    req_to_token_stride: tl.constexpr,
    block_pool_stride: tl.constexpr,
    # Constants
    DENSE_LEN: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    BLOCK_SIZE_SPARSE: tl.constexpr,  # sparse block size (64)
):
    """Build sparse kv_indices for FlashInfer decode.

    One program per request. For sparse requests: writes dense region [0, dense_len),
    then selected block tokens beyond dense_len, then window region.
    For dense requests: writes all tokens [0, seq_len).

    kv_indptr is an INPUT (pre-computed on CPU) that tells each program where
    to start writing in kv_indices_out.
    """
    COPY_BLOCK: tl.constexpr = 512
    pid = tl.program_id(0)

    req_pool_idx = tl.load(req_pool_indices_ptr + pid)
    sl = tl.load(seq_lens_ptr + pid).to(tl.int32)
    n_blocks = tl.load(n_blocks_ptr + req_pool_idx).to(tl.int32)
    out_offset = tl.load(kv_indptr_ptr + pid).to(tl.int64)

    req_base = req_pool_idx.to(tl.int64) * req_to_token_stride
    block_base = req_pool_idx.to(tl.int64) * block_pool_stride

    is_sparse = (n_blocks > 0) & (sl > DENSE_LEN)

    if is_sparse:
        # --- 1. Dense region [0, DENSE_LEN) ---
        for i in range(tl.cdiv(DENSE_LEN, COPY_BLOCK)):
            offsets = tl.arange(0, COPY_BLOCK).to(tl.int64) + i * COPY_BLOCK
            mask = offsets < DENSE_LEN
            phys = tl.load(req_to_token_ptr + req_base + offsets, mask=mask, other=0)
            tl.store(kv_indices_out_ptr + out_offset + offsets, phys, mask=mask)
        out_offset += DENSE_LEN

        # --- 2. Sparse blocks beyond dense_len ---
        for bi in range(n_blocks):
            block_idx = tl.load(block_pool_ptr + block_base + bi).to(tl.int32)
            block_start = block_idx * BLOCK_SIZE_SPARSE
            block_end = tl.minimum(block_start + BLOCK_SIZE_SPARSE, sl)

            if block_start >= DENSE_LEN and block_start < sl:
                n_tokens = block_end - block_start
                # BLOCK_SIZE_SPARSE (64) fits in one COPY_BLOCK (512), single pass
                offsets = tl.arange(0, BLOCK_SIZE_SPARSE).to(tl.int64)
                mask = offsets < n_tokens
                phys = tl.load(
                    req_to_token_ptr + req_base + block_start.to(tl.int64) + offsets,
                    mask=mask, other=0,
                )
                tl.store(
                    kv_indices_out_ptr + out_offset + offsets,
                    phys, mask=mask,
                )
                out_offset += n_tokens.to(tl.int64)

        # --- 3. Window region [max(DENSE_LEN, sl - WINDOW_SIZE), sl) ---
        window_start = tl.maximum(DENSE_LEN, sl - WINDOW_SIZE)
        window_len = sl - window_start
        if window_len > 0:
            for i in range(tl.cdiv(window_len, COPY_BLOCK)):
                offsets = tl.arange(0, COPY_BLOCK).to(tl.int64) + i * COPY_BLOCK
                mask = offsets < window_len
                phys = tl.load(
                    req_to_token_ptr + req_base + window_start.to(tl.int64) + offsets,
                    mask=mask, other=0,
                )
                tl.store(
                    kv_indices_out_ptr + out_offset + offsets,
                    phys, mask=mask,
                )
    else:
        # Dense fallback: copy all [0, sl)
        for i in range(tl.cdiv(sl, COPY_BLOCK)):
            offsets = tl.arange(0, COPY_BLOCK).to(tl.int64) + i * COPY_BLOCK
            mask = offsets < sl
            phys = tl.load(req_to_token_ptr + req_base + offsets, mask=mask, other=0)
            tl.store(kv_indices_out_ptr + out_offset + offsets, phys, mask=mask)


def get_num_page_per_block_flashmla(page_size: int = 64) -> int:
    num_page_per_block = _FLASHMLA_CREATE_KV_BLOCK_SIZE // page_size
    return num_page_per_block


@triton.jit
def create_flashmla_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
    kv_indices_ptr_stride: tl.constexpr,
    PAGED_SIZE: tl.constexpr = 64,
):
    NUM_PAGE_PER_BLOCK: tl.constexpr = (
        FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON // PAGED_SIZE
    )
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start

    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_paged = tl.cdiv(kv_end - kv_start, PAGED_SIZE)
    num_pages_loop = tl.cdiv(kv_end - kv_start, FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON)

    for i in range(num_pages_loop):
        # index into req_to_token_ptr needs to be int64
        paged_offset = (
            tl.arange(0, NUM_PAGE_PER_BLOCK).to(tl.int64) + i * NUM_PAGE_PER_BLOCK
        ) * PAGED_SIZE
        paged_offset_out = tl.arange(0, NUM_PAGE_PER_BLOCK) + i * NUM_PAGE_PER_BLOCK

        mask = paged_offset < num_paged * PAGED_SIZE
        mask_out = paged_offset_out < num_paged

        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + paged_offset,
            mask=mask,
        )
        tl.store(
            kv_indices_ptr + pid * kv_indices_ptr_stride + paged_offset_out,
            data // PAGED_SIZE,
            mask=mask_out,
        )


@triton.jit
def concat_and_cast_mha_k_kernel(
    k_ptr,
    k_nope_ptr,
    k_rope_ptr,
    head_cnt: tl.constexpr,
    k_stride0: tl.constexpr,
    k_stride1: tl.constexpr,
    nope_stride0: tl.constexpr,
    nope_stride1: tl.constexpr,
    rope_stride0: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    head_range = tl.arange(0, head_cnt)

    k_head_ptr = k_ptr + pid_loc * k_stride0 + head_range[:, None] * k_stride1

    nope_offs = tl.arange(0, nope_dim)

    src_nope_ptr = (
        k_nope_ptr
        + pid_loc * nope_stride0
        + head_range[:, None] * nope_stride1
        + nope_offs[None, :]
    )
    dst_nope_ptr = k_head_ptr + nope_offs[None, :]

    src_nope = tl.load(src_nope_ptr)
    tl.store(dst_nope_ptr, src_nope)

    rope_offs = tl.arange(0, rope_dim)
    src_rope_ptr = k_rope_ptr + pid_loc * rope_stride0 + rope_offs[None, :]
    dst_rope_ptr = k_head_ptr + nope_dim + rope_offs[None, :]
    src_rope = tl.load(src_rope_ptr)
    tl.store(dst_rope_ptr, src_rope)


def concat_and_cast_mha_k_triton(
    k: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
):
    # The source data type will be implicitly converted to the target data type.
    assert (
        len(k.shape) == 3 and len(k_nope.shape) == 3 and len(k_rope.shape) == 3
    ), f"shape should be 3d, but got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"
    assert (
        k.shape[0] == k_nope.shape[0] and k.shape[0] == k_rope.shape[0]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"
    assert (
        k.shape[1] == k_nope.shape[1] and 1 == k_rope.shape[1]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"
    assert (
        k.shape[-1] == k_nope.shape[-1] + k_rope.shape[-1]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"

    nope_dim = k_nope.shape[-1]
    rope_dim = k_rope.shape[-1]
    grid = (k.shape[0],)

    concat_and_cast_mha_k_kernel[grid](
        k,
        k_nope,
        k_rope,
        k.shape[1],
        k.stride(0),
        k.stride(1),
        k_nope.stride(0),
        k_nope.stride(1),
        k_rope.stride(0),
        nope_dim,
        rope_dim,
    )


@triton.jit
def pad_sequence_with_mask_kernel(
    input_ptr,  # (total_tokens, hidden)
    offsets_ptr,  # (B,)
    lengths_ptr,  # (B,)
    output_ptr,  # (B, max_len, hidden)
    mask_ptr,  # (B, max_len)
    max_len,
    hidden_dim,
    BLOCK_M: tl.constexpr,  # seq block
    BLOCK_D: tl.constexpr,  # hidden block
):
    b = tl.program_id(0)  # batch index
    m = tl.program_id(1)  # seq block index

    offset = tl.load(offsets_ptr + b)
    length = tl.load(lengths_ptr + b)

    seq_ids = m * BLOCK_M + tl.arange(0, BLOCK_M)
    hid_ids = tl.arange(0, BLOCK_D)

    seq_mask = seq_ids < max_len
    valid_token = seq_ids < length

    # input index
    in_token = offset + seq_ids
    in_ptr = input_ptr + in_token[:, None] * hidden_dim + hid_ids[None, :]

    # output index
    out_ptr = (
        output_ptr
        + b * max_len * hidden_dim
        + seq_ids[:, None] * hidden_dim
        + hid_ids[None, :]
    )

    values = tl.load(
        in_ptr,
        mask=valid_token[:, None] & (hid_ids[None, :] < hidden_dim),
        other=0.0,
    )

    tl.store(
        out_ptr,
        values,
        mask=seq_mask[:, None] & (hid_ids[None, :] < hidden_dim),
    )

    # attention mask
    if tl.program_id(2) == 0:
        mask_out_ptr = mask_ptr + b * max_len + seq_ids
        tl.store(mask_out_ptr, valid_token, mask=seq_mask)


def pad_sequence_with_mask(
    input_emb,  # (total_tokens, hidden)
    offsets,  # (B,)
    lengths,  # (B,)
    max_len,
):
    B = offsets.shape[0]
    hidden_dim = input_emb.shape[1]

    output = torch.zeros(
        (B, max_len, hidden_dim),
        device=input_emb.device,
        dtype=input_emb.dtype,
    )
    attn_mask = torch.empty(
        (B * max_len),
        device=input_emb.device,
        dtype=torch.bool,
    )

    BLOCK_M = 32
    BLOCK_D = triton.next_power_of_2(hidden_dim)

    grid = (
        B,
        triton.cdiv(max_len, BLOCK_M),
        1,
    )

    pad_sequence_with_mask_kernel[grid](
        input_emb,
        offsets,
        lengths,
        output,
        attn_mask,
        max_len,
        hidden_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
    )

    return B, output, attn_mask
