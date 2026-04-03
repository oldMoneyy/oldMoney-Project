"""Sparse prefill attention for MiniCPM4 layers on the FlashInfer backend.

This module adds sparse block selection to the PAGED prefix attention path
during chunked prefill. For chunks 2+ (which have a cached prefix), the
paged wrapper normally attends to ALL prefix tokens. Sparse prefill reduces
this to only the top-k selected blocks.

Flow:
1. MiniCPMAttention.forward(): compute KC1 from prefix keys + new keys,
   score Q against KC1 to select top-k blocks, attach to kwargs
2. FlashInfer forward_extend(): detect sparse metadata, build filtered
   kv_indices for the paged wrapper, compute sparse prefix attention

Only applies when:
- Layer is minicpm4 (layer_id in MINICPM4_LAYERS)
- Prefix length > dense_len (8192)
- SGLANG_SPARSE_PREFILL=1 environment variable is set
"""

import logging
import os
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Lazy imports for modules only available on GPU server
_infllm_imported = False
_infllmv2_attn_stage1 = None
_max_pooling_1d_varlen = None


def _lazy_import():
    global _infllm_imported, _infllmv2_attn_stage1, _max_pooling_1d_varlen
    if _infllm_imported:
        return
    from infllm_v2 import infllmv2_attn_stage1, max_pooling_1d_varlen
    _infllmv2_attn_stage1 = infllmv2_attn_stage1
    _max_pooling_1d_varlen = max_pooling_1d_varlen
    _infllm_imported = True


# Whether sparse prefill is enabled (set via env or server arg)
SPARSE_PREFILL_ENABLED = os.environ.get("SGLANG_SPARSE_PREFILL", "0") == "1"

# MiniCPM4 layer IDs (softmax attention layers that benefit from sparse)
MINICPM4_LAYERS = {0, 9, 16, 17, 22, 29, 30, 31}


def compress_keys_simple(k: torch.Tensor, kernel_size: int, kernel_stride: int) -> torch.Tensor:
    """Compress keys via sliding-window mean pooling.

    Matches the original compress_k_complete_kernel_new Triton kernel behavior.

    Args:
        k: Key tensor, shape (seq_len, num_kv_heads, head_dim), float32
        kernel_size: Window size (32 for KC1, 128 for KC2)
        kernel_stride: Stride (16 for KC1, 64 for KC2)

    Returns:
        Compressed keys, shape (num_chunks, num_kv_heads, head_dim)
    """
    seq_len = k.shape[0]
    num_chunks = max(0, (seq_len - kernel_size) // kernel_stride + 1)
    if num_chunks == 0:
        return k.new_empty(0, k.shape[1], k.shape[2])

    # k: (seq_len, heads, dim) -> (heads, dim, seq_len) for unfold
    k_t = k.permute(1, 2, 0).contiguous()
    k_unf = k_t.unfold(2, kernel_size, kernel_stride)  # (heads, dim, chunks, kernel_size)
    k_compressed = k_unf.mean(dim=-1)  # (heads, dim, chunks)
    return k_compressed.permute(2, 0, 1).contiguous()  # (chunks, heads, dim)


def compute_sparse_prefill_metadata(
    q: torch.Tensor,
    k_new: torch.Tensor,
    forward_batch,
    layer,
    config,
) -> Optional[dict]:
    """Compute sparse block indices for prefill attention.

    Called from MiniCPMAttention.forward() before self.attn().
    Reads prefix keys from KV cache, compresses with new keys,
    scores Q against KC1/KC2, and returns topk block indices.

    Args:
        q: Query tensor after RoPE, shape (total_new_tokens, num_q_heads * head_dim)
        k_new: New key tensor after RoPE, shape (total_new_tokens, num_kv_heads * head_dim)
        forward_batch: ForwardBatch
        layer: RadixAttention layer (or MiniCPMAttention for accessing attn props)
        config: Model config with sparse params

    Returns:
        Dict with sparse metadata or None if dense attention should be used
    """
    _lazy_import()

    # Get layer properties from the RadixAttention sublayer
    num_q_heads = layer.tp_q_head_num
    num_kv_heads = layer.tp_k_head_num
    head_dim = layer.head_dim
    heads_per_group = num_q_heads // num_kv_heads  # 16 for this model

    # Sparse config
    block_size = config.sparse_block_size       # 64
    kernel_size = config.sparse_kernel_size     # 32
    kernel_stride = config.sparse_kernel_stride  # 16
    dense_len = config.sparse_dense_len         # 8192
    topk = config.sparse_topk                   # 64
    window_size = config.sparse_window_size     # 2048
    init_blocks = config.sparse_init_blocks     # 1
    local_blocks = window_size // block_size    # 32
    sparse_topk = topk + local_blocks           # 96

    seq_lens_cpu = forward_batch.seq_lens_cpu.tolist() if hasattr(forward_batch.seq_lens_cpu, 'tolist') else list(forward_batch.seq_lens_cpu)

    # Check if any request has prefix > dense_len
    prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu if forward_batch.extend_prefix_lens_cpu is not None else [0] * len(seq_lens_cpu)
    extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu if forward_batch.extend_seq_lens_cpu is not None else seq_lens_cpu

    max_prefix = max(prefix_lens_cpu)
    if max_prefix < dense_len:
        return None  # No prefix long enough for sparse

    batch_size = len(seq_lens_cpu)
    device = q.device

    # Reshape
    q_3d = q.view(-1, num_q_heads, head_dim)
    k_new_3d = k_new.view(-1, num_kv_heads, head_dim)

    # For each request, read prefix keys from KV cache + new keys, compute KC1
    req_to_token = forward_batch.req_to_token_pool.req_to_token
    req_pool_indices = forward_batch.req_pool_indices
    k_cache, _ = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)

    kc1_list = []
    kc2_list = []
    new_offset = 0
    for b in range(batch_size):
        prefix_len = prefix_lens_cpu[b]
        extend_len = extend_seq_lens_cpu[b]

        if prefix_len < dense_len:
            # This request doesn't need sparse -- skip KC1 computation
            # We'll put empty KC1/KC2 entries, and the topk will be unused
            # because build_filtered_kv_indices checks prefix_len < dense_len
            total_len = prefix_len + extend_len
            num_kc1 = max(0, (total_len - kernel_size) // kernel_stride + 1)
            num_kc2 = max(0, (total_len - kernel_size * 4) // (kernel_stride * 4) + 1)
            # Fill with zeros -- won't affect final output since we use dense for this request
            kc1_list.append(q.new_zeros(num_kc1, num_kv_heads, head_dim))
            kc2_list.append(q.new_zeros(num_kc2, num_kv_heads, head_dim))
            new_offset += extend_len
            continue

        req_idx = req_pool_indices[b].item()

        if prefix_len > 0:
            # Read prefix keys from KV cache
            phys_locs = req_to_token[req_idx, :prefix_len]
            k_prefix = k_cache[phys_locs].float()  # (prefix_len, kv_heads, dim)
            # Concatenate with new keys for this request
            k_all = torch.cat([k_prefix, k_new_3d[new_offset:new_offset + extend_len].float()], dim=0)
        else:
            k_all = k_new_3d[new_offset:new_offset + extend_len].float()

        kc1 = compress_keys_simple(k_all, kernel_size, kernel_stride)
        kc2 = compress_keys_simple(k_all, kernel_size * 4, kernel_stride * 4)
        kc1_list.append(kc1.to(q_3d.dtype))
        kc2_list.append(kc2.to(q_3d.dtype))
        new_offset += extend_len

    kc1_all = torch.cat(kc1_list, dim=0) if kc1_list else q_3d.new_empty(0, num_kv_heads, head_dim)
    kc2_all = torch.cat(kc2_list, dim=0) if kc2_list else q_3d.new_empty(0, num_kv_heads, head_dim)

    # Build cu_seqlens for scoring
    # Q: only new tokens (extend)
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for i in range(batch_size):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + extend_seq_lens_cpu[i]

    # KC1/KC2 cu_seqlens: based on total seq lens
    cu_seqlens_k1 = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_k2 = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for i in range(batch_size):
        sl = seq_lens_cpu[i]
        cu_seqlens_k1[i + 1] = cu_seqlens_k1[i] + max(0, (sl - kernel_size) // kernel_stride + 1)
        cu_seqlens_k2[i + 1] = cu_seqlens_k2[i] + max(0, (sl - kernel_size * 4) // (kernel_stride * 4) + 1)

    max_seqlen_q = max(extend_seq_lens_cpu)
    max_context_len = max(seq_lens_cpu)

    # cache_lens = prefix_lens (tokens already in cache before this extend)
    cache_lens = torch.tensor(prefix_lens_cpu, dtype=torch.int32, device=device)

    # GQA head ratio check -- match original compressed_attention behavior
    q_for_scoring = q_3d
    current_ratio = num_q_heads // num_kv_heads
    required_ratio = 16
    if current_ratio < required_ratio:
        repeat_times = required_ratio // current_ratio
        q_for_scoring = q_3d.repeat_interleave(repeat_times, dim=1)

    # CRITICAL: infllmv2_attn_stage1 expects cu_seqlens_q multiplied by
    # heads_per_group to correctly handle GQA grouping. The kernel internally
    # views Q as (total_q * groups, kv_heads, head_dim). Without this,
    # the scoring produces wrong results and topk misses important blocks.
    cu_seqlens_q_adjusted = cu_seqlens_q * heads_per_group
    max_seqlen_q_adjusted = max_seqlen_q * heads_per_group

    # Score computation
    score = _infllmv2_attn_stage1(
        q_for_scoring.contiguous(),
        kc1_all.contiguous(),
        kc2_all.contiguous(),
        cu_seqlens_q=cu_seqlens_q_adjusted,
        cu_seqlens_k=cu_seqlens_k1,
        cu_seqlens_v=cu_seqlens_k2,
        max_seqlen_q=max_seqlen_q_adjusted,
        max_seqlen_k=max_context_len // kernel_stride,
        causal=True,
    )

    # Block-level max pooling -- uses NON-adjusted cu_seqlens_q (like original)
    block_score = _max_pooling_1d_varlen(
        score.contiguous(),
        cu_seqlens_q,
        cu_seqlens_k1,
        cache_lens,
        max_seqlen_q,
        max_context_len,
        local_blocks=local_blocks,
        init_blocks=init_blocks,
        block_size=block_size,
        stride=kernel_stride,
        total_q=-1,  # Match original default
    )

    # Select top-k blocks
    topk_idx = block_score.topk(sparse_topk, dim=-1).indices.sort(-1).values
    topk_idx = topk_idx.to(torch.int32)

    # Build per-request block sets
    # topk_idx shape: (num_kv_heads, total_q, sparse_topk)
    sparse_block_indices = []
    q_offset = 0
    for b in range(batch_size):
        extend_len = extend_seq_lens_cpu[b]
        if prefix_lens_cpu[b] < dense_len:
            # Dense request -- no sparse blocks needed
            sparse_block_indices.append(None)
            q_offset += extend_len
            continue
        req_blocks = []
        for h in range(num_kv_heads):
            req_topk_h = topk_idx[h, q_offset:q_offset + extend_len, :]  # (extend_len, topk)
            blocks = req_topk_h.reshape(-1).unique()
            blocks = blocks[blocks >= 0]  # Remove sentinels
            req_blocks.append(blocks)
        sparse_block_indices.append(req_blocks)
        q_offset += extend_len

    return {
        "sparse_block_indices": sparse_block_indices,
        "block_size": block_size,
        "prefix_lens": prefix_lens_cpu,
        "seq_lens": seq_lens_cpu,
        "dense_len": dense_len,
    }


def build_filtered_kv_indices(
    sparse_metadata: dict,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    paged_kernel_lens: torch.Tensor,
    kv_start_idx: torch.Tensor,
    original_kv_indices: torch.Tensor,
    original_kv_indptr: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build filtered kv_indices that only include selected sparse blocks.

    For requests with prefix > dense_len, replace the full kv_indices
    with indices for only the selected blocks. For short requests,
    keep the original indices.

    Returns:
        (filtered_kv_indices, filtered_kv_indptr)
    """
    sparse_block_indices = sparse_metadata["sparse_block_indices"]
    block_size = sparse_metadata["block_size"]
    prefix_lens = sparse_metadata["prefix_lens"]
    seq_lens = sparse_metadata["seq_lens"]
    dense_len = sparse_metadata["dense_len"]
    batch_size = len(seq_lens)

    # For each request, compute the filtered KV token indices
    all_phys_indices = []
    filtered_lens = []

    for b in range(batch_size):
        if prefix_lens[b] >= dense_len and sparse_block_indices[b] is not None:
            # Sparse: gather tokens from selected blocks (union across heads)
            all_blocks = torch.cat(sparse_block_indices[b], dim=0).unique()
            all_blocks = all_blocks[all_blocks >= 0].sort().values

            # Vectorized: compute all token positions from selected blocks
            req_idx = req_pool_indices[b].item()
            prefix_len = prefix_lens[b]

            # Build token positions from blocks
            block_starts = all_blocks.long() * block_size  # (num_blocks,)
            block_ends = torch.clamp(block_starts + block_size, max=prefix_len)
            valid_mask = block_starts < prefix_len

            if valid_mask.any():
                valid_starts = block_starts[valid_mask]
                valid_ends = block_ends[valid_mask]
                block_lens = valid_ends - valid_starts

                # Gather physical indices for all valid blocks
                token_positions = torch.cat([
                    torch.arange(s.item(), e.item(), device=device)
                    for s, e in zip(valid_starts, valid_ends)
                ])
                phys = req_to_token[req_idx, token_positions]
                all_phys_indices.append(phys)
                filtered_lens.append(len(phys))
            else:
                filtered_lens.append(0)
        else:
            # Dense: copy original indices
            orig_start = original_kv_indptr[b].item()
            orig_end = original_kv_indptr[b + 1].item()
            orig_len = orig_end - orig_start
            if orig_len > 0:
                all_phys_indices.append(original_kv_indices[orig_start:orig_end])
            filtered_lens.append(orig_len)

    # Build filtered kv_indptr
    filtered_kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for b in range(batch_size):
        filtered_kv_indptr[b + 1] = filtered_kv_indptr[b] + filtered_lens[b]

    total_kv = filtered_kv_indptr[batch_size].item()

    # Concatenate all physical indices
    if all_phys_indices:
        filtered_kv_indices = torch.cat(all_phys_indices, dim=0).to(torch.int32)
        # Pad to avoid FlashInfer out-of-bounds
        pad = torch.zeros(256, dtype=torch.int32, device=device)
        filtered_kv_indices = torch.cat([filtered_kv_indices, pad])
    else:
        filtered_kv_indices = torch.zeros(256, dtype=torch.int32, device=device)

    return filtered_kv_indices, filtered_kv_indptr
