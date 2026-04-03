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
import math
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F

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

    Args:
        k: Key tensor, shape (seq_len, num_kv_heads, head_dim), float32
        kernel_size: Window size (32 for KC1, 128 for KC2)
        kernel_stride: Stride (16 for KC1, 64 for KC2)

    Returns:
        Compressed keys, shape (num_chunks, num_kv_heads, head_dim)
    """
    seq_len = k.shape[0]
    if seq_len < kernel_size:
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
    scaling = layer.scaling

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
    # k_cache shape: (total_slots, num_kv_heads, head_dim) possibly in FP8

    kc1_list = []
    kc2_list = []
    new_offset = 0
    for b in range(batch_size):
        prefix_len = prefix_lens_cpu[b]
        extend_len = extend_seq_lens_cpu[b]
        total_len = seq_lens_cpu[b]
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

    # GQA head ratio check
    q_for_scoring = q_3d
    current_ratio = num_q_heads // num_kv_heads
    if current_ratio < 16:
        repeat_times = 16 // current_ratio
        q_for_scoring = q_3d.repeat_interleave(repeat_times, dim=1)

    # Score computation
    score = _infllmv2_attn_stage1(
        q_for_scoring.contiguous(),
        kc1_all.contiguous(),
        kc2_all.contiguous(),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k1,
        cu_seqlens_v=cu_seqlens_k2,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_context_len // kernel_stride,
        causal=True,
    )

    # Block-level max pooling
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
        total_q=q_3d.shape[0],
    )

    # Select top-k blocks
    topk_idx = block_score.topk(sparse_topk, dim=-1).indices.sort(-1).values
    topk_idx = topk_idx.to(torch.int32)

    # For the paged prefix path, we take the UNION of all selected blocks
    # per request per head. This is used to filter kv_indices.
    # topk_idx shape: (num_kv_heads, total_q, sparse_topk)

    # Build per-request block sets
    sparse_block_indices = []  # List of (num_kv_heads,) sets of block indices per request
    q_offset = 0
    for b in range(batch_size):
        extend_len = extend_seq_lens_cpu[b]
        req_blocks = []
        for h in range(num_kv_heads):
            req_topk_h = topk_idx[h, q_offset:q_offset + extend_len, :]  # (extend_len, topk)
            blocks = req_topk_h.reshape(-1).unique()
            blocks = blocks[blocks >= 0]  # Remove sentinels
            req_blocks.append(blocks)
        sparse_block_indices.append(req_blocks)
        q_offset += extend_len

    return {
        "sparse_block_indices": sparse_block_indices,  # per-request, per-head block indices
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

    # For each request, compute the filtered KV length
    filtered_lens = []
    for b in range(batch_size):
        if prefix_lens[b] >= dense_len:
            # Sparse: count total tokens from selected blocks (union across heads)
            # For the paged path, we use the same blocks for all heads
            # (union of both heads' selections)
            all_blocks = torch.cat([bi for bi in sparse_block_indices[b]], dim=0).unique()
            all_blocks = all_blocks[all_blocks >= 0]
            total_tokens = 0
            for block_idx in all_blocks:
                bi = block_idx.item()
                start = bi * block_size
                end = min(start + block_size, prefix_lens[b])
                if start < prefix_lens[b]:
                    total_tokens += (end - start)
            filtered_lens.append(total_tokens)
        else:
            # Dense: keep original length
            filtered_lens.append(paged_kernel_lens[b].item())

    # Build filtered kv_indptr
    total_kv = sum(filtered_lens)
    filtered_kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for b in range(batch_size):
        filtered_kv_indptr[b + 1] = filtered_kv_indptr[b] + filtered_lens[b]

    # Build filtered kv_indices
    filtered_kv_indices = torch.empty(total_kv + 256, dtype=torch.int32, device=device)

    offset = 0
    for b in range(batch_size):
        req_idx = req_pool_indices[b].item()
        if prefix_lens[b] >= dense_len:
            # Sparse: gather selected block tokens
            all_blocks = torch.cat([bi for bi in sparse_block_indices[b]], dim=0).unique()
            all_blocks = all_blocks[all_blocks >= 0].sort().values

            for block_idx in all_blocks:
                bi = block_idx.item()
                start = bi * block_size
                end = min(start + block_size, prefix_lens[b])
                if start >= prefix_lens[b]:
                    continue
                chunk_len = end - start
                # Physical indices from req_to_token
                phys = req_to_token[req_idx, start:end]
                filtered_kv_indices[offset:offset + chunk_len] = phys
                offset += chunk_len
        else:
            # Dense: copy original indices
            orig_start = original_kv_indptr[b].item()
            orig_end = original_kv_indptr[b + 1].item()
            orig_len = orig_end - orig_start
            filtered_kv_indices[offset:offset + orig_len] = original_kv_indices[orig_start:orig_end]
            offset += orig_len

    return filtered_kv_indices, filtered_kv_indptr
