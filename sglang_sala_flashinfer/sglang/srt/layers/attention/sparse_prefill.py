"""Sparse attention for MiniCPM4 layers on the FlashInfer backend.

This module adds sparse block selection to both PREFILL and DECODE paths.

PREFILL (sparse prefill):
  For chunks 2+ (which have a cached prefix), the paged wrapper normally
  attends to ALL prefix tokens. Sparse prefill reduces this to top-k blocks.

DECODE (sparse decode):
  During decode, each token attends to dense_len + window + topk blocks
  instead of the full KV cache. Block selection is amortized: computed every
  DECODE_SELECTION_INTERVAL steps in DECODE_ANCHOR_LAYER only, then shared
  across all 8 minicpm4 layers.

Incremental KC1: compressed keys are cached per (request, layer) across
chunks/decode steps. Only new tokens are compressed incrementally.

Gated by:
- SGLANG_SPARSE_PREFILL=1 for prefill
- SGLANG_SPARSE_DECODE=1 for decode
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

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

# Whether sparse decode is enabled
SPARSE_DECODE_ENABLED = os.environ.get("SGLANG_SPARSE_DECODE", "0") == "1"

# MiniCPM4 layer IDs (softmax attention layers that benefit from sparse)
MINICPM4_LAYERS = {0, 9, 16, 17, 22, 29, 30, 31}

# Sparse decode: anchor layer for block selection (computed here, shared to all)
DECODE_ANCHOR_LAYER = 0

# Sparse decode: recompute block selection every N decode steps
DECODE_SELECTION_INTERVAL = 16


# ---------------------------------------------------------------------------
# Incremental KC1/KC2 cache
# ---------------------------------------------------------------------------

@dataclass
class CachedKC:
    """Cached compressed keys for a (request, layer) pair."""
    kc1: torch.Tensor        # (num_kc1_chunks, kv_heads, head_dim)
    kc2: torch.Tensor        # (num_kc2_chunks, kv_heads, head_dim)
    total_len: int           # total tokens covered by this cache
    kc1_kernel_stride: int   # stride used for KC1 (16)
    kc2_kernel_stride: int   # stride used for KC2 (64)


# Module-level cache: (req_pool_index, layer_id) -> CachedKC
_kc1_cache: dict[tuple[int, int], CachedKC] = {}


# ---------------------------------------------------------------------------
# Sparse decode: cached block selection (shared across layers)
# ---------------------------------------------------------------------------

@dataclass
class CachedDecodeBlocks:
    """Cached block selection for a request during decode."""
    block_indices: List[torch.Tensor]   # per kv_head: 1D int32 tensor of block indices
    decode_step: int                    # decode step when this was computed
    seq_len_at_compute: int             # seq_len when blocks were selected


# Module-level cache: req_pool_index -> CachedDecodeBlocks
_decode_block_cache: Dict[int, CachedDecodeBlocks] = {}

# Module-level buffer for accumulating new keys during decode
# (req_pool_index, layer_id) -> list of key tensors (each shape (1, kv_heads, head_dim))
_decode_key_buffer: Dict[Tuple[int, int], List[torch.Tensor]] = {}


def clear_decode_caches(active_req_indices: Optional[Set[int]] = None):
    """Remove stale decode cache entries for finished requests."""
    global _decode_block_cache, _decode_key_buffer
    if active_req_indices is None:
        _decode_block_cache.clear()
        _decode_key_buffer.clear()
        return
    stale = [k for k in _decode_block_cache if k not in active_req_indices]
    for k in stale:
        del _decode_block_cache[k]
    stale_buf = [k for k in _decode_key_buffer if k[0] not in active_req_indices]
    for k in stale_buf:
        del _decode_key_buffer[k]


# Cached sparse decode config (set once from model config)
_sparse_decode_cfg: Optional[dict] = None


def set_sparse_decode_config(config) -> None:
    """Set sparse decode config from model config. Called during model init."""
    global _sparse_decode_cfg
    if not hasattr(config, 'sparse_dense_len'):
        return
    _sparse_decode_cfg = {
        "dense_len": config.sparse_dense_len,
        "window_size": config.sparse_window_size,
        "block_size": config.sparse_block_size,
        "topk": config.sparse_topk,
        "kernel_size": config.sparse_kernel_size,
        "kernel_stride": config.sparse_kernel_stride,
    }


def get_sparse_decode_config() -> Optional[dict]:
    """Get sparse decode config. Returns None if not set."""
    return _sparse_decode_cfg


def clear_kc1_cache(active_req_indices: Optional[Set[int]] = None):
    """Remove stale cache entries for requests no longer active.

    Args:
        active_req_indices: Set of currently active req_pool_indices.
            If None, clears the entire cache.
    """
    global _kc1_cache
    if active_req_indices is None:
        _kc1_cache.clear()
        return
    stale_keys = [k for k in _kc1_cache if k[0] not in active_req_indices]
    for k in stale_keys:
        del _kc1_cache[k]


# ---------------------------------------------------------------------------
# Key compression
# ---------------------------------------------------------------------------

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
    num_chunks = max(0, (seq_len - kernel_size) // kernel_stride + 1)
    if num_chunks == 0:
        return k.new_empty(0, k.shape[1], k.shape[2])

    k_t = k.permute(1, 2, 0).contiguous()
    k_unf = k_t.unfold(2, kernel_size, kernel_stride)  # (heads, dim, chunks, kernel_size)
    k_compressed = k_unf.mean(dim=-1)  # (heads, dim, chunks)
    return k_compressed.permute(2, 0, 1).contiguous()  # (chunks, heads, dim)


def _compress_incremental(
    req_pool_idx: int,
    layer_id: int,
    prefix_len: int,
    extend_len: int,
    k_new_request: torch.Tensor,  # (extend_len, kv_heads, head_dim), float32
    k_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    req_idx: int,
    kernel_size: int,
    kernel_stride: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compress keys incrementally using cached KC entries.

    Only reads boundary tokens from KV cache (typically 16 for KC1, 64 for KC2)
    instead of the entire prefix.

    Returns:
        Full compressed keys, shape (total_chunks, kv_heads, head_dim)
    """
    total_len = prefix_len + extend_len
    cache_key = (req_pool_idx, layer_id)
    cached = _kc1_cache.get(cache_key)

    if cached is not None and cached.total_len <= total_len:
        # Incremental: reuse cached chunks, only compress new tokens + boundary
        # Select the right cached tensor based on kernel_stride
        is_kc1 = (kernel_stride == cached.kc1_kernel_stride)
        cached_kc = cached.kc1 if is_kc1 else cached.kc2
        history_chunks = cached_kc.shape[0]

        # boundary_start: first token position not fully covered by cached chunks
        boundary_start = history_chunks * kernel_stride

        if boundary_start >= total_len:
            # Cached chunks already cover everything (shouldn't happen normally)
            total_needed = max(0, (total_len - kernel_size) // kernel_stride + 1)
            return cached_kc[:total_needed].to(dtype)

        # Gather boundary tokens: [boundary_start, prefix_len) from KV cache
        # + [0, extend_len) from k_new_request
        boundary_from_prefix = max(0, prefix_len - boundary_start)

        if boundary_from_prefix > 0:
            phys_locs = req_to_token[req_idx, boundary_start:prefix_len]
            k_boundary = k_cache[phys_locs].float()
            k_tail = torch.cat([k_boundary, k_new_request], dim=0)
        else:
            # boundary_start >= prefix_len: only new tokens needed
            start_in_new = boundary_start - prefix_len
            k_tail = k_new_request[start_in_new:]

        # Compress the tail portion
        new_kc = compress_keys_simple(k_tail, kernel_size, kernel_stride)

        if new_kc.shape[0] == 0:
            return cached_kc.to(dtype)

        # Concatenate: cached history + new chunks
        kc_full = torch.cat([cached_kc.to(new_kc.dtype), new_kc], dim=0)
        return kc_full.to(dtype)
    else:
        # First time or cache invalidated: full computation
        if prefix_len > 0:
            phys_locs = req_to_token[req_idx, :prefix_len]
            k_prefix = k_cache[phys_locs].float()
            k_all = torch.cat([k_prefix, k_new_request], dim=0)
        else:
            k_all = k_new_request

        kc = compress_keys_simple(k_all, kernel_size, kernel_stride)
        return kc.to(dtype)


# ---------------------------------------------------------------------------
# Main sparse prefill metadata computation
# ---------------------------------------------------------------------------

def compute_sparse_prefill_metadata(
    q: torch.Tensor,
    k_new: torch.Tensor,
    forward_batch,
    layer,
    config,
) -> Optional[dict]:
    """Compute sparse block indices for prefill attention.

    Uses incremental KC1/KC2 caching to avoid re-reading the entire prefix
    from the KV cache each chunk. Only boundary tokens (~16 for KC1, ~64
    for KC2) are read from the cache on subsequent chunks.
    """
    _lazy_import()

    num_q_heads = layer.tp_q_head_num
    num_kv_heads = layer.tp_k_head_num
    head_dim = layer.head_dim
    heads_per_group = num_q_heads // num_kv_heads
    layer_id = layer.layer_id

    block_size = config.sparse_block_size
    kernel_size = config.sparse_kernel_size
    kernel_stride = config.sparse_kernel_stride
    dense_len = config.sparse_dense_len
    topk = config.sparse_topk
    window_size = config.sparse_window_size
    init_blocks = config.sparse_init_blocks
    local_blocks = window_size // block_size
    sparse_topk = topk + local_blocks

    seq_lens_cpu = forward_batch.seq_lens_cpu.tolist() if hasattr(forward_batch.seq_lens_cpu, 'tolist') else list(forward_batch.seq_lens_cpu)
    prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu if forward_batch.extend_prefix_lens_cpu is not None else [0] * len(seq_lens_cpu)
    extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu if forward_batch.extend_seq_lens_cpu is not None else seq_lens_cpu

    max_prefix = max(prefix_lens_cpu)
    if max_prefix <= dense_len:
        return None

    batch_size = len(seq_lens_cpu)
    device = q.device

    q_3d = q.view(-1, num_q_heads, head_dim)
    k_new_3d = k_new.view(-1, num_kv_heads, head_dim)

    req_to_token = forward_batch.req_to_token_pool.req_to_token
    req_pool_indices = forward_batch.req_pool_indices
    k_cache, _ = forward_batch.token_to_kv_pool.get_kv_buffer(layer_id)

    kc1_list = []
    kc2_list = []
    new_offset = 0

    for b in range(batch_size):
        prefix_len = prefix_lens_cpu[b]
        extend_len = extend_seq_lens_cpu[b]
        total_len = seq_lens_cpu[b]

        if prefix_len <= dense_len:
            num_kc1 = max(0, (total_len - kernel_size) // kernel_stride + 1)
            num_kc2 = max(0, (total_len - kernel_size * 4) // (kernel_stride * 4) + 1)
            kc1_list.append(q.new_zeros(num_kc1, num_kv_heads, head_dim))
            kc2_list.append(q.new_zeros(num_kc2, num_kv_heads, head_dim))
            new_offset += extend_len
            continue

        req_idx = req_pool_indices[b].item()
        req_pool_idx = req_idx
        k_new_req = k_new_3d[new_offset:new_offset + extend_len].float()

        # Incremental KC1
        kc1 = _compress_incremental(
            req_pool_idx, layer_id, prefix_len, extend_len,
            k_new_req, k_cache, req_to_token, req_idx,
            kernel_size, kernel_stride, q_3d.dtype,
        )

        # Incremental KC2
        kc2 = _compress_incremental(
            req_pool_idx, layer_id, prefix_len, extend_len,
            k_new_req, k_cache, req_to_token, req_idx,
            kernel_size * 4, kernel_stride * 4, q_3d.dtype,
        )

        # Update cache
        cache_key = (req_pool_idx, layer_id)
        _kc1_cache[cache_key] = CachedKC(
            kc1=kc1.to(torch.bfloat16), kc2=kc2.to(torch.bfloat16),
            total_len=total_len,
            kc1_kernel_stride=kernel_stride,
            kc2_kernel_stride=kernel_stride * 4,
        )

        kc1_list.append(kc1)
        kc2_list.append(kc2)
        new_offset += extend_len

    kc1_all = torch.cat(kc1_list, dim=0) if kc1_list else q_3d.new_empty(0, num_kv_heads, head_dim)
    kc2_all = torch.cat(kc2_list, dim=0) if kc2_list else q_3d.new_empty(0, num_kv_heads, head_dim)

    # Build cu_seqlens for scoring
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for i in range(batch_size):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + extend_seq_lens_cpu[i]

    cu_seqlens_k1 = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_k2 = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for i in range(batch_size):
        sl = seq_lens_cpu[i]
        cu_seqlens_k1[i + 1] = cu_seqlens_k1[i] + max(0, (sl - kernel_size) // kernel_stride + 1)
        cu_seqlens_k2[i + 1] = cu_seqlens_k2[i] + max(0, (sl - kernel_size * 4) // (kernel_stride * 4) + 1)

    max_seqlen_q = max(extend_seq_lens_cpu)
    max_context_len = max(seq_lens_cpu)
    cache_lens = torch.tensor(prefix_lens_cpu, dtype=torch.int32, device=device)

    # GQA head ratio check
    q_for_scoring = q_3d
    current_ratio = num_q_heads // num_kv_heads
    if current_ratio < 16:
        q_for_scoring = q_3d.repeat_interleave(16 // current_ratio, dim=1)

    cu_seqlens_q_adjusted = cu_seqlens_q * heads_per_group
    max_seqlen_q_adjusted = max_seqlen_q * heads_per_group

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
        total_q=-1,
    )

    topk_idx = block_score.topk(sparse_topk, dim=-1).indices.sort(-1).values
    topk_idx = topk_idx.to(torch.int32)

    sparse_block_indices = []
    q_offset = 0
    for b in range(batch_size):
        extend_len = extend_seq_lens_cpu[b]
        if prefix_lens_cpu[b] <= dense_len:
            sparse_block_indices.append(None)
            q_offset += extend_len
            continue
        req_blocks = []
        for h in range(num_kv_heads):
            req_topk_h = topk_idx[h, q_offset:q_offset + extend_len, :]
            blocks = req_topk_h.reshape(-1).unique()
            blocks = blocks[blocks >= 0]
            req_blocks.append(blocks)
        sparse_block_indices.append(req_blocks)
        q_offset += extend_len

    # Seed _decode_block_cache from prefill so CUDA-graph decode has blocks.
    # Only save on the last minicpm4 layer (31) to avoid redundant writes.
    if SPARSE_DECODE_ENABLED and layer_id == 31:
        for b in range(batch_size):
            if sparse_block_indices[b] is not None:
                req_idx = req_pool_indices[b].item()
                _decode_block_cache[req_idx] = CachedDecodeBlocks(
                    block_indices=sparse_block_indices[b],
                    decode_step=0,
                    seq_len_at_compute=seq_lens_cpu[b],
                )

    return {
        "sparse_block_indices": sparse_block_indices,
        "block_size": block_size,
        "prefix_lens": prefix_lens_cpu,
        "seq_lens": seq_lens_cpu,
        "dense_len": dense_len,
    }


# ---------------------------------------------------------------------------
# Sparse DECODE: block selection for decode tokens
# ---------------------------------------------------------------------------

def _update_kc_for_decode(
    req_pool_idx: int,
    layer_id: int,
    k_new: torch.Tensor,       # (1, kv_heads, head_dim) or (kv_heads, head_dim)
    seq_len: int,
    k_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Incrementally update KC1 or KC2 with a single new decode key.

    Appends the new key to a buffer. When enough keys accumulate
    (kernel_stride tokens), produces a new compressed entry.
    For partial buffers, recomputes the last entry using boundary tokens.

    Returns:
        Full compressed keys, shape (total_chunks, kv_heads, head_dim)
    """
    cache_key = (req_pool_idx, layer_id)
    cached = _kc1_cache.get(cache_key)

    if cached is None:
        # First decode after prefill ended without caching (shouldn't happen
        # if prefill ran, but handle gracefully): full recompute
        phys_locs = req_to_token[req_pool_idx, :seq_len]
        k_all = k_cache[phys_locs].float()
        kc = compress_keys_simple(k_all, kernel_size, kernel_stride)
        return kc.to(dtype)

    # Get the right cached tensor
    is_kc1 = (kernel_stride == cached.kc1_kernel_stride)
    cached_kc = cached.kc1 if is_kc1 else cached.kc2
    history_chunks = cached_kc.shape[0]

    # boundary_start: first token not fully covered by cached chunks
    boundary_start = history_chunks * kernel_stride

    if seq_len < boundary_start + kernel_size:
        # Not enough new tokens for another full chunk yet — return as-is
        return cached_kc.to(dtype)

    # We have enough tokens for new chunks — read boundary from KV cache
    phys_locs = req_to_token[req_pool_idx, boundary_start:seq_len]
    k_tail = k_cache[phys_locs].float()
    new_kc = compress_keys_simple(k_tail, kernel_size, kernel_stride)

    if new_kc.shape[0] == 0:
        return cached_kc.to(dtype)

    kc_full = torch.cat([cached_kc.to(new_kc.dtype), new_kc], dim=0)
    return kc_full.to(dtype)


def compute_sparse_decode_metadata(
    q: torch.Tensor,
    k_new: torch.Tensor,
    forward_batch,
    layer,
    config,
) -> Optional[dict]:
    """Compute sparse block indices for decode attention.

    Amortized: block selection is computed every DECODE_SELECTION_INTERVAL
    steps in DECODE_ANCHOR_LAYER only. Other layers/steps reuse cached blocks.

    Args:
        q: (bs, num_q_heads * head_dim) — one token per request
        k_new: (bs, num_kv_heads * head_dim) — new key per request
        forward_batch: ForwardBatch
        layer: RadixAttention
        config: model config with sparse_* attributes

    Returns:
        dict with sparse_block_indices, block_size, etc. or None if no
        request needs sparse decode.
    """
    _lazy_import()

    num_q_heads = layer.tp_q_head_num
    num_kv_heads = layer.tp_k_head_num
    head_dim = layer.head_dim
    heads_per_group = num_q_heads // num_kv_heads
    layer_id = layer.layer_id

    block_size = config.sparse_block_size
    kernel_size = config.sparse_kernel_size
    kernel_stride = config.sparse_kernel_stride
    dense_len = config.sparse_dense_len
    topk = config.sparse_topk
    window_size = config.sparse_window_size
    init_blocks = config.sparse_init_blocks
    local_blocks = window_size // block_size
    sparse_topk = topk + local_blocks

    seq_lens_cpu = forward_batch.seq_lens_cpu.tolist() if hasattr(forward_batch.seq_lens_cpu, 'tolist') else list(forward_batch.seq_lens_cpu)
    batch_size = len(seq_lens_cpu)
    device = q.device

    # Check if any request needs sparse decode
    max_seq = max(seq_lens_cpu)
    if max_seq <= dense_len:
        return None

    req_pool_indices = forward_batch.req_pool_indices
    req_to_token = forward_batch.req_to_token_pool.req_to_token
    k_cache, _ = forward_batch.token_to_kv_pool.get_kv_buffer(layer_id)

    q_3d = q.view(-1, num_q_heads, head_dim)   # (bs, num_q_heads, head_dim)
    k_3d = k_new.view(-1, num_kv_heads, head_dim)  # (bs, num_kv_heads, head_dim)

    is_anchor = (layer_id == DECODE_ANCHOR_LAYER)

    # For non-anchor layers, just return cached block selection
    if not is_anchor:
        sparse_block_indices = []
        any_sparse = False
        for b in range(batch_size):
            req_idx = req_pool_indices[b].item()
            if seq_lens_cpu[b] <= dense_len:
                sparse_block_indices.append(None)
                continue
            cached_blocks = _decode_block_cache.get(req_idx)
            if cached_blocks is not None:
                sparse_block_indices.append(cached_blocks.block_indices)
                any_sparse = True
            else:
                sparse_block_indices.append(None)
        if not any_sparse:
            return None
        return {
            "sparse_block_indices": sparse_block_indices,
            "block_size": block_size,
            "seq_lens": seq_lens_cpu,
            "dense_len": dense_len,
            "window_size": window_size,
        }

    # Anchor layer: check if we need to recompute block selection
    needs_recompute = []
    for b in range(batch_size):
        req_idx = req_pool_indices[b].item()
        if seq_lens_cpu[b] <= dense_len:
            needs_recompute.append(False)
            continue
        cached_blocks = _decode_block_cache.get(req_idx)
        if cached_blocks is None:
            needs_recompute.append(True)
        elif (seq_lens_cpu[b] - cached_blocks.seq_len_at_compute) >= DECODE_SELECTION_INTERVAL:
            needs_recompute.append(True)
        else:
            needs_recompute.append(False)

    if not any(needs_recompute) and not any(seq_lens_cpu[b] > dense_len for b in range(batch_size)):
        return None

    # For requests that need recompute: update KC1/KC2, score, select topk
    # For others: use cached blocks
    recompute_indices = [b for b in range(batch_size) if needs_recompute[b]]

    if recompute_indices:
        # Update KC1/KC2 for requests that need recompute
        kc1_list = []
        kc2_list = []
        q_recompute = []
        recompute_seq_lens = []

        for b in recompute_indices:
            req_idx = req_pool_indices[b].item()
            sl = seq_lens_cpu[b]

            kc1 = _update_kc_for_decode(
                req_idx, layer_id, k_3d[b:b+1], sl,
                k_cache, req_to_token,
                kernel_size, kernel_stride, q_3d.dtype,
            )
            kc2 = _update_kc_for_decode(
                req_idx, layer_id, k_3d[b:b+1], sl,
                k_cache, req_to_token,
                kernel_size * 4, kernel_stride * 4, q_3d.dtype,
            )

            # Update the KC cache
            cache_key = (req_idx, layer_id)
            _kc1_cache[cache_key] = CachedKC(
                kc1=kc1.to(torch.bfloat16), kc2=kc2.to(torch.bfloat16),
                total_len=sl,
                kc1_kernel_stride=kernel_stride,
                kc2_kernel_stride=kernel_stride * 4,
            )

            kc1_list.append(kc1)
            kc2_list.append(kc2)
            q_recompute.append(q_3d[b:b+1])  # (1, num_q_heads, head_dim)
            recompute_seq_lens.append(sl)

        # Stack and score
        rc_bs = len(recompute_indices)
        kc1_all = torch.cat(kc1_list, dim=0)
        kc2_all = torch.cat(kc2_list, dim=0)
        q_rc = torch.cat(q_recompute, dim=0)  # (rc_bs, num_q_heads, head_dim)

        # Build cu_seqlens
        cu_seqlens_q = torch.zeros(rc_bs + 1, dtype=torch.int32, device=device)
        cu_seqlens_k1 = torch.zeros(rc_bs + 1, dtype=torch.int32, device=device)
        cu_seqlens_k2 = torch.zeros(rc_bs + 1, dtype=torch.int32, device=device)
        for i, b in enumerate(recompute_indices):
            cu_seqlens_q[i + 1] = cu_seqlens_q[i] + 1  # 1 query token per request
            sl = recompute_seq_lens[i]
            cu_seqlens_k1[i + 1] = cu_seqlens_k1[i] + max(0, (sl - kernel_size) // kernel_stride + 1)
            cu_seqlens_k2[i + 1] = cu_seqlens_k2[i] + max(0, (sl - kernel_size * 4) // (kernel_stride * 4) + 1)

        max_context_len = max(recompute_seq_lens)
        # cache_lens: for decode, all tokens are cached (prefix = full seq_len)
        cache_lens = torch.tensor(recompute_seq_lens, dtype=torch.int32, device=device)

        # GQA adjustment
        q_for_scoring = q_rc
        current_ratio = num_q_heads // num_kv_heads
        if current_ratio < 16:
            q_for_scoring = q_rc.repeat_interleave(16 // current_ratio, dim=1)

        cu_seqlens_q_adjusted = cu_seqlens_q * heads_per_group
        max_seqlen_q_adjusted = 1 * heads_per_group

        score = _infllmv2_attn_stage1(
            q_for_scoring.contiguous(),
            kc1_all.contiguous(),
            kc2_all.contiguous(),
            cu_seqlens_q=cu_seqlens_q_adjusted,
            cu_seqlens_k=cu_seqlens_k1,
            cu_seqlens_v=cu_seqlens_k2,
            max_seqlen_q=max_seqlen_q_adjusted,
            max_seqlen_k=max_context_len // kernel_stride,
            causal=False,  # decode: query is after all KV, no causal mask needed
        )

        block_score = _max_pooling_1d_varlen(
            score.contiguous(),
            cu_seqlens_q,
            cu_seqlens_k1,
            cache_lens,
            1,  # max_seqlen_q = 1 token
            max_context_len,
            local_blocks=local_blocks,
            init_blocks=init_blocks,
            block_size=block_size,
            stride=kernel_stride,
            total_q=-1,
        )

        topk_idx = block_score.topk(sparse_topk, dim=-1).indices.sort(-1).values
        topk_idx = topk_idx.to(torch.int32)

        # Extract per-request block indices and cache them
        q_offset = 0
        for i, b in enumerate(recompute_indices):
            req_idx = req_pool_indices[b].item()
            req_blocks = []
            for h in range(num_kv_heads):
                blocks = topk_idx[h, q_offset:q_offset + 1, :].reshape(-1).unique()
                blocks = blocks[blocks >= 0]
                req_blocks.append(blocks)
            _decode_block_cache[req_idx] = CachedDecodeBlocks(
                block_indices=req_blocks,
                decode_step=seq_lens_cpu[b],
                seq_len_at_compute=seq_lens_cpu[b],
            )
            q_offset += 1

    # Build final result for all requests
    sparse_block_indices = []
    any_sparse = False
    for b in range(batch_size):
        req_idx = req_pool_indices[b].item()
        if seq_lens_cpu[b] <= dense_len:
            sparse_block_indices.append(None)
            continue
        cached_blocks = _decode_block_cache.get(req_idx)
        if cached_blocks is not None:
            sparse_block_indices.append(cached_blocks.block_indices)
            any_sparse = True
        else:
            sparse_block_indices.append(None)

    if not any_sparse:
        return None

    return {
        "sparse_block_indices": sparse_block_indices,
        "block_size": block_size,
        "seq_lens": seq_lens_cpu,
        "dense_len": dense_len,
        "window_size": window_size,
    }


# ---------------------------------------------------------------------------
# Filtered KV indices (kept for reference, used by dead-code non-ragged path)
# ---------------------------------------------------------------------------

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
    """Build filtered kv_indices that only include selected sparse blocks."""
    sparse_block_indices = sparse_metadata["sparse_block_indices"]
    block_size = sparse_metadata["block_size"]
    prefix_lens = sparse_metadata["prefix_lens"]
    seq_lens = sparse_metadata["seq_lens"]
    dense_len = sparse_metadata["dense_len"]
    batch_size = len(seq_lens)

    all_phys_indices = []
    filtered_lens = []

    for b in range(batch_size):
        if prefix_lens[b] > dense_len and sparse_block_indices[b] is not None:
            req_idx = req_pool_indices[b].item()
            prefix_len = prefix_lens[b]

            dense_positions = torch.arange(0, min(dense_len, prefix_len), device=device)

            all_blocks = torch.cat(sparse_block_indices[b], dim=0).unique()
            all_blocks = all_blocks[all_blocks >= 0].sort().values

            block_starts = all_blocks.long() * block_size
            block_ends = torch.clamp(block_starts + block_size, max=prefix_len)
            sparse_mask = (block_starts >= dense_len) & (block_starts < prefix_len)

            if sparse_mask.any():
                sparse_starts = block_starts[sparse_mask]
                sparse_ends = block_ends[sparse_mask]
                sparse_positions = torch.cat([
                    torch.arange(s.item(), e.item(), device=device)
                    for s, e in zip(sparse_starts, sparse_ends)
                ])
                token_positions = torch.cat([dense_positions, sparse_positions])
            else:
                token_positions = dense_positions

            phys = req_to_token[req_idx, token_positions]
            all_phys_indices.append(phys)
            filtered_lens.append(len(phys))
        else:
            orig_start = original_kv_indptr[b].item()
            orig_end = original_kv_indptr[b + 1].item()
            orig_len = orig_end - orig_start
            if orig_len > 0:
                all_phys_indices.append(original_kv_indices[orig_start:orig_end])
            filtered_lens.append(orig_len)

    filtered_kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for b in range(batch_size):
        filtered_kv_indptr[b + 1] = filtered_kv_indptr[b] + filtered_lens[b]

    if all_phys_indices:
        filtered_kv_indices = torch.cat(all_phys_indices, dim=0).to(torch.int32)
        pad = torch.zeros(256, dtype=torch.int32, device=device)
        filtered_kv_indices = torch.cat([filtered_kv_indices, pad])
    else:
        filtered_kv_indices = torch.zeros(256, dtype=torch.int32, device=device)

    return filtered_kv_indices, filtered_kv_indptr
