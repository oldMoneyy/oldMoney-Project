# -*- coding: utf-8 -*-
# Indexed-access fused recurrent kernel for Simple GLA.
# Reads/writes state directly from the MambaPool via pointer indirection.
# Eliminates gather+contiguous+scatter overhead per layer per step.

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0_source'] is not None,
    'STORE_FINAL_STATE': lambda args: args['STORE_FINAL_STATE'],
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [4, 8]
    ],
    key=['BK', 'BV'],
)
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_simple_gla_indexed_fwd_kernel(
    q,
    k,
    v,
    g_gamma,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_k, i_nh = (
        tl.program_id(0).to(tl.int64),
        tl.program_id(1).to(tl.int64),
        tl.program_id(2).to(tl.int64),
    )
    i_n, i_h = i_nh // H, i_nh % H

    all = B * T
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + bos * H * K + i_h * K + o_k
    p_k = k + bos * H * K + i_h * K + o_k
    p_v = v + bos * H * V + i_h * V + o_v
    p_o = o + (i_k * all + bos) * H * V + i_h * V + o_v

    b_g_gamma = tl.load(g_gamma + i_h)

    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n).to(tl.int64)
        if idx >= 0:
            p_h0 = h0_source + idx * H * K * V + i_h * K * V + o_k[:, None] * V + o_v[None, :]
            b_h += tl.load(p_h0, mask=m_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=m_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)

        b_h = b_h * tl.exp(b_g_gamma)
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], axis=0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_v)

        p_q += H * K
        p_k += H * K
        p_v += H * V
        p_o += H * V

    if STORE_FINAL_STATE:
        idx = tl.load(h0_indices + i_n).to(tl.int64)
        if idx >= 0:
            p_ht = h0_source + idx * H * K * V + i_h * K * V + o_k[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=m_h)


def fused_recurrent_simple_gla_indexed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_gamma: torch.Tensor,
    scale: float,
    h0_source: torch.Tensor,
    h0_indices: torch.Tensor,
    store_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    """Fused recurrent Simple GLA with indexed state pool access.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        g_gamma: [H] head-wise log-decay
        scale: attention scale
        h0_source: [pool_size, H, K, V] full state pool (float32)
        h0_indices: [N] int indices into h0_source per sequence
        store_final_state: write updated state back to h0_source
        cu_seqlens: [N+1] cumulative seq lengths for varlen

    Returns:
        o: [B, T, H, V]
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)

    o = q.new_empty(NK, *v.shape, dtype=torch.float32)

    grid = (NV, NK, N * H)
    fused_recurrent_simple_gla_indexed_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g_gamma=g_gamma,
        o=o,
        h0_source=h0_source,
        h0_indices=h0_indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        STORE_FINAL_STATE=store_final_state,
    )
    o = o.sum(0).to(q.dtype)
    return o
