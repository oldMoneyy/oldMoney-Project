# Specialized SimpleGLA fused recurrent kernel for inference decode
# Adapted from fla.ops.common.fused_recurrent
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# Optimizations over the generic kernel:
# - Only forward pass (no backward)
# - Supports USE_G_GAMMA (head-wise constant decay) directly
# - Autotune with larger BK/BV options for K=V=128
# - Removed unused GK, GV, REVERSE branches

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.op import exp


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['BK', 'BV', 'USE_G', 'USE_G_GAMMA'],
)
@triton.jit(do_not_specialize=['B', 'T'])
def simple_gla_fused_recurrent_fwd_kernel(
    q,
    k,
    v,
    g,
    g_gamma,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    B,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    all = B * T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_q = q + bos * H * K + i_h * K + o_k
    p_k = k + bos * H * K + i_h * K + o_k
    p_v = v + bos * H * V + i_h * V + o_v
    p_o = o + (i_k * all + bos) * H * V + i_h * V + o_v

    if USE_G:
        p_g = g + bos * H + i_h
    if USE_G_GAMMA:
        b_g_gamma = tl.load(g_gamma + i_h)

    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=m_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=m_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_h = b_h * exp(b_g)
        if USE_G_GAMMA:
            b_h = b_h * exp(b_g_gamma)
        b_h += b_k[:, None] * b_v[None, :]
        b_o = b_h * b_q[:, None]
        b_o = tl.sum(b_o, axis=0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_v)
        p_q += H * K
        p_k += H * K
        p_v += H * V
        p_o += H * V
        if USE_G:
            p_g += H

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=m_h)


def fused_recurrent_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor = None,
    g_gamma: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: torch.LongTensor = None,
) -> tuple:
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    # Use larger block sizes for better throughput
    BK = min(triton.next_power_of_2(K), 128)
    BV = min(triton.next_power_of_2(V), 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    h0 = initial_state
    ht = q.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    o = q.new_empty(NK, *v.shape, dtype=torch.float32)

    grid = (NV, NK, N * H)
    simple_gla_fused_recurrent_fwd_kernel[grid](
        q=q, k=k, v=v,
        g=g, g_gamma=g_gamma,
        o=o, h0=h0, ht=ht,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T, B=B, H=H, K=K, V=V,
        BK=BK, BV=BV,
        USE_G=g is not None,
        USE_G_GAMMA=g_gamma is not None,
    )
    o = o.sum(0)
    return o.to(q.dtype), ht
