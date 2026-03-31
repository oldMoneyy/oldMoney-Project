# Vendored SimpleGLA ops (inference-only, forward path)
# Adapted from fla.ops.simple_gla
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton

from sglang.srt.layers.attention.fla.chunk_h import chunk_fwd_h
from sglang.srt.layers.attention.fla.chunk_o import chunk_fwd_o
from sglang.srt.layers.attention.fla.simple_gla_recurrent import fused_recurrent_simple_gla


def chunk_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor = None,
    g_gamma: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor = None,
    head_first: bool = False,
    chunk_size: int = 64,
) -> tuple:
    if scale is None:
        scale = k.shape[-1] ** -0.5

    T = q.shape[1]
    if chunk_size is None:
        chunk_size = min(64, max(16, triton.next_power_of_2(T)))

    # No need for chunk_local_cumsum when using g_gamma (head-wise constant decay)
    # g_gamma is handled directly by the kernels via USE_G_GAMMA

    h, ht = chunk_fwd_h(
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=None,
        gv=None,
        h0=initial_state,
        output_final_state=output_final_state,
        states_in_fp32=False,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    return o.to(q.dtype), ht
