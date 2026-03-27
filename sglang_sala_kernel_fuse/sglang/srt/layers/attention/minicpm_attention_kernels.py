"""Attention kernel abstraction for MiniCPM backend.

This module provides a unified interface for different attention kernels,
allowing MiniCPM to use either flash attention or flashinfer as the backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_world_size
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
    )

    from sglang.srt.layers.radix_attention import RadixAttention

from sglang.srt.layers.attention.minicpm_sparse_kernels import (
    convert_sparse_page_table_to_flashinfer,
)


@dataclass
class AttentionParams:
    """Parameters for attention computation.

    This dataclass contains all parameters needed for both flash attention
    and flashinfer backends.
    """

    # Query, key, value tensors
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor

    # Page table for paged KV cache
    page_table: torch.Tensor

    # Sequence lengths
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k_new: torch.Tensor
    max_seqlen_q: int

    # Attention parameters
    softmax_scale: float
    causal: bool = True
    window_size: Tuple[int, int] = (-1, -1)
    softcap: float = 0.0

    # Quantization
    k_descale: Optional[torch.Tensor] = None
    v_descale: Optional[torch.Tensor] = None

    # Other parameters
    num_splits: int = 0
    fa_impl_ver: int = 3

    # Flashinfer-specific pre-converted metadata
    flashinfer_kv_indptr: Optional[torch.Tensor] = None
    flashinfer_kv_indices: Optional[torch.Tensor] = None
    flashinfer_kv_last_page_len: Optional[torch.Tensor] = None

    # Flashinfer decode wrapper (for CUDA graph mode)
    decode_wrapper: Optional["BatchDecodeWithPagedKVCacheWrapper"] = None


class AttentionKernel(ABC):
    """Abstract base class for attention kernels.

    This class defines the interface that all attention kernels must implement.
    """

    @abstractmethod
    def forward(
        self,
        params: AttentionParams,
        layer: RadixAttention,
    ) -> torch.Tensor:
        """Perform attention computation.

        Args:
            params: Attention parameters
            layer: The attention layer

        Returns:
            Attention output tensor
        """
        pass

    @abstractmethod
    def init_metadata(
        self,
        forward_batch,
        layer,
    ) -> Optional[object]:
        """Initialize backend-specific metadata.

        Args:
            forward_batch: The forward batch
            layer: The attention layer

        Returns:
            Backend-specific metadata object, or None if not needed
        """
        pass


class FlashAttentionKernel(AttentionKernel):
    """Flash Attention kernel implementation.

    This class wraps the flash_attn_with_kvcache function from sgl_kernel.
    """

    def __init__(self):
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        self.flash_attn_func = flash_attn_with_kvcache

    def forward(
        self,
        params: AttentionParams,
        layer: RadixAttention,
    ) -> torch.Tensor:
        """Perform attention computation using flash attention."""
        # Prepare kwargs based on fa_impl_ver
        kwargs = {}
        if params.fa_impl_ver != 3:
            kwargs["ver"] = params.fa_impl_ver

        return self.flash_attn_func(
            q=params.q,
            k_cache=params.k_cache,
            v_cache=params.v_cache,
            page_table=params.page_table,
            cache_seqlens=params.cache_seqlens,
            cu_seqlens_q=params.cu_seqlens_q,
            cu_seqlens_k_new=params.cu_seqlens_k_new,
            max_seqlen_q=params.max_seqlen_q,
            softmax_scale=params.softmax_scale,
            causal=params.causal,
            window_size=params.window_size,
            softcap=params.softcap,
            k_descale=params.k_descale,
            v_descale=params.v_descale,
            return_softmax_lse=False,
            num_splits=params.num_splits,
            **kwargs,
        )

    def init_metadata(
        self,
        forward_batch,
        layer,
    ) -> Optional[object]:
        """Flash attention doesn't need special metadata initialization."""
        return None


class FlashInferKernel(AttentionKernel):
    """FlashInfer kernel implementation.

    This class wraps the flashinfer attention wrappers.
    """

    def __init__(self, model_runner):
        self.device = model_runner.device
        self.page_size = model_runner.page_size
        self.model_dtype = model_runner.dtype  # dotv

        # dotv
        # Plan caching for sparse extend path (skip redundant begin_forward)
        self._cached_plan_kv_indptr = None
        self._cached_kv_indptr = None
        self._cached_kv_indices = None
        self._cached_kv_last_page_len = None
        # FA2 prefill doesn't support FP8 KV, FA3 needs SM90+
        # Fall back to flash_attn for prefill with FP8 KV on non-Hopper GPUs
        self.is_fp8_kv = "fp8" in model_runner.server_args.kv_cache_dtype.lower()
        # if self.is_fp8_kv:
        #     from sgl_kernel.flash_attn import flash_attn_with_kvcache
        #     self.flash_attn_prefill = flash_attn_with_kvcache
        ################################################

        # KV cache attributes
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.data_type = self.kv_cache_dtype

        # Model config attributes
        self.num_qo_heads = model_runner.model_config.num_attention_heads
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_tensor_model_parallel_world_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        # Query data type (same as KV cache dtype, but flashinfer uses separate parameters)
        self.q_data_type = self.kv_cache_dtype

        # Create workspace buffers for flashinfer
        workspace_size = envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get()
        self.decode_workspace = torch.empty(
            workspace_size,
            dtype=torch.uint8,
            device=self.device,
        )
        self.prefill_workspace = torch.empty(
            workspace_size,
            dtype=torch.uint8,
            device=self.device,
        )

        # Data format for flashinfer (Num heads, Head dim, Seq length)
        self.kv_layout = "NHD"

        # Wrappers will be created lazily
        self.decode_wrapper: Optional[BatchDecodeWithPagedKVCacheWrapper] = None
        self.prefill_wrapper: Optional[BatchPrefillWithPagedKVCacheWrapper] = None

        # Track if wrappers have been pre-planned for CUDA graph
        self.decode_wrapper_planned = False
        self.prefill_wrapper_planned = False

    def _get_or_create_decode_wrapper(
        self,
    ) -> "BatchDecodeWithPagedKVCacheWrapper":
        """Get or create the decode wrapper."""
        if self.decode_wrapper is None:
            from flashinfer import BatchDecodeWithPagedKVCacheWrapper

            # NOTE: use_tensor_cores=True is required for num_kv_heads=1 to avoid plan_info=None bug
            self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self.decode_workspace,
                self.kv_layout,
                use_tensor_cores=True,
            )
        return self.decode_wrapper

    def _get_or_create_prefill_wrapper(
        self,
    ) -> "BatchPrefillWithPagedKVCacheWrapper":
        """Get or create the prefill wrapper."""
        if self.prefill_wrapper is None:
            from flashinfer import BatchPrefillWithPagedKVCacheWrapper

            # dotv ###########################################################
            # FA2 doesn't support FP8 KV cache, use FA3 instead
            # is_fp8 = "fp8" in str(self.kv_cache_dtype).lower() or self.kv_cache_dtype in (
            #     torch.float8_e5m2, torch.float8_e4m3fn
            # )
            # backend = "fa3" if is_fp8 else "fa2"

            # self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            #     self.prefill_workspace,
            #     self.kv_layout,
            #     backend=backend,
            # )
            self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self.prefill_workspace,
                self.kv_layout,
                backend="fa2",
            )
            ####################################################################
        return self.prefill_wrapper
        
    def _safe_resize_flashinfer_buffers(self, wrapper, num_seqs, num_qo, num_kv_indices):
        """Safely resizes FlashInfer's internal buffers when handling huge sparse batches"""
        for buf_name, req_size in [
            ("_kv_lens_buffer", num_seqs),
            ("_paged_kv_indptr_buf", num_seqs + 1),
            ("_paged_kv_last_page_len_buf", num_seqs),
            ("_qo_indptr_buf", num_qo + 1),
            ("_paged_kv_indices_buf", num_kv_indices),
        ]:
            if hasattr(wrapper, buf_name) and getattr(wrapper, buf_name) is not None:
                t = getattr(wrapper, buf_name)
                if t.shape[0] < req_size:
                    # resize_ modifies tensor inplace without risking property setter complications
                    t.resize_(int(req_size * 1.2) + 1024)

    def plan_decode_wrapper(
        self,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_last_page_len: torch.Tensor,
    ) -> None:
        """Pre-plan the decode wrapper for CUDA graph compatibility.

        This method calls begin_forward with the given metadata to set up
        the flashinfer plan without calling forward. This must be called
        before CUDA graph capture.

        Args:
            kv_indptr: Indptr array for the sparse indices
            kv_indices: Flattened page indices
            kv_last_page_len: Last page length for each sequence
        """
        self.decode_wrapper_planned = False
        self._get_or_create_decode_wrapper()
        
        self._safe_resize_flashinfer_buffers(
            self.decode_wrapper,
            num_seqs=kv_indptr.shape[0] - 1,
            num_qo=kv_indptr.shape[0] - 1,
            num_kv_indices=kv_indices.shape[0]
        )
        
        self.decode_wrapper.begin_forward(
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
            non_blocking=True,
        )
        self.decode_wrapper_planned = True

    def plan_prefill_wrapper(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        causal: bool,
    ) -> None:
        """Pre-plan the prefill wrapper for CUDA graph compatibility.

        This method calls begin_forward with the given metadata to set up
        the flashinfer plan without calling forward. This must be called
        before CUDA graph capture.

        Args:
            qo_indptr: Query indptr
            kv_indptr: Indptr array for the sparse indices
            kv_indices: Flattened page indices
            kv_last_page_len: Last page length for each sequence
            causal: Whether to apply causal masking
        """
        self.prefill_wrapper_planned = False
        self._get_or_create_prefill_wrapper()
        valid_pages = kv_indptr[-1].item()
        kv_indices_valid = kv_indices[:valid_pages]
        
        self._safe_resize_flashinfer_buffers(
            self.prefill_wrapper,
            num_seqs=kv_indptr.shape[0] - 1,
            num_qo=qo_indptr.shape[0] - 1,
            num_kv_indices=valid_pages
        )
        
        self.prefill_wrapper.begin_forward(
            qo_indptr,
            kv_indptr,
            kv_indices_valid,
            kv_last_page_len,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            # q_data_type=self.q_data_type,
            q_data_type=self.model_dtype,
            kv_data_type=self.data_type,
            non_blocking=True,
            causal=causal,
        )
        self.prefill_wrapper_planned = True

    def forward(
        self,
        params: AttentionParams,
        layer: RadixAttention,
    ) -> torch.Tensor:
        """Perform attention computation using flashinfer."""
        # Determine if this is prefill or decode based on max_seqlen_q
        is_prefill = params.max_seqlen_q > 1

        # dotv #############################################
        # 针对 FP8 且是 Prefill 阶段：
        if is_prefill and self.is_fp8_kv:
            wrapper = self._get_or_create_prefill_wrapper()
            bs = params.cache_seqlens.shape[0]
            max_sparse_tokens = params.page_table.shape[1]
            
            kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=params.page_table.device)
            kv_indices = torch.zeros(bs * max_sparse_tokens, dtype=torch.int32, device=params.page_table.device)
            kv_last_page_len = torch.zeros(bs, dtype=torch.int32, device=params.page_table.device)
            
            kv_indptr, kv_indices, kv_last_page_len = convert_sparse_page_table_to_flashinfer(
                params.page_table, params.cache_seqlens, kv_indptr, kv_indices, kv_last_page_len,
            )
            
            # Truncate to valid elements to avoid FlashInfer buffer overflow with padded zeros
            valid_pages = kv_indptr[-1].item()
            kv_indices_valid = kv_indices[:valid_pages]

            # --- Fused FP8 gather + dequant (replaces 3 separate kernels) ---
            used_pages, new_kv_indices = torch.unique(kv_indices_valid, return_inverse=True)
            new_kv_indices = new_kv_indices.to(torch.int32)

            try:
                import fused_kernel_extension
                # Fused: gather + fp8→bf16 + scale in one kernel
                k_scale_tensor = params.k_descale if params.k_descale is not None else torch.ones(1, device=params.k_cache.device)
                v_scale_tensor = params.v_descale if params.v_descale is not None else torch.ones(1, device=params.v_cache.device)
                mini_k_cache = fused_kernel_extension.fused_fp8_gather_dequant(params.k_cache, used_pages, k_scale_tensor)
                mini_v_cache = fused_kernel_extension.fused_fp8_gather_dequant(params.v_cache, used_pages, v_scale_tensor)
            except ImportError:
                # Fallback: original 3-step approach
                mini_k_cache = params.k_cache[used_pages].to(self.model_dtype)
                mini_v_cache = params.v_cache[used_pages].to(self.model_dtype)
                if params.k_descale is not None:
                    mini_k_cache = mini_k_cache * params.k_descale
                if params.v_descale is not None:
                    mini_v_cache = mini_v_cache * params.v_descale
            # ----------------------------------------
            
            self._safe_resize_flashinfer_buffers(
                wrapper,
                num_seqs=kv_indptr.shape[0] - 1,
                num_qo=params.cu_seqlens_q.shape[0] - 1,
                num_kv_indices=new_kv_indices.shape[0]
            )

            wrapper.begin_forward(
                params.cu_seqlens_q,
                kv_indptr, new_kv_indices, kv_last_page_len,  # 传入重新映射的紧凑索引
                self.num_qo_heads, self.num_kv_heads,
                self.head_dim, self.page_size,
                q_data_type=self.model_dtype,
                kv_data_type=self.model_dtype,   # Mini-pool 已经是 BF16 了
                non_blocking=True,
                causal=params.causal,
            )
            return wrapper.forward(
                params.q,
                (mini_k_cache, mini_v_cache),    # 传入 Mini-pool
                causal=params.causal,
                sm_scale=params.softmax_scale,
                window_left=params.window_size[0] if params.window_size[0] != -1 else -1,
                logits_soft_cap=params.softcap if params.softcap > 0 else None,
            )
        # dotv #############################################
        # import time
        # do_profile = (not torch.cuda.is_current_stream_capturing()) and params.cache_seqlens.shape[0] > 100

        # FP8 KV + prefill: flashinfer FA2 doesn't support FP8, FA3 needs SM90+
        # Fall back to flash_attn which handles FP8 KV with k_descale/v_descale
        # if is_prefill and self.is_fp8_kv:
        #     from sgl_kernel.flash_attn import flash_attn_with_kvcache
            
        #     wrapper = self._get_or_create_prefill_wrapper()
        #     bs = params.cache_seqlens.shape[0]
        #     max_sparse_tokens = params.page_table.shape[1]
        #     kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=params.page_table.device)
        #     kv_indices = torch.zeros(bs * max_sparse_tokens, dtype=torch.int32, device=params.page_table.device)
        #     kv_last_page_len = torch.zeros(bs, dtype=torch.int32, device=params.page_table.device)
        #     kv_indptr, kv_indices, kv_last_page_len = convert_sparse_page_table_to_flashinfer(
        #         params.page_table, params.cache_seqlens, kv_indptr, kv_indices, kv_last_page_len,
        #     )
        #     wrapper.begin_forward(
        #         params.cu_seqlens_q,
        #         kv_indptr, kv_indices, kv_last_page_len,
        #         self.num_qo_heads, self.num_kv_heads,
        #         self.head_dim, self.page_size,
        #         q_data_type=self.model_dtype,
        #         kv_data_type=self.model_dtype,
        #         non_blocking=True,
        #         causal=params.causal,
        #     )
        #     return wrapper.forward(
        #         params.q,
        #         (params.k_cache.to(self.model_dtype), params.v_cache.to(self.model_dtype)),
        #         causal=params.causal,
        #         sm_scale=params.softmax_scale,
        #         window_left=params.window_size[0] if params.window_size[0] != -1 else -1,
        #         logits_soft_cap=params.softcap if params.softcap > 0 else None,
        #     )
        # dotv #############################################


        # CUDA graph mode: use the pre-configured wrapper from params
        if params.decode_wrapper is not None and not is_prefill:
            wrapper = params.decode_wrapper
            # For CUDA graph mode, update wrapper's internal buffers
            # Convert sparse_page_table to flashinfer format
            bs = params.cache_seqlens.shape[0]
            max_sparse_tokens = params.page_table.shape[1]

            # Get wrapper's internal buffer shapes to avoid reallocation
            kv_indptr_shape = wrapper._paged_kv_indptr_buf.shape
            kv_indices_shape = wrapper._paged_kv_indices_buf.shape
            kv_last_page_len_shape = wrapper._paged_kv_last_page_len_buf.shape

            # Create temporary buffers for conversion (reuse if possible)
            if (
                kv_indptr_shape[0] >= bs + 1
                and kv_indices_shape[0] >= bs * max_sparse_tokens
                and kv_last_page_len_shape[0] >= bs
            ):
                # Use wrapper's internal buffers directly (no allocation)
                kv_indptr = wrapper._paged_kv_indptr_buf
                kv_indices = wrapper._paged_kv_indices_buf
                kv_last_page_len = wrapper._paged_kv_last_page_len_buf
            else:
                # Allocate new buffers if wrapper's buffers are too small
                kv_indptr = torch.zeros(
                    bs + 1, dtype=torch.int32, device=params.page_table.device
                )
                kv_indices = torch.zeros(
                    bs * max_sparse_tokens,
                    dtype=torch.int32,
                    device=params.page_table.device,
                )
                kv_last_page_len = torch.zeros(
                    bs, dtype=torch.int32, device=params.page_table.device
                )

            # Convert sparse_page_table to flashinfer format
            kv_indptr_, kv_indices_, kv_last_page_len_ = (
                convert_sparse_page_table_to_flashinfer(
                    params.page_table,
                    params.cache_seqlens,
                    kv_indptr,
                    kv_indices,
                    kv_last_page_len,
                )
            )

            # Update wrapper's internal buffers with converted data
            #wrapper._paged_kv_indptr_buf.copy_(kv_indptr)
            #wrapper._paged_kv_indices_buf.copy_(kv_indices)
            #wrapper._paged_kv_last_page_len_buf.copy_(kv_last_page_len)
        else:
            if is_prefill:
                wrapper = self._get_or_create_prefill_wrapper()
            else:
                wrapper = self._get_or_create_decode_wrapper()
            
            if (
                params.flashinfer_kv_indptr is not None
                and params.flashinfer_kv_indices is not None
                and params.flashinfer_kv_last_page_len is not None
                and params.flashinfer_kv_indptr.shape[0] == params.cache_seqlens.shape[0] + 1
            ):
                kv_indptr = params.flashinfer_kv_indptr
                kv_indices = params.flashinfer_kv_indices
                kv_last_page_len = params.flashinfer_kv_last_page_len
                using_preconverted = True
            else:
                using_preconverted = False
                bs = params.cache_seqlens.shape[0]
                max_sparse_tokens = params.page_table.shape[1]

                # Reuse pre-allocated buffers if correctly sized
                if (self._cached_kv_indptr is not None
                    and self._cached_kv_indptr.shape[0] == bs + 1
                    and self._cached_kv_indices.shape[0] >= bs * max_sparse_tokens):
                    kv_indptr = self._cached_kv_indptr
                    kv_indices = self._cached_kv_indices
                    kv_last_page_len = self._cached_kv_last_page_len
                else:
                    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=params.page_table.device)
                    kv_indices = torch.zeros(bs * max_sparse_tokens, dtype=torch.int32, device=params.page_table.device)
                    kv_last_page_len = torch.zeros(bs, dtype=torch.int32, device=params.page_table.device)
                    self._cached_kv_indptr = kv_indptr
                    self._cached_kv_indices = kv_indices
                    self._cached_kv_last_page_len = kv_last_page_len

                # Convert page table — updates kv_indptr, kv_indices, kv_last_page_len in-place
                kv_indptr, kv_indices, kv_last_page_len = (
                    convert_sparse_page_table_to_flashinfer(
                        params.page_table,
                        params.cache_seqlens,
                        kv_indptr,
                        kv_indices,
                        kv_last_page_len,
                    )
                )

                # Check if we can skip begin_forward (same batch structure as previous call)
                # kv_indptr encodes the ragged batch structure — if it matches, the plan is reusable
                # FlashInfer holds kv_indices by reference, so in-place updates are visible
                can_reuse_plan = False
                if (not is_prefill
                    and self._cached_plan_kv_indptr is not None
                    and self._cached_plan_kv_indptr.shape == kv_indptr.shape
                    and torch.equal(self._cached_plan_kv_indptr, kv_indptr)):
                    can_reuse_plan = True

                if not can_reuse_plan:
                    if is_prefill:
                        valid_pages = kv_indptr[-1].item()
                        kv_indices_valid = kv_indices[:valid_pages]
                        qo_indptr = params.cu_seqlens_q
                        
                        self._safe_resize_flashinfer_buffers(
                            wrapper,
                            num_seqs=kv_indptr.shape[0] - 1,
                            num_qo=qo_indptr.shape[0] - 1,
                            num_kv_indices=valid_pages
                        )
                        
                        wrapper.begin_forward(
                            qo_indptr,
                            kv_indptr, kv_indices_valid, kv_last_page_len,
                            self.num_qo_heads, self.num_kv_heads,
                            self.head_dim, self.page_size,
                            q_data_type=self.model_dtype,
                            kv_data_type=self.data_type,
                            non_blocking=True,
                            causal=params.causal,
                        )
                    else:
                        self._safe_resize_flashinfer_buffers(
                            wrapper,
                            num_seqs=kv_indptr.shape[0] - 1,
                            num_qo=kv_indptr.shape[0] - 1,
                            num_kv_indices=kv_indices.shape[0]
                        )
                        wrapper.begin_forward(
                            kv_indptr, kv_indices, kv_last_page_len,
                            self.num_qo_heads, self.num_kv_heads,
                            self.head_dim, self.page_size,
                            q_data_type=self.model_dtype,
                            kv_data_type=self.data_type,
                            non_blocking=True,
                        )
                        # Cache the plan structure for reuse
                        self._cached_plan_kv_indptr = kv_indptr.clone()

            if not using_preconverted and not is_prefill:
                pass  # plan already handled above
            elif using_preconverted:
                if is_prefill:
                    valid_pages = kv_indptr[-1].item()
                    kv_indices_valid = kv_indices[:valid_pages]
                    qo_indptr = params.cu_seqlens_q
                    
                    self._safe_resize_flashinfer_buffers(
                        wrapper,
                        num_seqs=kv_indptr.shape[0] - 1,
                        num_qo=qo_indptr.shape[0] - 1,
                        num_kv_indices=valid_pages
                    )
                    
                    wrapper.begin_forward(
                        qo_indptr,
                        kv_indptr, kv_indices_valid, kv_last_page_len,
                        self.num_qo_heads, self.num_kv_heads,
                        self.head_dim, self.page_size,
                        q_data_type=self.model_dtype,
                        kv_data_type=self.data_type,
                        non_blocking=True,
                        causal=params.causal,
                    )
                else:
                    self._safe_resize_flashinfer_buffers(
                        wrapper,
                        num_seqs=kv_indptr.shape[0] - 1,
                        num_qo=kv_indptr.shape[0] - 1,
                        num_kv_indices=kv_indices.shape[0]
                    )
                    
                    wrapper.begin_forward(
                        kv_indptr, kv_indices, kv_last_page_len,
                        self.num_qo_heads, self.num_kv_heads,
                        self.head_dim, self.page_size,
                        q_data_type=self.model_dtype,
                        kv_data_type=self.data_type,
                        non_blocking=True,
                    )
        
        # Perform attention
        q_data = params.q
        k_data = (params.k_cache, params.v_cache)

        if is_prefill:
            # Prefill mode: use prefill wrapper
            # flashinfer's forward doesn't need cu_seqlens, they are set in begin_forward
            o = wrapper.forward(
                q_data,
                k_data,
                causal=params.causal,
                sm_scale=params.softmax_scale,
                window_left=(
                    params.window_size[0] if params.window_size[0] != -1 else -1
                ),
                logits_soft_cap=params.softcap if params.softcap > 0 else None,
            )
        else:
            # Decode mode: use decode wrapper
            o = wrapper.forward(
                q_data,
                k_data,
                sm_scale=params.softmax_scale,
                logits_soft_cap=params.softcap if params.softcap > 0 else None,
                k_scale=params.k_descale if self.is_fp8_kv else None,  # dotv
                v_scale=params.v_descale if self.is_fp8_kv else None,  # dotv
            )

        # dotv
        # if do_profile:
        #     torch.cuda.synchronize()
        #     _t4 = time.perf_counter()
            # print(f"    [FlashInfer] alloc={1000*(_t1-_t0):.1f}ms convert={1000*(_t2-_t1):.1f}ms plan={1000*(_t3-_t2):.1f}ms forward={1000*(_t4-_t3):.1f}ms bs={params.cache_seqlens.shape[0]} is_prefill={is_prefill}", flush=True)
        # dotv
        
        return o

    def init_metadata(
        self,
        forward_batch,
        layer,
    ) -> Optional[object]:
        """Initialize flashinfer-specific metadata.

        For flashinfer, the metadata is initialized per forward call via begin_forward,
        so this method is a no-op.
        """
        return None


def create_attention_kernel(
    kernel_type: str,
    model_runner,
) -> AttentionKernel:
    """Factory function to create the appropriate attention kernel.

    Args:
        kernel_type: Type of kernel to create ('flash_attn' or 'flashinfer')
        model_runner: The model runner instance

    Returns:
        An instance of the requested attention kernel

    Raises:
        ValueError: If kernel_type is not recognized
    """
    if kernel_type == "flash_attn":
        return FlashAttentionKernel()
    elif kernel_type == "flashinfer":
        return FlashInferKernel(model_runner)
    else:
        raise ValueError(f"Unknown attention kernel type: {kernel_type}")