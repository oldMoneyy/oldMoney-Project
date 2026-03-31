# Uncommented version is for NVFP4 Lightning Attn + FP16 Minicpm Attn Architecture
# Also supports Full NVFP4 and Late-Layer Protected BF16 dynamically!

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inference-only MiniCPM model compatible with HuggingFace weights."""

import copy
import math
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.hybrid_linear_attn_backend import SimpleGLAAttnBackend
from sglang.srt.layers.attention.minicpm_sparse_utils import (
    SparseBatchAnalyzer,
    SparseConfig,
    SparseMetadata,
    SparseMetadataBuilder,
)
from sglang.srt.layers.fused_kernels import fused_scale_add_rmsnorm, fused_sigmoid_mul, fused_qk_rmsnorm, fused_rmsnorm_sigmoid_mul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix


class MiniCPMMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MiniCPMAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        attn_use_rope: bool = True,
        use_output_gate: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.attn_use_rope = attn_use_rope
        self.use_output_gate = use_output_gate

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        if self.attn_use_rope:
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
            )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        if self.use_output_gate:
            self.o_gate = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("o_gate", prefix),
            )

        self.layer_id = layer_id

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)

        # Pre-compute output gate while hidden_states is still hot in L2 cache
        if self.use_output_gate:
            o_gate_output, _ = self.o_gate(hidden_states)

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.attn_use_rope:
            orig_dtype = q.dtype
            q, k = q.float(), k.float()
            q, k = self.rotary_emb(positions, q, k)
            q, k = q.to(orig_dtype), k.to(orig_dtype)

        attn_output = self.attn(q, k, v, forward_batch)

        if self.use_output_gate:
            fused_sigmoid_mul(attn_output, o_gate_output)

        output, _ = self.o_proj(attn_output)
        return output


class MiniCPMLightningMixer(nn.Module):
    """Lightning attention mixer that uses SimpleGLAAttnBackend.

    This is a wrapper that prepares inputs for the backend and handles
    the QKV projection, normalization, RoPE, and output processing,
    while delegating the Simple GLA kernel calls to SimpleGLAAttnBackend.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_rope: bool = True,
        use_output_gate: bool = False,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        use_output_norm: bool = False,
        qk_norm: bool = True,
        rope_head_dim: Optional[int] = None,
        scale: str = "1/sqrt(d)",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        if scale == "1/sqrt(d)":
            self.scale = self.head_dim ** (-0.5)
        elif scale == "1/d":
            self.scale = self.head_dim ** (-1.0)
        else:
            self.scale = 1.0
        self.use_output_gate = use_output_gate
        self.attention_bias = attention_bias
        self.rms_norm_eps = rms_norm_eps
        self.use_rope = use_rope
        self.qk_norm = qk_norm
        self.use_output_norm = use_output_norm
        self.rope_head_dim = (
            rope_head_dim if rope_head_dim is not None else self.head_dim
        )
        assert self.rope_head_dim <= self.head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        if self.use_output_norm:
            self.o_norm = RMSNorm(self.num_heads * self.head_dim, eps=self.rms_norm_eps)

        if self.use_output_gate:
            self.z_proj = ColumnParallelLinear(
                self.hidden_size,
                self.total_num_heads * self.head_dim,
                bias=self.attention_bias,
                quant_config=quant_config,
                prefix=add_prefix("z_proj", prefix),
            )

        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=self.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=self.rms_norm_eps)

        if self.use_rope:
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
            )

        self.layer_id = layer_id
        self.state_shape = (self.num_kv_heads, self.head_dim, self.head_dim)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)

        # Pre-compute gate while hidden_states is still hot in L2 cache
        if self.use_output_gate:
            z, _ = self.z_proj(hidden_states)

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.qk_norm:
            q = q.reshape(-1, self.head_dim)
            k = k.reshape(-1, self.head_dim)
            fused_qk_rmsnorm(q, k, self.q_norm.weight.data, self.k_norm.weight.data, self.q_norm.variance_epsilon)

        if self.use_rope:
            q = q.reshape(-1, self.num_heads * self.head_dim)
            k = k.reshape(-1, self.num_kv_heads * self.head_dim)
            orig_dtype = q.dtype
            q, k = q.float(), k.float()
            q, k = self.rotary_emb(positions, q, k)
            q, k = q.to(orig_dtype), k.to(orig_dtype)

        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)

        # ALWAYS unsqueeze to (1, total_tokens, h, d)
        q = q.unsqueeze(0)  # (1, total_tokens, num_heads, head_dim)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

        # Get backend from forward batch
        attn_backend = forward_batch.attn_backend
        if not hasattr(attn_backend, "linear_attn_backend"):
            raise RuntimeError(
                "SimpleGLAAttnBackend requires HybridLinearAttnBackend but got "
                f"{type(attn_backend).__name__}. This mixer should only be used for "
                "MiniCPM hybrid models."
            )

        linear_attn_backend = attn_backend.linear_attn_backend
        if not isinstance(linear_attn_backend, SimpleGLAAttnBackend):
            raise RuntimeError(
                f"Expected SimpleGLAAttnBackend but got {type(linear_attn_backend).__name__}"
            )

        # Prepare backend inputs
        # Backend expects q, k, v, forward_batch, layer_id
        # It will handle state loading/saving internally
        o = linear_attn_backend.forward(
            q=q,
            k=k,
            v=v,
            forward_batch=forward_batch,
            layer_id=self.layer_id,
            output_attentions=False,
        )

        o = o.reshape(-1, self.num_heads * self.head_dim)

        # Fused: rmsnorm + sigmoid gate in single kernel (saves 1 memory round-trip)
        if self.use_output_gate and self.use_output_norm:
            fused_rmsnorm_sigmoid_mul(o, z, self.o_norm.weight.data, self.o_norm.variance_epsilon)
        elif self.use_output_norm:
            o = self.o_norm(o)
        elif self.use_output_gate:
            fused_sigmoid_mul(o, z)

        y, _ = self.o_proj(o)
        return y


class MiniCPMDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if quant_config is not None and layer_id == 0:
            print(f"quant_config type={type(quant_config).__name__}, get_name={quant_config.get_name() if hasattr(quant_config, 'get_name') else 'NO METHOD'}")
        
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        if hasattr(config, "mixer_types") and config.mixer_types is not None:
            self.mixer_type = config.mixer_types[layer_id]
        else:
            self.mixer_type = "minicpm4"

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        # ------------------------------------------------------------------------
        # DYNAMIC QUANTIZATION ROUTING:
        # Reads `exclude_modules` from the config to automatically support BOTH:
        # 1. Fully NVFP4 quantized models
        # 2. Mixed-precision models (e.g., NVFP4 MLP + BF16 Attention)
        # 3. Layer-protected models (e.g., keeping layers 26-28 entirely in BF16)
        # ------------------------------------------------------------------------
        attn_quant_config = quant_config
        mlp_quant_config = quant_config

        if quant_config is not None:
            # Safely extract exclude_modules from the model config
            exclude_modules = []
            if hasattr(config, "quantization_config"):
                q_cfg = config.quantization_config
                if isinstance(q_cfg, dict):
                    exclude_modules = q_cfg.get("exclude_modules", [])
                elif hasattr(q_cfg, "exclude_modules"):
                    exclude_modules = q_cfg.exclude_modules
                    
            if not exclude_modules and hasattr(config, "hf_quant_config"):
                h_cfg = config.hf_quant_config
                if isinstance(h_cfg, dict):
                    exclude_modules = h_cfg.get("exclude_modules", [])
                elif hasattr(h_cfg, "exclude_modules"):
                    exclude_modules = h_cfg.exclude_modules

            # Helper to check if a block (attn or mlp) is excluded.
            def is_excluded(target_prefix: str) -> bool:
                for ex in exclude_modules:
                    ex_clean = ex.replace(".*", "")
                    # Case 1: Broad exclusion (e.g., ex="model.layers.26", target="model.layers.26.self_attn")
                    if target_prefix.startswith(ex_clean):
                        return True
                    # Case 2: Narrow exclusion covering the target (e.g., ex="model.layers.0.self_attn.q_proj", target="model.layers.0.self_attn")
                    if ex_clean.startswith(target_prefix):
                        return True
                return False

            layer_prefix = f"model.layers.{layer_id}"
            
            # Check if Attention or MLP specifically are excluded
            if is_excluded(f"{layer_prefix}.self_attn"):
                attn_quant_config = None
            if is_excluded(f"{layer_prefix}.mlp"):
                mlp_quant_config = None

            # Legacy GPTQ override (forces uniform config)
            if hasattr(quant_config, "get_name") and quant_config.get_name().startswith("gptq"):
                attn_quant_config = quant_config
                mlp_quant_config = quant_config
        # ------------------------------------------------------------------------

        if self.mixer_type == "minicpm4":
            self.self_attn = MiniCPMAttention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                layer_id=layer_id,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=attn_quant_config,
                attn_use_rope=(
                    config.attn_use_rope if hasattr(config, "attn_use_rope") else True
                ),
                use_output_gate=(
                    config.attn_use_output_gate
                    if hasattr(config, "attn_use_output_gate")
                    else False
                ),
                prefix=add_prefix("self_attn", prefix),
            )
        elif self.mixer_type in ["lightning", "lightning_attn", "lightning-attn"]:
            assert (
                config.head_dim is not False
            ), "head_dim must be provided for LightningAttention"
            self.self_attn = MiniCPMLightningMixer(
                hidden_size=self.hidden_size,
                num_heads=config.lightning_nh,
                num_kv_heads=config.lightning_nkv,
                head_dim=config.lightning_head_dim,
                layer_id=layer_id,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=attn_quant_config,
                use_rope=config.lightning_use_rope,
                use_output_gate=config.use_output_gate,
                attention_bias=config.attention_bias,
                rms_norm_eps=config.rms_norm_eps,
                use_output_norm=config.use_output_norm,
                qk_norm=config.qk_norm,
                scale=config.lightning_scale,
                prefix=add_prefix("self_attn", prefix),
            )
        else:
            raise ValueError(f"Unsupported mixer type: {self.mixer_type}")
            
        self.mlp = MiniCPMMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=mlp_quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # Precompute the residual scaling factor once
        self._residual_scale = config.scale_depth / math.sqrt(config.num_hidden_layers)

    def _compute_topk(self, forward_batch, base_metadata, sparse_metadata):
        """Compute TopK indices for sparse attention.

        For decode mode, TopK is simple: just use the precomputed sparse_page_table.
        For prefill mode, we need to compute TopK using kernel calls (deferred for now).

        Args:
            forward_batch: Forward batch
            base_metadata: Base metadata
            sparse_metadata: SparseMetadata to update with topk_indices
        """
        if forward_batch.forward_mode.is_decode_or_idle():
            # Decode path: TopK is just the precomputed page table
            sparse_metadata.topk_indices = base_metadata.sparse_page_table
        else:
            # Prefill path: Complex - needs kernel calls with compressed K1/K2
            # For now, leave topk_indices as None, backend will compute it
            # TODO: Implement full TopK computation in prefill mode
            pass

    # dotv ######################################################################
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # --- Optimized: fused scale+add+rmsnorm Triton kernels ---
        #
        # Per block: replaces separate mul_ + fused_add_rmsnorm (2 kernels)
        # with single fused_scale_add_rmsnorm (1 kernel).
        # Saves 2 x [T,H] memory round-trips per layer.

        if residual is not None:
            # Fuse: residual = hidden_states * scale + residual; hidden_states = rmsnorm(residual)
            # This merges the previous block's deferred scale + this block's add+norm
            fused_scale_add_rmsnorm(
                hidden_states, residual,
                self.input_layernorm.weight.data,
                self._residual_scale,
                self.input_layernorm.variance_epsilon,
            )
        else:
            # First layer: no prior residual to fuse
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fused: residual = hidden_states * scale + residual; hidden_states = rmsnorm(residual)
        fused_scale_add_rmsnorm(
            hidden_states, residual,
            self.post_attention_layernorm.weight.data,
            self._residual_scale,
            self.post_attention_layernorm.variance_epsilon,
        )

        # MLP
        hidden_states = self.mlp(hidden_states)

        # MLP output with scale — deferred to next layer's fused_scale_add_rmsnorm
        # (no separate *= here, the scale is baked into the next fused kernel call)
        return hidden_states, residual
    # dotv ######################################################################


class MiniCPMModel(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(
            [
                MiniCPMDecoderLayer(
                    config,
                    i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
            hidden_states.mul_(self.config.scale_emb)
        else:
            hidden_states = input_embeds
        residual = None

        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        # Final: fuse last layer's deferred scale + residual-add + final norm
        fused_scale_add_rmsnorm(
            hidden_states, residual,
            self.norm.weight.data,
            self.layers[-1]._residual_scale,
            self.norm.variance_epsilon,
        )
        return hidden_states


class MiniCPMForCausalLM(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.num_experts = getattr(self.config, "num_experts", 0)
        self.quant_config = quant_config
        self.model = MiniCPMModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        # self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if not self.config.tie_word_embeddings:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                prefix=add_prefix("lm_head", prefix),
            )

        self.scale_width = self.config.hidden_size / self.config.dim_model_base

        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is not None:
            input_embeds = input_embeds * self.config.scale_emb
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        hidden_states.div_(self.scale_width)
        if self.config.tie_word_embeddings:
            lm_head = self.model.embed_tokens
        else:
            lm_head = self.lm_head
        return self.logits_processor(input_ids, hidden_states, lm_head, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        expert_params_mapping = [
            # (param_name, weight_name, expert_id)
            (
                "ws" if weight_name in ["w1", "w3"] else "w2s",
                f"experts.{expert_id}.{weight_name}.weight",
                expert_id,
            )
            for expert_id in range(self.num_experts)
            for weight_name in ["w1", "w2", "w3"]
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]

                # ====================================================================
                # [NVFP4 Ultimate Fix]: 捕获 SGLang 合并层的底层切片 Bug，手动填入显存
                # ====================================================================
                try:
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight, shard_id)
                except AssertionError as e:
                    # 如果 SGLang 报形状不匹配崩溃了，我们直接接管！
                    if param.data.numel() == 1:
                        # 处理标量 (如 input_scale / weight_scale_2)
                        lw = loaded_weight.to(param.data.device).reshape(param.data.shape)
                        if param.data.item() == 0.0 or param.data.item() == 1.0:
                            param.data.copy_(lw)
                        else:
                            param.data.copy_(torch.max(param.data, lw))
                    else:
                        # 处理张量 (如 weight_scale)，手动拼接 gate 和 up
                        if isinstance(shard_id, int):
                            dim0 = loaded_weight.shape[0]
                            param.data[shard_id * dim0 : (shard_id + 1) * dim0].copy_(loaded_weight)
                        else:
                            # 理论上只有 gate_up_proj 会进这里，如果是 QKV 崩溃则抛出
                            raise e
                # ====================================================================
                break
            else:
                for param_name, weight_name, expert_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param, loaded_weight, weight_name, expert_id=expert_id
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

class MiniCPMSALAForCausalLM(MiniCPMForCausalLM):
    pass


# ==============================================================================
# EAGLE (v1) — Simple MLP draft head (kept for backward compatibility)
# ==============================================================================

class MiniCPMEagleModel(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids) * self.config.scale_emb
        else:
            hidden_states = input_embeds
        x = torch.cat((hidden_states, forward_batch.spec_info.hidden_states), dim=-1)
        x = self.fc1(x)
        x = self.act(x)
        hidden_states = self.fc2(x) * (
            self.config.scale_depth / math.sqrt(self.config.num_hidden_layers)
        )
        return hidden_states


class MiniCPMSALAEagleForCausalLM(MiniCPMForCausalLM):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(config, quant_config, prefix)
        self.model = MiniCPMEagleModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        if not self.config.tie_word_embeddings:
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.hidden_size,
                org_num_embeddings=config.vocab_size,
                prefix=add_prefix("lm_head", prefix),
            )
        else:
            self.lm_head = self.model.embed_tokens
        self.scale_width = self.config.hidden_size / self.config.dim_model_base
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        super().load_weights(weights)

    def set_embed_and_head(self, embed, head):
        self.model.embed_tokens = embed
        if self.config.tie_word_embeddings:
            self.lm_head = embed
        else:
            self.lm_head = head


# ==============================================================================
# EAGLE3 — Transformer-based draft head (following llama_eagle3.py pattern)
# ==============================================================================

class MiniCPMDecoderLayerEagle3(MiniCPMDecoderLayer):
    """EAGLE3 decoder layer: takes concatenated [embed, hidden] as QKV input.

    Key differences from standard MiniCPMDecoderLayer:
    - QKV projection input is 2*hidden_size (embed + hidden concatenated)
    - Has hidden_norm for the hidden state branch
    - Forward signature includes separate embeds and hidden_states
    - Uses standard minicpm4 attention (RadixAttention with KV cache)
      NOT lightning attention
    """

    def __init__(
        self,
        config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # Force minicpm4 type (standard attention) regardless of config.mixer_types
        orig_mixer_types = getattr(config, "mixer_types", None)
        config.mixer_types = ["minicpm4"]

        super().__init__(config, layer_id=layer_id, quant_config=quant_config, prefix=prefix)

        # Restore original mixer_types
        config.mixer_types = orig_mixer_types

        # Override QKV to accept 2*hidden_size input (concatenated embed + hidden)
        self.self_attn.qkv_proj = QKVParallelLinear(
            2 * config.hidden_size,
            self.self_attn.head_dim,
            self.self_attn.total_num_heads,
            self.self_attn.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("self_attn.qkv_proj", prefix),
        )

        # Add hidden_norm for the hidden state branch
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Start with hidden_states as the residual stream
        residual = hidden_states

        # Normalize both branches
        embeds_normed = self.input_layernorm(embeds)
        hidden_normed = self.hidden_norm(hidden_states)

        # Concatenate for QKV: [embed, hidden] -> 2*hidden_size
        combined = torch.cat([embeds_normed, hidden_normed], dim=-1)

        # Self Attention (standard minicpm4 with RadixAttention + KV cache)
        attn_out = self.self_attn(
            positions=positions,
            hidden_states=combined,
            forward_batch=forward_batch,
        )

        # MiniCPM residual: residual + output * (scale_depth / sqrt(N))
        hidden_states = residual + attn_out * (
            self.config.scale_depth / math.sqrt(self.config.num_hidden_layers)
        )

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * (
            self.config.scale_depth / math.sqrt(self.config.num_hidden_layers)
        )

        return hidden_states, residual


class MiniCPMModelEagle3(nn.Module):
    """EAGLE3 model for MiniCPM-SALA.

    Architecture (following llama_eagle3.py):
    - embed_tokens: shared from target model
    - fc: projects concatenated target hidden states to hidden_size
    - midlayer: single MiniCPM decoder layer with [embed, hidden] QKV input
    - norm: final RMSNorm
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        # Determine input hidden size from target model
        if hasattr(config, "target_hidden_size"):
            self.hidden_size_in = config.target_hidden_size
        else:
            self.hidden_size_in = config.hidden_size

        # fc projects concatenated target hidden states to draft hidden_size
        # For EAGLE3 with 3 captured layers: input = 3 * target_hidden_size
        self.fc = torch.nn.Linear(
            self.hidden_size_in * 3,
            config.hidden_size,
            bias=getattr(config, "bias", False),
        )

        # Single decoder layer with standard attention (not lightning)
        self.midlayer = MiniCPMDecoderLayerEagle3(
            config, layer_id=0, quant_config=quant_config,
            prefix=add_prefix("midlayer", prefix),
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors=None,
    ) -> torch.Tensor:
        if input_embeds is None:
            embeds = self.embed_tokens(input_ids) * self.config.scale_emb
        else:
            embeds = input_embeds

        # Get hidden states from target model (via spec_info)
        hidden_states = forward_batch.spec_info.hidden_states

        # Project if dimensions don't match (EAGLE3: 3*hidden -> hidden)
        if hidden_states.shape[-1] != embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)

        # Handle idle batch
        if hidden_states.shape[0] == 0:
            return hidden_states, [hidden_states]

        # Run through the single decoder layer
        residual = None
        hidden_states, residual = self.midlayer(
            positions,
            embeds,
            hidden_states,
            forward_batch,
            residual,
        )

        # Apply final norm
        hidden_states_to_logits = self.norm(hidden_states)

        # Return both logits input and auxiliary hidden state for next draft step
        return hidden_states_to_logits, [hidden_states]


class MiniCPMSALAEagle3ForCausalLM(MiniCPMForCausalLM):
    """EAGLE3 CausalLM for MiniCPM-SALA.

    Follows the exact pattern of LlamaForCausalLMEagle3:
    - Single transformer decoder layer as draft head
    - Takes concatenated target hidden states as input
    - Uses standard attention (RadixAttention)
    - Shares embed_tokens and lm_head from target model
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config

        if self.config.num_hidden_layers != 1:
            raise ValueError("EAGLE3 currently only supports 1 layer")

        self.model = MiniCPMModelEagle3(
            config, quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        # Handle lm_head
        self.load_lm_head_from_target = False
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            draft_vocab_size = getattr(config, "draft_vocab_size", None)
            if draft_vocab_size is None:
                self.load_lm_head_from_target = True
                draft_vocab_size = config.vocab_size
                config.draft_vocab_size = draft_vocab_size
            self.lm_head = ParallelLMHead(
                draft_vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        config_ = copy.deepcopy(config)
        config_.vocab_size = getattr(config_, "draft_vocab_size", config_.vocab_size)
        self.logits_processor = LogitsProcessor(config_)

        self.scale_width = self.config.hidden_size / self.config.dim_model_base
        self.capture_aux_hidden_states = True
        self.hot_token_id = None
        self.num_experts = 0

    def set_embed_and_head(self, embed, head):
        """Set shared embed_tokens and lm_head from target model."""
        self.model.embed_tokens = embed
        if self.config.tie_word_embeddings:
            self.lm_head = embed
        else:
            self.lm_head = head

    def set_embed(self, embed):
        """Set shared embed_tokens from target model (EAGLE3 has its own lm_head)."""
        self.model.embed_tokens = embed

    def get_hot_token_id(self):
        return self.hot_token_id

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is not None:
            input_embeds = input_embeds * self.config.scale_emb

        model_output = self.model(input_ids, positions, forward_batch, input_embeds)

        if isinstance(model_output, tuple):
            hidden_states_to_logits, aux_hidden_states = model_output
        else:
            hidden_states_to_logits = model_output
            aux_hidden_states = None

        # Apply MiniCPM scale_width before logits
        hidden_states_to_logits = hidden_states_to_logits / self.scale_width

        return self.logits_processor(
            input_ids, hidden_states_to_logits, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        for name, loaded_weight in weights:
            if "d2t" in name:
                self.hot_token_id = loaded_weight + torch.arange(loaded_weight.shape[0])
                continue
            if "t2d" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param_name_full = f"model.{name}" if name not in params_dict else name
                if param_name_full in params_dict:
                    param = params_dict[param_name_full]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param_name_full = name if name in params_dict else f"model.{name}"
                if param_name_full in params_dict:
                    param = params_dict[param_name_full]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)


EntryClass = [
    MiniCPMForCausalLM,
    MiniCPMSALAForCausalLM,
    MiniCPMSALAEagleForCausalLM,
    MiniCPMSALAEagle3ForCausalLM,
]