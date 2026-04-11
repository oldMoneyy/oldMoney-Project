# Copyright 2025 SGLang Team
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

import torch

E2M1_MAX = 6.0
# Put constants directly on CUDA if available
_device = "cuda" if torch.cuda.is_available() else "cpu"
E2M1_VALUES = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=_device
)
E2M1_BOUNDS = torch.tensor(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=torch.float32, device=_device
)


class KVFP4QuantizeUtil:
    """Utility class for MXFP4 quantization and dequantization operations."""

    @staticmethod
    @torch.compile
    def batched_quantize(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to KVFP4 format
        Args:
            tensor: Input tensor of shape [B, M, N]

        Returns:
            quant_tensor: Quantized tensor of shape [B, M, N/2]
            scale_factors: Scale factors of shape [B, M*N/16]
        """
        b, m, n = tensor.shape

        # Reshape to [B, M*N/16, 16] for block-wise quantization
        reshaped = tensor.view(b, m * n // 16, 16)

        # Compute scale factors per block
        block_max = reshaped.abs().max(dim=-1, keepdim=True).values
        scale_exp = torch.ceil(torch.log2(torch.clamp(block_max / E2M1_MAX, min=1e-10)))
        scale_factors = (scale_exp + 127).squeeze(-1).to(torch.uint8)

        # Apply scaling
        scaled = reshaped / torch.exp2(scale_exp)

        # Quantize to FP4
        sign_bits = (scaled < 0).to(torch.uint8) << 3
        abs_vals = scaled.abs()

        # Pure tensor version (CUDA Graph safe)
        magnitude_bits = torch.sum(abs_vals.unsqueeze(-1) >= E2M1_BOUNDS, dim=-1)

        # Combine sign and magnitude
        fp4_vals = sign_bits + magnitude_bits.to(torch.uint8)

        # Pack two FP4 values into one uint8
        fp4_reshaped = fp4_vals.view(b, m, n)
        packed = (fp4_reshaped[..., 1::2] << 4) + fp4_reshaped[..., 0::2]

        return packed, scale_factors

    @staticmethod
    @torch.compile
    def batched_dequantize(
        quant_tensor: torch.Tensor,
        scale_factors: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        Dequantize KVFP4 tensor
        Args:
            quant_tensor: Quantized tensor of shape [B, M, N/2]
            scale_factors: Scale factors of shape [B, M*N/16]
            dtype: Target dtype for output

        Returns:
            Dequantized tensor of shape [B, M, N]
        """
        b, m, n_half = quant_tensor.shape
        n = n_half * 2

        # More efficient unpacking using bit operations
        fp4_vals = torch.empty(b, m, n, dtype=torch.uint8, device=quant_tensor.device)
        fp4_vals[..., 0::2] = quant_tensor & 0x0F
        fp4_vals[..., 1::2] = (quant_tensor >> 4) & 0x0F

        # Extract sign and magnitude
        sign_mask = (fp4_vals & 0x08) != 0
        magnitude_idx = fp4_vals & 0x07

        # Convert to float values
        float_vals = E2M1_VALUES[magnitude_idx.long()]
        float_vals = torch.where(sign_mask, -float_vals, float_vals)

        # Reshape for block-wise scaling
        reshaped = float_vals.view(b, m * n // 16, 16)

        # Apply scale factors
        scale_exp = scale_factors.float() - 127
        scaled = reshaped * torch.exp2(scale_exp.unsqueeze(-1))

        return scaled.view(b, m, n).to(dtype)


def dequant_fp4_to_fp8_inplace(packed, scale, out, chunk_size=131072):
    """Dequantize FP4 packed buffer to FP8 E4M3, writing into pre-allocated output.

    Args:
        packed: [T, H, D//2] uint8 (two FP4 values per byte)
        scale: [T, H*D//16] uint8 (E8M0 block scales)
        out: [T, H, D] uint8 (pre-allocated, will hold FP8 E4M3 data)
        chunk_size: tokens per chunk to avoid OOM on intermediates
    """
    T = packed.shape[0]
    if T == 0:
        return

    _lut = E2M1_VALUES  # [8] float32 on CUDA

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)

        p = packed[start:end]       # [cs, H, D//2] uint8
        s = scale[start:end]        # [cs, S] uint8

        # Unpack two FP4 values from each byte
        low = (p & 0x0F).to(torch.int64)
        high = ((p >> 4) & 0x0F).to(torch.int64)

        # Interleave into full dimension
        cs, H, half_D = p.shape
        D = half_D * 2
        full = torch.empty(cs, H, D, dtype=torch.int64, device=p.device)
        full[..., 0::2] = low
        full[..., 1::2] = high

        # Sign and magnitude
        sign = (full >> 3) & 1
        mag = full & 0x07

        # LUT lookup for magnitude -> float
        vals = _lut[mag]  # [cs, H, D] float32
        vals = torch.where(sign > 0, -vals, vals)

        # Apply block scales (16 elements per scale)
        vals_blocked = vals.reshape(cs, -1, 16)       # [cs, H*D//16, 16]
        scale_exp = s.float() - 127.0                  # [cs, H*D//16]
        vals_blocked = vals_blocked * torch.exp2(scale_exp.unsqueeze(-1))

        # Cast to FP8 E4M3 and store
        result = vals_blocked.reshape(cs, H, D)
        out[start:end] = result.to(torch.float8_e4m3fn).view(torch.uint8)
