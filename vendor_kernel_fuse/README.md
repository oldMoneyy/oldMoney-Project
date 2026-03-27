# vendor_kernel_fuse

Fused CUDA kernels for MiniCPM-SALA model inference optimization.

## Build

```bash
cd /opt/vendor_kernel_fuse   # or wherever this dir is on the target machine
pip install -e .
```

## Kernels

| Kernel | What it fuses | Impact |
|--------|--------------|--------|
| `fused_fp8_gather_dequant` | `kv_cache[pages].to(bf16) * scale` → 1 kernel | ~5% prefill |
| `fused_topk_block_table_decode` | `topk + sort + block_table_v3` → 1 kernel | ~5-10% decode@bs64 |
| `fused_sigmoid_gate` | `x * sigmoid(gate)` with vectorized 128-bit loads | ~2% decode (replaces Triton) |

## Usage

```python
import fused_kernel_extension

# FP8 gather + dequant
mini_k = fused_kernel_extension.fused_fp8_gather_dequant(k_cache, used_pages, k_descale)

# Fused topk + block table (decode only)
sparse_pt = fused_kernel_extension.fused_topk_block_table_decode(block_score, page_table, cache_seqlens, topk)

# Sigmoid gate
out = fused_kernel_extension.fused_sigmoid_gate(attn_output, gate_output)
```
