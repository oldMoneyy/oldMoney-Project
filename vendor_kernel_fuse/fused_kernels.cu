/**
 * vendor_kernel_fuse: Fused CUDA kernels for MiniCPM-SALA inference
 *
 * Kernel 1: fused_fp8_gather_dequant
 *   Fuses: k_cache[pages].to(bf16) * k_descale into one kernel
 *   Impact: ~5% prefill improvement (eliminates gather + cast + multiply)
 *
 * Kernel 2: fused_topk_block_table
 *   Fuses: topk sort + block_table construction into one kernel
 *   Impact: ~5-10% decode improvement at high concurrency
 *
 * Kernel 3: fused_sigmoid_gate_cuda
 *   High-throughput x * sigmoid(gate) with vectorized memory access
 *   Impact: ~2% decode improvement (replaces Triton version for better perf)
 */

#include "static_switch.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

// ============================================================================
// Kernel 1: Fused FP8 Gather + Dequantize
//
// Replaces the Python code in FlashInferKernel.forward:
//   mini_k_cache = params.k_cache[used_pages].to(model_dtype)
//   mini_k_cache = mini_k_cache * params.k_descale
//
// This kernel does: output[i] = fp8_to_bf16(src[page_indices[i]]) * scale
// in a single pass, avoiding:
//   1. The gather (index_select) kernel
//   2. The fp8→bf16 cast kernel
//   3. The multiply kernel
// ============================================================================

__global__ void fused_fp8_gather_dequant_kernel(
    __nv_bfloat16 *__restrict__ output,       // [num_pages, page_size, num_heads, head_dim]
    const __nv_fp8_e4m3 *__restrict__ src,     // [total_pages, page_size, num_heads, head_dim]
    const int32_t *__restrict__ page_indices,  // [num_pages]
    const float *__restrict__ scale,           // scalar or [num_heads]
    const int64_t page_stride,                 // page_size * num_heads * head_dim
    const int64_t num_pages,
    const int64_t elements_per_page,           // page_size * num_heads * head_dim
    const bool per_head_scale,
    const int64_t num_heads,
    const int64_t head_dim) {

  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total_elements = num_pages * elements_per_page;
  if (tid >= total_elements) return;

  // Determine which page and offset within the page
  const int64_t page_idx = tid / elements_per_page;
  const int64_t offset_in_page = tid % elements_per_page;

  // Get the source page index
  const int32_t src_page = page_indices[page_idx];

  // Load FP8 value
  const __nv_fp8_e4m3 fp8_val = src[src_page * page_stride + offset_in_page];

  // Get scale (per-head or scalar)
  float s;
  if (per_head_scale) {
    const int64_t head_idx = (offset_in_page / head_dim) % num_heads;
    s = scale[head_idx];
  } else {
    s = scale[0];
  }

  // Convert FP8 → float → scale → BF16
  float f = static_cast<float>(fp8_val) * s;
  output[tid] = __float2bfloat16(f);
}

torch::Tensor fused_fp8_gather_dequant(
    const torch::Tensor &src,          // [total_pages, page_size, num_heads, head_dim], fp8_e4m3
    const torch::Tensor &page_indices, // [num_pages], int32
    const torch::Tensor &scale         // [1] or [num_heads], float32
) {
  TORCH_CHECK(src.is_cuda(), "src must be CUDA");
  TORCH_CHECK(page_indices.is_cuda(), "page_indices must be CUDA");
  TORCH_CHECK(scale.is_cuda(), "scale must be CUDA");

  const int64_t num_pages = page_indices.size(0);
  const int64_t page_size = src.size(1);
  const int64_t num_heads = src.size(2);
  const int64_t head_dim = src.size(3);
  const int64_t elements_per_page = page_size * num_heads * head_dim;
  const int64_t page_stride = elements_per_page;
  const bool per_head_scale = (scale.numel() > 1);

  // Output in BF16
  auto output = torch::empty({num_pages, page_size, num_heads, head_dim},
                              src.options().dtype(torch::kBFloat16));

  const int64_t total = num_pages * elements_per_page;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  fused_fp8_gather_dequant_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16 *>(output.data_ptr()),
      reinterpret_cast<const __nv_fp8_e4m3 *>(src.data_ptr()),
      page_indices.data_ptr<int32_t>(),
      scale.data_ptr<float>(),
      page_stride, num_pages, elements_per_page,
      per_head_scale, num_heads, head_dim);

  return output;
}


// ============================================================================
// Kernel 2: Fused TopK → Block Table
//
// Fuses the topk_idx.topk().sort() + get_block_table_v3 into one kernel.
// Instead of:
//   1. topk_idx = block_score.topk(topk).indices.sort().values  (Python)
//   2. sparse_page_table = get_block_table_v3(topk_idx, page_table, ...)  (CUDA)
//
// This kernel does both in one launch:
//   Given block_score, directly produce sparse_page_table.
//
// For decode mode (bs tokens, each with 1 query):
//   block_score shape: [kv_heads, bs, pooled_k_len]
//   Output: sparse_page_table [bs * kv_heads, topk * block_size]
// ============================================================================

constexpr int kHeadGroup = 2;
constexpr int kSparseBlockSize = 64;

// Simple insertion-sort based topk for small K (K <= 128)
template <int K>
__device__ void insertion_topk(
    const float *scores, int n,
    int *topk_indices, float *topk_values) {
  // Initialize with -inf
  #pragma unroll
  for (int i = 0; i < K; i++) {
    topk_values[i] = -1e30f;
    topk_indices[i] = -1;
  }

  for (int i = 0; i < n; i++) {
    float val = scores[i];
    if (val > topk_values[K - 1]) {
      // Insert into sorted list
      topk_values[K - 1] = val;
      topk_indices[K - 1] = i;
      // Bubble up
      for (int j = K - 2; j >= 0; j--) {
        if (topk_values[j + 1] > topk_values[j]) {
          float tv = topk_values[j];
          topk_values[j] = topk_values[j + 1];
          topk_values[j + 1] = tv;
          int ti = topk_indices[j];
          topk_indices[j] = topk_indices[j + 1];
          topk_indices[j + 1] = ti;
        } else {
          break;
        }
      }
    }
  }

  // Sort indices ascending (for cache-friendly page table access)
  for (int i = 0; i < K - 1; i++) {
    for (int j = i + 1; j < K; j++) {
      if (topk_indices[i] > topk_indices[j]) {
        int ti = topk_indices[i];
        topk_indices[i] = topk_indices[j];
        topk_indices[j] = ti;
        float tv = topk_values[i];
        topk_values[i] = topk_values[j];
        topk_values[j] = tv;
      }
    }
  }
}

template <int kSparseTopK>
__global__ void fused_topk_block_table_decode_kernel(
    const float *__restrict__ block_score,    // [kv_heads, bs, pooled_k_len]
    const int *__restrict__ page_table,       // [bs, max_seq_len]
    const int *__restrict__ cache_seqlens,    // [bs]
    int *__restrict__ out_page_table,         // [bs * kHeadGroup, kSparseTopK * kSparseBlockSize]
    const int bs,
    const int pooled_k_len,
    const int max_seq_len) {

  // Each block handles one (batch, head_group) pair
  const int batch_idx = blockIdx.x;
  const int head_group = blockIdx.y;
  if (batch_idx >= bs) return;

  // Step 1: TopK selection from block_score
  const float *scores = block_score +
      head_group * bs * pooled_k_len +
      batch_idx * pooled_k_len;

  __shared__ int topk_idx[kSparseTopK];
  __shared__ float topk_val[kSparseTopK];

  // Thread 0 does the topk (small K, not worth parallelizing further)
  if (threadIdx.x == 0) {
    insertion_topk<kSparseTopK>(scores, pooled_k_len, topk_idx, topk_val);
  }
  __syncthreads();

  // Step 2: Build block table from topk indices
  // Each thread handles one element of the output
  const int seqlen = cache_seqlens[batch_idx];
  const int out_row = batch_idx * kHeadGroup + head_group;
  const int out_stride = kSparseTopK * kSparseBlockSize;

  for (int i = threadIdx.x; i < kSparseTopK * kSparseBlockSize; i += blockDim.x) {
    const int block_idx = i / kSparseBlockSize;
    const int offset_in_block = i % kSparseBlockSize;
    const int sparse_block = topk_idx[block_idx];

    int result = 0;
    if (sparse_block >= 0) {
      const int token_pos = sparse_block * kSparseBlockSize + offset_in_block;
      if (token_pos < seqlen) {
        result = kHeadGroup * page_table[batch_idx * max_seq_len + token_pos] + head_group;
      }
    }
    out_page_table[out_row * out_stride + i] = result;
  }
}

torch::Tensor fused_topk_block_table_decode(
    const torch::Tensor &block_score,   // [kv_heads, bs, pooled_k_len]
    const torch::Tensor &page_table,    // [bs, max_seq_len]
    const torch::Tensor &cache_seqlens, // [bs]
    const int topk) {

  TORCH_CHECK(block_score.is_cuda(), "block_score must be CUDA");
  const int bs = block_score.size(1);
  const int pooled_k_len = block_score.size(2);
  const int max_seq_len = page_table.size(1);
  const int out_cols = topk * kSparseBlockSize;

  auto out = torch::zeros({bs * kHeadGroup, out_cols},
                           page_table.options());

  dim3 grid(bs, kHeadGroup);
  const int threads = min(1024, out_cols);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  VALUE_SPLITS_SWITCH(topk, kSparseTopK, [&]() {
    fused_topk_block_table_decode_kernel<kSparseTopK><<<grid, threads, 0, stream>>>(
        block_score.data_ptr<float>(),
        page_table.data_ptr<int>(),
        cache_seqlens.data_ptr<int>(),
        out.data_ptr<int>(),
        bs, pooled_k_len, max_seq_len);
  });

  return out;
}


// ============================================================================
// Kernel 3: High-throughput Fused Sigmoid Gate (CUDA)
//
// output = x * sigmoid(gate)
// Uses vectorized 4-element loads/stores for maximum memory bandwidth.
// This is faster than the Triton version for large tensors because
// it uses float4 vectorization (128-bit loads/stores).
// ============================================================================

__global__ void fused_sigmoid_gate_cuda_kernel(
    __nv_bfloat16 *__restrict__ output,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ gate,
    const int64_t n) {

  // Process 8 bf16 elements per thread (128 bits = 4 x int32)
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = gridDim.x * blockDim.x;

  // Vectorized path: process 8 bf16 at a time
  const int64_t n8 = n / 8 * 8;
  for (int64_t i = tid * 8; i < n8; i += stride * 8) {
    // Load 8 bf16 values (128 bits) as 4 int32s
    uint4 x_vec = *reinterpret_cast<const uint4 *>(x + i);
    uint4 g_vec = *reinterpret_cast<const uint4 *>(gate + i);

    __nv_bfloat16 *x_arr = reinterpret_cast<__nv_bfloat16 *>(&x_vec);
    __nv_bfloat16 *g_arr = reinterpret_cast<__nv_bfloat16 *>(&g_vec);

    uint4 o_vec;
    __nv_bfloat16 *o_arr = reinterpret_cast<__nv_bfloat16 *>(&o_vec);

    #pragma unroll
    for (int j = 0; j < 8; j++) {
      float xf = __bfloat162float(x_arr[j]);
      float gf = __bfloat162float(g_arr[j]);
      float sig = 1.0f / (1.0f + expf(-gf));
      o_arr[j] = __float2bfloat16(xf * sig);
    }

    *reinterpret_cast<uint4 *>(output + i) = o_vec;
  }

  // Scalar tail
  for (int64_t i = n8 + tid; i < n; i += stride) {
    float xf = __bfloat162float(x[i]);
    float gf = __bfloat162float(gate[i]);
    float sig = 1.0f / (1.0f + expf(-gf));
    output[i] = __float2bfloat16(xf * sig);
  }
}

// FP16 version
__global__ void fused_sigmoid_gate_cuda_kernel_fp16(
    __half *__restrict__ output,
    const __half *__restrict__ x,
    const __half *__restrict__ gate,
    const int64_t n) {

  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = gridDim.x * blockDim.x;

  const int64_t n8 = n / 8 * 8;
  for (int64_t i = tid * 8; i < n8; i += stride * 8) {
    uint4 x_vec = *reinterpret_cast<const uint4 *>(x + i);
    uint4 g_vec = *reinterpret_cast<const uint4 *>(gate + i);

    __half *x_arr = reinterpret_cast<__half *>(&x_vec);
    __half *g_arr = reinterpret_cast<__half *>(&g_vec);

    uint4 o_vec;
    __half *o_arr = reinterpret_cast<__half *>(&o_vec);

    #pragma unroll
    for (int j = 0; j < 8; j++) {
      float xf = __half2float(x_arr[j]);
      float gf = __half2float(g_arr[j]);
      float sig = 1.0f / (1.0f + expf(-gf));
      o_arr[j] = __float2half(xf * sig);
    }

    *reinterpret_cast<uint4 *>(output + i) = o_vec;
  }

  for (int64_t i = n8 + tid; i < n; i += stride) {
    float xf = __half2float(x[i]);
    float gf = __half2float(gate[i]);
    float sig = 1.0f / (1.0f + expf(-gf));
    output[i] = __float2half(xf * sig);
  }
}

torch::Tensor fused_sigmoid_gate_cuda(
    const torch::Tensor &x,
    const torch::Tensor &gate) {
  TORCH_CHECK(x.is_cuda() && gate.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(x.sizes() == gate.sizes(), "x and gate must have same shape");
  TORCH_CHECK(x.is_contiguous() && gate.is_contiguous(), "inputs must be contiguous");

  auto output = torch::empty_like(x);
  const int64_t n = x.numel();
  const int threads = 256;
  // Each thread processes 8 elements
  const int blocks = min((int64_t)65535, (n + threads * 8 - 1) / (threads * 8));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (x.dtype() == torch::kBFloat16) {
    fused_sigmoid_gate_cuda_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16 *>(output.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(gate.data_ptr()),
        n);
  } else if (x.dtype() == torch::kFloat16) {
    fused_sigmoid_gate_cuda_kernel_fp16<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__half *>(output.data_ptr()),
        reinterpret_cast<const __half *>(x.data_ptr()),
        reinterpret_cast<const __half *>(gate.data_ptr()),
        n);
  } else {
    TORCH_CHECK(false, "Unsupported dtype, expected bf16 or fp16");
  }

  return output;
}


// ============================================================================
// Python bindings
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_fp8_gather_dequant", &fused_fp8_gather_dequant,
        "Fused FP8 gather + dequantize (CUDA)");
  m.def("fused_topk_block_table_decode", &fused_topk_block_table_decode,
        "Fused TopK + Block Table construction for decode (CUDA)");
  m.def("fused_sigmoid_gate", &fused_sigmoid_gate_cuda,
        "Fused x * sigmoid(gate) with vectorized memory access (CUDA)");
}
