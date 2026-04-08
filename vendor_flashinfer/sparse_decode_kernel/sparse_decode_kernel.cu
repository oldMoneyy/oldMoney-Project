#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

// ---------------------------------------------------------------------------
// Kernel 1: compute kv_lens per request
// ---------------------------------------------------------------------------
// 1 thread per request. For sparse requests: counts dense + clipped sparse
// blocks + window. For dense: kv_len = seq_len.
// Both kernels share identical clipping logic so counts are guaranteed to match.
__global__ void sparse_decode_kv_lens_kernel(
    const int *__restrict__ seq_lens,          // [bs]
    const int *__restrict__ req_pool_indices,  // [bs]
    const int *__restrict__ block_pool,        // [max_reqs, max_blocks_per_req]
    const int *__restrict__ n_blocks,          // [max_reqs]
    const int *__restrict__ is_sparse,         // [bs]
    int *__restrict__ kv_lens_out,             // [bs]
    int block_pool_stride,
    int dense_len,
    int window_size,
    int block_size,
    int bs) {

  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= bs)
    return;

  int sl = seq_lens[b];

  if (!is_sparse[b]) {
    kv_lens_out[b] = sl;
    return;
  }

  int req_idx = req_pool_indices[b];
  int window_start = max(dense_len, sl - window_size);
  int n_blks = n_blocks[req_idx];
  long long block_base = (long long)req_idx * block_pool_stride;

  int count = dense_len; // dense region [0, dense_len)
  for (int bi = 0; bi < n_blks; bi++) {
    int block_idx = block_pool[block_base + bi];
    int block_start = block_idx * block_size;
    int n_tokens = min(block_size, window_start - block_start);
    if (n_tokens > 0)
      count += n_tokens;
  }
  count += (sl - window_start); // window region [window_start, sl)

  kv_lens_out[b] = count;
}

// ---------------------------------------------------------------------------
// Kernel 2: build kv_indices
// ---------------------------------------------------------------------------
// 1 thread block per request, blockDim.x threads cooperate on copies.
// Thread 0 computes per-block output offsets into shared memory (serial over
// blocks, but n_blocks is small ~100). Then all threads copy in parallel.
__global__ void sparse_decode_kv_indices_kernel(
    const int *__restrict__ req_to_token,      // [max_reqs, max_seq_len]
    const int *__restrict__ req_pool_indices,   // [bs]
    const int *__restrict__ seq_lens,           // [bs]
    const int *__restrict__ block_pool,         // [max_reqs, max_blocks_per_req]
    const int *__restrict__ n_blocks,           // [max_reqs]
    const int *__restrict__ is_sparse,          // [bs]
    const int *__restrict__ kv_indptr,          // [bs+1]
    int *__restrict__ kv_indices_out,           // [total_tokens]
    int req_to_token_stride,
    int block_pool_stride,
    int dense_len,
    int window_size,
    int block_size,
    int bs) {

  int b = blockIdx.x;
  if (b >= bs)
    return;

  int sl = seq_lens[b];
  int req_idx = req_pool_indices[b];
  int out_base = kv_indptr[b];
  long long req_base = (long long)req_idx * req_to_token_stride;

  if (!is_sparse[b]) {
    // Dense: copy all [0, sl)
    for (int i = threadIdx.x; i < sl; i += blockDim.x) {
      kv_indices_out[out_base + i] = req_to_token[req_base + i];
    }
    return;
  }

  // --- Sparse path ---
  int window_start = max(dense_len, sl - window_size);
  int n_blks = n_blocks[req_idx];
  long long block_base_addr = (long long)req_idx * block_pool_stride;

  // Shared memory layout: [n_blks] block output offsets + [1] window offset
  // Max shared = (max_blocks_per_req + 1) * 4 bytes, passed as dynamic smem.
  extern __shared__ int smem[];

  // Thread 0: compute cumulative offsets for each block
  if (threadIdx.x == 0) {
    int off = dense_len;
    for (int bi = 0; bi < n_blks; bi++) {
      int block_idx = block_pool[block_base_addr + bi];
      int block_start = block_idx * block_size;
      int n_tokens = min(block_size, window_start - block_start);
      if (n_tokens > 0) {
        smem[bi] = off;
        off += n_tokens;
      } else {
        smem[bi] = -1; // skip this block
      }
    }
    smem[n_blks] = off; // window region starts here
  }
  __syncthreads();

  // 1. Dense region [0, dense_len)
  for (int i = threadIdx.x; i < dense_len; i += blockDim.x) {
    kv_indices_out[out_base + i] = req_to_token[req_base + i];
  }

  // 2. Sparse blocks
  for (int bi = 0; bi < n_blks; bi++) {
    int off = smem[bi];
    if (off < 0)
      continue;
    int block_idx = block_pool[block_base_addr + bi];
    int block_start = block_idx * block_size;
    int n_tokens = min(block_size, window_start - block_start);
    for (int i = threadIdx.x; i < n_tokens; i += blockDim.x) {
      kv_indices_out[out_base + off + i] =
          req_to_token[req_base + block_start + i];
    }
  }

  // 3. Window region [window_start, sl)
  int window_off = smem[n_blks];
  int window_len = sl - window_start;
  for (int i = threadIdx.x; i < window_len; i += blockDim.x) {
    kv_indices_out[out_base + window_off + i] =
        req_to_token[req_base + window_start + i];
  }
}

// ---------------------------------------------------------------------------
// Wrapper: compute kv_lens
// ---------------------------------------------------------------------------
torch::Tensor sparse_decode_kv_lens_wrapper(
    const torch::Tensor &seq_lens,          // [bs] int32
    const torch::Tensor &req_pool_indices,  // [bs] int32/int64
    const torch::Tensor &block_pool,        // [max_reqs, max_blocks_per_req] int32
    const torch::Tensor &n_blocks,          // [max_reqs] int32
    const torch::Tensor &is_sparse,         // [bs] int32
    int dense_len,
    int window_size,
    int block_size) {

  int bs = seq_lens.size(0);
  torch::Tensor kv_lens = torch::empty({bs}, seq_lens.options());

  if (bs == 0)
    return kv_lens;

  const int THREADS = 256;
  const int BLOCKS = (bs + THREADS - 1) / THREADS;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  sparse_decode_kv_lens_kernel<<<BLOCKS, THREADS, 0, stream>>>(
      seq_lens.data_ptr<int>(),
      req_pool_indices.data_ptr<int>(),
      block_pool.data_ptr<int>(),
      n_blocks.data_ptr<int>(),
      is_sparse.data_ptr<int>(),
      kv_lens.data_ptr<int>(),
      static_cast<int>(block_pool.size(1)),
      dense_len, window_size, block_size, bs);

  return kv_lens;
}

// ---------------------------------------------------------------------------
// Wrapper: build kv_indices
// ---------------------------------------------------------------------------
void sparse_decode_kv_indices_wrapper(
    const torch::Tensor &req_to_token,      // [max_reqs, max_seq_len] int32
    const torch::Tensor &req_pool_indices,   // [bs] int32/int64
    const torch::Tensor &seq_lens,           // [bs] int32
    const torch::Tensor &block_pool,         // [max_reqs, max_blocks_per_req] int32
    const torch::Tensor &n_blocks,           // [max_reqs] int32
    const torch::Tensor &is_sparse,          // [bs] int32
    const torch::Tensor &kv_indptr,          // [bs+1] int32
    torch::Tensor &kv_indices_out,           // [total_tokens] int32
    int dense_len,
    int window_size,
    int block_size,
    int max_blocks_per_req) {

  int bs = seq_lens.size(0);
  if (bs == 0)
    return;

  const int THREADS = 256;
  // Dynamic shared memory: (max_blocks_per_req + 1) ints for block offsets
  size_t smem_bytes = (max_blocks_per_req + 1) * sizeof(int);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  sparse_decode_kv_indices_kernel<<<bs, THREADS, smem_bytes, stream>>>(
      req_to_token.data_ptr<int>(),
      req_pool_indices.data_ptr<int>(),
      seq_lens.data_ptr<int>(),
      block_pool.data_ptr<int>(),
      n_blocks.data_ptr<int>(),
      is_sparse.data_ptr<int>(),
      kv_indptr.data_ptr<int>(),
      kv_indices_out.data_ptr<int>(),
      static_cast<int>(req_to_token.size(1)),
      static_cast<int>(block_pool.size(1)),
      dense_len, window_size, block_size, bs);
}

// ---------------------------------------------------------------------------
// pybind11
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_decode_kv_lens", &sparse_decode_kv_lens_wrapper,
        "Compute per-request kv_lens for sparse decode (CUDA)");
  m.def("sparse_decode_kv_indices", &sparse_decode_kv_indices_wrapper,
        "Build sparse decode kv_indices for FlashInfer (CUDA)");
}
