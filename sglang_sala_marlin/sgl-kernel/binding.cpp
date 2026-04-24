// Pybind11 binding for standalone modified Marlin GEMM kernel.
// Exports gptq_marlin_gemm as a Python-callable function.

#include <torch/extension.h>
#include "scalar_type.hpp"

// Forward declaration from gptq_marlin.cu
torch::Tensor gptq_marlin_gemm(
    torch::Tensor& a,
    std::optional<torch::Tensor> c_or_none,
    torch::Tensor& b_q_weight,
    torch::Tensor& b_scales,
    std::optional<torch::Tensor> const& global_scale_or_none,
    std::optional<torch::Tensor> const& b_zeros_or_none,
    std::optional<torch::Tensor> const& g_idx_or_none,
    std::optional<torch::Tensor> const& perm_or_none,
    torch::Tensor& workspace,
    sglang::ScalarTypeId const& b_q_type_id,
    int64_t size_m,
    int64_t size_n,
    int64_t size_k,
    bool is_k_full,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float);

// Forward declaration from gptq_marlin_repack.cu
torch::Tensor gptq_marlin_repack(
    torch::Tensor& b_q_weight,
    torch::Tensor& perm,
    int64_t size_k,
    int64_t size_n,
    int64_t num_bits);

// Forward declaration from awq_marlin_repack.cu
torch::Tensor awq_marlin_repack(
    torch::Tensor& b_q_weight,
    int64_t size_k,
    int64_t size_n,
    int64_t num_bits);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gptq_marlin_gemm", &gptq_marlin_gemm,
        "Optimized GPTQ Marlin GEMM with multi-tier tile configs and atomic_add decode");
  m.def("gptq_marlin_repack", &gptq_marlin_repack,
        "GPTQ Marlin weight repacking");
  m.def("awq_marlin_repack", &awq_marlin_repack,
        "AWQ Marlin weight repacking");
}
