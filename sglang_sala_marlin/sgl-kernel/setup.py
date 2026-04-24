"""
Standalone build for modified Marlin GEMM kernel.
Builds only the Marlin kernel files into a loadable Python extension.

Build on remote server:
  cd sglang_sala_marlin/sgl-kernel
  pip install --no-build-isolation .

Or for development:
  python setup.py build_ext --inplace
"""
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))
csrc_dir = os.path.join(this_dir, "csrc", "gemm", "marlin")
include_dir = os.path.join(this_dir, "include")

setup(
    name="marlin_custom",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="marlin_custom",
            sources=[
                os.path.join(this_dir, "binding.cpp"),
                os.path.join(csrc_dir, "gptq_marlin.cu"),
                os.path.join(csrc_dir, "gptq_marlin_repack.cu"),
                os.path.join(csrc_dir, "awq_marlin_repack.cu"),
            ],
            include_dirs=[
                include_dir,
                csrc_dir,
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--threads", "4",
                    "-gencode", "arch=compute_80,code=sm_80",
                    "-gencode", "arch=compute_86,code=sm_86",
                    "-gencode", "arch=compute_89,code=sm_89",
                    "-gencode", "arch=compute_90,code=sm_90",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
