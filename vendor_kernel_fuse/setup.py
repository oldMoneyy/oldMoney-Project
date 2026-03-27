from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

supported_archs = ["80", "89", "90", "120"]
cc_flag = []
for arch in supported_archs:
    cc_flag.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])

setup(
    name='fused_kernel_extension',
    ext_modules=[
        CUDAExtension(
            'fused_kernel_extension',
            sources=['fused_kernels.cu'],
            extra_compile_args={
                'nvcc': ['-O3', '--use_fast_math'] + cc_flag,
                'cxx': ['-O3'],
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
