from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

supported_archs = ["80", "90", "120"]
cc_flag = []
for arch in supported_archs:
    cc_flag.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])

setup(
    name='sparse_decode_extension',
    ext_modules=[
        CUDAExtension(
            'sparse_decode_extension',
            sources=['sparse_decode_kernel.cu'],
            extra_compile_args={'nvcc': ['-O3'] + cc_flag}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
