from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='nms3d',
    ext_modules=[
        CUDAExtension('nms3d_cuda', [
            'src/mathutil_cuda.cpp',
            'src/mathutil_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
