from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='flipcumsums',
    packages=find_packages(),
    version='0.0.0',
    author='juvi21',
    ext_modules=[
        CUDAExtension(
            'flipcumsum', # operator name
            ['./src/cuda/flipcumsum_cuda.cu',
             './src/cuda/flipcumsum.cpp']
        ),
        CUDAExtension(
            'flip',  # Flip operator name
            ['./src/cuda/flip_cuda.cu', './src/cuda/flip.cpp']
        ),
        CUDAExtension(
            'cumsum',  # Cumsum operator name
            ['./src/cuda/cumsum_cuda.cu', './src/cuda/cumsum.cpp']
        ),

    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

