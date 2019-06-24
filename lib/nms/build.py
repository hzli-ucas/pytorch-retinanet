import os
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension
from setuptools import setup


sources = ['src/nms_binding.cpp']
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    defines += [('WITH_CUDA', None)]
    from torch.utils.cpp_extension import CUDAExtension
    build_extension = CUDAExtension
else:
    build_extension = CppExtension

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
extra_objects = ['src/cuda/nms_kernel.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ext_module = build_extension(
    '_ext.nms',
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    extra_objects=extra_objects,
    extra_compile_args=['-std=c99']
)

if __name__ == '__main__':
    setup(
        name = '_ext.nms',
        ext_modules = [ext_module],
        cmdclass={'build_ext': BuildExtension}
    )
