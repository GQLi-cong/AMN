from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='reg_att_map_generator',
      version='1.1.0',
      ext_modules=[
          CUDAExtension('reg_att_map_generator',
                        ['reg_att_map_generator_cuda.cpp', 'reg_att_map_generator.cu']),
      ],
      cmdclass={'build_ext': BuildExtension})
