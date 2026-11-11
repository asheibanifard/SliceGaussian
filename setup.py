"""Optional ahead-of-time build of the CUDA extension.

    pip install -e .                              # the kernel is compiled on first use (just-in-time)
    SLICEGAUSSIAN_BUILD_CUDA=1 pip install -e .   # compile it now, as slicegaussian._C

All other metadata is in pyproject.toml.
"""
import os

from setuptools import setup

ext_modules, cmdclass = [], {}
if os.environ.get("SLICEGAUSSIAN_BUILD_CUDA") == "1":
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    csrc = os.path.join("src", "slicegaussian", "csrc")
    ext_modules = [CUDAExtension("slicegaussian._C",
                                 [os.path.join(csrc, "fieldcut.cpp"), os.path.join(csrc, "fieldcut_kernel.cu")],
                                 extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"]})]
    cmdclass = {"build_ext": BuildExtension}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
