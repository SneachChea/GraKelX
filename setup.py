"""Setuptools compatibility entrypoint and extension definitions."""

# Author: Ioannis Siglidis <y.siglidis@gmail.com>
# License: BSD 3 clause
import sys
from platform import system
from typing import List

from Cython.Build import build_ext
from numpy import get_include
from setuptools import Extension, setup


def _extra_compile_args() -> List[str]:
    """Return OS-specific compiler optimization flags."""
    os_name = system()
    if os_name == "Windows":
        return ["/O2", "/w"]
    if os_name in ["Linux", "Darwin"]:
        return ["-O3", "-w"]
    return []


# Remove the "-Wstrict-prototypes" compiler option, which isn't valid for C++.
import distutils.sysconfig

cfg_vars = distutils.sysconfig.get_config_vars()
for key, value in cfg_vars.items():
    if isinstance(value, str):
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")


def get_extensions() -> List[Extension]:
    """Build the list of Cython/C++ extension modules."""
    extra_compile_args = _extra_compile_args()

    ext_address = "./grakelx/kernels/_c_functions/"
    ext = Extension(
        name="grakelx.kernels._c_functions",
        sources=[
            ext_address + "functions.pyx",
            ext_address + "src/ArashPartov.cpp",
            ext_address + "src/sm_core.cpp",
        ],
        include_dirs=[ext_address + "include", get_include()],
        depends=[ext_address + "include/functions.hpp"],
        language="c++",
        extra_compile_args=extra_compile_args,
    )

    isodir = "./grakelx/kernels/_isomorphism/"
    blissdir = isodir + "bliss-0.50/"

    blisssrcs = ["graph.cc", "heap.cc", "orbit.cc", "partition.cc", "uintseqhash.cc"]
    blisssrcs = [blissdir + src for src in blisssrcs]
    pn = str(sys.version_info[0])

    intpybliss = Extension(
        name="grakelx.kernels._isomorphism.intpybliss",
        define_macros=[("MAJOR_VERSION", "0"), ("MINOR_VERSION", "50beta")],
        include_dirs=[blissdir],
        language="c++",
        extra_compile_args=extra_compile_args,
        sources=[isodir + "intpyblissmodule_" + pn + ".cc"] + blisssrcs,
    )

    bliss = Extension(
        name="grakelx.kernels._isomorphism.bliss",
        include_dirs=[isodir],
        language="c++",
        extra_compile_args=extra_compile_args,
        sources=[isodir + "bliss.pyx"],
    )

    return [intpybliss, bliss, ext]


setup(
    ext_modules=get_extensions(),
    cmdclass={"build_ext": build_ext},
)
