# -*- coding: utf-8 -*-
"""Deepcompressor Extension."""

import os

import torch
from torch.utils.cpp_extension import load

__all__ = ["_C"]

dirpath = os.path.dirname(__file__)

_ext_name = "deepcompressor_C"
# IMPORTANT:
# Torch extensions are cached under ~/.cache/torch_extensions/<name>.
# If the GPU changes (e.g. A100 sm80 -> H100 sm90), reusing an old cached build can fail
# or run sub-optimally. Encode compute capability into the extension name so new GPUs
# automatically trigger a rebuild without requiring users to manually clear caches.
if torch.cuda.is_available():
    try:
        major, minor = torch.cuda.get_device_capability()
        _ext_name = f"{_ext_name}_sm{major}{minor}"
    except Exception:
        pass

_C = load(
    name=_ext_name,
    sources=[f"{dirpath}/pybind.cpp", f"{dirpath}/quantize/quantize.cu"],
    # NOTE: Some CUDA toolchains (nvcc) do not accept `-std=c++20`.
    # Use C++17 for broad compatibility (e.g. CUDA 11.x environments).
    extra_cflags=["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_HALF2_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--ptxas-options=--allow-expensive-optimizations=true",
        "--threads=8",
    ],
)
