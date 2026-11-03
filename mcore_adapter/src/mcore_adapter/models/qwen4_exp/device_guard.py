"""Decide whether a tensor really lives on CUDA.

torch_npu's transfer_to_npu reports ``Tensor.is_cuda`` as True on Ascend so that
CUDA-targeted call sites keep working. The kernels behind those call sites --
Triton, flash-linear-attention, flex attention, ``mm(out_dtype=...)`` -- do not
exist there, so ``is_cuda`` selects an implementation that cannot run. Device
type is the honest test, and on a real CUDA device it agrees with ``is_cuda``,
so CUDA paths are unchanged.
"""
from __future__ import annotations

import torch


def is_cuda_device(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "cuda"
