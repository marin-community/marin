"""Build-only probe for compiling a PyTorch CUDA extension with Marin's staged nvcc."""

import torch.utils.cpp_extension


def _skip_cuda_toolkit_version_check(*_args, **_kwargs) -> None:
    return None


torch.utils.cpp_extension._check_cuda_version = _skip_cuda_toolkit_version_check
