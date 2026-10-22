# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin override of the nvidia-cudnn-frontend 1.31.0 fused MXFP8 SwiGLU grouped-GEMM kernel (SM100).

The module subclasses the upstream kernel class and replaces only the methods it changes, copied
from upstream with "Modified by Marin" notes. It imports the CUDA-only CuTe DSL and the cuDNN
frontend at import time; import it only after checking for an SM100 GPU.
"""
