# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA cudnn-frontend fused MXFP8 MoE grouped GEMM kernels (MIT), vendored for SM100.

Upstream is cudnn-frontend commit 3041f3e88bf8f627b2ada1f471d7ee1aa437f76b
(``python/cudnn/grouped_gemm/``), by way of Marin research commit c3cb334f8e. Each file keeps its
MIT header and records its upstream path and any Marin modification. The modules import the
CUDA-only CuTe DSL at import time; import them only after checking for an SM100 GPU.
"""
