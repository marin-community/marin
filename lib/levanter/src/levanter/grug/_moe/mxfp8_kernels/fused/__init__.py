# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Vendored NVIDIA cudnn-frontend fused MXFP8 MoE grouped GEMM kernels (MIT).

Upstream: github.com/NVIDIA/cudnn-frontend @ 3041f3e88bf8f627b2ada1f471d7ee1aa437f76b
(python/cudnn/grouped_gemm/), vendored 2026-07-16 for MXFP8-004a (issue #7282).

Files and their upstream paths:

- ``grouped_gemm_swiglu_quant.py``  <- grouped_gemm_swiglu/grouped_gemm_swiglu_quant.py
  (fwd w13: blockscaled contiguous grouped GEMM -> SwiGLU -> dual-orientation
  MXFP8 quantize; emits c + d + d_col + swizzled sfd_row/sfd_col)
- ``grouped_gemm_dswiglu_quant.py`` <- grouped_gemm_dswiglu/grouped_gemm_dswiglu_quant.py
  (bwd: dgrad-w2 GEMM -> dSwiGLU chain rule vs the fwd pre-activation C ->
  dual-orientation MXFP8 quantize of dC)
- ``grouped_gemm_quant.py``         <- grouped_gemm_quant/grouped_gemm_quant.py
  (blockscaled grouped GEMM with optional quantizing epilogue; bf16 output mode
  used for fwd-w2 and dgrad-w13)
- ``moe_blockscaled_grouped_gemm_wgrad.py`` <- grouped_gemm_wgrad/moe_blockscaled_grouped_gemm_wgrad.py
  (2Dx2D grouped wgrad, raw cumsum offsets, online SF tensormaps)
- ``moe_persistent_scheduler.py`` / ``moe_sched_extension.py`` / ``moe_utils.py``
  / ``moe_kernel_helpers.py`` / ``utils.py``: shared helpers. NEWER revisions of
  the same-named files vendored in ``../mxfp8_grouped`` (renamed classes,
  incompatible) -- this package is deliberately self-contained.

Local modifications (all mechanical):

- relative imports (``from ..utils`` -> ``from .utils``)
- torch stripped: ``import torch`` plus the torch-only host reference helpers
  ``sigmoid`` / ``compute_reference_amax`` / ``compare_and_report_mismatches``
  (moe_kernel_helpers.py) and ``logical_shape_fp4x2_aware`` (utils.py). The
  kernel classes themselves take cute.Tensors and never touch torch.
- nvidia-cutlass-dsl 4.5.x compat: upstream targets >=4.6, where
  ``nvvm.atomicrmw`` infers its result type; 4.5.x requires it as the first
  positional arg. The six atomicrmw call sites (utils.py,
  moe_kernel_helpers.py, moe_persistent_scheduler.py) pass an explicit
  ``T.i32()``/``T.f32()``. This is the ONLY >=4.6 API the kernels use -- the
  repo pin ``nvidia-cutlass-dsl>=4.5.2,<4.6`` stands (no bump, so the
  MXFP8-002 suite's validation of that wheel still applies).

The torch-facing ``api.py`` wrappers were not vendored; ``adapter.py`` (ours)
drives the kernel classes from JAX via ``cutlass.jax.cutlass_call``.
"""
