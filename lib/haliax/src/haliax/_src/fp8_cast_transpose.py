# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Dual-output FP8 cast-transpose: one logical quantize -> row-major FP8 + token-major FP8.

The FP8 ``wgrad`` (``drhs_e[k,n] = sum_m lhs_e[m,k]*g_e[m,n]``) contracts the
token axis M, so both operands must be token-major (M contiguous) for the FP8
WGMMA (which only supports contraction-major/TN operands). ``cast_transpose``
produces both the row-major FP8 (the forward/dgrad operand, bit-identical to
``fp8.in_q``'s output) and the token-major FP8 (``[K,M]``/``[N,M]``, the wgrad
operand) from a single quantize, so the caller pays one cast for two layouts.

``cast_transpose_reference`` is the pure-JAX definition (``quantize`` then
``swapaxes``); it is correct on any backend and CPU-testable. It divides by the
scale and clips, matching ``fp8.quantize`` exactly (NOT ``x*scale``), so the
row-major output is bit-consistent with the forward operand ``in_q`` builds.
"""

import jax.numpy as jnp

from .fp8 import quantize
from .ragged_dot_cute import cute_available, cute_cast_transpose

# Fused GPU kernel handles bf16 inputs with an f32 accumulator and an N tile of 64.
_CUTE_CAST_TRANSPOSE_TILE_N = 64


def cast_transpose_reference(x, scale, *, out_dtype, compute_dtype=jnp.float32):
    """Pure-JAX cast-transpose: (row-major FP8, token-major FP8). Correct on any backend.

    ``x`` is ``[M, F]`` (tokens M leading). Returns ``(xq[M,F], xq_t[F,M])`` where
    ``xq`` is bit-identical to ``fp8.quantize(x, out_dtype, scale, compute_dtype)``.
    """
    xq = quantize(x, out_dtype, scale, compute_dtype)
    return xq, jnp.swapaxes(xq, -2, -1)


def _fused_conforming(x, compute_dtype) -> bool:
    """The fused CuTe kernel handles bf16 inputs, an f32 accumulator, and N%tile_n==0."""
    return (
        cute_available()
        and x.dtype == jnp.bfloat16
        and compute_dtype == jnp.float32
        and x.ndim == 2
        and x.shape[1] % _CUTE_CAST_TRANSPOSE_TILE_N == 0
    )


def cast_transpose(x, scale, *, out_dtype, compute_dtype=jnp.float32):
    """Cast-transpose ``x[M,F]`` to FP8, returning both row-major and token-major layouts.

    On a GPU with the CuTe DSL (and a conforming bf16 ``x``), a single fused kernel
    reads ``x`` once and emits both the row-major FP8 (bit-identical to
    ``quantize(x, out_dtype, scale)``) and the token-major FP8 transpose, replacing
    the reference's ``quantize + swapaxes`` two-pass. Otherwise the pure-JAX
    reference runs (exact on any backend).
    """
    if _fused_conforming(x, compute_dtype):
        return cute_cast_transpose(x, scale, out_dtype=out_dtype)
    return cast_transpose_reference(x, scale, out_dtype=out_dtype, compute_dtype=compute_dtype)
