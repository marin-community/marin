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
from .ragged_dot_cute import cute_available, cute_cast_transpose, cute_cast_transpose_weight

# Fused GPU kernel handles bf16 inputs with an f32 accumulator and an N tile of 64.
_CUTE_CAST_TRANSPOSE_TILE_N = 64


def cast_transpose_reference(x, scale, *, out_dtype, compute_dtype=jnp.float32):
    """Pure-JAX cast-transpose: (row-major FP8, token-major FP8). Correct on any backend.

    ``x`` is ``[..., M, F]`` (tokens M leading). Returns ``(xq, xq_t)`` where ``xq``
    is bit-identical to ``fp8.quantize(x, out_dtype, scale, compute_dtype)`` and
    ``xq_t`` swaps its last two axes (per-expert transpose for a batched ``x``).
    """
    xq = quantize(x, out_dtype, scale, compute_dtype)
    return xq, jnp.swapaxes(xq, -2, -1)


def _fused_conforming(x, compute_dtype) -> bool:
    """The fused CuTe kernel handles bf16 inputs, an f32 accumulator, and the two
    tiled dims (F, and per-expert M for a batched weight) divisible by the tile."""
    return (
        cute_available()
        and x.dtype == jnp.bfloat16
        and compute_dtype == jnp.float32
        and x.ndim in (2, 3)
        and all(x.shape[d] % _CUTE_CAST_TRANSPOSE_TILE_N == 0 for d in (x.ndim - 2, x.ndim - 1))
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


def cast_transpose_weight(rhs, scale, *, out_dtype, compute_dtype=jnp.float32):
    """Per-expert dual-write weight quantize: ``rhs[E,K,N]`` -> ``(q_rhs[E,K,N], q_rhs_t[E,N,K])``.

    ``q_rhs`` is bit-identical to ``quantize(rhs, out_dtype, scale)`` (the dgrad
    operand, natural layout) and ``q_rhs_t`` is its per-expert transpose (the
    K-contiguous forward operand). One fused HBM read produces both on a conforming
    GPU; otherwise the pure-JAX reference runs.
    """
    if _fused_conforming(rhs, compute_dtype):
        return cute_cast_transpose_weight(rhs, scale, out_dtype=out_dtype)
    return cast_transpose_reference(rhs, scale, out_dtype=out_dtype, compute_dtype=compute_dtype)
