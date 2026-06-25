# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""TE-style FP8 cast-transpose: one read of a high-precision tensor -> both f8 layouts.

Hopper f8 ``wgmma`` needs the contraction axis contiguous, so the f8 weight-gradient must consume
token-contiguous (transposed) operands. Producing that transpose with a second ``quantize`` re-reads
the bf16 source (1 read + 2 writes -> 2 reads + 2 writes), which costs a full re-cast and sinks the
f8 wgrad e2e (logbook GFP8-030). Transformer Engine's fix is a single ``cast_transpose`` kernel that
reads the source **once** and writes **both** the rowwise and transposed f8 quantizations; the
transposed copy then costs ~one extra f8 write rather than a re-cast.

This module holds the vanilla-JAX reference (the bit-exact correctness oracle) and the public
``cast_transpose`` wrapper. The Mosaic-GPU fast path that actually fuses the two stores lands in a
follow-up (logbook GFP8-033, M2).
"""

import jax
import jax.numpy as jnp

from haliax._src.fp8 import quantize

__all__ = ["cast_transpose", "cast_transpose_reference"]


def cast_transpose_reference(
    x: jax.Array,
    scale: jax.Array,
    *,
    out_dtype: jnp.dtype = jnp.float8_e4m3fn,
    compute_dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[jax.Array, jax.Array]:
    """Vanilla-JAX reference for :func:`cast_transpose` (the bit-exact oracle).

    Quantizes ``x`` to ``out_dtype`` with the existing per-tensor ``quantize`` and returns both the
    rowwise result and its transpose. Two separate ops here; the fast path produces the same bytes
    from a single read of ``x``.
    """
    if x.ndim != 2:
        raise ValueError(f"cast_transpose expects a 2D array, got shape {x.shape}")
    q = quantize(x, out_dtype, scale, compute_dtype)
    return q, q.T


def cast_transpose(
    x: jax.Array,  # [M, K] high-precision (bf16/f32)
    scale: jax.Array,  # per-tensor delayed-scaling scale (scalar)
    *,
    out_dtype: jnp.dtype = jnp.float8_e4m3fn,
    compute_dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[jax.Array, jax.Array]:  # (q[M, K] rowwise, qT[K, M] transposed), both out_dtype
    """One HBM read of ``x`` -> both the rowwise and transposed f8 quantizations (TE cast-transpose).

    Bit-identical to :func:`cast_transpose_reference` — the transpose is produced from the same
    quantized values, not by a second independent cast. ``scale`` is an input (delayed scaling: the
    step's scale comes from the previous amax history, known before the cast).

    Currently delegates to the reference; the Mosaic-GPU fused kernel lands in M2 (GFP8-033).
    """
    return cast_transpose_reference(x, scale, out_dtype=out_dtype, compute_dtype=compute_dtype)
