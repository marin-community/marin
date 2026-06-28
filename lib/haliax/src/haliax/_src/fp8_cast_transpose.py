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
``cast_transpose`` wrapper. On H100 the wrapper routes to the fused Mosaic-GPU kernel
(``fp8_cast_transpose_mgpu.py``), which produces the transpose via the ``WGMMA -> WGMMA_TRANSPOSED``
register layout cast and is bit-exact to the reference (logbook GFP8-033); elsewhere (CPU/TPU, odd
shapes, or no Pallas) it delegates to the reference.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import custom_vjp

from haliax._src.fp8 import _new_scale_and_history, quantize

# Guard the Mosaic-GPU cast-transpose kernel: H100-only (Hopper wgmma layouts), absent on TPU/CPU.
_has_pallas_mosaic = False
try:
    from haliax._src.fp8_cast_transpose_mgpu import cast_transpose_mgpu  # type: ignore[assignment]

    _has_pallas_mosaic = True
except (ImportError, ModuleNotFoundError):
    pass

# The Mosaic kernel tiles 128x128; non-conforming shapes fall back to the reference.
_MOSAIC_BLOCK = 128

__all__ = ["cast_transpose", "cast_transpose_reference", "in_q_transpose"]


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

    Uses the fused Mosaic-GPU kernel on H100 for 128-tileable shapes; otherwise the reference.
    """
    m, k = x.shape if x.ndim == 2 else (0, 0)
    use_mosaic = (
        _has_pallas_mosaic
        and jax.default_backend() == "gpu"
        and x.ndim == 2
        and m % _MOSAIC_BLOCK == 0
        and k % _MOSAIC_BLOCK == 0
    )
    if use_mosaic:
        return cast_transpose_mgpu(
            x, scale, out_dtype=out_dtype, compute_dtype=compute_dtype, block_m=_MOSAIC_BLOCK, block_k=_MOSAIC_BLOCK
        )
    return cast_transpose_reference(x, scale, out_dtype=out_dtype, compute_dtype=compute_dtype)


# --- Delayed-scaling dual-output quantize (the cast-transpose analog of fp8.in_q) ---------------
# fp8.in_q quantizes an operand to E4M3 and propagates the delayed-scaling state (new scale/amax) as
# the gradients of its scale/amax_history inputs. in_q_transpose does the same but ALSO returns the
# transposed quantization from the same single read of the input, so the f8 weight-gradient consumes
# a token-contiguous operand without a separate XLA transpose (logbook GFP8-033 M3).


@partial(custom_vjp, nondiff_argnums=(0,))
def in_q_transpose(compute_dtype, inp, scale, amax_history):
    """Quantize ``inp`` to E4M3 and also return its transpose: ``(q, q.T, new_scale)`` from one read.

    ``q``/``q.T`` are the rowwise and token-contiguous f8 operands the forward and weight-gradient
    consume; ``new_scale`` is the live delayed-scaling scale (as in :func:`haliax._src.fp8.in_q`).
    """
    new_scale, _ = _new_scale_and_history(inp, jnp.float8_e4m3fn, scale, amax_history)
    q, q_t = cast_transpose(inp, new_scale, out_dtype=jnp.float8_e4m3fn, compute_dtype=compute_dtype)
    return q, q_t, new_scale


def in_q_transpose_fwd(compute_dtype, inp, scale, amax_history):
    new_scale, new_history = _new_scale_and_history(inp, jnp.float8_e4m3fn, scale, amax_history)
    q, q_t = cast_transpose(inp, new_scale, out_dtype=jnp.float8_e4m3fn, compute_dtype=compute_dtype)
    return (q, q_t, new_scale), (new_scale, new_history)


def in_q_transpose_bwd(compute_dtype, res, _g):
    # inp gets no gradient; the freshly computed scale/history overwrite the scale/amax_history grads
    # (delayed-scaling state), exactly as in fp8.in_q. The cotangent of q.T is unused.
    new_scale, new_history = res
    return None, new_scale, new_history


in_q_transpose.defvjp(in_q_transpose_fwd, in_q_transpose_bwd)
