# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fuse Inkling score construction, softmax and value projection on TPU."""

import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from experiments.grug.belay_tpu.inkling_pallas import TILE, _backward, _dot


def _fused_probability(qr, kr, vr, br, qir, kir, outr, pr):
    q, k, v, band = qr[...], kr[...], vr[...], br[...]
    length, dim = k.shape
    width = band.shape[-1]
    start = pl.program_id(1) * TILE
    padded_width = max(length, width)
    padded = jnp.pad(band, ((0, 0), (0, padded_width - width)))
    # The rotation is in VMEM. Mask wrapped entries to implement a zero-filled
    # shift of the band, without constructing a dense bias in HBM.
    bias = pltpu.roll(padded, (start - (width - TILE)) % padded_width, axis=1)[:, :length]
    relative = jnp.arange(length) - start + width - TILE
    bias = jnp.where((relative >= 0) & (relative < width), bias, 0)
    scaled_q = (q * (1.0 / math.sqrt(dim))).astype(q.dtype)
    logits = _dot(scaled_q, k.T).astype(q.dtype) + bias
    causal = start + jnp.arange(TILE)[:, None] >= jnp.arange(length)[None, :]
    valid = causal & (qir[0, :][:, None] == kir[0, :][None, :])
    logits = jnp.where(valid, logits, jnp.asarray(-1e9, logits.dtype))
    probabilities = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    outr[...] = _dot(probabilities.astype(v.dtype), v).astype(v.dtype)
    pr[...] = probabilities


def fused_probability(q, k, v, band, qids, kids, *, interpret=False):
    """Return attention output and FP32 probabilities without an HBM score array."""
    groups, length, dim = q.shape
    width = band.shape[-1]
    if dim != TILE or length % TILE or width % TILE or width <= TILE:
        raise ValueError("Fused Inkling requires width 128 and aligned sequence/band dimensions")
    if q.shape != k.shape or q.shape != v.shape or band.shape[:2] != q.shape[:2]:
        raise ValueError("Fused Inkling requires equal local Q/K/V shapes and aligned bias rows")
    if any(x.dtype != q.dtype for x in (k, v, band)):
        raise ValueError("Fused Inkling requires equal activation dtypes")
    rows = pl.BlockSpec((None, TILE, dim), lambda g, i: (g, i, 0))
    full = pl.BlockSpec((None, length, dim), lambda g, i: (g, 0, 0))
    return pl.pallas_call(
        _fused_probability,
        out_shape=(jax.ShapeDtypeStruct(q.shape, v.dtype), jax.ShapeDtypeStruct((groups, length, length), jnp.float32)),
        grid=(groups, length // TILE),
        in_specs=(
            rows,
            full,
            full,
            pl.BlockSpec((None, TILE, width), lambda g, i: (g, i, 0)),
            pl.BlockSpec((None, 1, TILE), lambda g, i: (g, 0, i)),
            pl.BlockSpec((None, 1, length), lambda g, i: (g, 0, 0)),
        ),
        out_specs=(rows, pl.BlockSpec((None, TILE, length), lambda g, i: (g, i, 0))),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        interpret=interpret,
        name="inkling_fused_probability_save",
    )(q, k, v, band, qids[:, None, :], kids[:, None, :])


def _forward(q, k, v, band, qids, kids):
    output, probability = fused_probability(q, k, v, band, qids, kids)
    return output, (q, k, v, probability, jnp.zeros((0, band.shape[-1]), band.dtype))


@jax.custom_vjp
def fused_attention_local(q, k, v, band, qids, kids):
    """Save FP32 probabilities and reuse the qualified Pallas attention transpose."""
    return _forward(q, k, v, band, qids, kids)[0]


fused_attention_local.defvjp(_forward, _backward)
