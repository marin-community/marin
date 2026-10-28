# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Saved-probability attention over causal query buckets, without QK recomputation."""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from experiments.grug.belay_tpu.inkling_pallas import (
    TILE,
    _cost,
    _dot,
    _probability,
    _probability_math,
    _query_grad,
    _query_grad_math,
    _scores_math,
    _shape,
)


def _scores(qr, kr, br, qir, kir, sr, *, query_start, extent, width):
    query_block = query_start // TILE + pl.program_id(1)
    key_block = pl.program_id(2)
    band_block = key_block - query_block + extent // TILE
    bias = jnp.where((band_block >= 0) & (band_block < width // TILE), br[...], 0)
    score = _scores_math(qr[...], kr[...], bias, qir[0, :], kir[0, :])
    causal = query_block * TILE + jnp.arange(TILE)[:, None] >= key_block * TILE + jnp.arange(TILE)[None, :]
    sr[...] = jnp.where(causal, score, jnp.asarray(-1e9, score.dtype))


def _partial_kv_math(ds, prob, q, dout):
    scaled = (q * (1.0 / math.sqrt(q.shape[-1]))).astype(q.dtype)
    return _dot(ds.T, scaled), _dot(prob.astype(dout.dtype).T, dout)


def _partial_kv(dsr, pr, qr, dor, dkr, dvr):
    dk, dv = _partial_kv_math(dsr[...], pr[...], qr[...], dor[...])
    dkr[...] = dk
    dvr[...] = dv


def _band_grad(dsr, dbr, *, query_start, extent, key_length):
    key_block = query_start // TILE + pl.program_id(1) + pl.program_id(2) - extent // TILE
    dbr[...] = jnp.where((key_block >= 0) & (key_block < key_length // TILE), dsr[...], 0)


def bucket_forward(q, k, v, band, qids, kids, *, query_start, interpret=False):
    """Evaluate query rows against their inclusive causal key prefix."""
    groups, query_length, dim = q.shape
    key_length, width = k.shape[1], band.shape[-1]
    extent = width - TILE
    grid = (groups, query_length // TILE, key_length // TILE)
    inputs = (q, k, band, qids[:, None, :], kids[:, None, :])
    score_shape = _shape((groups, query_length, key_length), q.dtype)
    tile = _shape((TILE, dim), q.dtype)
    ids = _shape((TILE,), jnp.int32)
    scores = pl.pallas_call(
        functools.partial(_scores, query_start=query_start, extent=extent, width=width),
        out_shape=score_shape,
        grid=grid,
        in_specs=(
            pl.BlockSpec((None, TILE, dim), lambda g, i, j: (g, i, 0)),
            pl.BlockSpec((None, TILE, dim), lambda g, i, j: (g, j, 0)),
            pl.BlockSpec(
                (None, TILE, TILE),
                lambda g, i, j: (g, i, jnp.clip(j - i - query_start // TILE + extent // TILE, 0, width // TILE - 1)),
            ),
            pl.BlockSpec((None, 1, TILE), lambda g, i, j: (g, 0, i)),
            pl.BlockSpec((None, 1, TILE), lambda g, i, j: (g, 0, j)),
        ),
        out_specs=pl.BlockSpec((None, TILE, TILE), lambda g, i, j: (g, i, j)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel")),
        cost_estimate=_cost(_scores_math, inputs, (score_shape,), (tile, tile, tile, ids, ids), grid),
        interpret=interpret,
        name="inkling_causal_bucket_scores",
    )(*inputs)
    grid = (groups, query_length // TILE)
    shapes = (_shape(q.shape, q.dtype), _shape(score_shape.shape, jnp.float32))
    return pl.pallas_call(
        _probability,
        out_shape=shapes,
        grid=grid,
        in_specs=(
            pl.BlockSpec((None, TILE, key_length), lambda g, i: (g, i, 0)),
            pl.BlockSpec((None, key_length, dim), lambda g, i: (g, 0, 0)),
        ),
        out_specs=(
            pl.BlockSpec((None, TILE, dim), lambda g, i: (g, i, 0)),
            pl.BlockSpec((None, TILE, key_length), lambda g, i: (g, i, 0)),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        cost_estimate=_cost(
            _probability_math,
            (scores, v),
            shapes,
            (_shape((TILE, key_length), q.dtype), _shape((key_length, dim), v.dtype)),
            grid,
        ),
        interpret=interpret,
        name="inkling_causal_bucket_probability_save",
    )(scores, v)


def bucket_backward(q, k, v, prob, dout, *, width, query_start, interpret=False):
    """Return query/band derivatives and FP32 partial key/value derivatives."""
    groups, query_length, dim = q.shape
    key_length, extent = k.shape[1], width - TILE
    grid = (groups, query_length // TILE)
    rows = pl.BlockSpec((None, TILE, dim), lambda g, i: (g, i, 0))
    keys = pl.BlockSpec((None, key_length, dim), lambda g, i: (g, 0, 0))
    probabilities = pl.BlockSpec((None, TILE, key_length), lambda g, i: (g, i, 0))
    shapes = (_shape(q.shape, q.dtype), _shape(prob.shape, q.dtype))
    inputs = (prob, k, v, dout)
    dq, ds = pl.pallas_call(
        _query_grad,
        out_shape=shapes,
        grid=grid,
        in_specs=(probabilities, keys, keys, rows),
        out_specs=(rows, probabilities),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        cost_estimate=_cost(
            _query_grad_math,
            inputs,
            shapes,
            (
                _shape((TILE, key_length), jnp.float32),
                _shape((key_length, dim), k.dtype),
                _shape((key_length, dim), v.dtype),
                _shape((TILE, dim), dout.dtype),
            ),
            grid,
        ),
        interpret=interpret,
        name="inkling_causal_bucket_query_gradient",
    )(*inputs)
    grid = (groups, key_length // TILE)
    columns = pl.BlockSpec((None, query_length, TILE), lambda g, i: (g, 0, i))
    queries = pl.BlockSpec((None, query_length, dim), lambda g, i: (g, 0, 0))
    shapes = (_shape(k.shape, jnp.float32), _shape(v.shape, jnp.float32))
    inputs = (ds, prob, q, dout)
    dk, dv = pl.pallas_call(
        _partial_kv,
        out_shape=shapes,
        grid=grid,
        in_specs=(columns, columns, queries, queries),
        out_specs=(rows, rows),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        cost_estimate=_cost(
            _partial_kv_math,
            inputs,
            shapes,
            (
                _shape((query_length, TILE), q.dtype),
                _shape((query_length, TILE), jnp.float32),
                _shape((query_length, dim), q.dtype),
                _shape((query_length, dim), dout.dtype),
            ),
            grid,
        ),
        interpret=interpret,
        name="inkling_causal_bucket_partial_kv_gradient",
    )(*inputs)
    band_shape = _shape((groups, query_length, width), q.dtype)
    dband = pl.pallas_call(
        functools.partial(_band_grad, query_start=query_start, extent=extent, key_length=key_length),
        out_shape=band_shape,
        grid=(groups, query_length // TILE, width // TILE),
        in_specs=(
            pl.BlockSpec(
                (None, TILE, TILE),
                lambda g, i, j: (
                    g,
                    i,
                    jnp.clip(query_start // TILE + i + j - extent // TILE, 0, key_length // TILE - 1),
                ),
            ),
        ),
        out_specs=pl.BlockSpec((None, TILE, TILE), lambda g, i, j: (g, i, j)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel")),
        interpret=interpret,
        name="inkling_causal_bucket_band_gradient",
    )(ds)
    return dq, dk, dv, dband


def _forward(q, k, v, band, qids, kids, bucket_size, interpret):
    if bucket_size < TILE or bucket_size % TILE or q.shape[1] % TILE:
        raise ValueError("Query buckets and sequence length must be positive multiples of 128")
    if q.shape != k.shape or q.shape != v.shape or q.shape[-1] != TILE:
        raise ValueError("Causal attention requires equal local Q/K/V shapes and head width 128")
    if band.shape[:2] != q.shape[:2] or band.shape[-1] <= TILE or band.shape[-1] % TILE:
        raise ValueError("Inkling bias must have matching query rows and a 128-aligned band")
    if q.dtype not in (jnp.float32, jnp.bfloat16) or any(x.dtype != q.dtype for x in (k, v, band)):
        raise ValueError("Q/K/V and bias must have a common FP32 or BF16 dtype")
    if qids.shape != q.shape[:2] or kids.shape != q.shape[:2]:
        raise ValueError("Segment IDs must match the local token arrays")
    outputs, probabilities = [], []
    for start in range(0, q.shape[1], bucket_size):
        end = min(start + bucket_size, q.shape[1])
        out, prob = bucket_forward(
            q[:, start:end],
            k[:, :end],
            v[:, :end],
            band[:, start:end],
            qids[:, start:end],
            kids[:, :end],
            query_start=start,
            interpret=interpret,
        )
        outputs.append(out)
        probabilities.append(prob)
    residual = (q, k, v, tuple(probabilities), jnp.zeros((0, band.shape[-1]), band.dtype))
    return jnp.concatenate(outputs, axis=1), residual


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def causal_attention_local(q, k, v, band, qids, kids, bucket_size=512, interpret=False):
    """Save probabilities over causal prefixes without repeating forward products."""
    return _forward(q, k, v, band, qids, kids, bucket_size, interpret)[0]


def _backward(bucket_size, interpret, residual, dout):
    q, k, v, probabilities, shape_token = residual
    dq, dband = [], []
    dk, dv = jnp.zeros(k.shape, jnp.float32), jnp.zeros(v.shape, jnp.float32)
    for start, prob in zip(range(0, q.shape[1], bucket_size), probabilities, strict=True):
        end = min(start + bucket_size, q.shape[1])
        q_grad, k_grad, v_grad, band_grad = bucket_backward(
            q[:, start:end],
            k[:, :end],
            v[:, :end],
            prob,
            dout[:, start:end],
            width=shape_token.shape[-1],
            query_start=start,
            interpret=interpret,
        )
        dq.append(q_grad)
        dband.append(band_grad)
        padding = ((0, 0), (0, q.shape[1] - end), (0, 0))
        dk = dk + jnp.pad(k_grad, padding)
        dv = dv + jnp.pad(v_grad, padding)
    return (
        jnp.concatenate(dq, axis=1),
        dk.astype(k.dtype),
        dv.astype(v.dtype),
        jnp.concatenate(dband, axis=1),
        None,
        None,
    )


causal_attention_local.defvjp(_forward, _backward)
