# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Device-local TPU Inkling attention retaining probabilities for backward.

Call only on fully local arrays, inside inkling_tpu's explicit shard_map in
the model. Four matrix-product stages and a band-layout copy replace generic
dense gather/scatter and autodiff. No forward contraction is recomputed.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float, Int

TILE = 128


def _dot(left, right):
    return jnp.matmul(
        left,
        right,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST if left.dtype == jnp.float32 else jax.lax.Precision.DEFAULT,
    )


def _shape(shape, dtype):
    return jax.ShapeDtypeStruct(shape, dtype)


def _cost(body, inputs, outputs, local_inputs, grid):
    estimate = pl.estimate_cost(body, *local_inputs)
    return pl.CostEstimate(
        flops=estimate.flops * math.prod(grid),
        transcendentals=estimate.transcendentals * math.prod(grid),
        bytes_accessed=sum(math.prod(x.shape) * jnp.dtype(x.dtype).itemsize for x in (*inputs, *outputs)),
    )


def _scores_math(q, k, bias, qids, kids):
    scaled = (q * (1.0 / math.sqrt(q.shape[-1]))).astype(q.dtype)
    score = _dot(scaled, k.T).astype(q.dtype) + bias.astype(q.dtype)
    return jnp.where(qids[:, None] == kids[None, :], score, jnp.asarray(-1e9, score.dtype))


def _scores(qr, kr, br, qir, kir, sr, *, extent, width):
    query_block, key_block = pl.program_id(1), pl.program_id(2)
    band_block = key_block - query_block + extent // TILE
    bias = jnp.where((band_block >= 0) & (band_block < width // TILE), br[...], 0)
    score = _scores_math(qr[...], kr[...], bias, qir[0, :], kir[0, :])
    causal = query_block * TILE + jnp.arange(TILE)[:, None] >= key_block * TILE + jnp.arange(TILE)[None, :]
    sr[...] = jnp.where(causal, score, jnp.asarray(-1e9, score.dtype))


def _probability_math(score, v):
    prob = jax.nn.softmax(score.astype(jnp.float32), axis=-1)
    return _dot(prob.astype(v.dtype), v).astype(v.dtype), prob


def _probability(sr, vr, outr, pr):
    out, prob = _probability_math(sr[...], vr[...])
    outr[...] = out
    pr[...] = prob


def _query_grad_math(prob, k, v, dout):
    dp = _dot(dout, v.T).astype(v.dtype).astype(jnp.float32)
    weighted = prob * dp
    ds = (weighted - prob * jnp.sum(weighted, axis=-1, keepdims=True)).astype(k.dtype)
    dq = _dot(ds, k).astype(k.dtype) * (1.0 / math.sqrt(k.shape[-1]))
    return dq, ds


def _query_grad(pr, kr, vr, dor, dqr, dsr):
    dq, ds = _query_grad_math(pr[...], kr[...], vr[...], dor[...])
    dqr[...] = dq
    dsr[...] = ds


def _kv_grad_math(ds, prob, q, dout):
    scaled = (q * (1.0 / math.sqrt(q.shape[-1]))).astype(q.dtype)
    return _dot(ds.T, scaled).astype(q.dtype), _dot(prob.astype(dout.dtype).T, dout).astype(dout.dtype)


def _kv_grad(dsr, pr, qr, dor, dkr, dvr):
    dk, dv = _kv_grad_math(dsr[...], pr[...], qr[...], dor[...])
    dkr[...] = dk
    dvr[...] = dv


def _band_grad(dsr, dbr, *, extent, length):
    key_block = pl.program_id(1) + pl.program_id(2) - extent // TILE
    dbr[...] = jnp.where((key_block >= 0) & (key_block < length // TILE), dsr[...], 0)


def score_tiles(q, k, band, qids, kids):
    """Return local causal score tiles with the source activation rounding."""
    groups, length, dim = q.shape
    width = band.shape[-1]
    extent = width - TILE
    grid = (groups, length // TILE, length // TILE)
    # A singleton sublane dimension lets each program read one group while
    # satisfying TPU's 8-by-128 trailing tile contract.
    inputs = (q, k, band, qids[:, None, :], kids[:, None, :])
    score_shape = _shape((groups, length, length), q.dtype)
    tile = _shape((TILE, dim), q.dtype)
    ids = _shape((TILE,), jnp.int32)
    return pl.pallas_call(
        functools.partial(_scores, extent=extent, width=width),
        out_shape=score_shape,
        grid=grid,
        in_specs=(
            pl.BlockSpec((None, TILE, dim), lambda g, i, j: (g, i, 0)),
            pl.BlockSpec((None, TILE, dim), lambda g, i, j: (g, j, 0)),
            pl.BlockSpec(
                (None, TILE, TILE), lambda g, i, j: (g, i, jnp.clip(j - i + extent // TILE, 0, width // TILE - 1))
            ),
            pl.BlockSpec((None, 1, TILE), lambda g, i, j: (g, 0, i)),
            pl.BlockSpec((None, 1, TILE), lambda g, i, j: (g, 0, j)),
        ),
        out_specs=pl.BlockSpec((None, TILE, TILE), lambda g, i, j: (g, i, j)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel")),
        cost_estimate=_cost(_scores_math, inputs, (score_shape,), (tile, tile, tile, ids, ids), grid),
        name="inkling_scores_banded",
    )(*inputs)


def _forward(q, k, v, band, qids, kids):
    groups, length, dim = q.shape
    width = band.shape[-1]
    if jax.default_backend() != "tpu":
        raise ValueError("Inkling Pallas requires TPU")
    if dim != 128 or length % TILE or width % TILE or width <= TILE:
        raise ValueError("Inkling Pallas requires head width 128 and 128-aligned sequence/band")
    if q.shape != k.shape or q.shape != v.shape or any(x.dtype != q.dtype for x in (k, v, band)):
        raise ValueError("Inkling Pallas requires equal local Q/K/V shapes and activation dtypes")
    if q.dtype not in (jnp.float32, jnp.bfloat16):
        raise ValueError("Inkling Pallas supports FP32 and BF16 activations")
    scores = score_tiles(q, k, band, qids, kids)
    grid2 = (groups, length // TILE)
    outputs = (_shape(q.shape, v.dtype), _shape(scores.shape, jnp.float32))
    out, prob = pl.pallas_call(
        _probability,
        out_shape=outputs,
        grid=grid2,
        in_specs=(
            pl.BlockSpec((None, TILE, length), lambda g, i: (g, i, 0)),
            pl.BlockSpec((None, length, dim), lambda g, i: (g, 0, 0)),
        ),
        out_specs=(
            pl.BlockSpec((None, TILE, dim), lambda g, i: (g, i, 0)),
            pl.BlockSpec((None, TILE, length), lambda g, i: (g, i, 0)),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        cost_estimate=_cost(
            _probability_math,
            (scores, v),
            outputs,
            (_shape((TILE, length), q.dtype), _shape((length, dim), v.dtype)),
            grid2,
        ),
        name="inkling_probability_save",
    )(scores, v)
    return out, (q, k, v, prob, jnp.zeros((0, width), band.dtype))


@jax.custom_vjp
def pallas_attention_local(
    q: Float[Array, "G S D"],
    k: Float[Array, "G S D"],
    v: Float[Array, "G S D"],
    band: Float[Array, "G S W"],
    qids: Int[Array, "G S"],
    kids: Int[Array, "G S"],
) -> Float[Array, "G S D"]:
    """Compute causal attention on fully local arrays and save probabilities."""
    return _forward(q, k, v, band, qids, kids)[0]


def _backward(residual, dout):
    q, k, v, prob, shape_token = residual
    groups, length, dim = q.shape
    width = shape_token.shape[-1]
    extent = width - TILE
    grid = (groups, length // TILE)
    full = pl.BlockSpec((None, length, dim), lambda g, i: (g, 0, 0))
    rows = pl.BlockSpec((None, TILE, dim), lambda g, i: (g, i, 0))
    prob_rows = pl.BlockSpec((None, TILE, length), lambda g, i: (g, i, 0))
    shapes = (_shape(q.shape, q.dtype), _shape(prob.shape, q.dtype))
    inputs = (prob, k, v, dout)
    dq, ds = pl.pallas_call(
        _query_grad,
        out_shape=shapes,
        grid=grid,
        in_specs=(prob_rows, full, full, rows),
        out_specs=(rows, prob_rows),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        cost_estimate=_cost(
            _query_grad_math,
            inputs,
            shapes,
            (
                _shape((TILE, length), jnp.float32),
                _shape((length, dim), k.dtype),
                _shape((length, dim), v.dtype),
                _shape((TILE, dim), dout.dtype),
            ),
            grid,
        ),
        name="inkling_query_gradient",
    )(*inputs)
    columns = pl.BlockSpec((None, length, TILE), lambda g, i: (g, 0, i))
    shapes = (_shape(k.shape, k.dtype), _shape(v.shape, v.dtype))
    inputs = (ds, prob, q, dout)
    dk, dv = pl.pallas_call(
        _kv_grad,
        out_shape=shapes,
        grid=grid,
        in_specs=(columns, columns, full, full),
        out_specs=(rows, rows),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        cost_estimate=_cost(
            _kv_grad_math,
            inputs,
            shapes,
            (
                _shape((length, TILE), ds.dtype),
                _shape((length, TILE), jnp.float32),
                _shape((length, dim), q.dtype),
                _shape((length, dim), dout.dtype),
            ),
            grid,
        ),
        name="inkling_key_value_gradient",
    )(*inputs)
    band_shape = _shape((groups, length, width), shape_token.dtype)
    grid3 = (groups, length // TILE, width // TILE)
    dband = pl.pallas_call(
        functools.partial(_band_grad, extent=extent, length=length),
        out_shape=band_shape,
        grid=grid3,
        in_specs=(
            pl.BlockSpec(
                (None, TILE, TILE), lambda g, i, j: (g, i, jnp.clip(i + j - extent // TILE, 0, length // TILE - 1))
            ),
        ),
        out_specs=pl.BlockSpec((None, TILE, TILE), lambda g, i, j: (g, i, j)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel")),
        cost_estimate=_cost(
            lambda x: jnp.where(jnp.isfinite(x), x, 0), (ds,), (band_shape,), (_shape((TILE, TILE), ds.dtype),), grid3
        ),
        name="inkling_band_gradient",
    )(ds)
    return dq, dk, dv, dband, None, None


pallas_attention_local.defvjp(_forward, _backward)
