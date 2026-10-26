# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TPU KDA preparation with saved solves and tile-local reverse-mode algebra.

Forward keeps prefix, attention, inverse construction and RHS products in one
Pallas program. Backward reuses the saved prefix, inverse and solutions. It
re-evaluates elementwise factors in VMEM but does not repeat forward matrix
products or the state recurrence.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from experiments.grug.belay_tpu.kda_triangular_tpu import (
    _inverse,
    _prep_factors,
    _product,
    _solve_gradients,
    _transpose,
)
from experiments.grug.moe.kda import _SegmentMasks
from experiments.grug.moe.kda_prep_pallas import ChunkPrep


def _prefix_product(left, right):
    return jax.lax.dot_general(
        left.astype(jnp.float32),
        right,
        (((2,), (1,)), ((0,), (0,))),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )


def _masks(same, carry, last, through):
    return _SegmentMasks(same, carry[..., 0], last[..., 0], through[..., 0, 0])


def _factors(q, k, e, v, cumulative, beta, masks, dtype):
    def mm(_spec, left, right):
        return _product(left, _transpose(right), dtype)

    return _prep_factors(q, k, e, v, cumulative, beta[..., 0], mm, q.shape[-2], masks)


def _forward_values(q, k, e, v, g, beta, same, carry, last, through, *, dtype):
    lower = same & jnp.tril(jnp.ones(same.shape[-2:], dtype=bool))
    cumulative = _prefix_product(lower, g)
    qi, attention, kw, decay, a, rhs = _factors(q, k, e, v, cumulative, beta, _masks(same, carry, last, through), dtype)
    inverse = _inverse(a, dtype)
    solved = _product(inverse, rhs, dtype)
    return qi, solved[..., 128:], solved[..., :128], attention, kw, decay[:, None, :], inverse, cumulative


def _forward(*refs, dtype):
    values = _forward_values(*(r[...] for r in refs[:10]), dtype=dtype)
    for ref, value in zip(refs[10:], values, strict=True):
        ref[...] = value


def _backward_values(*inputs, dtype):
    q, k, e, v, cumulative, beta, same, carry, last, through, inverse, kcd, vp = inputs[:13]
    dqi, dkcd, dvp, dattention, dkw, ddecay = inputs[13:]
    da, drhs = _solve_gradients(inverse, jnp.concatenate((vp, kcd), -1), jnp.concatenate((dvp, dkcd), -1), dtype)
    masks = _masks(same, carry, last, through)
    function = functools.partial(_factors, masks=masks, dtype=dtype)
    # Transpose the elementwise and bilinear preparation graph inside VMEM.
    # Its unused primal outputs are eliminated; no inverse/solve is traced here.
    _, backward = jax.vjp(function, q, k, e, v, cumulative, beta)
    dq, dk, de, dv, dcumulative, dbeta = backward((dqi, dattention, dkw, ddecay[:, 0, :], da, drhs))
    lower = same & jnp.tril(jnp.ones(same.shape[-2:], dtype=bool))
    dg = _prefix_product(_transpose(lower), dcumulative)
    return dq, dk, de, dv, dg, dbeta


def _backward(*refs, dtype):
    values = _backward_values(*(r[...] for r in refs[:19]), dtype=dtype)
    for ref, value in zip(refs[19:], values, strict=True):
        ref[...] = value


def _spec(shape, groups):
    return pl.BlockSpec((groups, *shape[1:]), lambda block: (block, 0, 0))


def _shape(value):
    return jax.ShapeDtypeStruct(value.shape, value.dtype)


def _forward_call(inputs, dtype, groups, interpret):
    q, _k, _e, _v, _g, _b, same, *_masks = inputs
    outputs = (
        _shape(q),
        _shape(q),
        _shape(q),
        jax.ShapeDtypeStruct(same.shape, jnp.float32),
        _shape(q),
        jax.ShapeDtypeStruct((q.shape[0], 1, 128), jnp.float32),
        jax.ShapeDtypeStruct(same.shape, jnp.float32),
        _shape(q),
    )
    return pl.pallas_call(
        functools.partial(_forward, dtype=dtype),
        out_shape=outputs,
        grid=(q.shape[0] // groups,),
        in_specs=tuple(_spec(x.shape, groups) for x in inputs),
        out_specs=tuple(_spec(x.shape, groups) for x in outputs),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="kda_tpu_prep_forward",
    )(*inputs)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def _prepare(inputs, dtype, groups, interpret):
    return _forward_call(inputs, dtype, groups, interpret)[:6]


def _prepare_fwd(inputs, dtype, groups, interpret):
    *outputs, inverse, cumulative = _forward_call(inputs, dtype, groups, interpret)
    residual = (*inputs[:4], cumulative, *inputs[5:], inverse, outputs[1], outputs[2])
    return tuple(outputs), residual


def _prepare_bwd(dtype, groups, interpret, residual, cotangents):
    inputs = (*residual, *cotangents)
    outputs = tuple(_shape(x) for x in residual[:6])
    gradients = pl.pallas_call(
        functools.partial(_backward, dtype=dtype),
        out_shape=outputs,
        grid=(residual[0].shape[0] // groups,),
        in_specs=tuple(_spec(x.shape, groups) for x in inputs),
        out_specs=tuple(_spec(x.shape, groups) for x in outputs),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="kda_tpu_prep_backward",
    )(*inputs)
    return ((*gradients, None, None, None, None),)


_prepare.defvjp(_prepare_fwd, _prepare_bwd)


def chunk_prep_fused(qc, kc, ec, vc, gc, bc, _mm, mm_dtype, c, masks, *, groups=4, interpret=False):
    """Prepare local KDA chunks, retaining FP32 solve factors and gate prefixes."""
    if c not in (16, 32, 64) or qc.shape[-1] != 128 or groups not in (1, 4):
        raise ValueError("Fused preparation requires C16/32/64, D128 and one/four systems")
    if masks is None:
        lead = qc.shape[:-2]
        masks = _SegmentMasks(
            jnp.ones((*lead, c, c), dtype=bool),
            jnp.ones((*lead, c), jnp.float32),
            jnp.ones((*lead, c), jnp.float32),
            jnp.ones(lead, jnp.float32),
        )
    count = math.prod(qc.shape[:-2])
    arrays = (
        qc,
        kc,
        ec,
        vc,
        gc,
        bc[..., None],
        masks.same,
        masks.carry_in[..., None],
        masks.last_doc[..., None],
        masks.carry_through[..., None, None],
    )
    arrays = tuple(x.reshape(count, *x.shape[-2:]) for x in arrays)
    padding = -count % groups
    if padding:
        arrays = tuple(jnp.pad(x, ((0, padding), (0, 0), (0, 0))) for x in arrays)
    outputs = tuple(x[:count] for x in _prepare(arrays, mm_dtype, groups, interpret))
    shaped = tuple(x.reshape(*qc.shape[:-2], *x.shape[-2:]) for x in outputs)
    return ChunkPrep(*shaped[:5], shaped[5][..., 0, :])
