# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experimental, device-local TPU KDA state pass with an explicit reverse pass.

Inputs are prepared FP32 chunk tensors, including document masks from kda.py.
One ordered chunk stream owns each FP32 state in VMEM. This module is not
selected by the model: callers must explicitly shard/map local tensors first.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _dot(a, b, precision=jax.lax.Precision.HIGHEST):
    precision = precision if a.dtype == jnp.float32 else jax.lax.Precision.DEFAULT
    return jnp.matmul(a, b, precision=precision, preferred_element_type=jnp.float32)


def _intra(a, b):
    # Match the reference's BF16 operands AND BF16 result, then upcast.
    return _dot(a.astype(jnp.bfloat16), b.astype(jnp.bfloat16)).astype(jnp.bfloat16).astype(jnp.float32)


def _chunk(q, a, k, w, v, decay, state, *, precision):
    vn = v - _dot(w, state, precision)
    out = _dot(q, state, precision) + _intra(a, vn)
    return out, decay.T * state + _dot(k.T, vn, precision), vn


def _forward(qr, ar, kr, wr, vr, dr, s0r, outr, hr, vnr, str_, scratch, *, precision):
    @pl.when(pl.program_id(1) == 0)
    def initialize():
        scratch[...] = s0r[...]

    state = scratch[...]
    out, following, vn = _chunk(qr[...], ar[...], kr[...], wr[...], vr[...], dr[...], state, precision=precision)
    hr[...] = state
    vnr[...] = vn
    outr[...] = out
    scratch[...] = following
    str_[...] = following


def _chunk_backward(q, a, k, w, decay, state, vn, dout, ds, *, precision):
    dv = _intra(a.T, dout) + _dot(k, ds, precision)
    before = decay.T * ds - _dot(w.T, dv, precision) + _dot(q.T, dout, precision)
    return (
        _dot(dout, state.T, precision),
        _intra(dout, vn.T),
        _dot(vn, ds.T, precision),
        -_dot(dv, state.T, precision),
        dv,
        # Reduce over sublanes so the result already occupies lanes. The
        # opposite layout needs a sublane gather unavailable on TPU v4.
        jnp.sum((state * ds).T, axis=0)[None, :],
        before,
    )


def _backward(qr, ar, kr, wr, dr, hr, vnr, dor, dstr, dqr, dar, dkr, dwr, dvr, ddr, ds0r, scratch, *, precision):
    @pl.when(pl.program_id(1) == 0)
    def initialize():
        scratch[...] = dstr[...]

    grads = _chunk_backward(
        qr[...], ar[...], kr[...], wr[...], dr[...], hr[...], vnr[...], dor[...], scratch[...], precision=precision
    )
    for ref, value in zip((dqr, dar, dkr, dwr, dvr, ddr, ds0r), grads, strict=True):
        ref[...] = value
    scratch[...] = grads[-1]


def _spec(rows, cols, chunks, *, reverse=False):
    return pl.BlockSpec(
        (None, None, rows, cols),
        lambda g, n: (g, chunks - 1 - n if reverse else n, 0, 0),
    )


def _state_spec():
    return pl.BlockSpec((None, 128, 128), lambda g, n: (g, 0, 0))


def _cost(body, operands, grid_size, inputs, outputs):
    cost = pl.estimate_cost(body, *operands)
    io = sum(math.prod(x.shape) * jnp.dtype(x.dtype).itemsize for x in (*inputs, *outputs))
    return pl.CostEstimate(
        flops=cost.flops * grid_size, transcendentals=cost.transcendentals * grid_size, bytes_accessed=io
    )


def _forward_call(q, a, k, w, v, decay, initial, precision):
    groups, chunks, c, d = q.shape
    if d != 128 or v.shape != q.shape or c not in (16, 32):
        raise ValueError("TPU state prototype requires K=V=128 and C=16 or 32")
    if any(x.dtype != jnp.float32 for x in (q, a, k, w, v, decay, initial)):
        raise ValueError("Prepared state-pass inputs must be FP32")
    if jax.default_backend() != "tpu":
        raise ValueError("TPU state prototype requires a TPU backend")
    inputs = (q, a, k, w, v, decay, initial)

    def shape(dims):
        return jax.ShapeDtypeStruct(dims, jnp.float32)

    outputs = (shape(q.shape), shape((groups, chunks, d, d)), shape(v.shape), shape(initial.shape))
    chunk_shapes = [shape(x.shape[-2:]) for x in inputs]
    return pl.pallas_call(
        functools.partial(_forward, precision=precision),
        out_shape=outputs,
        grid=(groups, chunks),
        in_specs=[*[_spec(*x.shape[-2:], chunks) for x in inputs[:-1]], _state_spec()],
        out_specs=[_spec(c, d, chunks), _spec(d, d, chunks), _spec(c, d, chunks), _state_spec()],
        scratch_shapes=[pltpu.VMEM((d, d), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        cost_estimate=_cost(
            functools.partial(_chunk, precision=precision), chunk_shapes, groups * chunks, inputs, outputs
        ),
        name="kda_tpu_state_forward",
    )(*inputs)


@functools.partial(jax.custom_vjp, nondiff_argnums=(7,))
def state_pass(q, a, k, w, v, decay, initial, precision=jax.lax.Precision.HIGHEST):
    """Return output/final state for fully local [G,N,C,128] prepared chunks."""
    out, _, _, final = _forward_call(q, a, k, w, v, decay, initial, precision)
    return out, final


def _fwd(q, a, k, w, v, decay, initial, precision):
    out, states, vn, final = _forward_call(q, a, k, w, v, decay, initial, precision)
    return (out, final), (q, a, k, w, decay, states, vn)


def _bwd(precision, residual, cotangents):
    q, a, k, w, decay, _states, vn = residual
    dout, dfinal = cotangents
    groups, chunks, _c, d = q.shape
    inputs = (*residual, dout, dfinal)
    shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in (q, a, k, w, vn, decay, dfinal)]
    chunk_shapes = [jax.ShapeDtypeStruct(x.shape[-2:], x.dtype) for x in inputs]
    return pl.pallas_call(
        functools.partial(_backward, precision=precision),
        out_shape=tuple(shapes),
        grid=(groups, chunks),
        in_specs=[*[_spec(*x.shape[-2:], chunks, reverse=True) for x in inputs[:-1]], _state_spec()],
        out_specs=[*[_spec(*x.shape[-2:], chunks, reverse=True) for x in shapes[:-1]], _state_spec()],
        scratch_shapes=[pltpu.VMEM((d, d), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        cost_estimate=_cost(
            functools.partial(_chunk_backward, precision=precision), chunk_shapes, groups * chunks, inputs, shapes
        ),
        name="kda_tpu_state_backward",
    )(*inputs)


state_pass.defvjp(_fwd, _bwd)
