# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Device-local TPU KDA state pass with saved states and an explicit reverse pass.

Inputs are prepared FP32 chunk tensors, including document masks from kda.py.
One or four head streams share each ordered VMEM program. Saved states and
corrected values retain the original external layout. This module is not
selected by the model: callers must explicitly shard/map local tensors first.
Ported from belay-base16-tpu-v4 aef63f29d; Ckpt 1 qualification is separate.
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


def _intra(a, b, dtype):
    # Match the reference's operand AND result casts, then upcast.
    return _dot(a.astype(dtype), b.astype(dtype)).astype(dtype).astype(jnp.float32)


def _transpose(x):
    return jnp.swapaxes(x, -1, -2)


def _chunk(q, a, k, w, v, decay, state, *, precision, matmul_dtype):
    vn = v - _dot(w, state, precision)
    out = _dot(q, state, precision) + _intra(a, vn, matmul_dtype)
    return out, _transpose(decay) * state + _dot(_transpose(k), vn, precision), vn


def _forward(qr, ar, kr, wr, vr, dr, s0r, outr, hr, vnr, str_, scratch, *, precision, matmul_dtype):
    @pl.when(pl.program_id(1) == 0)
    def initialize():
        scratch[...] = s0r[...]

    state = scratch[...]
    out, following, vn = _chunk(
        qr[...],
        ar[...],
        kr[...],
        wr[...],
        vr[...],
        dr[...],
        state,
        precision=precision,
        matmul_dtype=matmul_dtype,
    )
    hr[...] = state
    vnr[...] = vn
    outr[...] = out
    scratch[...] = following
    str_[...] = following


def _chunk_backward(q, a, k, w, decay, state, vn, dout, ds, *, precision, matmul_dtype):
    dv = _intra(_transpose(a), dout, matmul_dtype) + _dot(k, ds, precision)
    before = _transpose(decay) * ds - _dot(_transpose(w), dv, precision) + _dot(_transpose(q), dout, precision)
    return (
        _dot(dout, _transpose(state), precision),
        _intra(dout, _transpose(vn), matmul_dtype),
        _dot(vn, _transpose(ds), precision),
        -_dot(dv, _transpose(state), precision),
        dv,
        # Reduce over sublanes so the result already occupies lanes. The
        # opposite layout needs a sublane gather unavailable on TPU v4.
        jnp.sum(_transpose(state * ds), axis=-2)[..., None, :],
        before,
    )


def _backward(
    qr, ar, kr, wr, dr, hr, vnr, dor, dstr, dqr, dar, dkr, dwr, dvr, ddr, ds0r, scratch, *, precision, matmul_dtype
):
    @pl.when(pl.program_id(1) == 0)
    def initialize():
        scratch[...] = dstr[...]

    grads = _chunk_backward(
        qr[...],
        ar[...],
        kr[...],
        wr[...],
        dr[...],
        hr[...],
        vnr[...],
        dor[...],
        scratch[...],
        precision=precision,
        matmul_dtype=matmul_dtype,
    )
    for ref, value in zip((dqr, dar, dkr, dwr, dvr, ddr, ds0r), grads, strict=True):
        ref[...] = value
    scratch[...] = grads[-1]


def _spec(rows, cols, chunks, *, reverse=False, head_group=1):
    return pl.BlockSpec(
        (None if head_group == 1 else head_group, None, rows, cols),
        lambda g, n: (g, chunks - 1 - n if reverse else n, 0, 0),
    )


def _state_spec(head_group=1):
    return pl.BlockSpec((None if head_group == 1 else head_group, 128, 128), lambda g, n: (g, 0, 0))


def _cost(body, operands, grid_size, inputs, outputs):
    cost = pl.estimate_cost(body, *operands)
    io = sum(math.prod(x.shape) * jnp.dtype(x.dtype).itemsize for x in (*inputs, *outputs))
    return pl.CostEstimate(
        flops=cost.flops * grid_size, transcendentals=cost.transcendentals * grid_size, bytes_accessed=io
    )


def _forward_call(q, a, k, w, v, decay, initial, precision, matmul_dtype, head_group, interpret):
    groups, chunks, c, d = q.shape
    if d != 128 or v.shape != q.shape or c not in (16, 32, 64, 128):
        raise ValueError("TPU state prototype requires K=V=128 and C=16, 32, 64 or 128")
    if any(x.dtype != jnp.float32 for x in (q, a, k, w, v, decay, initial)):
        raise ValueError("Prepared state-pass inputs must be FP32")
    if head_group not in (1, 4) or groups % head_group:
        raise ValueError("State streams must form complete groups of one or four heads")
    if not interpret and jax.default_backend() != "tpu":
        raise ValueError("TPU state prototype requires a TPU backend")
    inputs = (q, a, k, w, v, decay, initial)

    def shape(dims):
        return jax.ShapeDtypeStruct(dims, jnp.float32)

    outputs = (shape(q.shape), shape((groups, chunks, d, d)), shape(v.shape), shape(initial.shape))
    prefix = () if head_group == 1 else (head_group,)
    chunk_shapes = [shape((*prefix, *x.shape[-2:])) for x in inputs]
    return pl.pallas_call(
        functools.partial(_forward, precision=precision, matmul_dtype=matmul_dtype),
        out_shape=outputs,
        grid=(groups // head_group, chunks),
        in_specs=[*[_spec(*x.shape[-2:], chunks, head_group=head_group) for x in inputs[:-1]], _state_spec(head_group)],
        out_specs=[
            _spec(c, d, chunks, head_group=head_group),
            _spec(d, d, chunks, head_group=head_group),
            _spec(c, d, chunks, head_group=head_group),
            _state_spec(head_group),
        ],
        scratch_shapes=[pltpu.VMEM((*prefix, d, d), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        cost_estimate=_cost(
            functools.partial(_chunk, precision=precision, matmul_dtype=matmul_dtype),
            chunk_shapes,
            groups // head_group * chunks,
            inputs,
            outputs,
        ),
        interpret=interpret,
        name="kda_tpu_state_forward" if head_group == 1 else "kda_tpu_state_heads_forward",
    )(*inputs)


@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10))
def state_pass(
    q,
    a,
    k,
    w,
    v,
    decay,
    initial,
    precision=jax.lax.Precision.HIGHEST,
    matmul_dtype=jnp.bfloat16,
    head_group=1,
    interpret=False,
):
    """Return output/final state for local [G,N,C,128] chunks, grouping consecutive heads."""
    out, _, _, final = _forward_call(q, a, k, w, v, decay, initial, precision, matmul_dtype, head_group, interpret)
    return out, final


def _fwd(q, a, k, w, v, decay, initial, precision, matmul_dtype, head_group, interpret):
    out, states, vn, final = _forward_call(q, a, k, w, v, decay, initial, precision, matmul_dtype, head_group, interpret)
    return (out, final), (q, a, k, w, decay, states, vn)


def _bwd(precision, matmul_dtype, head_group, interpret, residual, cotangents):
    q, a, k, w, decay, _states, vn = residual
    dout, dfinal = cotangents
    groups, chunks, _c, d = q.shape
    inputs = (*residual, dout, dfinal)
    shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in (q, a, k, w, vn, decay, dfinal)]
    prefix = () if head_group == 1 else (head_group,)
    chunk_shapes = [jax.ShapeDtypeStruct((*prefix, *x.shape[-2:]), x.dtype) for x in inputs]
    return pl.pallas_call(
        functools.partial(_backward, precision=precision, matmul_dtype=matmul_dtype),
        out_shape=tuple(shapes),
        grid=(groups // head_group, chunks),
        in_specs=[
            *[_spec(*x.shape[-2:], chunks, reverse=True, head_group=head_group) for x in inputs[:-1]],
            _state_spec(head_group),
        ],
        out_specs=[
            *[_spec(*x.shape[-2:], chunks, reverse=True, head_group=head_group) for x in shapes[:-1]],
            _state_spec(head_group),
        ],
        scratch_shapes=[pltpu.VMEM((*prefix, d, d), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        cost_estimate=_cost(
            functools.partial(_chunk_backward, precision=precision, matmul_dtype=matmul_dtype),
            chunk_shapes,
            groups // head_group * chunks,
            inputs,
            shapes,
        ),
        interpret=interpret,
        name="kda_tpu_state_backward" if head_group == 1 else "kda_tpu_state_heads_backward",
    )(*inputs)


state_pass.defvjp(_fwd, _bwd)
