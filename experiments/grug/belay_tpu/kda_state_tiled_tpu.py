# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Batch consecutive C=16 KDA chunks into one TPU DMA tile; save every state."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from experiments.grug.belay_tpu.kda_state_tpu import _chunk, _chunk_backward, _cost, _state_spec


def _forward(qr, ar, kr, wr, vr, dr, s0r, outr, hr, vnr, str_, scratch, *, precision, matmul_dtype, tile):
    @pl.when(pl.program_id(1) == 0)
    def initialize():
        scratch[...] = s0r[...]

    state = scratch[...]
    for index in range(tile):
        out, following, vn = _chunk(
            qr[index],
            ar[index],
            kr[index],
            wr[index],
            vr[index],
            dr[index],
            state,
            precision=precision,
            matmul_dtype=matmul_dtype,
        )
        hr[index] = state
        vnr[index] = vn
        outr[index] = out
        state = following
    scratch[...] = state
    str_[...] = state


def _backward(
    qr,
    ar,
    kr,
    wr,
    dr,
    hr,
    vnr,
    dor,
    dstr,
    dqr,
    dar,
    dkr,
    dwr,
    dvr,
    ddr,
    ds0r,
    scratch,
    *,
    precision,
    matmul_dtype,
    tile,
):
    @pl.when(pl.program_id(1) == 0)
    def initialize():
        scratch[...] = dstr[...]

    state_grad = scratch[...]
    for index in reversed(range(tile)):
        grads = _chunk_backward(
            qr[index],
            ar[index],
            kr[index],
            wr[index],
            dr[index],
            hr[index],
            vnr[index],
            dor[index],
            state_grad,
            precision=precision,
            matmul_dtype=matmul_dtype,
        )
        for ref, value in zip((dqr, dar, dkr, dwr, dvr, ddr), grads[:-1], strict=True):
            ref[index] = value
        state_grad = grads[-1]
    scratch[...] = state_grad
    ds0r[...] = state_grad


def _spec(shape, chunks, tile, *, reverse=False):
    return pl.BlockSpec(
        (None, tile, *shape),
        lambda group, block: (group, chunks // tile - 1 - block if reverse else block, 0, 0),
    )


def _forward_call(q, a, k, w, v, decay, initial, precision, matmul_dtype, tile, interpret):
    groups, chunks, c, d = q.shape
    if d != 128 or c != 16 or chunks % tile:
        raise ValueError("Tiled KDA needs C=16, D=128, and a chunk count divisible by the DMA tile")
    inputs = (q, a, k, w, v, decay, initial)
    shape = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.float32)
    outputs = (shape(q.shape), shape((groups, chunks, d, d)), shape(v.shape), shape(initial.shape))
    chunk_shapes = [shape(x.shape[-2:]) for x in inputs]
    return pl.pallas_call(
        functools.partial(_forward, precision=precision, matmul_dtype=matmul_dtype, tile=tile),
        out_shape=outputs,
        grid=(groups, chunks // tile),
        in_specs=[*[_spec(x.shape[-2:], chunks, tile) for x in inputs[:-1]], _state_spec()],
        out_specs=[_spec((c, d), chunks, tile), _spec((d, d), chunks, tile), _spec((c, d), chunks, tile), _state_spec()],
        scratch_shapes=[pltpu.VMEM((d, d), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        cost_estimate=_cost(
            functools.partial(_chunk, precision=precision, matmul_dtype=matmul_dtype),
            chunk_shapes,
            groups * chunks,
            inputs,
            outputs,
        ),
        interpret=interpret,
        name="kda_tpu_tiled_state_forward",
    )(*inputs)


@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10))
def state_pass_tiled(
    q,
    a,
    k,
    w,
    v,
    decay,
    initial,
    precision=jax.lax.Precision.HIGHEST,
    matmul_dtype=jnp.bfloat16,
    tile=4,
    interpret=False,
):
    """Preserve ordered C=16 recurrence while loading multiple chunks per program."""
    out, _, _, final = _forward_call(q, a, k, w, v, decay, initial, precision, matmul_dtype, tile, interpret)
    return out, final


def _fwd(q, a, k, w, v, decay, initial, precision, matmul_dtype, tile, interpret):
    out, states, vn, final = _forward_call(q, a, k, w, v, decay, initial, precision, matmul_dtype, tile, interpret)
    return (out, final), (q, a, k, w, decay, states, vn)


def _bwd(precision, matmul_dtype, tile, interpret, residual, cotangents):
    q, a, k, w, decay, _states, vn = residual
    dout, dfinal = cotangents
    groups, chunks, _c, d = q.shape
    inputs = (*residual, dout, dfinal)
    shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in (q, a, k, w, vn, decay, dfinal)]
    chunk_shapes = [jax.ShapeDtypeStruct(x.shape[-2:], x.dtype) for x in inputs]
    return pl.pallas_call(
        functools.partial(_backward, precision=precision, matmul_dtype=matmul_dtype, tile=tile),
        out_shape=tuple(shapes),
        grid=(groups, chunks // tile),
        in_specs=[*[_spec(x.shape[-2:], chunks, tile, reverse=True) for x in inputs[:-1]], _state_spec()],
        out_specs=[*[_spec(x.shape[-2:], chunks, tile, reverse=True) for x in shapes[:-1]], _state_spec()],
        scratch_shapes=[pltpu.VMEM((d, d), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        cost_estimate=_cost(
            functools.partial(_chunk_backward, precision=precision, matmul_dtype=matmul_dtype),
            chunk_shapes,
            groups * chunks,
            inputs,
            shapes,
        ),
        interpret=interpret,
        name="kda_tpu_tiled_state_backward",
    )(*inputs)


state_pass_tiled.defvjp(_fwd, _bwd)
