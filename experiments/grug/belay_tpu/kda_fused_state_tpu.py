# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""KDA preparation and saved-state recurrence in one ordered TPU program.

Backward retains prefix, inverse, solutions, attention, states and corrected
values. It reconstructs elementwise state operands, consumes their cotangents
in VMEM, and never repeats the forward solve or state recurrence.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from experiments.grug.belay_tpu.kda_fused_prep_tpu import _backward_values, _factors, _forward_values, _masks
from experiments.grug.belay_tpu.kda_state_tpu import _chunk, _chunk_backward
from experiments.grug.moe.kda import _SegmentMasks


def _forward(*refs, dtype, precision):
    inputs = tuple(r[...] for r in refs[:10])
    initial = refs[10]
    out, final, states, corrected, attention_ref, inverse_ref, prefix_ref, kcd_ref, vp_ref = refs[11:20]
    scratch = refs[20]

    @pl.when(pl.program_id(1) == 0)
    def initialize():
        scratch[...] = initial[...]

    qi, kcd, vp, attention, kw, decay, inverse, prefix = _forward_values(*inputs, dtype=dtype)
    state = scratch[...]
    output, following, vn = _chunk(qi, attention, kw, kcd, vp, decay, state, precision=precision, matmul_dtype=dtype)
    for ref, value in zip(
        (out, final, states, corrected, attention_ref, inverse_ref, prefix_ref, kcd_ref, vp_ref),
        (output, following, state, vn, attention, inverse, prefix, kcd, vp),
        strict=True,
    ):
        ref[...] = value
    scratch[...] = following


def _backward(*refs, dtype, precision):
    residual = tuple(r[...] for r in refs[:16])
    q, k, e, v, prefix, beta, same, carry, last, through, _inverse, kcd, _vp, attention, state, vn = residual
    dout, dfinal = refs[16:18]
    scratch = refs[25]

    @pl.when(pl.program_id(1) == 0)
    def initialize():
        scratch[...] = dfinal[...]

    # Only these three elementwise outputs are live. The unused attention/RHS
    # products are eliminated before machine compilation.
    qi, _, kw, decay, _, _ = _factors(q, k, e, v, prefix, beta, _masks(same, carry, last, through), dtype)
    dqi, da, dkw, dkcd, dvp, ddecay, before = _chunk_backward(
        qi,
        attention,
        kw,
        kcd,
        decay[:, None, :],
        state,
        vn,
        dout[...],
        scratch[...],
        precision=precision,
        matmul_dtype=dtype,
    )
    gradients = _backward_values(*residual[:13], dqi, dkcd, dvp, da, dkw, ddecay, dtype=dtype)
    for ref, value in zip(refs[18:25], (*gradients, before), strict=True):
        ref[...] = value
    scratch[...] = before


def _spec(shape, chunks, heads, reverse=False):
    return pl.BlockSpec(
        (heads, None, *shape[-2:]),
        lambda group, chunk: (group, chunks - 1 - chunk if reverse else chunk, 0, 0),
    )


def _state_spec(heads):
    return pl.BlockSpec((heads, 128, 128), lambda group, chunk: (group, 0, 0))


def _shape(x):
    return jax.ShapeDtypeStruct(x.shape, x.dtype)


def _forward_call(inputs, dtype, precision, heads, interpret):
    q, _k, _e, _v, _g, _beta, same, *_ = inputs
    streams, chunks, _c, d = q.shape
    outputs = (
        _shape(q),
        _shape(inputs[-1]),
        jax.ShapeDtypeStruct((streams, chunks, d, d), jnp.float32),
        _shape(q),
        jax.ShapeDtypeStruct(same.shape, jnp.float32),
        jax.ShapeDtypeStruct(same.shape, jnp.float32),
        _shape(q),
        _shape(q),
        _shape(q),
    )
    return pl.pallas_call(
        functools.partial(_forward, dtype=dtype, precision=precision),
        out_shape=outputs,
        grid=(streams // heads, chunks),
        in_specs=(*(_spec(x.shape, chunks, heads) for x in inputs[:-1]), _state_spec(heads)),
        out_specs=tuple(_state_spec(heads) if i == 1 else _spec(x.shape, chunks, heads) for i, x in enumerate(outputs)),
        scratch_shapes=[pltpu.VMEM((heads, d, d), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        interpret=interpret,
        name="kda_tpu_fused_state_forward",
    )(*inputs)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4))
def _run(inputs, dtype, precision, heads, interpret):
    return _forward_call(inputs, dtype, precision, heads, interpret)[:2]


def _run_fwd(inputs, dtype, precision, heads, interpret):
    out, final, states, vn, attention, inverse, prefix, kcd, vp = _forward_call(
        inputs, dtype, precision, heads, interpret
    )
    residual = (*inputs[:4], prefix, *inputs[5:10], inverse, kcd, vp, attention, states, vn)
    return (out, final), residual


def _run_bwd(dtype, precision, heads, interpret, residual, cotangents):
    inputs = (*residual, *cotangents)
    streams, chunks, _c, d = residual[0].shape
    outputs = tuple(_shape(x) for x in (*residual[:6], cotangents[1]))
    gradients = pl.pallas_call(
        functools.partial(_backward, dtype=dtype, precision=precision),
        out_shape=outputs,
        grid=(streams // heads, chunks),
        in_specs=(*(_spec(x.shape, chunks, heads, True) for x in inputs[:-1]), _state_spec(heads)),
        out_specs=(*(_spec(x.shape, chunks, heads, True) for x in outputs[:-1]), _state_spec(heads)),
        scratch_shapes=[pltpu.VMEM((heads, d, d), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        interpret=interpret,
        name="kda_tpu_fused_state_backward",
    )(*inputs)
    return ((*gradients[:6], None, None, None, None, gradients[6]),)


_run.defvjp(_run_fwd, _run_bwd)


def fused_chunk_state(q, k, e, v, g, beta, masks, initial, *, dtype, precision, heads=1, interpret=False):
    """Return local chunk output/final state with preparation kept inside VMEM."""
    if q.shape[-1] != 128 or q.shape[-2] not in (16, 32, 64) or heads not in (1, 4):
        raise ValueError("Fused KDA requires C16/32/64, D128 and one/four heads per program")
    if masks is None:
        lead, c = q.shape[:-2], q.shape[-2]
        masks = _SegmentMasks(
            jnp.ones((*lead, c, c), dtype=bool),
            jnp.ones((*lead, c), jnp.float32),
            jnp.ones((*lead, c), jnp.float32),
            jnp.ones(lead, jnp.float32),
        )
    n = q.shape[-3]
    arrays = (
        q,
        k,
        e,
        v,
        g,
        beta[..., None],
        masks.same,
        masks.carry_in[..., None],
        masks.last_doc[..., None],
        masks.carry_through[..., None, None],
    )
    arrays = tuple(x.reshape(-1, n, *x.shape[-2:]) for x in arrays)
    if arrays[0].shape[0] % heads:
        raise ValueError("KDA streams must form complete head groups")
    return _run((*arrays, initial), dtype, precision, heads, interpret)
