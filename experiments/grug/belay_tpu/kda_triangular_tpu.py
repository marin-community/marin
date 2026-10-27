# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fused unit-lower solve for KDA, retaining the inverse and solution for VJP.

The input is strictly lower A and a concatenated value/state RHS. The result
solves (I-A) Y = RHS. Forward uses the stable block-doubling construction;
backward contracts the saved inverse and solution without replaying its AD graph.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from experiments.grug.moe.kda_prep_pallas import ChunkPrep


def _product(left, right, dtype):
    precision = jax.lax.Precision.HIGHEST if dtype == jnp.float32 else jax.lax.Precision.DEFAULT
    value = jax.lax.dot_general(
        left.astype(dtype),
        right.astype(dtype),
        (((2,), (1,)), ((0,), (0,))),
        precision=precision,
        preferred_element_type=jnp.float32,
    )
    return value.astype(dtype).astype(jnp.float32)


def _transpose(value):
    return jnp.swapaxes(value, -1, -2)


def _inverse(a, dtype):
    c = a.shape[-1]
    row = jax.lax.broadcasted_iota(jnp.int32, a.shape, 1)
    col = jax.lax.broadcasted_iota(jnp.int32, a.shape, 2)
    inverse = (row == col).astype(jnp.float32)
    for level in range(int(math.log2(c))):
        size = 1 << level
        mask = (row // (2 * size) == col // (2 * size)) & (row // size % 2 == 1) & (col // size % 2 == 0)
        inverse += _product(_product(inverse, jnp.where(mask, a, 0.0), dtype), inverse, dtype)
    return inverse


def _forward(ar, br, yr, tr, *, dtype):
    inverse = _inverse(ar[...], dtype)
    yr[...] = _product(inverse, br[...], dtype)
    tr[...] = inverse


def _solve_gradients(inverse, solution, dy, dtype):
    precision = jax.lax.Precision.HIGHEST if dtype == jnp.float32 else jax.lax.Precision.HIGH

    def dot(x, y):
        return jax.lax.dot_general(
            x, y, (((2,), (1,)), ((0,), (0,))), precision=precision, preferred_element_type=jnp.float32
        )

    db = dot(_transpose(inverse), dy)
    da = dot(db, _transpose(solution))
    row = jax.lax.broadcasted_iota(jnp.int32, da.shape, 1)
    col = jax.lax.broadcasted_iota(jnp.int32, da.shape, 2)
    return jnp.where(row > col, da, 0.0), db


def _backward(tr, yr, dyr, dar, dbr, *, dtype):
    da, db = _solve_gradients(tr[...], yr[...], dyr[...], dtype)
    dar[...] = da
    dbr[...] = db


def _spec(rows, width, groups):
    return pl.BlockSpec((groups, rows, width), lambda block: (block, 0, 0))


def _forward_call(a, rhs, dtype, groups, interpret):
    count, c, _ = a.shape
    inverse = jax.ShapeDtypeStruct(a.shape, jnp.float32)
    solution = jax.ShapeDtypeStruct(rhs.shape, jnp.float32)
    return pl.pallas_call(
        functools.partial(_forward, dtype=dtype),
        out_shape=(solution, inverse),
        grid=(count // groups,),
        in_specs=(_spec(c, c, groups), _spec(c, rhs.shape[-1], groups)),
        out_specs=(_spec(c, rhs.shape[-1], groups), _spec(c, c, groups)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="kda_triangular_solve_forward",
    )(a, rhs)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def _solve(a, rhs, dtype, groups, interpret):
    return _forward_call(a, rhs, dtype, groups, interpret)[0]


def _solve_fwd(a, rhs, dtype, groups, interpret):
    solution, inverse = _forward_call(a, rhs, dtype, groups, interpret)
    return solution, (inverse, solution)


def _solve_bwd(dtype, groups, interpret, residual, dy):
    inverse, solution = residual
    count, c, _ = inverse.shape
    return pl.pallas_call(
        functools.partial(_backward, dtype=dtype),
        out_shape=(jax.ShapeDtypeStruct(inverse.shape, jnp.float32), jax.ShapeDtypeStruct(solution.shape, jnp.float32)),
        grid=(count // groups,),
        in_specs=(_spec(c, c, groups), _spec(c, solution.shape[-1], groups), _spec(c, solution.shape[-1], groups)),
        out_specs=(_spec(c, c, groups), _spec(c, solution.shape[-1], groups)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="kda_triangular_solve_backward",
    )(inverse, solution, dy)


_solve.defvjp(_solve_fwd, _solve_bwd)


def solve_unit_lower(a, rhs, *, dtype=jnp.bfloat16, groups=4, interpret=False):
    """Solve independent KDA chunk systems; ignore diagonal and upper A entries.

    Inputs/outputs are FP32. BF16 controls forward products only; the analytic
    backward uses saved FP32 factors with HIGH products. It approximates the
    real solve derivative, not the sequence of rounded block-inverse products.
    """
    if a.ndim != 3 or rhs.ndim != 3 or a.shape[:2] != rhs.shape[:2] or a.shape[1] != a.shape[2]:
        raise ValueError("Expected A[N,C,C] and RHS[N,C,D]")
    if a.shape[1] not in (16, 32, 64) or groups not in (1, 4):
        raise ValueError("Solve requires C16/32/64 and one/four systems per program")
    if a.dtype != jnp.float32 or rhs.dtype != jnp.float32 or dtype not in (jnp.float32, jnp.bfloat16):
        raise ValueError("Solve inputs must be FP32 and products FP32/BF16")
    count = a.shape[0]
    pad = -count % groups
    if pad:
        a = jnp.pad(a, ((0, pad), (0, 0), (0, 0)))
        rhs = jnp.pad(rhs, ((0, pad), (0, 0), (0, 0)))
    return _solve(a, rhs, dtype, groups, interpret)[:count]


def _prep_factors(qc, kc, ec, vc, cumulative, bc, mm, c, masks):
    lower = jnp.tril(jnp.ones((c, c), dtype=bool))
    if masks is not None:
        lower = lower & masks.same
    inflate = jnp.exp(cumulative)
    qi = qc * inflate
    beta_keys = ec * bc[..., None]
    state_keys = beta_keys * inflate
    strict = lower & jnp.tril(jnp.ones((c, c), dtype=bool), -1)
    if c == 16:
        kd = kc * jnp.exp(jnp.minimum(-cumulative, 80.0))
        a_raw = -mm("...rd,...id->...ri", state_keys, kd)
        attention_raw = mm("...rd,...id->...ri", qi, kd)
    else:
        a_rows, attention_rows = [], []
        for start in range(0, c, 16):
            stop = start + 16
            anchor = cumulative[..., start : start + 1, :]
            row_anchor = jnp.broadcast_to(anchor, cumulative[..., start:stop, :].shape)
            col_anchor = jnp.broadcast_to(anchor, cumulative[..., :stop, :].shape)
            if masks is not None:
                row_anchor = jnp.where(masks.same[..., start:stop, start : start + 1], row_anchor, 0.0)
                col_anchor = jnp.where(jnp.swapaxes(masks.same[..., start : start + 1, :stop], -1, -2), col_anchor, 0.0)
                eligible = jnp.swapaxes(jnp.any(masks.same[..., start:stop, :stop], axis=-2, keepdims=True), -1, -2)
            else:
                eligible = jnp.ones(cumulative[..., :stop, :1].shape, dtype=bool)
            row_factor = jnp.exp(cumulative[..., start:stop, :] - row_anchor + 15.0)
            # Exclude older documents before exponentiation; their prefixes
            # may be arbitrarily far from the current document's zero anchor.
            col_exponent = jnp.where(eligible, col_anchor - cumulative[..., :stop, :] - 15.0, 0.0)
            columns = kc[..., :stop, :] * jnp.exp(col_exponent)
            ar = -mm("...rd,...id->...ri", beta_keys[..., start:stop, :] * row_factor, columns)
            qr = mm("...rd,...id->...ri", qc[..., start:stop, :] * row_factor, columns)
            if stop < c:
                zeros = jnp.zeros((*ar.shape[:-1], c - stop), ar.dtype)
                ar = jnp.concatenate((ar, zeros), axis=-1)
                qr = jnp.concatenate((qr, zeros), axis=-1)
            a_rows.append(ar)
            attention_rows.append(qr)
        a_raw = jnp.concatenate(a_rows, axis=-2)
        attention_raw = jnp.concatenate(attention_rows, axis=-2)
    a = jnp.where(strict, a_raw, 0.0)
    attention = jnp.where(lower, attention_raw, 0.0)
    tail = cumulative[..., -1, :]
    kw = kc * jnp.exp(jnp.minimum(tail[..., None, :] - cumulative, 0.0))
    decay = jnp.exp(tail)
    if masks is not None:
        qi = qi * masks.carry_in[..., None]
        state_keys = state_keys * masks.carry_in[..., None]
        kw = kw * masks.last_doc[..., None]
        decay = decay * masks.carry_through[..., None]
    rhs = jnp.concatenate((vc * bc[..., None], state_keys), axis=-1)
    return qi, attention, kw, decay, a, rhs


def chunk_prep_solve(qc, kc, ec, vc, gc, bc, mm, mm_dtype, c, masks, *, groups=4, interpret=False):
    """KDA adapter with causal 16-token factors and a saved fused solve.

    C16 retains its existing factors. C32/64 use the first gate of each
    subblock with a fixed offset, avoiding data-dependent future anchors.
    Remaining gate/mask/attention derivatives use standard AD.
    """
    if c not in (16, 32, 64):
        raise ValueError("The bounded solve adapter supports C16/32/64")
    lower = jnp.tril(jnp.ones((c, c), dtype=bool))
    if masks is not None:
        lower = lower & masks.same
    cumulative = jnp.einsum("...tj,...jd->...td", lower.astype(jnp.float32), gc, precision=jax.lax.Precision.HIGHEST)
    qi, attention, kw, decay, a, rhs = _prep_factors(qc, kc, ec, vc, cumulative, bc, mm, c, masks)
    solved = solve_unit_lower(
        a.reshape(-1, c, c), rhs.reshape(-1, c, rhs.shape[-1]), dtype=mm_dtype, groups=groups, interpret=interpret
    ).reshape(rhs.shape)
    return ChunkPrep(qi, solved[..., vc.shape[-1] :], solved[..., : vc.shape[-1]], attention, kw, decay)
