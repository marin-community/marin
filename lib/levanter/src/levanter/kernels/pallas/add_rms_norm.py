# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Residual add + weighted RMS norm, forward and backward, one row-wise pass each.

Forward, for rows of ``x`` (and optionally ``residual``) and a weight ``w`` over the last dimension::

    s = f32(x) + f32(residual)                    (s = f32(x) without a residual)
    h = s                                         (rounded to x.dtype)
    inv_rms = rsqrt(mean(s^2) + eps)              (float32, one per row)
    n = (s * inv_rms) * f32(w)                    (rounded to x.dtype)

The statistic and ``n`` use the unrounded sum, as XLA computes them when it fuses ``x + residual`` (in ``x.dtype``)
into the norm (it may skip a rounding between fused ops); the backward uses the rounded ``h``.

Backward, given the cotangent of ``n`` as ``d_n * gate + d_n_extra`` (the form a gated consumer such as the June
model's GatedNorm hands back; ``gate`` and ``d_n_extra`` are optional) and optionally the cotangent ``d_h`` of ``h``::

    g = f32(d_n) * f32(gate) + f32(d_n_extra)
    dx = inv_rms * (g w - f32(h) inv_rms^2 mean(g w f32(h))) + f32(d_h)      (rounded to h.dtype)
    dw = sum over rows of g * f32(h) * inv_rms                               (float32)

``dx`` is also the residual's cotangent. Written as separate JAX ops, XLA runs the backward as a row reduction plus an
elementwise kernel (with the weight-gradient column sums), each reading every operand again, and the forward's
statistic stays a separate kernel wherever another kernel needs it. Here each direction is one kernel: a program
holds whole rows in registers, so every operand is read once.

``impl``: ``"xla"`` (the same math in JAX ops, which XLA fuses as it can) or ``"pallas"`` (Pallas-Triton kernels for
GPU; on CPU they run in the Pallas interpreter, on TPU they raise). Both compute in float32 and round only the outputs,
so they agree up to the float32 summation order of the row statistics.
"""

import functools
import math
from typing import Literal

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from jaxtyping import Array, Float

AddRmsNormImpl = Literal["xla", "pallas"]

# Columns per register tile (the kernels loop over the row in tiles of this many columns; Triton tiles must be powers
# of two) and the tuning of each kernel: rows per program (forward) or per loop step (backward), rows per backward
# program (one float32 weight-gradient partial sum per program), warps. The best of a sweep at [32768, 2560] bf16 on
# MI355X: forward with a residual 145 us (XLA's fusion of the same ops: 213 us), without 89 us (107 us); backward
# with gate, extra and d_h 267 us (359 us), about 4.4 TB/s.
_MAX_CHUNK = 512
_MIN_CHUNK = 16
_FWD_BLOCK_ROWS = 1
_FWD_NUM_WARPS = 2
_BWD_BLOCK_ROWS = 2
_BWD_GROUP_ROWS = 32
_BWD_NUM_WARPS = 2


def _chunk(hidden: int) -> int | None:
    """The largest power of two up to ``_MAX_CHUNK`` that divides ``hidden``, or None below the minimum."""
    c = math.gcd(hidden, _MAX_CHUNK)
    c = 1 << (c.bit_length() - 1)
    return c if c >= _MIN_CHUNK else None


def pallas_add_rms_norm_supported(hidden: int) -> bool:
    """Whether the kernels can tile a hidden width (a power-of-two divisor of at least 16)."""
    return _chunk(hidden) is not None


def _check(x: jax.Array, weight: jax.Array, *others: jax.Array | None) -> None:
    if x.ndim != 2:
        raise ValueError(f"rows must be rank-2 [rows, hidden], got shape={x.shape}")
    if weight.shape != (x.shape[1],):
        raise ValueError(f"weight must be [{x.shape[1]}], got {weight.shape}")
    for o in others:
        if o is not None and o.shape != x.shape:
            raise ValueError(f"operand shape {o.shape} does not match {x.shape}")


# --------------------------------------------------------------------------------------------------------------------
# JAX formulation (the "xla" implementation and the reference for the kernels).


def _fwd_xla(x, residual, weight, eps):
    hf = x.astype(jnp.float32)
    if residual is not None:
        hf = hf + residual.astype(jnp.float32)
    h = hf.astype(x.dtype)
    inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(hf), axis=-1) + eps)
    n = ((hf * inv_rms[:, None]) * weight.astype(jnp.float32)).astype(x.dtype)
    return h, n, inv_rms


def _cotangent(d_n, gate, d_n_extra):
    g = d_n.astype(jnp.float32)
    if gate is not None:
        g = g * gate.astype(jnp.float32)
    if d_n_extra is not None:
        g = g + d_n_extra.astype(jnp.float32)
    return g


def _bwd_xla(h, inv_rms, weight, d_n, gate, d_n_extra, d_h):
    g = _cotangent(d_n, gate, d_n_extra)
    hf = h.astype(jnp.float32)
    inv = inv_rms[:, None]
    gw = g * weight.astype(jnp.float32)
    projection = jnp.mean(gw * hf, axis=-1, keepdims=True)
    dx = inv * (gw - hf * (inv * inv) * projection)
    if d_h is not None:
        dx = dx + d_h.astype(jnp.float32)
    dw = jnp.sum(g * (hf * inv), axis=0)
    return dx.astype(h.dtype), dw


# --------------------------------------------------------------------------------------------------------------------
# Pallas-Triton kernels. Refs are whole arrays (no block specs); each program loads its rows in column tiles.


def _load(ref, row0, block_rows, col0, chunk, ok):
    return plgpu.load(ref.at[pl.ds(row0, block_rows), pl.ds(col0, chunk)], mask=ok[:, None], other=0).astype(
        jnp.float32
    )


def _store(ref, row0, col0, value, ok):
    block_rows, chunk = value.shape
    plgpu.store(ref.at[pl.ds(row0, block_rows), pl.ds(col0, chunk)], value.astype(ref.dtype), mask=ok[:, None])


def _fwd_kernel(*refs, has_residual: bool, block_rows: int, chunk: int, eps: float):
    if has_residual:
        x_ref, r_ref, w_ref, h_ref, n_ref, inv_ref = refs
    else:
        x_ref, w_ref, n_ref, inv_ref = refs
    num_rows, hidden = x_ref.shape
    row0 = pl.program_id(0) * block_rows
    ok = row0 + jnp.arange(block_rows) < num_rows
    tiles = []
    sum_sq = jnp.zeros((block_rows,), jnp.float32)
    for c in range(hidden // chunk):
        h = _load(x_ref, row0, block_rows, c * chunk, chunk, ok)
        if has_residual:
            h = h + _load(r_ref, row0, block_rows, c * chunk, chunk, ok)
            _store(h_ref, row0, c * chunk, h, ok)
        tiles.append(h)
        sum_sq = sum_sq + jnp.sum(h * h, axis=1)
    inv = jax.lax.rsqrt(sum_sq / hidden + eps)
    plgpu.store(inv_ref.at[pl.ds(row0, block_rows)], inv, mask=ok)
    for c, h in enumerate(tiles):
        w = plgpu.load(w_ref.at[pl.ds(c * chunk, chunk)]).astype(jnp.float32)
        _store(n_ref, row0, c * chunk, (h * inv[:, None]) * w[None, :], ok)


def _bwd_kernel(*refs, has_gate: bool, has_extra: bool, has_dh: bool, block_rows: int, group_rows: int, chunk: int):
    refs = list(refs)
    h_ref, inv_ref, w_ref, dn_ref = refs[:4]
    rest = refs[4:]
    gate_ref = rest.pop(0) if has_gate else None
    extra_ref = rest.pop(0) if has_extra else None
    dh_ref = rest.pop(0) if has_dh else None
    dx_ref, dw_ref = rest
    num_rows, hidden = h_ref.shape
    num_chunks = hidden // chunk
    group0 = pl.program_id(0) * group_rows
    weights = [plgpu.load(w_ref.at[pl.ds(c * chunk, chunk)]).astype(jnp.float32) for c in range(num_chunks)]

    def step(i, dw):
        row0 = group0 + i * block_rows
        ok = row0 + jnp.arange(block_rows) < num_rows
        inv = plgpu.load(inv_ref.at[pl.ds(row0, block_rows)], mask=ok, other=0)[:, None]
        hs, gs = [], []
        projection = jnp.zeros((block_rows,), jnp.float32)
        for c in range(num_chunks):
            col0 = c * chunk
            g = _load(dn_ref, row0, block_rows, col0, chunk, ok)
            if gate_ref is not None:
                g = g * _load(gate_ref, row0, block_rows, col0, chunk, ok)
            if extra_ref is not None:
                g = g + _load(extra_ref, row0, block_rows, col0, chunk, ok)
            h = _load(h_ref, row0, block_rows, col0, chunk, ok)
            projection = projection + jnp.sum(g * weights[c][None, :] * h, axis=1)
            hs.append(h)
            gs.append(g)
        projection = (projection / hidden)[:, None]
        out = []
        for c in range(num_chunks):
            col0 = c * chunk
            h, g = hs[c], gs[c]
            dx = inv * (g * weights[c][None, :] - h * (inv * inv) * projection)
            if dh_ref is not None:
                dx = dx + _load(dh_ref, row0, block_rows, col0, chunk, ok)
            _store(dx_ref, row0, col0, dx, ok)
            out.append(dw[c] + jnp.sum(g * (h * inv), axis=0))
        return tuple(out)

    steps = group_rows // block_rows
    dw0 = tuple(jnp.zeros((chunk,), jnp.float32) for _ in range(num_chunks))
    dw = jax.lax.fori_loop(0, steps, step, dw0)
    for c in range(num_chunks):
        plgpu.store(dw_ref.at[pl.program_id(0), pl.ds(c * chunk, chunk)], dw[c])


def _interpret(interpret: bool | None) -> bool:
    """Whether to run the kernels in the Pallas interpreter: by default on CPU (tests), never on GPU.

    The kernels are Pallas-Triton, so on TPU only the "xla" implementation runs."""
    if interpret is not None:
        return interpret
    backend = jax.default_backend()
    if backend == "tpu":
        raise ValueError('the Pallas RMS norm kernels are Pallas-Triton (GPU) kernels; use impl="xla" on TPU')
    return backend != "gpu"


def _chunk_or_raise(hidden: int) -> int:
    chunk = _chunk(hidden)
    if chunk is None:
        raise ValueError(f"the Pallas RMS norm needs a hidden width with a power-of-two divisor >= 16, got {hidden}")
    return chunk


def _fwd_pallas(x, residual, weight, eps, interpret):
    num_rows, hidden = x.shape
    chunk = _chunk_or_raise(hidden)
    itemsize = jnp.dtype(x.dtype).itemsize
    has_residual = residual is not None
    row_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    out_shape = [row_shape, jax.ShapeDtypeStruct((num_rows,), jnp.float32)]
    if has_residual:
        out_shape = [row_shape, *out_shape]
    operands = (x, residual, weight) if has_residual else (x, weight)
    outs = pl.pallas_call(
        functools.partial(_fwd_kernel, has_residual=has_residual, block_rows=_FWD_BLOCK_ROWS, chunk=chunk, eps=eps),
        grid=(pl.cdiv(num_rows, _FWD_BLOCK_ROWS),),
        in_specs=[pl.no_block_spec] * len(operands),
        out_specs=[pl.no_block_spec] * len(out_shape),
        out_shape=out_shape,
        compiler_params=plgpu.CompilerParams(num_warps=_FWD_NUM_WARPS, num_stages=1),
        cost_estimate=pl.CostEstimate(
            flops=4 * num_rows * hidden,
            transcendentals=num_rows,
            bytes_accessed=(2 + 2 * has_residual) * num_rows * hidden * itemsize + 4 * num_rows,
        ),
        interpret=_interpret(interpret),
        name="add_rms_norm_fwd" if has_residual else "rms_norm_fwd",
    )(*operands)
    if has_residual:
        return tuple(outs)
    n, inv = outs
    return x, n, inv


def _bwd_pallas(h, inv_rms, weight, d_n, gate, d_n_extra, d_h, interpret):
    num_rows, hidden = h.shape
    chunk = _chunk_or_raise(hidden)
    itemsize = jnp.dtype(h.dtype).itemsize
    optional = [t for t in (gate, d_n_extra, d_h) if t is not None]
    operands = (h, inv_rms, weight, d_n, *optional)
    groups = pl.cdiv(num_rows, _BWD_GROUP_ROWS)
    dx, dw = pl.pallas_call(
        functools.partial(
            _bwd_kernel,
            has_gate=gate is not None,
            has_extra=d_n_extra is not None,
            has_dh=d_h is not None,
            block_rows=_BWD_BLOCK_ROWS,
            group_rows=_BWD_GROUP_ROWS,
            chunk=chunk,
        ),
        grid=(groups,),
        in_specs=[pl.no_block_spec] * len(operands),
        out_specs=[pl.no_block_spec, pl.no_block_spec],
        out_shape=[
            jax.ShapeDtypeStruct(h.shape, h.dtype),
            jax.ShapeDtypeStruct((groups, hidden), jnp.float32),
        ],
        compiler_params=plgpu.CompilerParams(num_warps=_BWD_NUM_WARPS, num_stages=1),
        cost_estimate=pl.CostEstimate(
            flops=10 * num_rows * hidden,
            transcendentals=0,
            bytes_accessed=(3 + len(optional)) * num_rows * hidden * itemsize + 4 * (num_rows + groups * hidden),
        ),
        interpret=_interpret(interpret),
        name="rms_norm_bwd",
    )(*operands)
    return dx, jnp.sum(dw, axis=0)


# --------------------------------------------------------------------------------------------------------------------
# Public entry points.


def add_rms_norm_fwd(
    x: Float[Array, "R D"],
    residual: Float[Array, "R D"] | None,
    weight: Float[Array, "D"],
    *,
    eps: float,
    impl: AddRmsNormImpl,
    interpret: bool | None = None,
) -> tuple[Float[Array, "R D"], Float[Array, "R D"], Float[Array, "R"]]:
    """``(h, n, inv_rms)`` for rows ``x`` (plus ``residual``), see the module docstring. Without a residual ``h`` is
    ``x`` itself. ``interpret=None`` uses the Pallas interpreter on CPU."""
    _check(x, weight, residual)
    if impl == "xla":
        return _fwd_xla(x, residual, weight, eps)
    if impl == "pallas":
        return _fwd_pallas(x, residual, weight, eps, interpret)
    raise ValueError(f"unknown add_rms_norm impl {impl!r}")


def add_rms_norm_bwd(
    h: Float[Array, "R D"],
    inv_rms: Float[Array, "R"],
    weight: Float[Array, "D"],
    d_n: Float[Array, "R D"],
    *,
    gate: Float[Array, "R D"] | None = None,
    d_n_extra: Float[Array, "R D"] | None = None,
    d_h: Float[Array, "R D"] | None = None,
    impl: AddRmsNormImpl,
    interpret: bool | None = None,
) -> tuple[Float[Array, "R D"], Float[Array, "D"]]:
    """``(dx, dw)``: the cotangent of ``x`` (and of the residual) in ``h.dtype`` and the float32 weight gradient, for
    the cotangent ``d_n * gate + d_n_extra`` of ``n`` and ``d_h`` of ``h``, see the module docstring."""
    _check(h, weight, d_n, gate, d_n_extra, d_h)
    if inv_rms.shape != (h.shape[0],):
        raise ValueError(f"inv_rms must be [{h.shape[0]}], got {inv_rms.shape}")
    if impl == "xla":
        return _bwd_xla(h, inv_rms, weight, d_n, gate, d_n_extra, d_h)
    if impl == "pallas":
        return _bwd_pallas(h, inv_rms, weight, d_n, gate, d_n_extra, d_h, interpret)
    raise ValueError(f"unknown add_rms_norm impl {impl!r}")
