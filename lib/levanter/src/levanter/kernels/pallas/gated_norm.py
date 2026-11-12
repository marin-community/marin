# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Fuse GatedNorm's up projection, sigmoid and product, and its gate backward.

For normalized rows ``n`` and down-projection output ``a``, the forward returns ``n * g`` and ``g``, where
``g = sigmoid(silu(a) @ w_up)``. The backward returns the up-projection cotangent and the cotangent of ``a``. The
down projection and weight-gradient GEMMs remain in XLA. ``xla`` supplies the JAX reference; ``pallas`` uses Triton
kernels on GPU, the Pallas interpreter on CPU, and raises on TPU.
"""

import functools
from typing import Literal

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from jaxtyping import Array, Float

GatedNormImpl = Literal["xla", "pallas"]

# Tuning of each kernel: rows per program, the column tile it loops over, warps, pipeline stages. The best of a sweep
# at [32768, 2560], rank 128, bf16 on MI355X (device time: forward 148 us, backward 136 us).
_FWD_BLOCK_ROWS = 32
_FWD_BLOCK_N = 128
_FWD_NUM_WARPS = 4
_FWD_NUM_STAGES = 1
_BWD_BLOCK_ROWS = 64
_BWD_BLOCK_N = 128
_BWD_NUM_WARPS = 8
_BWD_NUM_STAGES = 2


def pallas_gated_norm_supported(hidden: int, rank: int) -> bool:
    """Whether the kernels can tile the shapes: the hidden width a multiple of the column tiles, a power-of-two rank
    of at least 16."""
    return hidden % _FWD_BLOCK_N == 0 and hidden % _BWD_BLOCK_N == 0 and rank >= 16 and rank & (rank - 1) == 0


def _check(n, a, w_up, *others):
    if n.ndim != 2 or a.ndim != 2 or a.shape[0] != n.shape[0]:
        raise ValueError(f"n [rows, hidden] and a [rows, rank] must share rows, got {n.shape} and {a.shape}")
    if w_up.shape != (a.shape[1], n.shape[1]):
        raise ValueError(f"w_up must be [{a.shape[1]}, {n.shape[1]}], got {w_up.shape}")
    for o in others:
        if o.shape != n.shape:
            raise ValueError(f"operand shape {o.shape} does not match {n.shape}")


def _sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))


def _silu_and_grad(a):
    """``silu(a)`` and its derivative, float32."""
    sig = _sigmoid(a)
    return a * sig, sig * (1.0 + a * (1.0 - sig))


def _dot(x, y):
    return jnp.dot(x, y, preferred_element_type=jnp.float32)


def _round(x, dtype):
    return x.astype(dtype).astype(jnp.float32)


def gated_norm_silu(a: Float[Array, "... K"]) -> Float[Array, "... K"]:
    """``silu(a)`` in float32, rounded once to ``a.dtype`` (the kernels' ``s``)."""
    return _silu_and_grad(a.astype(jnp.float32))[0].astype(a.dtype)


# JAX reference for the kernels.


def _fwd_xla(n, a, w_up):
    dtype = n.dtype
    g = _sigmoid(_dot(gated_norm_silu(a.astype(dtype)), w_up.astype(dtype))).astype(dtype)
    return (n.astype(jnp.float32) * g.astype(jnp.float32)).astype(dtype), g


def _bwd_xla(n, a, g, w_up, d_out):
    dtype = n.dtype
    gf = g.astype(jnp.float32)
    dz = (_round(d_out.astype(jnp.float32) * n.astype(jnp.float32), dtype) * gf * (1.0 - gf)).astype(dtype)
    dsilu = _silu_and_grad(a.astype(dtype).astype(jnp.float32))[1]
    da = (_round(_dot(dz, w_up.astype(dtype).T), dtype) * dsilu).astype(dtype)
    return dz, da


# Pallas-Triton programs use whole-array refs.


def _fwd_kernel(n_ref, a_ref, wu_ref, out_ref, g_ref, *, block_rows: int, block_n: int):
    num_rows, hidden = n_ref.shape
    dtype = n_ref.dtype
    row0 = pl.program_id(0) * block_rows
    ok = (row0 + jnp.arange(block_rows) < num_rows)[:, None]
    rows = pl.ds(row0, block_rows)
    s = gated_norm_silu(plgpu.load(a_ref.at[rows, :], mask=ok, other=0))

    def step(j, carry):
        cols = pl.ds(pl.multiple_of(j * block_n, block_n), block_n)
        g = _sigmoid(_dot(s, plgpu.load(wu_ref.at[:, cols]))).astype(dtype)
        plgpu.store(g_ref.at[rows, cols], g, mask=ok)
        x = plgpu.load(n_ref.at[rows, cols], mask=ok, other=0).astype(jnp.float32)
        plgpu.store(out_ref.at[rows, cols], (x * g.astype(jnp.float32)).astype(dtype), mask=ok)
        return carry

    jax.lax.fori_loop(0, hidden // block_n, step, 0)


def _bwd_kernel(n_ref, a_ref, g_ref, do_ref, wut_ref, dz_ref, da_ref, *, block_rows: int, block_n: int):
    num_rows, hidden = n_ref.shape
    dtype = n_ref.dtype
    row0 = pl.program_id(0) * block_rows
    ok = (row0 + jnp.arange(block_rows) < num_rows)[:, None]
    rows = pl.ds(row0, block_rows)

    def step(j, acc):
        cols = pl.ds(pl.multiple_of(j * block_n, block_n), block_n)
        g = plgpu.load(g_ref.at[rows, cols], mask=ok, other=0).astype(jnp.float32)
        x = plgpu.load(n_ref.at[rows, cols], mask=ok, other=0).astype(jnp.float32)
        do = plgpu.load(do_ref.at[rows, cols], mask=ok, other=0).astype(jnp.float32)
        dz = (_round(do * x, dtype) * g * (1.0 - g)).astype(dtype)
        plgpu.store(dz_ref.at[rows, cols], dz, mask=ok)
        return acc + _dot(dz, plgpu.load(wut_ref.at[cols, :]))

    rank = a_ref.shape[1]
    acc = jax.lax.fori_loop(0, hidden // block_n, step, jnp.zeros((block_rows, rank), jnp.float32))
    dsilu = _silu_and_grad(plgpu.load(a_ref.at[rows, :], mask=ok, other=0).astype(jnp.float32))[1]
    plgpu.store(da_ref.at[rows, :], (_round(acc, dtype) * dsilu).astype(dtype), mask=ok)


def _interpret(interpret: bool | None) -> bool:
    """Whether to run the kernels in the Pallas interpreter: by default on CPU (tests), never on GPU.

    The kernels are Pallas-Triton, so on TPU only the "xla" implementation runs."""
    if interpret is not None:
        return interpret
    backend = jax.default_backend()
    if backend == "tpu":
        raise ValueError('the Pallas GatedNorm kernels are Pallas-Triton (GPU) kernels; use impl="xla" on TPU')
    return backend != "gpu"


def _require_tiles(hidden: int, rank: int) -> None:
    if not pallas_gated_norm_supported(hidden, rank):
        raise ValueError(f"the Pallas GatedNorm cannot tile hidden={hidden}, rank={rank}")


def _fwd_pallas(n, a, w_up, interpret):
    num_rows, hidden = n.shape
    rank = a.shape[1]
    _require_tiles(hidden, rank)
    dtype = n.dtype
    itemsize = jnp.dtype(dtype).itemsize
    rows = jax.ShapeDtypeStruct(n.shape, dtype)
    out, g = pl.pallas_call(
        functools.partial(_fwd_kernel, block_rows=_FWD_BLOCK_ROWS, block_n=_FWD_BLOCK_N),
        grid=(pl.cdiv(num_rows, _FWD_BLOCK_ROWS),),
        in_specs=[pl.no_block_spec] * 3,
        out_specs=[pl.no_block_spec] * 2,
        out_shape=[rows, rows],
        compiler_params=plgpu.CompilerParams(num_warps=_FWD_NUM_WARPS, num_stages=_FWD_NUM_STAGES),
        cost_estimate=pl.CostEstimate(
            flops=2 * num_rows * hidden * rank,
            transcendentals=num_rows * hidden,
            bytes_accessed=(3 * num_rows * hidden + num_rows * rank + hidden * rank) * itemsize,
        ),
        interpret=_interpret(interpret),
        name="gated_norm_gate_fwd",
    )(n, a.astype(dtype), w_up.astype(dtype))
    return out, g


def _bwd_pallas(n, a, g, w_up, d_out, interpret):
    num_rows, hidden = n.shape
    rank = a.shape[1]
    _require_tiles(hidden, rank)
    dtype = n.dtype
    itemsize = jnp.dtype(dtype).itemsize
    dz, da = pl.pallas_call(
        functools.partial(_bwd_kernel, block_rows=_BWD_BLOCK_ROWS, block_n=_BWD_BLOCK_N),
        grid=(pl.cdiv(num_rows, _BWD_BLOCK_ROWS),),
        in_specs=[pl.no_block_spec] * 5,
        out_specs=[pl.no_block_spec] * 2,
        out_shape=[jax.ShapeDtypeStruct(n.shape, dtype), jax.ShapeDtypeStruct(a.shape, dtype)],
        compiler_params=plgpu.CompilerParams(num_warps=_BWD_NUM_WARPS, num_stages=_BWD_NUM_STAGES),
        cost_estimate=pl.CostEstimate(
            flops=2 * num_rows * hidden * rank,
            transcendentals=num_rows * rank,
            bytes_accessed=(4 * num_rows * hidden + 2 * num_rows * rank + hidden * rank) * itemsize,
        ),
        interpret=_interpret(interpret),
        name="gated_norm_gate_bwd",
    )(n, a.astype(dtype), g, d_out, w_up.astype(dtype).T)
    return dz, da


def gated_norm_gate_fwd(
    n: Float[Array, "R D"],
    a: Float[Array, "R K"],
    w_up: Float[Array, "K D"],
    *,
    impl: GatedNormImpl,
    interpret: bool | None = None,
) -> tuple[Float[Array, "R D"], Float[Array, "R D"]]:
    """``(out, g)``: the gate ``g = sigmoid(silu(a) w_up)`` and ``out = n * g``, in ``n.dtype`` (``a`` and ``w_up``
    are cast to it), see the module docstring. ``interpret=None`` uses the Pallas interpreter on CPU."""
    _check(n, a, w_up)
    if impl == "xla":
        return _fwd_xla(n, a, w_up)
    if impl == "pallas":
        return _fwd_pallas(n, a, w_up, interpret)
    raise ValueError(f"unknown gated_norm impl {impl!r}")


def gated_norm_gate_bwd(
    n: Float[Array, "R D"],
    a: Float[Array, "R K"],
    g: Float[Array, "R D"],
    w_up: Float[Array, "K D"],
    d_out: Float[Array, "R D"],
    *,
    impl: GatedNormImpl,
    interpret: bool | None = None,
) -> tuple[Float[Array, "R D"], Float[Array, "R K"]]:
    """``(dz, da)`` for the cotangent ``d_out`` of ``out``: the cotangents of the up projection's output and of ``a``,
    in ``n.dtype``, see the module docstring."""
    _check(n, a, w_up, g, d_out)
    if impl == "xla":
        return _bwd_xla(n, a, g, w_up, d_out)
    if impl == "pallas":
        return _bwd_pallas(n, a, g, w_up, d_out, interpret)
    raise ValueError(f"unknown gated_norm impl {impl!r}")
