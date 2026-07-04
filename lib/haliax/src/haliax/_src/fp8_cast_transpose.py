# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Fused FP8 cast-transpose (+ amax) for the ragged FP8 path.

FP8 ``wgmma`` needs the contracting dim contiguous for both operands, so several
operands are needed in *two* layouts (natural and transposed). Naively that is a
quantize pass plus a separate (slow, strided) FP8 transpose, and the amax pass
for delayed scaling on top -- three reads of the operand.

These Pallas-Triton kernels do it in **one read**: load the bf16 tile, emit a
per-tile amax partial (reduced to the per-tensor amax by XLA, which avoids
serializing every tile on one global ``atomic_max`` location), quantize with the
delayed-scaling scale, and store *both* the natural and the transposed FP8 tile
(the transpose is expressed on the store side, which Triton lowers to a
coalesced tiled transpose -- far cheaper than XLA's strided FP8 transpose).
``in_q_ct`` wraps this in the same ``OverwriteWithGradient`` custom-VJP
threading as ``in_q`` so the per-tensor scale / amax-history state still flows
through ``apply_updates``.
"""

import functools

import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from .fp8 import get_fp8_max, roll_amax_history, scale_from_history

_CT_BLOCKS = (128, 64, 32, 16)


def _block_size(dim: int) -> int:
    block = next((b for b in _CT_BLOCKS if dim % b == 0), None)
    if block is None:
        raise ValueError(f"dim={dim} must be a multiple of 16 for the cast-transpose tile sizes {_CT_BLOCKS}")
    return block


# ---------------------------------------------------------------------------
# Cast-transpose + amax kernels: 2D (activations, gradients) and 3D (weights).
# ---------------------------------------------------------------------------


def _ct_kernel(a_ref, inv_ref, nat_ref, t_ref, amax_ref, *, q_dtype, dtype_max):
    """Fused cast-transpose+amax kernel body.

    Loads the bf16 tile once, emits the tile's amax partial, quantizes, and
    stores both the natural and the transposed FP8 tile.  The partials are
    reduced to the per-tensor amax outside the kernel; a single global
    ``atomic_max`` location would serialize every tile of the grid on one
    cacheline.  The same body is shared by the 2D and 3D dispatch functions;
    Pallas presents the inner ``(bm, bn)`` tile uniformly regardless of whether
    the call grids over an outer expert dimension.
    """
    a = plgpu.load(a_ref).astype(jnp.float32)
    amax_ref[0, 0] = jnp.max(jnp.abs(a))
    q = jnp.clip(a * inv_ref[0], -dtype_max, dtype_max).astype(q_dtype)
    plgpu.store(nat_ref, q)
    plgpu.store(t_ref, q.T)


def _ct_body_cost(q_dtype, dtype_max, x_struct):
    """Flops/transcendentals estimate of the fused amax+quantize body (bytes are computed separately)."""

    def _body(x_, inv_):
        a_f32 = x_.astype(jnp.float32)
        _ = jnp.max(jnp.abs(a_f32))  # unused value: keeps the amax reduction in the cost estimate
        return jnp.clip(a_f32 * inv_[0], -dtype_max, dtype_max).astype(q_dtype)

    inv_struct = jax.ShapeDtypeStruct((1,), jnp.float32)
    return pl.estimate_cost(_body, x_struct, inv_struct)


@functools.lru_cache(maxsize=16)
def _make_ct_call_2d(m: int, k: int, q_dtype, x_dtype):
    """Build and cache the cast-transpose+amax pallas_call for 2-D inputs."""
    # A 64-row tile on the (token) M axis measures ~7% more bandwidth than 128
    # at the d2560 shapes; K keeps the widest tile.
    bm = 64 if m % 64 == 0 else _block_size(m)
    bk = _block_size(k)
    dtype_max = float(get_fp8_max(q_dtype, jnp.float32))

    grid_m, grid_k = m // bm, k // bk
    body_cost = _ct_body_cost(q_dtype, dtype_max, jax.ShapeDtypeStruct((m, k), x_dtype))
    input_bytes = m * k * jnp.dtype(x_dtype).itemsize + jnp.dtype(jnp.float32).itemsize
    # Two output stores: natural [m,k] + transposed [k,m] = 2*m*k FP8 bytes, plus amax partials.
    output_bytes = 2 * m * k * jnp.dtype(q_dtype).itemsize + grid_m * grid_k * jnp.dtype(jnp.float32).itemsize
    cost = pl.CostEstimate(
        flops=body_cost.flops,
        transcendentals=body_cost.transcendentals,
        bytes_accessed=input_bytes + output_bytes,
    )

    return pl.pallas_call(
        functools.partial(_ct_kernel, q_dtype=q_dtype, dtype_max=dtype_max),
        out_shape=[
            jax.ShapeDtypeStruct((m, k), q_dtype),
            jax.ShapeDtypeStruct((k, m), q_dtype),
            jax.ShapeDtypeStruct((grid_m, grid_k), jnp.float32),
        ],
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j: (i, j)),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=[
            pl.BlockSpec((bm, bk), lambda i, j: (i, j)),
            pl.BlockSpec((bk, bm), lambda i, j: (j, i)),
            pl.BlockSpec((1, 1), lambda i, j: (i, j)),
        ],
        grid=(grid_m, grid_k),
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=2),
        cost_estimate=cost,
    )


def cast_transpose_amax_2d(x, inv_scale, q_dtype):
    """``x[M,K]`` bf16 → (fp8 ``[M,K]``, fp8 ``[K,M]``, amax ``[1]``) in one read.

    Loads the bf16 tile once, emits per-tile amax partials (reduced by XLA),
    quantizes, and stores both the natural and the transposed FP8 tiles.

    Args:
        x: ``[M, K]`` bfloat16 input.
        inv_scale: ``[1]`` float32 inverse scale (``1 / new_scale``).
        q_dtype: target FP8 dtype (``jnp.float8_e4m3fn`` or ``jnp.float8_e5m2``).

    Returns:
        ``(q_natural, q_transposed, amax)``: natural ``[M, K]``, transposed ``[K, M]``,
        and per-tensor amax ``[1]``.
    """
    m, k = x.shape
    q_nat, q_t, amax_parts = _make_ct_call_2d(m, k, q_dtype, x.dtype)(x, inv_scale)
    return q_nat, q_t, jnp.max(amax_parts).reshape(1)


@functools.lru_cache(maxsize=16)
def _make_ct_call_3d(e: int, k: int, n: int, q_dtype, x_dtype):
    """Build and cache the cast-transpose+amax pallas_call for 3-D weight inputs.

    Grids over the expert dimension ``E``; each grid cell processes one
    ``(bm, bn)`` tile of the ``[K, N]`` slice.  The transposed out-spec
    ``(None, bn, bm)`` indexed ``(e, j, i)`` writes the ``[K, N]`` block into
    the ``[N, K]`` transposed layout, which Triton lowers as a coalesced tiled
    transpose.
    """
    bm, bn = _block_size(k), _block_size(n)
    dtype_max = float(get_fp8_max(q_dtype, jnp.float32))

    grid_k, grid_n = k // bm, n // bn
    body_cost = _ct_body_cost(q_dtype, dtype_max, jax.ShapeDtypeStruct((e, k, n), x_dtype))
    input_bytes = e * k * n * jnp.dtype(x_dtype).itemsize + jnp.dtype(jnp.float32).itemsize
    # Two output stores: natural [e,k,n] + transposed [e,n,k] = 2*e*k*n FP8 bytes, plus amax partials.
    output_bytes = 2 * e * k * n * jnp.dtype(q_dtype).itemsize + e * grid_k * grid_n * jnp.dtype(jnp.float32).itemsize
    cost = pl.CostEstimate(
        flops=body_cost.flops,
        transcendentals=body_cost.transcendentals,
        bytes_accessed=input_bytes + output_bytes,
    )

    return pl.pallas_call(
        functools.partial(_ct_kernel, q_dtype=q_dtype, dtype_max=dtype_max),
        out_shape=[
            jax.ShapeDtypeStruct((e, k, n), q_dtype),
            jax.ShapeDtypeStruct((e, n, k), q_dtype),
            jax.ShapeDtypeStruct((e, grid_k, grid_n), jnp.float32),
        ],
        in_specs=[
            pl.BlockSpec((None, bm, bn), lambda e, i, j: (e, i, j)),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=[
            pl.BlockSpec((None, bm, bn), lambda e, i, j: (e, i, j)),
            pl.BlockSpec((None, bn, bm), lambda e, i, j: (e, j, i)),
            pl.BlockSpec((None, 1, 1), lambda e, i, j: (e, i, j)),
        ],
        grid=(e, k // bm, n // bn),
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=2),
        cost_estimate=cost,
    )


def cast_transpose_amax_3d(x, inv_scale, q_dtype):
    """``x[E,K,N]`` bf16 → (fp8 ``[E,K,N]``, fp8 ``[E,N,K]``, amax ``[1]``) in one read.

    Like ``cast_transpose_amax_2d`` but grids over the expert dimension ``E``.

    Args:
        x: ``[E, K, N]`` bfloat16 weight tensor.
        inv_scale: ``[1]`` float32 inverse scale (``1 / new_scale``).
        q_dtype: target FP8 dtype.

    Returns:
        ``(q_natural, q_transposed, amax)``: natural ``[E, K, N]``, transposed
        ``[E, N, K]``, and per-tensor amax ``[1]``.
    """
    e, k, n = x.shape
    q_nat, q_t, amax_parts = _make_ct_call_3d(e, k, n, q_dtype, x.dtype)(x, inv_scale)
    return q_nat, q_t, jnp.max(amax_parts).reshape(1)


# ---------------------------------------------------------------------------
# Delayed-scaling helper.
# ---------------------------------------------------------------------------


def _next_scale_and_inv(q_dtype, scale, amax_history):
    """Delayed-scaling scale for this step (shared TE recipe from ``_src/fp8``) + its reciprocal."""
    new_scale = scale_from_history(q_dtype, scale, amax_history)
    inv_scale = (1.0 / new_scale).astype(jnp.float32).reshape(1)
    return new_scale, inv_scale


# ---------------------------------------------------------------------------
# Cast-transpose analog of in_q: produces both layouts + threads delayed-scaling state.
# ---------------------------------------------------------------------------


def cast_transpose_delayed(q_dtype, mode, inp, scale, amax_history):
    """Cast-transpose ``inp`` with delayed scaling.

    Returns ``(q_natural, q_transposed, new_scale, new_history)``.  The amax is
    folded into the kernel (per-tile partials reduced by XLA), so
    ``new_history`` reflects the current step's observed magnitude without a
    separate read of the input.

    Args:
        q_dtype: target FP8 dtype.
        mode: ``"2d"`` for activations ``[M, K]`` or ``"3d"`` for weights ``[E, K, N]``.
        inp: bf16 tensor to quantize.
        scale: current delayed-scaling scale ``[1]``.
        amax_history: rolling amax history.
    """
    new_scale, inv_scale = _next_scale_and_inv(q_dtype, scale, amax_history)
    ct = cast_transpose_amax_3d if mode == "3d" else cast_transpose_amax_2d
    q_nat, q_t, cur_amax = ct(inp, inv_scale, q_dtype)
    new_history = roll_amax_history(cur_amax[0], amax_history)
    return q_nat, q_t, new_scale, new_history


@functools.partial(custom_vjp, nondiff_argnums=(0, 1))
def in_q_ct(q_dtype, mode, inp, scale, amax_history):
    """Cast-transpose analog of ``in_q``: returns ``(q_natural, q_transposed, new_scale)``.

    Produces both FP8 layouts (natural and transposed) in a single bf16 read
    and threads the delayed-scaling ``OverwriteWithGradient`` state (``scale``
    and ``amax_history`` are overwritten by the backward via the returned
    cotangents, just like ``in_q``).

    Args:
        q_dtype: target FP8 dtype (non-differentiable static arg).
        mode: ``"2d"`` or ``"3d"`` (non-differentiable static arg).
        inp: bf16 tensor to quantize.
        scale: current delayed-scaling scale ``[1]``.
        amax_history: rolling amax history.

    Returns:
        ``(q_natural, q_transposed, new_scale)``.
    """
    q_nat, q_t, new_scale, _ = cast_transpose_delayed(q_dtype, mode, inp, scale, amax_history)
    return q_nat, q_t, new_scale


def _in_q_ct_fwd(q_dtype, mode, inp, scale, amax_history):
    q_nat, q_t, new_scale, new_history = cast_transpose_delayed(q_dtype, mode, inp, scale, amax_history)
    return (q_nat, q_t, new_scale), (new_scale, new_history)


def _in_q_ct_bwd(q_dtype, mode, res, _cts):
    new_scale, new_history = res
    # No grad to inp; pass updated scale/history through as cotangents so
    # OverwriteWithGradient overwrites the persisted delayed-scaling state.
    return None, new_scale, new_history


in_q_ct.defvjp(_in_q_ct_fwd, _in_q_ct_bwd)
