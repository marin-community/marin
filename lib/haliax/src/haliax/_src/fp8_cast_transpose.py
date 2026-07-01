# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Fused FP8 cast-transpose (+ amax) for the ragged FP8 path.

FP8 ``wgmma`` needs the contracting dim contiguous for both operands, so several
operands are needed in *two* layouts (natural and transposed). Naively that is a
quantize pass plus a separate (slow, strided) FP8 transpose, and the amax pass
for delayed scaling on top -- three reads of the operand.

These Pallas-Triton kernels do it in **one read**: load the bf16 tile, fold the
amax reduction (``atomic_max``), quantize with the delayed-scaling scale, and
store *both* the natural and the transposed FP8 tile (the transpose is expressed
on the store side, which Triton lowers to a coalesced tiled transpose -- far
cheaper than XLA's strided FP8 transpose). ``in_q_ct`` wraps this in the same
``OverwriteWithGradient`` custom-VJP threading as ``in_q`` so the per-tensor
scale / amax-history state still flows through ``apply_updates``.

``quantize_amax_2d`` and ``in_q_qa`` (natural layout only, no transpose) are
retained for backward compatibility; new callers should prefer
``cast_transpose_amax_2d`` / ``in_q_ct`` which produce both layouts in a single
read.
"""

import functools

import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from .fp8 import compute_scale, get_fp8_max

_CT_BLOCKS = (128, 64, 32, 16)


def _block_size(dim: int) -> int:
    return next(b for b in _CT_BLOCKS if dim % b == 0)


def _qa_kernel(a_ref, inv_ref, _amax_in_ref, nat_ref, amax_ref, *, q_dtype, dtype_max):
    """Fused quantize+amax kernel body (no transpose).

    Loads the bf16 tile, accumulates the per-tile amax into ``amax_ref`` via
    ``atomic_max``, and writes the clipped-and-cast FP8 tile to ``nat_ref``.
    """
    a = plgpu.load(a_ref).astype(jnp.float32)
    plgpu.atomic_max(amax_ref, (0,), jnp.max(jnp.abs(a)))
    q = jnp.clip(a * inv_ref[0], -dtype_max, dtype_max).astype(q_dtype)
    plgpu.store(nat_ref, q)


@functools.lru_cache(maxsize=16)
def _make_qa_call(m: int, k: int, q_dtype, x_dtype):
    """Build and cache the ``pl.pallas_call`` for a given ``(m, k, q_dtype, x_dtype)`` config.

    ``pl.pallas_call`` creates a new JIT-compiled callable each invocation; without
    this cache the Triton kernel would recompile on every eager call, adding ~150 ms
    overhead.  The cache is keyed on the four static parameters that determine the
    compiled kernel: m, k, q_dtype, x_dtype.  The resulting callable is reused
    across all calls with the same config.
    """
    bm, bk = _block_size(m), _block_size(k)
    dtype_max = float(get_fp8_max(q_dtype, jnp.float32))

    # Cost estimate via a body-equivalent JAX fn on abstract inputs.
    x_struct = jax.ShapeDtypeStruct((m, k), x_dtype)
    inv_struct = jax.ShapeDtypeStruct((1,), jnp.float32)

    def _body(x_, inv_):
        a_f32 = x_.astype(jnp.float32)
        _amax = jnp.max(jnp.abs(a_f32))
        return jnp.clip(a_f32 * inv_[0], -dtype_max, dtype_max).astype(q_dtype)

    body_cost = pl.estimate_cost(_body, x_struct, inv_struct)
    input_bytes = m * k * jnp.dtype(x_dtype).itemsize + jnp.dtype(jnp.float32).itemsize
    output_bytes = m * k * jnp.dtype(q_dtype).itemsize + jnp.dtype(jnp.float32).itemsize
    cost = pl.CostEstimate(
        flops=body_cost.flops,
        transcendentals=body_cost.transcendentals,
        bytes_accessed=input_bytes + output_bytes,
    )

    return pl.pallas_call(
        functools.partial(_qa_kernel, q_dtype=q_dtype, dtype_max=dtype_max),
        out_shape=[
            jax.ShapeDtypeStruct((m, k), q_dtype),
            jax.ShapeDtypeStruct((1,), jnp.float32),
        ],
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j: (i, j)),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=[
            pl.BlockSpec((bm, bk), lambda i, j: (i, j)),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        grid=(m // bm, k // bk),
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=2),
        input_output_aliases={2: 1},
        cost_estimate=cost,
    )


def quantize_amax_2d(x, inv_scale, q_dtype):
    """``x[M,K]`` bf16 → (fp8 ``[M,K]``, amax ``[1]``) in one read.

    Loads the bf16 tile once, folds the amax reduction via ``atomic_max``,
    quantizes with the delayed-scaling reciprocal, and stores the natural FP8
    tile.

    Args:
        x: ``[M, K]`` bfloat16 input.
        inv_scale: ``[1]`` float32 inverse scale (``1 / new_scale``).
        q_dtype: target FP8 dtype (``jnp.float8_e4m3fn`` or ``jnp.float8_e5m2``).

    Returns:
        ``(q_natural, amax)``: natural-layout FP8 ``[M, K]`` and per-tensor amax ``[1]``.
    """
    m, k = x.shape
    amax0 = jnp.zeros((1,), jnp.float32)
    return _make_qa_call(m, k, q_dtype, x.dtype)(x, inv_scale, amax0)


# ---------------------------------------------------------------------------
# Cast-transpose + amax: produces both natural and transposed FP8 layouts in
# one bf16 read.  The transpose is expressed on the store side; Triton lowers
# it to a coalesced tiled transpose (far cheaper than XLA's strided transpose).
# ---------------------------------------------------------------------------


def _ct_kernel(a_ref, inv_ref, _amax_in_ref, nat_ref, t_ref, amax_ref, *, q_dtype, dtype_max):
    """Fused cast-transpose+amax kernel body.

    Loads the bf16 tile once, accumulates the per-tile amax, quantizes, and
    stores both the natural and the transposed FP8 tile.  The same body is
    shared by the 2D and 3D dispatch functions; Pallas presents the inner
    ``(bm, bn)`` tile uniformly regardless of whether the call grids over an
    outer expert dimension.
    """
    a = plgpu.load(a_ref).astype(jnp.float32)
    plgpu.atomic_max(amax_ref, (0,), jnp.max(jnp.abs(a)))
    q = jnp.clip(a * inv_ref[0], -dtype_max, dtype_max).astype(q_dtype)
    plgpu.store(nat_ref, q)
    plgpu.store(t_ref, q.T)


@functools.lru_cache(maxsize=16)
def _make_ct_call_2d(m: int, k: int, q_dtype, x_dtype):
    """Build and cache the cast-transpose+amax pallas_call for 2-D inputs."""
    bm, bk = _block_size(m), _block_size(k)
    dtype_max = float(get_fp8_max(q_dtype, jnp.float32))

    x_struct = jax.ShapeDtypeStruct((m, k), x_dtype)
    inv_struct = jax.ShapeDtypeStruct((1,), jnp.float32)

    def _body(x_, inv_):
        a_f32 = x_.astype(jnp.float32)
        _amax = jnp.max(jnp.abs(a_f32))
        return jnp.clip(a_f32 * inv_[0], -dtype_max, dtype_max).astype(q_dtype)

    body_cost = pl.estimate_cost(_body, x_struct, inv_struct)
    input_bytes = m * k * jnp.dtype(x_dtype).itemsize + jnp.dtype(jnp.float32).itemsize
    # Two output stores: natural [m,k] + transposed [k,m] = 2*m*k FP8 bytes, plus amax.
    output_bytes = 2 * m * k * jnp.dtype(q_dtype).itemsize + jnp.dtype(jnp.float32).itemsize
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
            jax.ShapeDtypeStruct((1,), jnp.float32),
        ],
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j: (i, j)),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=[
            pl.BlockSpec((bm, bk), lambda i, j: (i, j)),
            pl.BlockSpec((bk, bm), lambda i, j: (j, i)),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        grid=(m // bm, k // bk),
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=2),
        input_output_aliases={2: 2},
        cost_estimate=cost,
    )


def cast_transpose_amax_2d(x, inv_scale, q_dtype):
    """``x[M,K]`` bf16 → (fp8 ``[M,K]``, fp8 ``[K,M]``, amax ``[1]``) in one read.

    Loads the bf16 tile once, accumulates the amax via ``atomic_max``,
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
    amax0 = jnp.zeros((1,), jnp.float32)
    return _make_ct_call_2d(m, k, q_dtype, x.dtype)(x, inv_scale, amax0)


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

    x_struct = jax.ShapeDtypeStruct((e, k, n), x_dtype)
    inv_struct = jax.ShapeDtypeStruct((1,), jnp.float32)

    def _body(x_, inv_):
        a_f32 = x_.astype(jnp.float32)
        _amax = jnp.max(jnp.abs(a_f32))
        return jnp.clip(a_f32 * inv_[0], -dtype_max, dtype_max).astype(q_dtype)

    body_cost = pl.estimate_cost(_body, x_struct, inv_struct)
    input_bytes = e * k * n * jnp.dtype(x_dtype).itemsize + jnp.dtype(jnp.float32).itemsize
    # Two output stores: natural [e,k,n] + transposed [e,n,k] = 2*e*k*n FP8 bytes, plus amax.
    output_bytes = 2 * e * k * n * jnp.dtype(q_dtype).itemsize + jnp.dtype(jnp.float32).itemsize
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
            jax.ShapeDtypeStruct((1,), jnp.float32),
        ],
        in_specs=[
            pl.BlockSpec((None, bm, bn), lambda e, i, j: (e, i, j)),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=[
            pl.BlockSpec((None, bm, bn), lambda e, i, j: (e, i, j)),
            pl.BlockSpec((None, bn, bm), lambda e, i, j: (e, j, i)),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        grid=(e, k // bm, n // bn),
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=2),
        input_output_aliases={2: 2},
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
    amax0 = jnp.zeros((1,), jnp.float32)
    return _make_ct_call_3d(e, k, n, q_dtype, x.dtype)(x, inv_scale, amax0)


# ---------------------------------------------------------------------------
# Delayed-scaling helpers shared by both the natural-only and cast-transpose paths.
# ---------------------------------------------------------------------------


def _next_scale_and_inv(q_dtype, scale, amax_history):
    """Delayed-scaling scale for this step (from history) + its reciprocal."""
    dtype_max = get_fp8_max(q_dtype, jnp.float32)
    amax_from_history = jnp.max(amax_history, axis=0)
    new_scale = compute_scale(amax_from_history, scale, dtype_max)
    inv_scale = (1.0 / new_scale).astype(jnp.float32).reshape(1)
    return new_scale, inv_scale


def _quantize_delayed(q_dtype, inp, scale, amax_history):
    """Quantize ``inp`` with delayed scaling via the fused kernel.

    Returns ``(q, new_scale, new_history)``.
    """
    new_scale, inv_scale = _next_scale_and_inv(q_dtype, scale, amax_history)
    q, cur_amax = quantize_amax_2d(inp, inv_scale, q_dtype)
    new_history = jnp.roll(amax_history, shift=-1, axis=0).at[0].set(cur_amax[0])
    return q, new_scale, new_history


@functools.partial(custom_vjp, nondiff_argnums=(0,))
def in_q_qa(q_dtype, inp, scale, amax_history):
    """``in_q`` variant using the fused quantize+amax kernel.

    Same interface and ``OverwriteWithGradient`` VJP threading as ``in_q``,
    but the quantization step goes through ``quantize_amax_2d`` instead of the
    naive XLA cast.
    """
    q, new_scale, _ = _quantize_delayed(q_dtype, inp, scale, amax_history)
    return q, new_scale


def _in_q_qa_fwd(q_dtype, inp, scale, amax_history):
    q, new_scale, new_history = _quantize_delayed(q_dtype, inp, scale, amax_history)
    return (q, new_scale), (new_scale, new_history)


def _in_q_qa_bwd(q_dtype, res, _):
    new_scale, new_history = res
    # No gradient to inp; pass updated scale/history as cotangents so
    # OverwriteWithGradient overwrites the persisted delayed-scaling state.
    return None, new_scale, new_history


in_q_qa.defvjp(_in_q_qa_fwd, _in_q_qa_bwd)


# ---------------------------------------------------------------------------
# Cast-transpose analog of in_q: produces both layouts + threads delayed-scaling state.
# ---------------------------------------------------------------------------


def cast_transpose_delayed(q_dtype, mode, inp, scale, amax_history):
    """Cast-transpose ``inp`` with delayed scaling.

    Returns ``(q_natural, q_transposed, new_scale, new_history)``.  The amax is
    folded into the kernel (via ``atomic_max``), so ``new_history`` reflects the
    current step's observed magnitude without a separate reduction pass.

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
    new_history = jnp.roll(amax_history, shift=-1, axis=0).at[0].set(cur_amax[0])
    return q_nat, q_t, new_scale, new_history


@functools.partial(custom_vjp, nondiff_argnums=(0, 1))
def in_q_ct(q_dtype, mode, inp, scale, amax_history):
    """Cast-transpose analog of ``in_q``: returns ``(q_natural, q_transposed, new_scale)``.

    Produces both FP8 layouts (natural and transposed) in a single bf16 read
    and threads the delayed-scaling ``OverwriteWithGradient`` state (``scale``
    and ``amax_history`` are overwritten by the backward via the returned
    cotangents, just like ``in_q`` / ``in_q_qa``).

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
