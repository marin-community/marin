# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Fused FP8 quantize + amax kernel for the ragged FP8 path.

Reads a bf16 tile once, folds the amax reduction via ``atomic_max``, quantizes
with the delayed-scaling reciprocal, and stores the natural FP8 tile.  This is
the elementwise precursor to a fused cast-transpose kernel that will
additionally store a transposed FP8 tile in the same pass.
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
