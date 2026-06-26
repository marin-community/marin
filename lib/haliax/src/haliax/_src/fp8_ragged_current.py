# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Per-tensor *current*-scaling FP8 for the grouped (ragged) matmul.

The current-scaling analog of :mod:`haliax._src.fp8_ragged` (which uses delayed scaling
with amax-history state threaded as overwrite-with-gradient parameters). Here the scale is
recomputed from the live tensor every step -- ``amax(x) / fp8_max`` -- so the op is a pure
function with no carried state. Operands and the output gradient are quantized to E4M3 (the
all-E4M3 recipe); the backward runs the two grad-dots (dlhs/drhs) on f8 operands.

This is the best case for E4M3's narrow dynamic range (an ideal per-tensor scale every step,
no staleness). It is the validation vehicle for whether E4M3 has enough range for MoE
gradients over a real training trajectory -- if loss drifts from bf16 even here, coarser
scaling cannot save it (logbook GFP8-035).
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import custom_vjp

from haliax._src.fp8 import dequantize, get_fp8_max, quantize
from haliax._src.fp8_ragged import _ragged_dot_layout
from haliax.nn.ragged_dot import (
    _DEFAULT_DIM_NUMS,
    _DLHS_DIM_NUMS,
    _DRHS_DIM_NUMS,
    Implementation,
)

_E4M3 = jnp.float8_e4m3fn


def _current_scale(x: jax.Array, q_dtype: jnp.dtype) -> jax.Array:
    """Per-tensor current scale ``amax(x) / fp8_max``, recomputed from the live tensor.

    amax == 0 (an all-zero tensor) falls back to a unit scale.
    """
    amax = jnp.max(jnp.abs(x)).astype(jnp.float32)
    fp8_max = get_fp8_max(q_dtype, jnp.float32)
    return jnp.where(amax > 0, amax / fp8_max, jnp.ones_like(amax))


def _quantize_current(
    x: jax.Array, q_dtype: jnp.dtype, compute_dtype: jnp.dtype
) -> tuple[jax.Array, jax.Array]:
    scale = _current_scale(x, q_dtype)
    return quantize(x, q_dtype, scale, compute_dtype), scale


@partial(custom_vjp, nondiff_argnums=(3, 4))
def fp8_current_scaled_ragged_dot(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    preferred_element_type: jnp.dtype,
    implementation: Implementation,
) -> jax.Array:
    """``ragged_dot`` drop-in for per-tensor current-scaling all-E4M3 FP8.

    Args:
        lhs: ``[tokens, in]`` activations.
        rhs: ``[experts, in, out]`` expert weights.
        group_sizes: ``[experts]`` tokens per expert.
        preferred_element_type: output/accumulation dtype of the grouped GEMM.
        implementation: ragged-dot backend (see :func:`haliax.nn.ragged_dot.ragged_dot`).

    Returns:
        ``[tokens, out]`` activations dequantized back to ``preferred_element_type``.
    """
    q_lhs, lhs_scale = _quantize_current(lhs, _E4M3, preferred_element_type)
    q_rhs, rhs_scale = _quantize_current(rhs, _E4M3, preferred_element_type)
    out = _ragged_dot_layout(
        q_lhs,
        q_rhs,
        group_sizes,
        _DEFAULT_DIM_NUMS,
        preferred_element_type,
        implementation,
    )
    return dequantize(out, preferred_element_type, lhs_scale * rhs_scale)


def _fwd(lhs, rhs, group_sizes, preferred_element_type, implementation):
    q_lhs, lhs_scale = _quantize_current(lhs, _E4M3, preferred_element_type)
    q_rhs, rhs_scale = _quantize_current(rhs, _E4M3, preferred_element_type)
    out = _ragged_dot_layout(
        q_lhs,
        q_rhs,
        group_sizes,
        _DEFAULT_DIM_NUMS,
        preferred_element_type,
        implementation,
    )
    out = dequantize(out, preferred_element_type, lhs_scale * rhs_scale)
    # Tiny typed sentinels carry the operand dtypes to the backward as real array leaves
    # (a raw numpy.dtype is not a valid custom_vjp residual leaf).
    lhs_dt = jnp.zeros((), lhs.dtype)
    rhs_dt = jnp.zeros((), rhs.dtype)
    return out, (q_lhs, lhs_scale, lhs_dt, q_rhs, rhs_scale, rhs_dt, group_sizes)


def _bwd(preferred_element_type, implementation, res, g):
    q_lhs, lhs_scale, lhs_dt, q_rhs, rhs_scale, rhs_dt, group_sizes = res
    # all-E4M3: the output grad is quantized to E4M3 too -- the dynamic-range risk under test.
    grad_scale = _current_scale(g, _E4M3)
    q_g = quantize(g, _E4M3, grad_scale, preferred_element_type)

    # dlhs[M,K] = dout[M,N] @ rhs[G,K,N]^T. The cotangent must carry the operand's dtype (the
    # f8 GEMM accumulates in preferred_element_type, but custom_vjp requires grad dtype == input dtype).
    grad_lhs = _ragged_dot_layout(
        q_g, q_rhs, group_sizes, _DLHS_DIM_NUMS, preferred_element_type, implementation
    )
    grad_lhs = dequantize(grad_lhs, lhs_dt.dtype, rhs_scale * grad_scale)

    # drhs[G,K,N] = lhs[M,K]^T @ dout[M,N], contracting the ragged token axis.
    grad_rhs = _ragged_dot_layout(
        q_lhs, q_g, group_sizes, _DRHS_DIM_NUMS, preferred_element_type, implementation
    )
    grad_rhs = dequantize(grad_rhs, rhs_dt.dtype, lhs_scale * grad_scale)

    return grad_lhs, grad_rhs, None  # lhs, rhs, group_sizes (non-diff int)


fp8_current_scaled_ragged_dot.defvjp(_fwd, _bwd)
