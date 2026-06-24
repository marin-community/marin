# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Per-tensor delayed-scaling FP8 for the grouped (ragged) matmul.

The ragged analog of :func:`haliax._src.fp8.fp8_scaled_dot_general`. Operands are
quantized to E4M3 and fed straight into the grouped GEMM; the output is dequantized by
the operand scales. The backward quantizes the output grad to E5M2 and runs the two
grouped grad-dots (dlhs/drhs) on f8 operands. Scale/amax-history state flows in as
arguments and back out as gradients (overwrite-with-gradient), exactly as in the dense
path. Each contraction is dispatched to the production ragged-dot backend (Triton on
GPU, XLA fallback) via the same layout dimension-numbers the bf16 backward uses.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import custom_vjp

from haliax._src.fp8 import _new_scale_and_history, dequantize, in_q, out_dq, quantize
from haliax.nn.ragged_dot import (
    _AUTO_FALLBACK_EXCEPTIONS,
    _DEFAULT_DIM_NUMS,
    _DLHS_DIM_NUMS,
    _DRHS_DIM_NUMS,
    _preferred_implementations,
    _triton_pallas_call,
    Implementation,
)

# Token dimension is padded to a multiple of this before the grouped GEMM, matching
# `haliax.nn.ragged_dot.ragged_dot`; keeps the fp8 path's shape semantics identical to bf16.
_RAGGED_PAD_MULTIPLE = 512


def _ragged_dot_layout(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    dimension_numbers: jax.lax.RaggedDotDimensionNumbers,
    preferred_element_type: jnp.dtype,
    implementation: Implementation,
) -> jax.Array:
    """Single grouped matmul for one ragged-dot layout (forward/dlhs/drhs), backend-dispatched
    like ``ragged_dot`` but with an explicit output dtype so the f8 operands accumulate into
    ``preferred_element_type`` rather than being truncated back to f8 on store."""
    last_exc: Exception | None = None
    for impl in _preferred_implementations(implementation):
        try:
            if impl == "triton":
                return _triton_pallas_call(lhs, rhs, group_sizes, dimension_numbers, out_dtype=preferred_element_type)
            if impl == "xla":
                return jax.lax.ragged_dot_general(
                    lhs,
                    rhs,
                    group_sizes,
                    ragged_dot_dimension_numbers=dimension_numbers,
                    preferred_element_type=preferred_element_type,
                )
            raise NotImplementedError(f"FP8 ragged dot does not support implementation {impl!r}")
        except _AUTO_FALLBACK_EXCEPTIONS as exc:
            if implementation == "auto" and impl != "xla":
                last_exc = exc
                continue
            raise
    raise RuntimeError(f"No ragged_dot implementation selected (last error: {last_exc})")


@partial(custom_vjp, nondiff_argnums=(9, 10, 11))
def quantized_ragged_dot(
    lhs,
    q_lhs,
    lhs_scale,
    rhs,
    q_rhs,
    rhs_scale,
    out_grad_scale,
    out_grad_amax_history,
    group_sizes,
    preferred_element_type,
    implementation,
    grad_dtype,
):
    """Forward f8 grouped matmul on already-quantized operands; the custom_vjp carries the
    ``grad_dtype`` output-grad backward. Full-precision ``lhs``/``rhs`` are passed only to route
    their gradients (the forward consumes the quantized operands).

    ``grad_dtype`` selects the backward gradient format: E5M2 (the Transformer-Engine hybrid
    recipe, default) or E4M3 (the all-E4M3 recipe; with coarse per-tensor scaling this is the
    DeepSeek-style format *without* its fine-grained scaling — see logbook GFP8-023)."""
    return _ragged_dot_layout(q_lhs, q_rhs, group_sizes, _DEFAULT_DIM_NUMS, preferred_element_type, implementation)


def quantized_ragged_dot_fwd(
    lhs,
    q_lhs,
    lhs_scale,
    rhs,
    q_rhs,
    rhs_scale,
    out_grad_scale,
    out_grad_amax_history,
    group_sizes,
    preferred_element_type,
    implementation,
    grad_dtype,
):
    out = _ragged_dot_layout(q_lhs, q_rhs, group_sizes, _DEFAULT_DIM_NUMS, preferred_element_type, implementation)
    res = (q_lhs, lhs_scale, q_rhs, rhs_scale, out_grad_scale, out_grad_amax_history, group_sizes)
    return out, res


def quantized_ragged_dot_bwd(preferred_element_type, implementation, grad_dtype, res, g):
    q_lhs, lhs_scale, q_rhs, rhs_scale, out_grad_scale, out_grad_amax_history, group_sizes = res

    new_out_grad_scale, new_out_grad_amax_history = _new_scale_and_history(
        g, grad_dtype, out_grad_scale, out_grad_amax_history
    )
    q_g = quantize(g, grad_dtype, new_out_grad_scale, preferred_element_type)

    # dlhs[M,K] = dout[M,N] @ rhs[G,K,N]^T  (f8 operands, dequantized by the rhs and grad scales)
    grad_lhs = _ragged_dot_layout(q_g, q_rhs, group_sizes, _DLHS_DIM_NUMS, preferred_element_type, implementation)
    grad_lhs = dequantize(grad_lhs, preferred_element_type, rhs_scale * new_out_grad_scale)

    # drhs[G,K,N] = lhs[M,K]^T @ dout[M,N]  (f8 operands, dequantized by the lhs and grad scales)
    grad_rhs = _ragged_dot_layout(q_lhs, q_g, group_sizes, _DRHS_DIM_NUMS, preferred_element_type, implementation)
    grad_rhs = dequantize(grad_rhs, preferred_element_type, lhs_scale * new_out_grad_scale)

    return (grad_lhs, None, None, grad_rhs, None, None, new_out_grad_scale, new_out_grad_amax_history, None)


quantized_ragged_dot.defvjp(quantized_ragged_dot_fwd, quantized_ragged_dot_bwd)


def fp8_scaled_ragged_dot(
    lhs,
    rhs,
    group_sizes,
    *,
    preferred_element_type,
    lhs_scale,
    rhs_scale,
    grad_scale,
    lhs_amax_history,
    rhs_amax_history,
    grad_amax_history,
    quantize_compute_type,
    grad_dtype=jnp.float8_e5m2,
    implementation: Implementation = "auto",
):
    """``ragged_dot`` drop-in for per-tensor delayed-scaling FP8 (direct quantization).

    Args:
        lhs: ``[tokens, in]`` activations.
        rhs: ``[experts, in, out]`` expert weights.
        group_sizes: ``[experts]`` tokens per expert.
        preferred_element_type: output/accumulation dtype of the grouped GEMM.
        lhs_scale, rhs_scale, grad_scale: per-tensor scales for the two operands and the
            output gradient (delayed-scaling state).
        lhs_amax_history, rhs_amax_history, grad_amax_history: amax windows for the same.
        quantize_compute_type: dtype operands are cast to before E4M3 quantization.
        grad_dtype: backward output-grad format — ``float8_e5m2`` (hybrid recipe, default) or
            ``float8_e4m3fn`` (all-E4M3 recipe).
        implementation: ragged-dot backend (see :func:`haliax.nn.ragged_dot.ragged_dot`).

    Returns:
        ``[tokens, out]`` activations. Scale/amax state is returned as gradients of the
        corresponding arguments (overwrite-with-gradient).
    """
    tokens = lhs.shape[0]
    pad = (-tokens) % _RAGGED_PAD_MULTIPLE
    if pad:
        lhs = jax.lax.pad(lhs, jnp.zeros((), dtype=lhs.dtype), [(0, pad, 0), (0, 0, 0)])

    q_lhs, new_lhs_scale = in_q(quantize_compute_type, lhs, lhs_scale, lhs_amax_history)
    q_rhs, new_rhs_scale = in_q(quantize_compute_type, rhs, rhs_scale, rhs_amax_history)
    y = quantized_ragged_dot(
        lhs,
        q_lhs,
        new_lhs_scale,
        rhs,
        q_rhs,
        new_rhs_scale,
        grad_scale,
        grad_amax_history,
        group_sizes,
        preferred_element_type,
        implementation,
        grad_dtype,
    )
    y = out_dq(preferred_element_type, new_lhs_scale, new_rhs_scale, y)
    if pad:
        y = y[:tokens]
    return y
