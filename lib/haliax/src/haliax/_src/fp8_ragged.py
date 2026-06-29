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

from enum import StrEnum
from functools import partial

import jax
import jax.numpy as jnp
from jax import custom_vjp

from haliax._src import mixed_fp8_wgmma_shim
from haliax._src.fp8 import _new_scale_and_history, dequantize, in_q, out_dq, quantize
from haliax._src.fp8_cast_transpose import cast_transpose, in_q_transpose, in_q_transpose_3d
from haliax._src.fp8_ragged_guards import Fp8Contraction, assert_fp8_contraction
from haliax.nn.ragged_dot import (
    _AUTO_FALLBACK_EXCEPTIONS,
    _DEFAULT_DIM_NUMS,
    _DLHS_DIM_NUMS,
    _DRHS_DIM_NUMS,
    _mosaic_pallas_call,
    _mosaic_wgrad_transposed,
    _preferred_implementations,
    _triton_pallas_call,
    Implementation,
)

# Token dimension is padded to a multiple of this before the grouped GEMM, matching
# `haliax.nn.ragged_dot.ragged_dot`; keeps the fp8 path's shape semantics identical to bf16.
_RAGGED_PAD_MULTIPLE = 512

_WGRAD_FALLBACK_DTYPE = jnp.bfloat16


class MosaicWgradMode(StrEnum):
    """Weight-gradient strategy for the Mosaic FP8 ragged backward (the ``mosaic`` implementation only).

    Hopper f8 ``wgmma`` cannot transpose the contraction (token) axis in-kernel, so the wgrad needs
    token-contiguous f8 operands. The two ways to get the weight gradient:
    """

    BF16 = "bf16"  # dequantize the f8 operands and run the wgrad in bf16 (default; simpler, ~1.27× e2e)
    FP8 = "fp8"  # fused cast-transpose -> genuine f8 wgmma weight-gradient (~1.33× e2e; GFP8-033 M3)


def _f8_wgrad_active(
    implementation: Implementation, mosaic_wgrad: MosaicWgradMode
) -> bool:
    """The f8 cast-transpose weight-gradient runs only on the mosaic backend in FP8 mode."""
    return implementation == "mosaic" and mosaic_wgrad == MosaicWgradMode.FP8


def _ragged_dot_layout(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    dimension_numbers: jax.lax.RaggedDotDimensionNumbers,
    preferred_element_type: jnp.dtype,
    implementation: Implementation,
    rhs_pretransposed: jax.Array | None = None,
) -> jax.Array:
    """Single grouped matmul for one ragged-dot layout (forward/dlhs/drhs), backend-dispatched
    like ``ragged_dot`` but with an explicit output dtype so the f8 operands accumulate into
    ``preferred_element_type`` rather than being truncated back to f8 on store.

    ``rhs_pretransposed`` is the K-contiguous f8 weights (from the fused cast-transpose) for the mosaic
    forward layout; when given, the mosaic kernel skips the XLA ``swapaxes`` (GFP8-OPT-R4). Ignored by
    the triton/xla backends, which consume the natural ``rhs`` layout."""
    last_exc: Exception | None = None
    for impl in _preferred_implementations(implementation):
        try:
            if impl == "mosaic":
                return _mosaic_pallas_call(
                    lhs,
                    rhs,
                    group_sizes,
                    dimension_numbers,
                    out_dtype=preferred_element_type,
                    rhs_pretransposed=rhs_pretransposed,
                )
            if impl == "triton":
                return _triton_pallas_call(
                    lhs,
                    rhs,
                    group_sizes,
                    dimension_numbers,
                    out_dtype=preferred_element_type,
                )
            if impl == "xla":
                return jax.lax.ragged_dot_general(
                    lhs,
                    rhs,
                    group_sizes,
                    ragged_dot_dimension_numbers=dimension_numbers,
                    preferred_element_type=preferred_element_type,
                )
            raise NotImplementedError(
                f"FP8 ragged dot does not support implementation {impl!r}"
            )
        except _AUTO_FALLBACK_EXCEPTIONS as exc:
            if implementation == "auto" and impl != "xla":
                last_exc = exc
                continue
            raise
    raise RuntimeError(
        f"No ragged_dot implementation selected (last error: {last_exc})"
    )


@partial(custom_vjp, nondiff_argnums=(11, 12, 13, 14))
def quantized_ragged_dot(
    lhs,
    q_lhs,
    q_lhs_t,
    lhs_scale,
    rhs,
    q_rhs,
    q_rhs_t,
    rhs_scale,
    out_grad_scale,
    out_grad_amax_history,
    group_sizes,
    preferred_element_type,
    implementation,
    grad_dtype,
    mosaic_wgrad,
):
    """Forward f8 grouped matmul on already-quantized operands; the custom_vjp carries the
    ``grad_dtype`` output-grad backward. Full-precision ``lhs``/``rhs`` are passed only to route
    their gradients (the forward consumes the quantized operands). ``q_lhs_t`` is the cast-transposed
    (token-contiguous) f8 activations the f8 weight-gradient consumes; it is unused on the bf16-wgrad
    / triton / xla backward (where the caller passes ``q_lhs`` as a placeholder). ``q_rhs_t`` is the
    cast-transposed (K-contiguous) f8 weights the mosaic forward consumes in place of an XLA swapaxes
    (GFP8-OPT-R4); on the triton/xla forward it is a placeholder (the caller passes ``q_rhs``).

    ``grad_dtype`` selects the backward gradient format: E5M2 (the Transformer-Engine hybrid
    recipe, default) or E4M3 (the all-E4M3 recipe; with coarse per-tensor scaling this is the
    DeepSeek-style format *without* its fine-grained scaling — see logbook GFP8-023)."""
    assert_fp8_contraction(
        q_lhs, q_rhs, contraction=Fp8Contraction.FORWARD, grad_dtype=grad_dtype
    )
    return _ragged_dot_layout(
        q_lhs,
        q_rhs,
        group_sizes,
        _DEFAULT_DIM_NUMS,
        preferred_element_type,
        implementation,
        rhs_pretransposed=q_rhs_t if implementation == "mosaic" else None,
    )


def quantized_ragged_dot_fwd(
    lhs,
    q_lhs,
    q_lhs_t,
    lhs_scale,
    rhs,
    q_rhs,
    q_rhs_t,
    rhs_scale,
    out_grad_scale,
    out_grad_amax_history,
    group_sizes,
    preferred_element_type,
    implementation,
    grad_dtype,
    mosaic_wgrad,
):
    assert_fp8_contraction(
        q_lhs, q_rhs, contraction=Fp8Contraction.FORWARD, grad_dtype=grad_dtype
    )
    out = _ragged_dot_layout(
        q_lhs,
        q_rhs,
        group_sizes,
        _DEFAULT_DIM_NUMS,
        preferred_element_type,
        implementation,
        rhs_pretransposed=q_rhs_t if implementation == "mosaic" else None,
    )
    res = (
        q_lhs,
        q_lhs_t,
        lhs_scale,
        q_rhs,
        rhs_scale,
        out_grad_scale,
        out_grad_amax_history,
        group_sizes,
    )
    return out, res


def quantized_ragged_dot_bwd(
    preferred_element_type, implementation, grad_dtype, mosaic_wgrad, res, g
):
    (
        q_lhs,
        q_lhs_t,
        lhs_scale,
        q_rhs,
        rhs_scale,
        out_grad_scale,
        out_grad_amax_history,
        group_sizes,
    ) = res

    new_out_grad_scale, new_out_grad_amax_history = _new_scale_and_history(
        g, grad_dtype, out_grad_scale, out_grad_amax_history
    )

    f8_wgrad = _f8_wgrad_active(implementation, mosaic_wgrad)
    if f8_wgrad:
        # Cast-transpose the output grad once -> rowwise q_g (for dlhs) + token-contiguous q_g_t (for
        # the f8 wgrad), no separate XLA transpose (GFP8-033 M3).
        q_g, q_g_t = cast_transpose(
            g,
            new_out_grad_scale,
            out_dtype=grad_dtype,
            compute_dtype=preferred_element_type,
        )
    else:
        q_g = quantize(g, grad_dtype, new_out_grad_scale, preferred_element_type)

    # dlhs[M,K] = dout[M,N] @ rhs[G,K,N]^T  (f8 operands, dequantized by the rhs and grad scales)
    assert_fp8_contraction(
        q_g, q_rhs, contraction=Fp8Contraction.DLHS, grad_dtype=grad_dtype
    )
    grad_lhs = _ragged_dot_layout(
        q_g, q_rhs, group_sizes, _DLHS_DIM_NUMS, preferred_element_type, implementation
    )
    grad_lhs = dequantize(
        grad_lhs, preferred_element_type, rhs_scale * new_out_grad_scale
    )

    # drhs[G,K,N] = lhs[M,K]^T @ dout[M,N], contracting the ragged token axis.
    if f8_wgrad:
        # mosaic f8 wgrad: both operands already cast-transposed (token-contiguous) -> the wgrad kernel
        # directly, no XLA `swapaxes` (the GFP8-033 win). Then dequantize by the lhs and grad scales.
        assert_fp8_contraction(
            q_lhs_t, q_g_t, contraction=Fp8Contraction.DRHS, grad_dtype=grad_dtype
        )
        grad_rhs = _mosaic_wgrad_transposed(
            q_lhs_t, q_g_t, group_sizes, out_dtype=preferred_element_type
        )
        grad_rhs = dequantize(
            grad_rhs, preferred_element_type, lhs_scale * new_out_grad_scale
        )
    elif implementation == "mosaic":
        # mosaic, f8 wgrad off: bf16 fallback (the shipped ~1.27× hybrid). Hopper f8 wgmma can't
        # transpose the token axis in-kernel, so run the GEMM in bf16 on the dequantized f8 operands.
        # Dequantizing folds in lhs_scale/grad_scale, so the result is already in real units.
        lhs_b = dequantize(q_lhs, _WGRAD_FALLBACK_DTYPE, lhs_scale)
        g_b = dequantize(q_g, _WGRAD_FALLBACK_DTYPE, new_out_grad_scale)
        grad_rhs = _ragged_dot_layout(
            lhs_b, g_b, group_sizes, _DRHS_DIM_NUMS, preferred_element_type, "auto"
        )
    else:
        # triton / xla: f8 wgrad via the standard layout dispatch (these backends transpose internally).
        assert_fp8_contraction(
            q_lhs, q_g, contraction=Fp8Contraction.DRHS, grad_dtype=grad_dtype
        )
        grad_rhs = _ragged_dot_layout(
            q_lhs,
            q_g,
            group_sizes,
            _DRHS_DIM_NUMS,
            preferred_element_type,
            implementation,
        )
        grad_rhs = dequantize(
            grad_rhs, preferred_element_type, lhs_scale * new_out_grad_scale
        )

    return (
        grad_lhs,
        None,  # q_lhs
        None,  # q_lhs_t
        None,  # lhs_scale
        grad_rhs,
        None,  # q_rhs
        None,  # q_rhs_t
        None,  # rhs_scale
        new_out_grad_scale,
        new_out_grad_amax_history,
        None,  # group_sizes
    )


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
    mosaic_wgrad: MosaicWgradMode = MosaicWgradMode.BF16,
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
        mosaic_wgrad: weight-gradient strategy for the ``mosaic`` backend — ``BF16`` (default; the
            wgrad runs in bf16 on the dequantized operands) or ``FP8`` (fused cast-transpose f8 wgrad,
            ~1.33× e2e; requires ``grad_dtype=float8_e4m3fn`` so both wgrad operands share one f8 type).

    Returns:
        ``[tokens, out]`` activations. Scale/amax state is returned as gradients of the
        corresponding arguments (overwrite-with-gradient).
    """
    # Enable mixed E4M3/E5M2 Hopper wgmma on stock jax/jaxlib before any mosaic kernel traces.
    # Idempotent; a no-op on the bf16-wgrad / triton / xla backward, which never mixes f8 types.
    mixed_fp8_wgmma_shim.activate_if_supported()
    tokens = lhs.shape[0]
    pad = (-tokens) % _RAGGED_PAD_MULTIPLE
    if pad:
        lhs = jax.lax.pad(lhs, jnp.zeros((), dtype=lhs.dtype), [(0, pad, 0), (0, 0, 0)])

    if _f8_wgrad_active(implementation, mosaic_wgrad):
        # One read of the activations -> rowwise q_lhs (forward GEMM) + token-contiguous q_lhs_t
        # (f8 weight-gradient), no separate XLA transpose (GFP8-033 M3).
        q_lhs, q_lhs_t, new_lhs_scale = in_q_transpose(
            quantize_compute_type, lhs, lhs_scale, lhs_amax_history
        )
    else:
        q_lhs, new_lhs_scale = in_q(
            quantize_compute_type, lhs, lhs_scale, lhs_amax_history
        )
        q_lhs_t = (
            q_lhs  # placeholder; the bf16-wgrad / triton / xla backward never reads it
        )
    if implementation == "mosaic":
        # Fused cast-transpose of the weights -> rowwise q_rhs (dlhs backward) + K-contiguous q_rhs_t
        # (forward), replacing the per-step XLA swapaxes in the mosaic forward (GFP8-OPT-R4).
        q_rhs, q_rhs_t, new_rhs_scale = in_q_transpose_3d(
            quantize_compute_type, rhs, rhs_scale, rhs_amax_history
        )
    else:
        q_rhs, new_rhs_scale = in_q(quantize_compute_type, rhs, rhs_scale, rhs_amax_history)
        q_rhs_t = q_rhs  # placeholder; the triton/xla forward consumes the natural rhs layout
    y = quantized_ragged_dot(
        lhs,
        q_lhs,
        q_lhs_t,
        new_lhs_scale,
        rhs,
        q_rhs,
        q_rhs_t,
        new_rhs_scale,
        grad_scale,
        grad_amax_history,
        group_sizes,
        preferred_element_type,
        implementation,
        grad_dtype,
        mosaic_wgrad,
    )
    y = out_dq(preferred_element_type, new_lhs_scale, new_rhs_scale, y)
    if pad:
        y = y[:tokens]
    return y
