# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""FP8 grouped matmul (``ragged_dot``) for Hopper H100 via the CuTe DSL.

FP8 forward + mixed-dtype FP8 ``dgrad`` (E5M2 x E4M3); the weight gradient
(``wgrad``) runs in bf16 here (a later change adds the FP8 cast-transpose wgrad).
Delayed per-tensor scaling reuses ``_src/fp8.py``: ``in_q`` threads the two
forward-operand scale/amax states through its own custom_vjp; the output-grad
state is threaded by ``quantized_ragged_dot``'s custom_vjp. The weight is
transposed in bf16 (hardware-legal) then cast to FP8 for the k-contiguous forward.
"""

import functools

import jax
import jax.numpy as jnp
from jax import custom_vjp

from .fp8 import in_q, quantize, update_fp8_meta
from .ragged_dot_cute import cute_ragged_dot

_E4M3 = jnp.float8_e4m3fn
_E5M2 = jnp.float8_e5m2


@functools.partial(custom_vjp, nondiff_argnums=(9, 10))
def quantized_ragged_dot(
    q_lhs, q_rhs_t, q_rhs, out_scale, rhs_scale, grad_scale, grad_amax_history, lhs, rhs, group_sizes, rev_dtype
):
    """FP8 forward of pre-quantized operands; FP8 dgrad + bf16 wgrad backward.

    ``q_lhs[T,K]`` / ``q_rhs_t[E,N,K]`` are E4M3 (k-contiguous, forward). ``q_rhs``
    is the natural-layout ``[E,K,N]`` E4M3 weight the dgrad contracts over N.
    ``out_scale = 1/(lhs_scale*rhs_scale)``: the raw fp8 product is
    ``(a/s_a)·(b/s_b) = (a·b)/(s_a·s_b)``, so recovering ``a·b`` MULTIPLIES by
    ``s_a·s_b`` — but the CuTe epilogue DIVIDES the accumulator by ``out_scale``,
    hence we pass the reciprocal. ``grad_scale``/``grad_amax_history`` carry
    the output-grad delayed-scaling state; the backward returns their updates as
    cotangents (OverwriteWithGradient). ``lhs``/``rhs`` are the bf16 operands the
    bf16 wgrad differentiates. ``rev_dtype`` is the output-grad FP8 dtype (E5M2).
    """
    return cute_ragged_dot(q_lhs, q_rhs_t, group_sizes, out_dtype=lhs.dtype, out_scale=out_scale)


def _qrd_fwd(
    q_lhs, q_rhs_t, q_rhs, out_scale, rhs_scale, grad_scale, grad_amax_history, lhs, rhs, group_sizes, rev_dtype
):
    out = cute_ragged_dot(q_lhs, q_rhs_t, group_sizes, out_dtype=lhs.dtype, out_scale=out_scale)
    return out, (q_rhs, rhs_scale, grad_scale, grad_amax_history, lhs, rhs)


def _qrd_bwd(group_sizes, rev_dtype, res, g):
    q_rhs, rhs_scale, grad_scale, grad_amax_history, lhs, rhs = res

    # --- output-grad delayed scaling: next-step scale + rolled history ---
    new_grad_scale, new_grad_history = update_fp8_meta(g, rev_dtype, grad_scale, grad_amax_history)
    q_g = quantize(g, rev_dtype, new_grad_scale, jnp.float32)  # [M,N] E5M2

    # --- FP8 dgrad: g(E5M2) @ rhs(E4M3), contract N (natural layout) -> dlhs[M,K] ---
    # q_rhs is [E,K,N]; the CuTe kernel maps b[E,N_tile,K_contr], so passing q_rhs
    # contracts the last axis N (the fwd output dim) producing dlhs[M,K].
    # dgrad_scale = 1/(grad_scale*rhs_scale): dequant is a MULTIPLY by the scale
    # product, but the kernel epilogue DIVIDES by out_scale, so pass the reciprocal
    # (shape (1,) float32 to match the epilogue divisor).
    dgrad_scale = jnp.reshape((1.0 / (new_grad_scale * rhs_scale)).astype(jnp.float32), (1,))
    grad_lhs = cute_ragged_dot(q_g, q_rhs, group_sizes, out_dtype=lhs.dtype, out_scale=dgrad_scale)

    # --- bf16-exact wgrad: differentiate the reference bf16 ragged_dot for grad_rhs only ---
    from haliax.nn.ragged_dot import ragged_dot as _bf16_ragged_dot  # noqa: PLC0415

    _, vjp_fn = jax.vjp(lambda ro: _bf16_ragged_dot(lhs, ro, group_sizes, op=None), rhs)
    (grad_rhs,) = vjp_fn(g)

    # Cotangents for (q_lhs, q_rhs_t, q_rhs, out_scale, rhs_scale, grad_scale, grad_amax_history, lhs, rhs).
    # grad_scale / grad_amax_history return the updated state (OverwriteWithGradient pattern).
    return (None, None, None, None, None, new_grad_scale, new_grad_history, grad_lhs, grad_rhs)


quantized_ragged_dot.defvjp(_qrd_fwd, _qrd_bwd)


def fp8_scaled_ragged_dot(
    lhs,
    rhs,
    group_sizes,
    *,
    lhs_scale,
    rhs_scale,
    grad_scale,
    lhs_amax_history,
    rhs_amax_history,
    grad_amax_history,
    quantize_compute_type=jnp.float32,
    fwd_dtype=_E4M3,
    rev_dtype=_E5M2,
):
    """FP8 ``ragged_dot``: E4M3 forward, mixed-FP8 dgrad, bf16 wgrad.

    Args:
        lhs: [T, K] activation matrix (bf16).
        rhs: [E, K, N] expert weight matrix (bf16), natural layout.
        group_sizes: [E] integer token counts per expert.
        lhs_scale: Per-tensor scale for the lhs operand (shape (1,) float32).
        rhs_scale: Per-tensor scale for the rhs operand (shape (1,) float32).
        grad_scale: Per-tensor scale for the output gradient (shape (1,) float32).
        lhs_amax_history: Rolling amax history for the lhs operand.
        rhs_amax_history: Rolling amax history for the rhs operand.
        grad_amax_history: Rolling amax history for the output gradient.
        quantize_compute_type: Accumulator dtype for quantize/dequantize ops.
        fwd_dtype: FP8 dtype for forward operands (default E4M3).
        rev_dtype: FP8 dtype for the output gradient (default E5M2).

    Returns:
        [T, N] output array in ``lhs.dtype``.
    """
    comp = quantize_compute_type
    q_lhs, new_lhs_scale = in_q(comp, fwd_dtype, lhs, lhs_scale, lhs_amax_history)
    # Quantize rhs in natural layout [E,K,N] for the dgrad (contracts N).
    q_rhs, new_rhs_scale = in_q(comp, fwd_dtype, rhs, rhs_scale, rhs_amax_history)
    # Transpose to [E,N,K] for the forward pass (CuTe kernel contracts K, the last axis).
    q_rhs_t = quantize(jnp.swapaxes(rhs, 1, 2), fwd_dtype, new_rhs_scale, comp)
    # out_scale = 1/(lhs_scale*rhs_scale): dequant is a MULTIPLY by the scale product,
    # but the kernel epilogue DIVIDES by out_scale, so pass the reciprocal (shape (1,)
    # float32 to match the epilogue divisor).
    out_scale = jnp.reshape((1.0 / (new_lhs_scale * new_rhs_scale)).astype(jnp.float32), (1,))
    return quantized_ragged_dot(
        q_lhs,
        q_rhs_t,
        q_rhs,
        out_scale,
        new_rhs_scale,
        grad_scale,
        grad_amax_history,
        lhs,
        rhs,
        group_sizes,
        rev_dtype,
    )
