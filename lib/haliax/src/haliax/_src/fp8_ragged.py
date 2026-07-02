# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""FP8 grouped matmul (``ragged_dot``) for Hopper H100 via the CuTe DSL.

FP8 forward + mixed-dtype FP8 ``dgrad`` (E5M2 x E4M3) and, by default, a genuine
mixed-FP8 ``wgrad``: the weight gradient ``drhs_e[k,n] = sum_m lhs_e[m,k]·g_e[m,n]``
contracts the token axis M, so both operands are cast-transposed to token-major
FP8 (activations E4M3, output-grad E5M2) and fed to ``cute_wgrad`` -- a real
E4M3×E5M2 grouped GEMM (the TE recipe), not a bf16 fallback. ``WgradMode.BF16``
keeps the (bf16-exact, dequantize-based) Task-2 wgrad for A/B testing.

Delayed per-tensor scaling reuses ``_src/fp8.py``: ``in_q`` threads the two
forward-operand scale/amax states through its own custom_vjp; the output-grad
state is threaded by ``quantized_ragged_dot``'s custom_vjp. The weight is
transposed in bf16 (hardware-legal) then cast to FP8 for the k-contiguous forward.
"""

import functools
from enum import StrEnum

import jax
import jax.numpy as jnp
from jax import custom_vjp

from .fp8 import in_q, quantize, update_fp8_meta
from .fp8_cast_transpose import cast_transpose
from .ragged_dot_cute import cute_ragged_dot, cute_wgrad

_E4M3 = jnp.float8_e4m3fn
_E5M2 = jnp.float8_e5m2


class WgradMode(StrEnum):
    """Weight-gradient backend for the FP8 ``ragged_dot`` backward.

    ``FP8`` (default) cast-transposes both operands to token-major FP8 and runs a
    genuine E4M3×E5M2 grouped GEMM (``cute_wgrad``) contracting the token axis.
    ``BF16`` dequantizes and differentiates the reference bf16 ``ragged_dot`` --
    bit-exact vs bf16, used for A/B testing and below the FP8 crossover point.
    """

    BF16 = "bf16"
    FP8 = "fp8"


@functools.partial(custom_vjp, nondiff_argnums=(11, 12, 13))
def quantized_ragged_dot(
    q_lhs,
    q_lhs_t,
    q_rhs_t,
    q_rhs,
    out_scale,
    lhs_scale,
    rhs_scale,
    grad_scale,
    grad_amax_history,
    lhs,
    rhs,
    group_sizes,
    rev_dtype,
    wgrad_mode,
):
    """FP8 forward of pre-quantized operands; FP8 dgrad + (FP8 or bf16) wgrad backward.

    ``q_lhs[T,K]`` / ``q_rhs_t[E,N,K]`` are E4M3 (k-contiguous, forward). ``q_lhs_t``
    is the token-major ``[K,T]`` activation the FP8 wgrad contracts over T; ``q_rhs``
    is the natural-layout ``[E,K,N]`` E4M3 weight the dgrad contracts over N.
    ``out_scale = 1/(lhs_scale*rhs_scale)``: the raw fp8 product is
    ``(a/s_a)·(b/s_b) = (a·b)/(s_a·s_b)``, so recovering ``a·b`` MULTIPLIES by
    ``s_a·s_b`` -- but the CuTe epilogue DIVIDES the accumulator by ``out_scale``,
    hence we pass the reciprocal. ``lhs_scale``/``rhs_scale`` are the applied
    forward scales (needed to build the reciprocal wgrad/dgrad divisors);
    ``grad_scale``/``grad_amax_history`` carry the output-grad delayed-scaling
    state (returned as cotangents, OverwriteWithGradient). ``lhs``/``rhs`` are the
    bf16 operands the bf16 wgrad differentiates. ``rev_dtype`` is the output-grad
    FP8 dtype (E5M2); ``wgrad_mode`` selects the FP8 or bf16 wgrad.
    """
    return cute_ragged_dot(q_lhs, q_rhs_t, group_sizes, out_dtype=lhs.dtype, out_scale=out_scale)


def _qrd_fwd(
    q_lhs,
    q_lhs_t,
    q_rhs_t,
    q_rhs,
    out_scale,
    lhs_scale,
    rhs_scale,
    grad_scale,
    grad_amax_history,
    lhs,
    rhs,
    group_sizes,
    rev_dtype,
    wgrad_mode,
):
    out = cute_ragged_dot(q_lhs, q_rhs_t, group_sizes, out_dtype=lhs.dtype, out_scale=out_scale)
    return out, (q_lhs_t, q_rhs, lhs_scale, rhs_scale, grad_scale, grad_amax_history, lhs, rhs)


def _qrd_bwd(group_sizes, rev_dtype, wgrad_mode, res, g):
    q_lhs_t, q_rhs, lhs_scale, rhs_scale, grad_scale, grad_amax_history, lhs, rhs = res

    # --- output-grad delayed scaling: next-step scale + rolled history ---
    new_grad_scale, new_grad_history = update_fp8_meta(g, rev_dtype, grad_scale, grad_amax_history)
    # Cast-transpose g once: q_g[M,N] E5M2 (dgrad, row-major, bit-identical to
    # quantize(g,...)) and q_g_t[N,M] E5M2 (wgrad, token-major).
    q_g, q_g_t = cast_transpose(g, new_grad_scale, out_dtype=rev_dtype, compute_dtype=jnp.float32)

    # --- FP8 dgrad: g(E5M2) @ rhs(E4M3), contract N (natural layout) -> dlhs[M,K] ---
    # q_rhs is [E,K,N]; the CuTe kernel maps b[E,N_tile,K_contr], so passing q_rhs
    # contracts the last axis N (the fwd output dim) producing dlhs[M,K].
    # dgrad_scale = 1/(grad_scale*rhs_scale): dequant is a MULTIPLY by the scale
    # product, but the kernel epilogue DIVIDES by out_scale, so pass the reciprocal.
    dgrad_scale = jnp.reshape((1.0 / (new_grad_scale * rhs_scale)).astype(jnp.float32), (1,))
    grad_lhs = cute_ragged_dot(q_g, q_rhs, group_sizes, out_dtype=lhs.dtype, out_scale=dgrad_scale)

    # --- wgrad: drhs_e[k,n] = sum_m lhs_e[m,k]·g_e[m,n], contracting the token axis M ---
    if wgrad_mode == WgradMode.FP8:
        # Genuine mixed E4M3(lhs)×E5M2(grad) grouped GEMM on the token-major operands.
        # wgrad_scale = 1/(lhs_scale*grad_scale): reciprocal for the same divide-in-
        # epilogue reason as the forward/dgrad (NOT the raw scale product).
        wgrad_scale = jnp.reshape((1.0 / (lhs_scale * new_grad_scale)).astype(jnp.float32), (1,))
        grad_rhs = cute_wgrad(q_lhs_t, q_g_t, group_sizes, out_dtype=rhs.dtype, out_scale=wgrad_scale)
    else:
        # bf16-exact wgrad: differentiate the reference bf16 ragged_dot for grad_rhs only.
        from haliax.nn.ragged_dot import ragged_dot as _bf16_ragged_dot  # noqa: PLC0415

        _, vjp_fn = jax.vjp(lambda ro: _bf16_ragged_dot(lhs, ro, group_sizes, op=None), rhs)
        (grad_rhs,) = vjp_fn(g)

    # Cotangents for (q_lhs, q_lhs_t, q_rhs_t, q_rhs, out_scale, lhs_scale, rhs_scale,
    # grad_scale, grad_amax_history, lhs, rhs). grad_scale / grad_amax_history return
    # the updated state (OverwriteWithGradient pattern).
    return (None, None, None, None, None, None, None, new_grad_scale, new_grad_history, grad_lhs, grad_rhs)


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
    wgrad_mode=WgradMode.FP8,
):
    """FP8 ``ragged_dot``: E4M3 forward, mixed-FP8 dgrad, FP8 (or bf16) wgrad.

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
        wgrad_mode: FP8 (genuine cast-transpose wgrad, default) or BF16 (reference).

    Returns:
        [T, N] output array in ``lhs.dtype``.
    """
    comp = quantize_compute_type
    q_lhs, new_lhs_scale = in_q(comp, fwd_dtype, lhs, lhs_scale, lhs_amax_history)
    # Token-major activation for the FP8 wgrad. This is the transpose of the same
    # E4M3 forward operand (bit-identical to cast_transpose's token-major output),
    # so no second quantize is paid; a fused cast-transpose kernel would later
    # fold in_q's quantize and this transpose into one HBM pass.
    q_lhs_t = jnp.swapaxes(q_lhs, 0, 1)
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
        q_lhs_t,
        q_rhs_t,
        q_rhs,
        out_scale,
        new_lhs_scale,
        new_rhs_scale,
        grad_scale,
        grad_amax_history,
        lhs,
        rhs,
        group_sizes,
        rev_dtype,
        wgrad_mode,
    )
