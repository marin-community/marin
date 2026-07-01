# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""FP8 grouped matmul (``ragged_dot``) for Hopper H100 -- FP8 forward, bf16 backward.

The MoE-grouped analog of the dense ``fp8_scaled_dot_general`` in ``_src/fp8.py``.
The forward contracts activations and expert weights as same-dtype ``e4m3 x e4m3``
on the Hopper FP8 tensor cores via the genuine ragged Mosaic ``wgmma`` kernel
(``_src/ragged_dot_mgpu``), which handles genuinely **non-uniform, dynamic**
``group_sizes`` (no equal-size / batched-dense reshape). The backward is computed
in **bf16** (numerically exact) by differentiating the reference bf16
``ragged_dot`` against the operands, so the operand gradients match the bf16
training default bit-for-bit.

Delayed per-tensor scaling (TE-style scale + amax history) is reused verbatim
from the dense helpers in ``_src/fp8.py`` (``in_q`` / ``quantize``): the input and
kernel scale + amax_history update through ``in_q``'s custom VJP as
``OverwriteWithGradient`` overwrites. The output-gradient scaling state is unused
here (the backward is bf16) and threads through unchanged.

To get the expert weights contracting-dim-contiguous -- as FP8 ``wgmma`` requires
-- the weight is transposed in **bf16** (a hardware-legal transpose) and only then
cast to FP8; no strided FP8 transpose or fused cast-transpose kernel is used yet.
"""

import functools

import jax
import jax.numpy as jnp
from jax import custom_vjp

from .fp8 import in_q, quantize
from .ragged_dot_mgpu import mgpu_ragged_dot

_E4M3 = jnp.float8_e4m3fn


def _fixed_config(m: int, n: int, k: int) -> dict:
    """Conservative (untuned) Mosaic block config for the ragged FP8 wgmma kernel.

    Block sizes divide the operand shapes: ``block_k`` must divide ``k``; ``block_n``
    must divide ``n`` (and ``n`` must be swizzle-aligned -- a multiple of 128 for the
    bf16 output store's TMA descriptor); ``block_m`` divides ``m`` here, though the
    ragged kernel also masks a non-dividing final m-tile. Autotuning (including the
    tuned ``block_m=192`` and ``max_concurrent_steps`` sweep) is a later change.
    """
    block_m = next((b for b in (128, 64, 32, 16) if m % b == 0), 16)
    block_n = next((b for b in (128, 64, 32, 16) if n % b == 0), 16)
    block_k = next((b for b in (64, 32, 16) if k % b == 0), 16)
    grid_block_n = next((gb for gb in (4, 2, 1) if n % (gb * block_n) == 0), 1)
    return dict(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        max_concurrent_steps=6,
        grid_block_n=grid_block_n,
    )


def _ragged_fp8(lhs, rhs_nk, group_sizes, out_dtype, out_scale):
    """Genuine ragged FP8 wgmma: ``lhs[M,K] . rhs[G,K,N]`` contracting K.

    ``rhs_nk`` is laid out ``[G, N, K]`` (the ``transpose_rhs`` layout) so the
    contracting dim K is contiguous for both operands, as FP8 ``wgmma`` requires.
    The per-tensor dequant ``out_scale`` is folded into the kernel store.
    """
    g, n, k = rhs_nk.shape
    m = lhs.shape[0]
    if n % 128 != 0:
        raise ValueError(
            f"n={n} must be a multiple of 128 for bf16 TMA swizzle alignment in the Mosaic ragged wgmma kernel"
        )
    cfg = _fixed_config(m, n, k)
    return mgpu_ragged_dot(
        lhs,
        rhs_nk,
        group_sizes=group_sizes,
        transpose_rhs=True,
        out_dtype=out_dtype,
        out_scale=out_scale,
        **cfg,
    )


@functools.partial(custom_vjp, nondiff_argnums=(7,))
def quantized_ragged_dot(q_lhs, q_rhs_t, out_scale, lhs, rhs, grad_scale, grad_amax_history, group_sizes):
    """FP8 ragged forward of pre-quantized E4M3 operands; bf16 (exact) backward.

    ``q_lhs`` is ``[T,K]`` E4M3, ``q_rhs_t`` is ``[E,N,K]`` E4M3 (the transpose_rhs
    layout). ``out_scale`` is the folded per-tensor dequant ``lhs_scale*rhs_scale``.
    ``lhs``/``rhs`` are the original compute-dtype (bf16) operands, carried so the
    backward can differentiate the reference bf16 ``ragged_dot``.
    ``grad_scale``/``grad_amax_history`` are the output-gradient delayed-scaling state;
    threaded as differentiable args so the backward can return identity cotangents
    (preserving the current values) until the FP8-backward step updates them.
    """
    return _ragged_fp8(q_lhs, q_rhs_t, group_sizes, lhs.dtype, out_scale)


def _qrd_fwd(q_lhs, q_rhs_t, out_scale, lhs, rhs, grad_scale, grad_amax_history, group_sizes):
    out = _ragged_fp8(q_lhs, q_rhs_t, group_sizes, lhs.dtype, out_scale)
    return out, (lhs, rhs, grad_scale, grad_amax_history)


def _qrd_bwd(group_sizes, res, g):
    lhs, rhs, grad_scale, grad_amax_history = res
    # bf16-exact operand gradients: differentiate the reference bf16 ragged_dot.
    # Local import breaks the quantization <-> nn.ragged_dot import cycle.
    from haliax.nn.ragged_dot import ragged_dot as _bf16_ragged_dot  # noqa: PLC0415

    _, vjp_fn = jax.vjp(lambda lo, ro: _bf16_ragged_dot(lo, ro, group_sizes, op=None), lhs, rhs)
    grad_lhs, grad_rhs = vjp_fn(g)
    # Cotangents for (q_lhs, q_rhs_t, out_scale, lhs, rhs, grad_scale, grad_amax_history):
    # - quantized operands and the folded scale are intermediate (overwrites handled by in_q)
    # - operand grads flow to lhs/rhs
    # - grad_scale/grad_amax_history are preserved unchanged (bf16 backward; the
    #   FP8-backward follow-up will update them to new_scale/new_history instead)
    return (None, None, None, grad_lhs, grad_rhs, grad_scale, grad_amax_history)


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
    rev_dtype=_E4M3,
):
    """FP8 ``ragged_dot`` with an E4M3 forward and a bf16 (exact) backward.

    Args:
        lhs: ``[T, K]`` activations (rows sorted by expert / contiguous groups).
        rhs: ``[E, K, N]`` expert weights.
        group_sizes: ``[E]`` token count per expert (``sum == T``); fully dynamic
            and non-uniform.
        lhs_scale, rhs_scale, grad_scale: ``[1]`` delayed-scaling scales.
        lhs_amax_history, rhs_amax_history, grad_amax_history: amax histories.
        fwd_dtype: forward-operand FP8 dtype (E4M3).
        rev_dtype: output-gradient FP8 dtype; unused here (backward is bf16), kept
            for interface parity with the dense op and the FP8-backward follow-up.

    Quantizes activations and expert weights to ``fwd_dtype`` with delayed
    per-tensor scaling and contracts them on the FP8 tensor cores via the genuine
    ragged ``wgmma`` kernel, then dequantizes. The input and kernel scale /
    amax-history update through ``in_q``'s custom VJP as ``OverwriteWithGradient``
    overwrites; the output-gradient state passes through unchanged.
    """
    del rev_dtype  # FP8 output-grad dtype is reserved for the FP8-backward path; unused while backward is bf16.
    comp = quantize_compute_type
    q_lhs, new_lhs_scale = in_q(comp, fwd_dtype, lhs, lhs_scale, lhs_amax_history)
    # in_q on rhs threads the kernel scale + amax_history overwrite; the naturally
    # quantized weight it returns is unused (the forward needs the transposed
    # layout), so it is dropped and DCE'd.
    _q_rhs, new_rhs_scale = in_q(comp, fwd_dtype, rhs, rhs_scale, rhs_amax_history)
    # bf16-transpose-then-cast: transpose in bf16 (hardware-legal), then cast to FP8.
    q_rhs_t = quantize(jnp.swapaxes(rhs, 1, 2), fwd_dtype, new_rhs_scale, comp)
    out_scale = (new_lhs_scale * new_rhs_scale).astype(jnp.float32)
    return quantized_ragged_dot(q_lhs, q_rhs_t, out_scale, lhs, rhs, grad_scale, grad_amax_history, group_sizes)
