# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""FP8 grouped matmul (``ragged_dot``) for Hopper H100 -- FP8 forward, bf16 backward.

The MoE-grouped analog of the dense ``fp8_scaled_dot_general`` in ``_src/fp8.py``.
The forward contracts activations and expert weights as same-dtype ``e4m3 x e4m3``
on the Hopper FP8 tensor cores via the genuine ragged Mosaic ``wgmma`` kernel
(``_src/ragged_dot_mgpu``), which handles genuinely **non-uniform, dynamic**
``group_sizes`` (no equal-size / batched-dense reshape). The backward is computed
in **bf16** (numerically exact) by directly invoking the Triton kernels from the
bf16 ``ragged_dot`` backward (``_DLHS_DIM_NUMS`` / ``_DRHS_DIM_NUMS`` via
``_triton_pallas_call``), so the operand gradients match the bf16 training default
bit-for-bit with no forward recompute.

Delayed per-tensor scaling (TE-style scale + amax history) is reused verbatim
from the dense helpers in ``_src/fp8.py``: the input and kernel scale + amax_history
update through ``in_q_ct``'s custom VJP as ``OverwriteWithGradient`` overwrites.
The output-gradient scaling state is unused here (the backward is bf16) and threads
through unchanged.

Both the activation (lhs) and expert weight (rhs) quantize go through ``in_q_ct``
(``_src/fp8_cast_transpose``), a fused Pallas-Triton kernel that reads the bf16 tile
once, folds the amax reduction via ``atomic_max``, and stores **both** the natural
and the transposed FP8 tile in one pass.  The transposed-weight layout (``[E,N,K]``)
is consumed by the forward; the natural-weight layout (``[E,K,N]``) and the
transposed-activation layout (``[K,T]``) are produced but deferred to the FP8
backward follow-up.
"""

import functools

import jax
import jax.numpy as jnp
from jax import custom_vjp

from .fp8_cast_transpose import in_q_ct
from .ragged_dot_mgpu import mgpu_ragged_dot

_E4M3 = jnp.float8_e4m3fn
_E5M2 = jnp.float8_e5m2

# H100 per-block SMEM ceiling (bytes) for the Mosaic operand pipeline; see the
# forward config note.
_H100_SMEM_LIMIT = 232448

# Candidate wgmma block sizes, largest first; a dim picks the largest that
# divides it (falling back to 16 when none does).
_GEMM_BLOCKS = (128, 64, 32, 16)


def _autotuned_config(m: int, n: int, k: int) -> dict:
    """Static Mosaic block config for the ragged FP8 wgmma forward kernel.

    Tuned on H100 at the d2560 grug-MoE shapes (per-leg sweeps):
    ``(block_m, block_n, block_k) = (128, 128, 128)`` with a six-step pipeline
    beats a ``192 x 5`` block/pipeline by ~10-14% -- the shorter accumulator
    halves register pressure, buying a deeper pipeline. Operand tiles are FP8
    (1 byte), so the pipeline SMEM footprint is
    ``(block_m + block_n) * block_k * max_concurrent_steps``
    bytes; the loop drops steps until it fits the H100 per-block limit.
    """
    block_m = 128 if m >= 128 else next((b for b in _GEMM_BLOCKS if m % b == 0), 16)
    block_n = next((b for b in _GEMM_BLOCKS if n % b == 0), 16)
    block_k = next((b for b in _GEMM_BLOCKS if k % b == 0), 16)
    grid_block_n = next((gb for gb in (4, 2, 1) if n % (gb * block_n) == 0), 1)
    max_concurrent_steps = 6
    while max_concurrent_steps > 1 and (block_m + block_n) * block_k * max_concurrent_steps > _H100_SMEM_LIMIT:
        max_concurrent_steps -= 1
    return dict(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        max_concurrent_steps=max_concurrent_steps,
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
    cfg = _autotuned_config(m, n, k)
    return mgpu_ragged_dot(
        lhs,
        rhs_nk,
        group_sizes=group_sizes,
        transpose_rhs=True,
        out_dtype=out_dtype,
        out_scale=out_scale,
        **cfg,
    )


@custom_vjp
def quantized_ragged_dot(q_lhs, q_rhs_t, out_scale, lhs, rhs, grad_scale, grad_amax_history, group_sizes):
    """FP8 ragged forward of pre-quantized E4M3 operands; bf16 (exact) backward.

    ``q_lhs`` is ``[T,K]`` E4M3, ``q_rhs_t`` is ``[E,N,K]`` E4M3 (the transpose_rhs
    layout). ``out_scale`` is the folded per-tensor dequant ``lhs_scale*rhs_scale``.
    ``lhs``/``rhs`` are the original compute-dtype (bf16) operands, carried so the
    backward can compute gradients directly with the bf16 Triton kernels.
    ``grad_scale``/``grad_amax_history`` are the output-gradient delayed-scaling state;
    threaded as differentiable args so the backward can return identity cotangents
    (preserving the current values) until the FP8-backward step updates them.
    ``group_sizes`` is a regular operand with a ``None`` cotangent rather than a
    ``nondiff_argnums`` entry: nondiff positions reject traced arrays under ``jit``.
    """
    return _ragged_fp8(q_lhs, q_rhs_t, group_sizes, lhs.dtype, out_scale)


def _qrd_fwd(q_lhs, q_rhs_t, out_scale, lhs, rhs, grad_scale, grad_amax_history, group_sizes):
    out = _ragged_fp8(q_lhs, q_rhs_t, group_sizes, lhs.dtype, out_scale)
    return out, (lhs, rhs, grad_scale, grad_amax_history, group_sizes)


def _qrd_bwd(res, g):
    lhs, rhs, grad_scale, grad_amax_history, group_sizes = res
    # Direct bf16 operand gradients: call the same Triton kernels used by the bf16
    # ragged_dot backward. lhs/rhs are already in the residuals, so there is no
    # forward recompute.
    # Local import breaks the quantization <-> nn.ragged_dot import cycle.
    from haliax.nn.ragged_dot import _DLHS_DIM_NUMS, _DRHS_DIM_NUMS, _triton_pallas_call  # noqa: PLC0415

    # dlhs[M,K] = dout[M,N] @ rhs[G,K,N]^T  (contracts N)
    grad_lhs = _triton_pallas_call(g, rhs, group_sizes, _DLHS_DIM_NUMS)
    # drhs[G,K,N] = lhs[M,K]^T @ dout[M,N]  (contracts M, ragged)
    grad_rhs = _triton_pallas_call(lhs, g, group_sizes, _DRHS_DIM_NUMS)
    # Cotangents for (q_lhs, q_rhs_t, out_scale, lhs, rhs, grad_scale,
    # grad_amax_history, group_sizes):
    # - quantized operands and the folded scale are intermediate (overwrites handled by in_q)
    # - operand grads flow to lhs/rhs
    # - grad_scale/grad_amax_history are preserved unchanged (bf16 backward; the
    #   FP8-backward follow-up will update them to new_scale/new_history instead)
    return (None, None, None, grad_lhs, grad_rhs, grad_scale, grad_amax_history, None)


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
    fwd_dtype=_E4M3,
    rev_dtype=_E5M2,
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
    amax-history update through ``in_q_ct``'s custom VJP as ``OverwriteWithGradient``
    overwrites; the output-gradient state passes through unchanged.
    """
    device = jax.devices()[0]
    if device.platform != "gpu" or not device.compute_capability.startswith("9."):
        raise NotImplementedError(
            f"fp8_scaled_ragged_dot requires Hopper (SM90): the Mosaic wgmma kernels are "
            f"sm_90a-specific. Got {device.device_kind}."
        )
    del rev_dtype  # FP8 output-grad dtype is reserved for the FP8-backward path; unused while backward is bf16.
    # Fused cast-transpose+amax: reads each bf16 tile once, folds the amax reduction, and
    # stores both the natural and the transposed FP8 tile in one pass.  VJP threading matches in_q.
    # _q_lhs_t [K,T] and _q_rhs [E,K,N] are produced in the same fused read as the used layouts;
    # they will be wired into the FP8 backward in the follow-up commit.
    q_lhs, _q_lhs_t, new_lhs_scale = in_q_ct(fwd_dtype, "2d", lhs, lhs_scale, lhs_amax_history)
    _q_rhs, q_rhs_t, new_rhs_scale = in_q_ct(fwd_dtype, "3d", rhs, rhs_scale, rhs_amax_history)
    out_scale = (new_lhs_scale * new_rhs_scale).astype(jnp.float32)
    return quantized_ragged_dot(q_lhs, q_rhs_t, out_scale, lhs, rhs, grad_scale, grad_amax_history, group_sizes)
