# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""FP8 grouped matmul (``ragged_dot``) for Hopper -- all three GEMMs on FP8.

The MoE-grouped analog of the dense ``fp8_scaled_dot_general`` in ``_src/fp8.py``.
All three contractions run on the FP8 tensor cores via the ragged Mosaic ``wgmma``
kernels in ``_src/ragged_dot_mgpu``, which handle non-uniform dynamic
``group_sizes`` (no equal-size / batched-dense reshape):

  * forward   ``out[T,N]  = lhs[T,K] . rhs[E,K,N]``  -- ``mgpu_ragged_dot``
  * grad_lhs  ``dl[T,K]   = g[T,N] . rhs[E,K,N]``    -- ``mgpu_ragged_dot``
  * grad_rhs  ``dr[E,K,N] = lhs[T,K] . g[T,N]``      -- ``mgpu_dwgrad``

FP8 ``wgmma`` needs the contracting dim contiguous in both operands and cannot
transpose an operand at runtime, so each bf16 input is quantized into *both* the
natural and the transposed FP8 layout by one fused cast-transpose+amax read
(``in_q_ct`` in ``_src/fp8_cast_transpose``): the forward consumes ``q_lhs [T,K]``
and ``q_rhs_t [E,N,K]``, the dgrad consumes ``q_rhs [E,K,N]``, and the wgrad
consumes ``q_lhs_t [K, T+E*256]`` (with the boundary appendix, see
``write_boundary_appendix``) and ``q_g_t [N,T]``.

Delayed per-tensor scaling (TE-style scale + amax history) follows the dense
helpers in ``_src/fp8.py``: activation, weight, and output-gradient scale +
amax-history state all update through the custom VJPs as ``OverwriteWithGradient``
cotangents.

``rev_dtype`` defaults to E5M2 (the numerically correct output-gradient dtype),
so both backward GEMMs are genuine mixed ``e5m2 x e4m3`` contractions.  Mixed
``wgmma`` needs jax >= 0.11.0 (jax-ml/jax#38859; see ``Fp8RaggedDotOp``).
"""

import functools

import jax
import jax.numpy as jnp
from jax import custom_vjp

from .fp8 import roll_amax_history
from .fp8_cast_transpose import (
    WGRAD_TOKEN_BLOCK,
    _next_scale_and_inv,
    cast_transpose_amax_2d,
    in_q_ct,
    write_boundary_appendix,
)
from .ragged_dot_mgpu import mgpu_dwgrad, mgpu_ragged_dot

_E4M3 = jnp.float8_e4m3fn
_E5M2 = jnp.float8_e5m2

# H100 per-block SMEM ceiling (bytes) for the Mosaic operand pipeline; see the
# forward config note. The wgrad operand tiles are FP8 (1 byte), so the pipeline
# footprint is ``(block_m + block_n) * block_k * max_concurrent_steps`` bytes.
_H100_SMEM_LIMIT = 232448

# Candidate wgmma block sizes, largest first; a dim picks the largest that
# divides it (falling back to 16 when none does).
_GEMM_BLOCKS = (128, 64, 32, 16)


def _autotuned_config(m: int, n: int, k: int) -> dict:
    """Static Mosaic block config for the ragged FP8 wgmma forward/dgrad kernel.

    Tuned on H100 at the d=2560 MoE shapes (hidden 2560, intermediate 1280) (per-leg sweeps, both the
    forward and the dgrad orientation): ``(block_m, block_n, block_k) =
    (128, 128, 128)`` with a six-step pipeline beats a ``192 x 5``
    block/pipeline by ~10-14% -- the shorter accumulator
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


def _dwgrad_config(k: int, n: int) -> dict:
    """Static Mosaic block config for the ragged FP8 weight-gradient kernel.

    Tuned on H100 at the d=2560 MoE shapes (hidden 2560, intermediate 1280). ``block_m`` tiles the output K,
    ``block_n`` the output N, and ``block_k=128`` the ragged token (contracting)
    dim. The tile *pair* matters more than either dimension alone: wide-N
    ``(64, 256)`` beats ``(128, 128)``, but ``(128, 256)`` collapses, so the
    wide-N tile is only taken with the short 64-row block. Tile pairs that
    exceed the H100 per-block SMEM ceiling drop ``max_concurrent_steps`` (as
    the forward config does at ``block_k=128``); the ``next(..., fallback)``
    guards keep the config valid when K/N are not multiples of the preferred
    tile.
    """
    if k % 64 == 0 and n % 256 == 0:
        block_m, block_n = 64, 256
    else:
        block_m = next((b for b in _GEMM_BLOCKS if k % b == 0), 16)
        block_n = next((b for b in _GEMM_BLOCKS if n % b == 0), 16)
    # Plain row-major tile order (no n-snake) wins at the d2560 shapes: the wgrad's
    # operand slices for one expert group fit L2 anyway, and the snake's split n-walk
    # costs more in lhs re-reads than it saves (measured ~6-8%).
    grid_block_n = 1
    block_k = 128
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
    if m % 64 != 0:
        raise ValueError(
            f"m={m} (token count) must be a multiple of 64 for the FP8 wgmma accumulator tiling in the Mosaic ragged kernel"
        )
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


@functools.partial(custom_vjp, nondiff_argnums=(11,))
def quantized_ragged_dot(
    q_lhs,  # [T, K] e4m3      -- forward A
    q_lhs_t,  # [K, T+E*256] e4m3 -- grad_rhs A (token-contiguous, + boundary appendix)
    q_rhs,  # [E, K, N] e4m3   -- grad_lhs B (natural layout)
    q_rhs_t,  # [E, N, K] e4m3 -- forward B (transpose_rhs layout)
    lhs_scale,  # [1] delayed-scaling scale for the activation quantize
    rhs_scale,  # [1] delayed-scaling scale for the weight quantize
    grad_scale,  # [1] output-grad scale from the previous step
    grad_amax_history,  # output-grad amax history from the previous step
    lhs,  # [T, K] original operand: differentiable, receives grad_lhs
    rhs,  # [E, K, N] original operand: differentiable, receives grad_rhs
    group_sizes,  # [E]
    rev_dtype,  # static: output-gradient FP8 dtype (E5M2 by default)
):
    """FP8 ragged forward of pre-quantized E4M3 operands; FP8 dgrad and wgrad.

    All FP8 layouts are precomputed by ``in_q_ct``, so no transpose happens here
    (see the module docstring for which GEMM consumes which layout). ``lhs`` and
    ``rhs`` are the differentiable args -- the operand gradients flow to them --
    and also supply the compute dtype; the ``q_*`` and scale args get ``None``
    cotangents (their state updates through ``in_q_ct``'s VJP). The backward
    quantizes the output gradient to ``rev_dtype`` with delayed scaling and
    returns the rolled ``grad_scale`` / ``grad_amax_history`` as
    ``OverwriteWithGradient`` cotangents.
    """
    combined = (lhs_scale * rhs_scale).astype(jnp.float32)
    return _ragged_fp8(q_lhs, q_rhs_t, group_sizes, lhs.dtype, combined)


def _qrd_fwd(
    q_lhs,
    q_lhs_t,
    q_rhs,
    q_rhs_t,
    lhs_scale,
    rhs_scale,
    grad_scale,
    grad_amax_history,
    lhs,
    rhs,
    group_sizes,
    rev_dtype,
):
    combined = (lhs_scale * rhs_scale).astype(jnp.float32)
    out = _ragged_fp8(q_lhs, q_rhs_t, group_sizes, lhs.dtype, combined)
    res = (q_lhs_t, q_rhs, lhs_scale, rhs_scale, grad_scale, grad_amax_history, lhs, rhs, group_sizes)
    return out, res


def _qrd_bwd(rev_dtype, res, g):
    q_lhs_t, q_rhs, lhs_scale, rhs_scale, grad_scale, grad_amax_history, lhs, rhs, group_sizes = res
    out_dtype = lhs.dtype

    # Delayed scaling for the output gradient: one fused cast-transpose+amax read
    # produces both the natural [T,N] layout q_g (for the dgrad) and the transposed
    # [N,T] layout q_g_t (for the wgrad) plus the current-step amax to roll into
    # the history -- a single bf16 read of ``g``.
    new_g_scale, inv_g = _next_scale_and_inv(rev_dtype, grad_scale, grad_amax_history)
    q_g, q_g_t, cur_amax = cast_transpose_amax_2d(g, inv_g, rev_dtype)  # [T,N], [N,T]
    new_g_hist = roll_amax_history(cur_amax[0], grad_amax_history)

    # grad_lhs[T,K] = g[T,N] . rhs[E,K,N] (contract N), on the pre-cast natural
    # weight layout q_rhs. The dgrad has the same shape structure as the forward
    # and shares its tuned config (per-leg sweeps: the same block config wins both).
    dlhs_scale = (rhs_scale * new_g_scale).astype(jnp.float32)
    grad_lhs = _ragged_fp8(q_g, q_rhs, group_sizes, out_dtype, dlhs_scale)

    # grad_rhs[E,K,N] = lhs[T,K] . g[T,N] (contracts the ragged token dim) via
    # mgpu_dwgrad on the pre-cast token-contiguous layouts q_lhs_t [K, T+E*256]
    # and q_g_t [N,T]. Group boundaries falling mid-token-tile are handled by the
    # boundary appendix: fill q_lhs_t's appendix slots with pre-masked boundary
    # blocks (in place) and the kernel's index_map reads them for the first/last
    # pipeline step of each group.
    t = lhs.shape[0]
    if t % 128 != 0:
        raise ValueError(
            f"T={t} (token count) must be a multiple of 128 for the FP8 weight-gradient kernel: "
            "the token dim is the wgmma contracting dim and its GMEM/TMA tile is 128 tokens wide."
        )
    _, k, n = rhs.shape
    drhs_cfg = _dwgrad_config(k, n)
    drhs_scale = (lhs_scale * new_g_scale).astype(jnp.float32)
    q_lhs_t = write_boundary_appendix(q_lhs_t, group_sizes)
    grad_rhs = mgpu_dwgrad(
        q_lhs_t,
        q_g_t,
        group_sizes=group_sizes,
        out_dtype=out_dtype,
        out_scale=drhs_scale,
        **drhs_cfg,
    )
    grad_rhs = grad_rhs.astype(rhs.dtype)

    # Operand grads flow to lhs/rhs; the output-grad scale and amax history are
    # overwritten with the rolled delayed-scaling state; everything else (handled
    # by in_q_ct's VJP) gets None.
    return (None, None, None, None, None, None, new_g_scale, new_g_hist, grad_lhs, grad_rhs, None)


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
    """FP8 ``ragged_dot`` with an E4M3 forward and an all-FP8 (dgrad + wgrad) backward.

    Args:
        lhs: ``[T, K]`` activations (rows sorted by expert / contiguous groups).
        rhs: ``[E, K, N]`` expert weights.
        group_sizes: ``[E]`` token count per expert (``sum == T``); fully dynamic
            and non-uniform.
        lhs_scale, rhs_scale, grad_scale: ``[1]`` delayed-scaling scales.
        lhs_amax_history, rhs_amax_history, grad_amax_history: amax histories.
        fwd_dtype: forward-operand FP8 dtype (E4M3).
        rev_dtype: output-gradient FP8 dtype. Defaults to E5M2, making both
            gradients mixed ``e5m2 x e4m3`` contractions (needs jax >= 0.11.0,
            jax-ml/jax#38859); E4M3 gives uniform GEMMs that lower on jax 0.10.x.

    Quantizes activations and expert weights to ``fwd_dtype`` with delayed
    per-tensor scaling and runs the forward and both gradients on the FP8 tensor
    cores (see the module docstring for the kernel/layout mapping). All scale /
    amax-history state updates through the custom VJPs as
    ``OverwriteWithGradient`` overwrites.

    The dual-layout quantize runs unconditionally, so a forward that is never
    differentiated still pays the extra transposed-layout write. This op targets
    training, where the backward consumes every layout; a single-layout
    inference path would need its own cast kernel and is not worth the extra
    machinery here.
    """
    t, k = lhs.shape
    _, _, n = rhs.shape
    # Fail fast on the whole op's alignment contract -- forward AND backward --
    # at call time. The op= path bypasses ragged_dot's 512-row padding, and the
    # backward's constraints are stricter than the forward's (T % 64), so
    # without this check a forward-only run would succeed and the first
    # jax.grad would fail at trace time. Checked before the device check so the
    # shape contract surfaces on any backend.
    if t % 128 != 0:
        raise ValueError(
            f"T={t} (token count) must be a multiple of 128 for fp8_scaled_ragged_dot: the FP8 "
            "weight-gradient kernel contracts the token dim in 128-token TMA tiles."
        )
    if k % 128 != 0:
        raise ValueError(
            f"K={k} must be a multiple of 128 for fp8_scaled_ragged_dot: K is the dgrad output's "
            "minor dim and needs bf16 TMA swizzle alignment in the Mosaic ragged wgmma kernel."
        )
    if n % 128 != 0:
        raise ValueError(
            f"N={n} must be a multiple of 128 for fp8_scaled_ragged_dot: bf16 TMA swizzle "
            "alignment of the forward output store in the Mosaic ragged wgmma kernel."
        )
    device = jax.devices()[0]
    if device.platform != "gpu" or not device.compute_capability.startswith("9."):
        raise NotImplementedError(
            f"fp8_scaled_ragged_dot requires Hopper (SM90): the Mosaic wgmma kernels are "
            f"sm_90a-specific. Got {device.device_kind}."
        )
    # One fused cast-transpose+amax read per operand produces both FP8 layouts.
    # The activation's transposed layout carries the wgrad boundary appendix
    # (2*WGRAD_TOKEN_BLOCK extra columns per expert), filled in the backward.
    e = rhs.shape[0]
    q_lhs, q_lhs_t, new_lhs_scale = in_q_ct(
        fwd_dtype, "2d", e * 2 * WGRAD_TOKEN_BLOCK, lhs, lhs_scale, lhs_amax_history
    )
    q_rhs, q_rhs_t, new_rhs_scale = in_q_ct(fwd_dtype, "3d", 0, rhs, rhs_scale, rhs_amax_history)
    return quantized_ragged_dot(
        q_lhs,
        q_lhs_t,
        q_rhs,
        q_rhs_t,
        new_lhs_scale,
        new_rhs_scale,
        grad_scale,
        grad_amax_history,
        lhs,
        rhs,
        group_sizes,
        rev_dtype,
    )
