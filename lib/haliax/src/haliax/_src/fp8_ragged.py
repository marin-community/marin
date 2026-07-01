# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""FP8 grouped matmul (``ragged_dot``) for Hopper H100 -- all three GEMMs on FP8.

The MoE-grouped analog of the dense ``fp8_scaled_dot_general`` in ``_src/fp8.py``.
The forward contracts activations and expert weights as same-dtype ``e4m3 x e4m3``
on the Hopper FP8 tensor cores via the genuine ragged Mosaic ``wgmma`` kernel
(``_src/ragged_dot_mgpu``), which handles genuinely **non-uniform, dynamic**
``group_sizes`` (no equal-size / batched-dense reshape).

The backward runs **both** gradients on the FP8 tensor cores:

  * grad_lhs  ``dl[T,K]  = g[T,N] . rhs[E,K,N]`` (contract N)  -- FP8 ``wgmma``.
  * grad_rhs  ``dr[E,K,N] = lhs[T,K] . g[T,N]`` (contract the ragged token dim)
    -- FP8 ``wgmma`` via ``mgpu_dwgrad``.

The output gradient ``g`` is quantized to ``rev_dtype`` with delayed per-tensor
scaling; on stock jaxlib Mosaic ``wgmma`` rejects mixed operand dtypes, so
``rev_dtype`` defaults to E4M3 -- the same dtype as the E4M3 weight, making both
gradients uniform ``e4m3 x e4m3`` contractions that lower on unpatched Mosaic.
The dgrad consumes the **pre-produced natural weight layout** ``q_rhs [E,K,N]``;
the wgrad contracts the ragged token dim, which FP8 ``wgmma`` cannot transpose at
runtime, so it consumes the **pre-produced token-contiguous layouts**
``q_lhs_t [K,T]`` (cast once by the forward, no re-cast) and ``q_g_t [N,T]`` (the
transposed output gradient), emitting ``[E,K,N]`` directly with the dequant scale
folded. This is the approximate same-dtype backward; the genuine mixed
``e5m2 x e4m3`` backward is a separate follow-up.

Delayed per-tensor scaling (TE-style scale + amax history) is reused verbatim from
the dense helpers in ``_src/fp8.py``: the input and kernel scale + amax_history
update through ``in_q_ct``'s custom VJP as ``OverwriteWithGradient`` overwrites.
The output-gradient scale + amax history update too -- the backward quantizes
``g`` with delayed scaling and returns the rolled ``grad_scale`` / ``grad_amax_history``
as the ``OverwriteWithGradient`` cotangents.

Both the activation (lhs) and expert weight (rhs) quantize go through ``in_q_ct``
(``_src/fp8_cast_transpose``), a fused Pallas-Triton kernel that reads the bf16 tile
once, folds the amax reduction via ``atomic_max``, and stores **both** the natural
and the transposed FP8 tile in one pass.  The forward consumes the transposed-weight
layout (``[E,N,K]``); the dgrad consumes the natural-weight layout (``[E,K,N]``); the
wgrad consumes the transposed-activation layout (``[K,T]``).  The output gradient is
likewise cast-transposed once, yielding ``q_g [T,N]`` (dgrad) and ``q_g_t [N,T]``
(wgrad) from a single read.
"""

import functools

import jax.numpy as jnp
from jax import custom_vjp

from .fp8_cast_transpose import _next_scale_and_inv, cast_transpose_amax_2d, in_q_ct
from .ragged_dot_mgpu import mgpu_dwgrad, mgpu_ragged_dot

_E4M3 = jnp.float8_e4m3fn

# H100 per-block SMEM ceiling (bytes) for the Mosaic operand pipeline; see the
# forward config note. The wgrad operand tiles are FP8 (1 byte), so the pipeline
# footprint is ``(block_m + block_n) * block_k * max_concurrent_steps`` bytes.
_H100_SMEM_LIMIT = 232448


@functools.cache
def _autotuned_config(num_experts: int, m: int, n: int, k: int) -> dict:
    """Static Mosaic block config for the ragged FP8 wgmma forward kernel.

    Tuned on H100 for the d2560 grug-MoE shapes. Cached per ``(num_experts, m, n, k)``
    so the selection cost is paid once. Key choices:

    - ``block_m=192``: the ragged kernel masks a non-dividing final m-tile, so 192
      is safe even when ``m`` is not divisible by 192; measured ~832 vs ~739 TFLOP/s
      at ``block_m=128`` on the d2560 operating point.
    - ``block_k=128``: measured ~852 vs ~832 TFLOP/s at ``block_k=64``. With
      ``block_m=192`` and ``max_concurrent_steps=6`` the SMEM budget is exceeded
      (``(192+128)*128*6 = 245760 > 232448``); using ``max_concurrent_steps=5``
      reduces it to 204800 bytes and keeps the ``block_k=128`` gain.
    """
    block_m = 192 if m >= 192 else next((b for b in (128, 64, 32, 16) if m % b == 0), 16)
    block_n = next((b for b in (128, 64, 32, 16) if n % b == 0), 16)
    block_k = 128 if k % 128 == 0 else next((b for b in (64, 32, 16) if k % b == 0), 16)
    grid_block_n = next((gb for gb in (4, 2, 1) if n % (gb * block_n) == 0), 1)
    # With block_m=192 and block_k=128, SMEM = (block_m+block_n)*block_k*mcs.
    # mcs=6 gives 245760 bytes which exceeds H100's 232448-byte per-block limit;
    # drop to mcs=5 (204800 bytes) when block_k=128 is in use.
    max_concurrent_steps = 5 if block_k == 128 else 6
    return dict(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        max_concurrent_steps=max_concurrent_steps,
        grid_block_n=grid_block_n,
    )


@functools.cache
def _dwgrad_config(num_experts: int, k: int, n: int) -> dict:
    """Static Mosaic block config for the ragged FP8 weight-gradient kernel.

    ``block_m`` tiles the output K, ``block_n`` the output N, and ``block_k=128``
    the ragged token (contracting) dim -- 128 measured ~732 vs ~679 TFLOP/s at 64.
    The operand tiles are FP8 (1 byte), so the pipeline SMEM footprint is
    ``(block_m + block_n) * block_k * max_concurrent_steps``; at the d2560 shapes
    (``block_m=block_n=128``) that is ``256*128*6 = 196608 <= 232448``, so
    ``max_concurrent_steps=6`` fits. Larger K/N tiles that would exceed the H100
    per-block ceiling drop to a smaller ``max_concurrent_steps`` (as the forward
    does at ``block_k=128``). The ``next(..., fallback)`` guards keep the config
    valid on shapes whose K/N are not multiples of the preferred tile.
    """
    block_m = next((b for b in (128, 64, 32, 16) if k % b == 0), 16)
    block_n = next((b for b in (128, 64, 32, 16) if n % b == 0), 16)
    grid_block_n = next((gb for gb in (2, 1) if n % (gb * block_n) == 0), 1)
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
    cfg = _autotuned_config(g, m, n, k)
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
    q_lhs_t,  # [K, T] e4m3    -- grad_rhs A (token-contiguous)
    q_rhs,  # [E, K, N] e4m3   -- grad_lhs B (natural layout)
    q_rhs_t,  # [E, N, K] e4m3 -- forward B (transpose_rhs layout)
    lhs_scale,  # [1] delayed-scaling scale for the activation quantize
    rhs_scale,  # [1] delayed-scaling scale for the weight quantize
    grad_scale,  # [1] output-grad scale from the previous step
    grad_amax_history,  # output-grad amax history from the previous step
    lhs,  # [T, K] compute-dtype original (bf16 grad_rhs routing)
    rhs,  # [E, K, N] compute-dtype original (grad dtype routing)
    group_sizes,  # [E]
    rev_dtype,  # static: output-gradient FP8 dtype (E4M3 here)
):
    """FP8 ragged forward of pre-quantized E4M3 operands; FP8 dgrad and wgrad.

    Every FP8 operand layout is precomputed once by the fused cast-transpose+amax
    kernel, so no transpose happens here. The forward contracts ``q_lhs`` against the
    transpose_rhs layout ``q_rhs_t`` (E4M3 x E4M3) and dequantizes by the folded
    ``lhs_scale*rhs_scale``. The backward quantizes the output gradient to
    ``rev_dtype`` (static arg 11; E4M3 here) with delayed scaling and contracts it
    against the pre-produced natural weight layout ``q_rhs`` for grad_lhs and the
    pre-produced transposed activation ``q_lhs_t`` for grad_rhs -- all three GEMMs
    on the FP8 tensor cores. ``grad_scale``/``grad_amax_history`` are differentiable so
    the backward returns the rolled delayed-scaling state as ``OverwriteWithGradient``
    cotangents.
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
    new_g_hist = jnp.roll(grad_amax_history, shift=-1, axis=0).at[0].set(cur_amax[0])

    # grad_lhs[T,K] = g[T,N] . rhs[E,K,N]  (contract N).  FP8 e4m3 x e4m3 on stock
    # jaxlib; consumes the pre-cast natural weight layout q_rhs (no re-cast). The
    # dequant folds both operands' scales: rhs_scale * new_g_scale.
    # Config: reuses _autotuned_config (the forward's block_m=192/block_k=128/mcs=5).
    # At the d2560 operating point (w13, K=N=2560) the dgrad shape -- g[T,N], rhs[E,K,N] --
    # is identical to the forward shape, so _autotuned_config selects the same tuned
    # values.  Validated: no separate _dgrad_config is needed.
    dlhs_scale = (rhs_scale * new_g_scale).astype(jnp.float32)
    grad_lhs = _ragged_fp8(q_g, q_rhs, group_sizes, out_dtype, dlhs_scale)

    # grad_rhs[E,K,N] = lhs[T,K] . g[T,N]  (contracts the ragged token dim).  FP8
    # e4m3 x e4m3 via mgpu_dwgrad, which consumes the pre-cast token-contiguous
    # layouts q_lhs_t [K,T] (no re-cast of the activation) and q_g_t [N,T], emits
    # [E,K,N] directly, and folds the dequant scale lhs_scale * new_g_scale. The
    # f16-upcast boundary mask inside the kernel zeros tokens outside each group,
    # so group boundaries falling mid-block_k-tile are handled correctly.
    t = lhs.shape[0]
    if t % 128 != 0:
        raise ValueError(
            f"T={t} (token count) must be a multiple of 128 for the FP8 weight-gradient kernel: "
            "the token dim is the wgmma contracting dim and its GMEM/TMA tile is 128 tokens wide."
        )
    e, k, n = rhs.shape
    drhs_cfg = _dwgrad_config(e, k, n)
    drhs_scale = (lhs_scale * new_g_scale).astype(jnp.float32)
    grad_rhs = mgpu_dwgrad(
        q_lhs_t,
        q_g_t,
        group_sizes=group_sizes,
        out_dtype=out_dtype,
        out_scale=drhs_scale,
        **drhs_cfg,
    )
    grad_rhs = grad_rhs.astype(rhs.dtype)

    # Cotangents for the 11 differentiable args (q_lhs, q_lhs_t, q_rhs, q_rhs_t,
    # lhs_scale, rhs_scale, grad_scale, grad_amax_history, lhs, rhs, group_sizes):
    # the q's are intermediate (overwrites handled by in_q_ct); lhs_scale/rhs_scale
    # overwrite via in_q_ct; operand grads flow to lhs/rhs; the output-grad scale and
    # amax history are overwritten here with the rolled delayed-scaling state.
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
    quantize_compute_type=jnp.float32,
    fwd_dtype=_E4M3,
    rev_dtype=_E4M3,
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
        rev_dtype: output-gradient FP8 dtype. Defaults to E4M3 so both gradients are
            uniform ``e4m3 x e4m3`` contractions that lower on stock jaxlib.

    Quantizes activations and expert weights to ``fwd_dtype`` with delayed
    per-tensor scaling and contracts them on the FP8 tensor cores via the genuine
    ragged ``wgmma`` kernel, then dequantizes. Both gradients run on the FP8 tensor
    cores: the input gradient against the pre-cast natural weight layout and the
    weight gradient against the pre-cast transposed activation and transposed
    output gradient (gradient quantized to ``rev_dtype`` with delayed scaling). The
    input, kernel, and output-gradient scale / amax-history all update through the
    custom VJPs as ``OverwriteWithGradient`` overwrites.
    """
    del quantize_compute_type  # CT kernel uses float32 internally; parameter kept for API parity.
    # Fused cast-transpose+amax: reads each bf16 tile once, folds the amax reduction, and
    # stores both the natural and the transposed FP8 tile in one pass.  VJP threading matches in_q.
    # The forward uses q_lhs [T,K] and q_rhs_t [E,N,K]; the dgrad uses q_rhs [E,K,N] (natural
    # weight layout); the wgrad uses q_lhs_t [K,T] (transposed activation) -- all cast once here.
    q_lhs, q_lhs_t, new_lhs_scale = in_q_ct(fwd_dtype, "2d", lhs, lhs_scale, lhs_amax_history)
    q_rhs, q_rhs_t, new_rhs_scale = in_q_ct(fwd_dtype, "3d", rhs, rhs_scale, rhs_amax_history)
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
