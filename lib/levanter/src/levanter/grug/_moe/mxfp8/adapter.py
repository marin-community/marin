# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX entry points for the cudnn-frontend fused MXFP8 MoE grouped kernels (SM100 only).

The kernels come from the nvidia-cudnn-frontend package (``cudnn.gemm.cutedsl.grouped``). The
SwiGLU kernel is subclassed in :mod:`levanter.grug._moe.mxfp8.kernels` to gain the forward-only
variant. The backward runs the unified dGLU kernel in its deterministic mode, which writes the
routing-weight gradient into one slot per N tile instead of adding across tiles with float32
atomics.

Four calls cover one gated expert MLP, forward and backward:

- :func:`mxfp8_fused_swiglu`: ``x_q @ w13`` followed in the epilogue by SwiGLU and MXFP8
  quantization of the hidden activation in both orientations. :func:`mxfp8_fused_swiglu_forward`
  is the same kernel storing only what the next forward GEMM reads, for a forward pass whose
  backward recomputes it.
- :func:`mxfp8_fused_gemm`: a plain grouped GEMM with a bf16 epilogue (forward w2, dgrad w13).
- :func:`mxfp8_fused_dswiglu`: ``g_q @ w2^T`` followed by the SwiGLU backward against the saved
  pre-activation, MXFP8 quantization of its result in both orientations, and the per-row routing
  weight gradient.
- :func:`mxfp8_fused_wgrad`: the grouped weight gradient from two token-axis quantized operands.

Row layout: tokens are stored contiguously per expert, and every expert's row count is padded to
a multiple of :data:`GROUP_PAD`. ``offs`` holds the int32 END offset of each expert in that padded
buffer, so ``offs[-1]`` equals the padded row count. Padding rows must be finite (zeros), since a
non-finite row poisons the 32-token column scales it shares.

Gate/up contract: the fused SwiGLU pairs gate and up columns in alternating 32-column blocks of the
``2 * I`` axis (even blocks gate, odd blocks up). ``w13`` must be laid out that way before it is
quantized, and the pre-activation ``c13`` and its gradient come back in the same order.

Scale-factor layouts, all float8_e8m0fnu bytes:

- row orientation (scales along features): the whole ``(rows, features / 32)`` scale matrix in the
  SM100 32x4x4 swizzle, flat. Because each expert spans whole 128-row blocks this equals the
  per-expert swizzle concatenation.
- column orientation (scales along tokens): per-expert swizzles of each ``(features,
  tokens_e / 32)`` scale matrix, concatenated in expert order. This is what :func:`mxfp8_fused_wgrad`
  consumes.
- weights: one swizzled ``(N, K / 32)`` block per expert, stacked.

The routing-weight knobs of the kernels (alpha, beta, prob, norm_const) are pinned to 1.

Launchers are memoized through :func:`levanter.cutlass_kernel_cache.cute_launcher_factory`, so
shape-identical calls share one CuTeDSL compile and the persistent kernel cache can serve them.
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
import jax
import jax.numpy as jnp
from cudnn.gemm.cutedsl.grouped.dglu.moe_blockscaled_grouped_gemm_dglu_dbias import (
    BlockScaledMoEGroupedGemmDgluDbiasKernel,
)
from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode, WGradInputOrder, WgradSfTensormapConstructor
from cudnn.gemm.cutedsl.grouped.quant.grouped_gemm_quant import BlockScaledMoEGroupedGemmQuantKernel
from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad import BlockScaledMoEGroupedGemmWgradKernel
from quack.cute_dsl_utils import get_max_active_clusters

from levanter.cutlass_kernel_cache import cute_launcher_factory, cutlass_call, gpu_compute_capability
from levanter.grug._moe.mxfp8.kernels.swiglu_forward_only import SwigluKernel
from levanter.grug._moe.mxfp8.layout import _GATE_UP_BLOCK, GROUP_PAD
from levanter.grug._moe.mxfp8.quantize import MXFP8_BLOCK_SIZE, SF_TILE_ROWS

# (mma_tiler_mn, cluster_shape_mn) per kernel: the winners of the research ablations in
# c3cb334f8e:experiments/grug/moe/mxfp8.py. A sweep of tile and
# cluster shapes at the hero expert chunk (H = I = 3072, 3 experts, 302,336 rows) on one B200
# found no other configuration more than 3% faster for any kernel.
_SWIGLU_TILING = ((256, 256), (2, 1))
_DSWIGLU_TILING = ((256, 256), (2, 1))
_GEMM_TILING = ((256, 256), (2, 1))
_WGRAD_TILING = ((256, 256), (2, 2))
# The scalar dSwiGLU epilogue takes 3.98 ms at that chunk against 4.52 ms for the packed-f32 one.
# Its rounding differs, so dc13 bytes differ from the packed epilogue's.
_DSWIGLU_VECTORIZED_F32 = False
# An MMA tile this many rows tall runs as a 2-CTA instruction over a pair of SMs.
_TWO_CTA_TILE_M = 256

# The wgrad operands: both token-major 2D tensors, one dense (E, K, N) output.
_WGRAD_INPUT_ORDER = WGradInputOrder.Tensor2D
_WGRAD_WEIGHT_MODE = MoEWeightMode.DENSE
# The dense, statically scheduled quant GEMM and dGLU kernels ignore their workspace but still take
# a pointer.
_UNUSED_WORKSPACE_BYTES = 16
# The dGLU kernel adds each routing-gradient partial into its slot, so the slots go in zeroed as the
# last input and come back as the last output in the same buffer.
_DPROB_SLOTS_INPUT = 10
_DPROB_SLOTS_OUTPUT = 5

_SPEC = cjax.TensorSpec()
# A row-major (L, M, K) buffer presented as the kernels' logical (M, K, L), with L the batch or
# expert mode. Token operands are passed as (1, M, K) so they carry the same trailing unit mode.
_BATCH_LAST = cjax.TensorSpec(mode=(1, 2, 0))
# A row-major (T, K) buffer presented as (K, T): the wgrad kernel's M-major A operand.
_TRANSPOSED = cjax.TensorSpec(mode=(1, 0))


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _round_up(a: int, b: int) -> int:
    return _ceil_div(a, b) * b


def _sf_row_bytes(rows: int, features: int) -> int:
    """Byte count of a row-orientation swizzled scale tensor for a ``(rows, features)`` operand."""
    return _round_up(rows, SF_TILE_ROWS) * _round_up(_ceil_div(features, MXFP8_BLOCK_SIZE), 4)


def _sf_col_bytes(rows: int, features: int) -> int:
    """Byte count of a column-orientation (per-expert, token-axis) scale tensor for ``(rows, features)``."""
    return _round_up(features, SF_TILE_ROWS) * _round_up(_ceil_div(rows, MXFP8_BLOCK_SIZE), 4)


def _dprob_slot_count(intermediate: int) -> int:
    """Number of routing-gradient slots per row: one per N tile the dGLU scheduler can emit.

    The scheduler counts whole clusters along N, so the count rounds up to a multiple of the
    cluster's N extent (upstream's ``dsrelu.api._dprob_n_slots``).
    """
    (_tile_m, tile_n), (_cluster_m, cluster_n) = _DSWIGLU_TILING
    return _ceil_div(intermediate, tile_n * cluster_n) * cluster_n


def _require_sm100() -> None:
    capability = gpu_compute_capability()
    if capability // 10 != 10:
        raise RuntimeError(f"The fused MXFP8 kernels require an SM100 GPU, got compute capability {capability}")


def _max_active_clusters(cluster_shape_mn: tuple[int, int]) -> int:
    return get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1])


def _check_scales(*scales: jax.Array) -> None:
    for scale in scales:
        if scale.dtype != jnp.float8_e8m0fnu:
            raise ValueError(f"scale factors must be float8_e8m0fnu, got {scale.dtype}")


def _check_quantized(*operands: jax.Array) -> None:
    for operand in operands:
        if operand.dtype != jnp.float8_e4m3fn:
            raise ValueError(f"quantized operands must be float8_e4m3fn, got {operand.dtype}")


def _check_padded_rows(rows: int, offs: jax.Array, experts: int) -> None:
    if rows % GROUP_PAD:
        raise ValueError(f"the padded row count {rows} must be a multiple of {GROUP_PAD}")
    if offs.shape != (experts,):
        raise ValueError(f"offs must have shape ({experts},), got {offs.shape}")


def _as_batch_last(t: jax.Array) -> jax.Array:
    return t.reshape(1, *t.shape)


def _unit_vector(n: int) -> jax.Array:
    return jnp.ones((n,), jnp.float32)


class SwigluOutputs(NamedTuple):
    """:func:`mxfp8_fused_swiglu`'s outputs for ``M`` rows and intermediate size ``I``."""

    c13: jax.Array
    """(M, 2I) bf16 pre-activation, gate and up interleaved in 32-column blocks."""
    h: jax.Array
    """(M, I) e4m3 SwiGLU output quantized along features, the down GEMM's operand."""
    h_col: jax.Array
    """(M, I) e4m3 SwiGLU output quantized along tokens, the down weight gradient's operand."""
    h_scales: jax.Array
    """Flat row-orientation scales of ``h``."""
    h_col_scales: jax.Array
    """Flat column-orientation scales of ``h_col``."""


class DSwigluOutputs(NamedTuple):
    """:func:`mxfp8_fused_dswiglu`'s outputs for ``M`` rows and intermediate size ``I``."""

    dc13: jax.Array
    """(M, 2I) e4m3 gradient of the interleaved pre-activation, quantized along features."""
    dc13_col: jax.Array
    """(M, 2I) e4m3 gradient of the interleaved pre-activation, quantized along tokens."""
    dc13_scales: jax.Array
    """Flat row-orientation scales of ``dc13``."""
    dc13_col_scales: jax.Array
    """Flat column-orientation scales of ``dc13_col``."""
    row_dot: jax.Array
    """(M,) float32 routing-weight gradient ``<swiglu(c13), dh>`` with ``dh = g @ w2^T``."""


# --------------------------------------------------------------------------- #
# fwd w13: grouped GEMM + SwiGLU + dual-orientation MXFP8 quantize
# --------------------------------------------------------------------------- #


@cute_launcher_factory
def _swiglu_launcher(*, expert_count, mma_tiler_mn, cluster_shape_mn, max_active_clusters):
    """Signature: (stream, x, w13, sfx, sfw13, offs, alpha, prob, norm, c13, h, h_col, sfh_row, sfh_col)."""

    @cute.jit
    def launcher(stream, mA, mB, mSFA, mSFB, mOffs, mAlpha, mProb, mNorm, mC, mD, mDCol, mSFDRow, mSFDCol):
        kernel = SwigluKernel(
            sf_vec_size=MXFP8_BLOCK_SIZE,
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=mma_tiler_mn[0] == _TWO_CTA_TILE_M,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            vector_f32=False,
            generate_sfd=True,
            discrete_col_sfd=True,
            expert_cnt=expert_count,
            forward_only=False,
        )
        kernel(
            mA,
            mB,
            mC,
            mD,
            mDCol,
            mSFA,
            mSFB,
            mSFDRow,
            mSFDCol,
            None,  # amax: only produced on the bf16-output path
            mNorm,
            mOffs,
            mAlpha,
            mProb,
            max_active_clusters,
            stream,
        )

    return launcher


@cute_launcher_factory
def _swiglu_forward_launcher(*, expert_count, mma_tiler_mn, cluster_shape_mn, max_active_clusters):
    """Signature: (stream, x, w13, sfx, sfw13, offs, alpha, prob, norm, h, sfh_row)."""

    @cute.jit
    def launcher(stream, mA, mB, mSFA, mSFB, mOffs, mAlpha, mProb, mNorm, mD, mSFDRow):
        kernel = SwigluKernel(
            sf_vec_size=MXFP8_BLOCK_SIZE,
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=mma_tiler_mn[0] == _TWO_CTA_TILE_M,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            vector_f32=False,
            generate_sfd=True,
            discrete_col_sfd=False,
            expert_cnt=expert_count,
            forward_only=True,
        )
        kernel(
            mA,
            mB,
            None,  # c: not stored
            mD,
            None,  # d_col: not stored
            mSFA,
            mSFB,
            mSFDRow,
            None,  # sfd_col: not stored
            None,  # amax: only produced on the bf16-output path
            mNorm,
            mOffs,
            mAlpha,
            mProb,
            max_active_clusters,
            stream,
        )

    return launcher


_SWIGLU_INPUT_SPEC = (_BATCH_LAST, _BATCH_LAST, _SPEC, _SPEC, _SPEC, _SPEC, _SPEC, _SPEC)


def _swiglu_problem(x_q, x_sf, w13i_q, w13i_sf, offs) -> tuple[int, int, int]:
    """Check the fused SwiGLU operands and return ``(rows, experts, intermediate)``."""
    _require_sm100()
    m, hidden = x_q.shape
    experts, n2, k = w13i_q.shape
    if k != hidden:
        raise ValueError(f"contraction mismatch: x_q {x_q.shape}, w13i_q {w13i_q.shape}")
    if n2 % (2 * _GATE_UP_BLOCK):
        raise ValueError(f"2I={n2} must be a multiple of {2 * _GATE_UP_BLOCK} (32-column gate/up block pairs)")
    _check_padded_rows(m, offs, experts)
    _check_quantized(x_q, w13i_q)
    _check_scales(x_sf, w13i_sf)
    return m, experts, n2 // 2


def _swiglu_inputs(x_q, x_sf, w13i_q, w13i_sf, offs, *, rows: int, experts: int) -> tuple[jax.Array, ...]:
    """The fused SwiGLU kernels' inputs in ``_SWIGLU_INPUT_SPEC`` order, routing-weight knobs at 1."""
    return (
        _as_batch_last(x_q),
        w13i_q,
        x_sf.reshape(-1),
        w13i_sf.reshape(-1),
        offs.astype(jnp.int32),
        _unit_vector(experts),
        jnp.ones((rows, 1, 1), jnp.float32),
        _unit_vector(1),
    )


def _swiglu_launcher_config(experts: int) -> dict:
    mma_tiler_mn, cluster_shape_mn = _SWIGLU_TILING
    return {
        "expert_count": experts,
        "mma_tiler_mn": mma_tiler_mn,
        "cluster_shape_mn": cluster_shape_mn,
        "max_active_clusters": _max_active_clusters(cluster_shape_mn),
    }


def mxfp8_fused_swiglu(x_q, x_sf, w13i_q, w13i_sf, offs) -> SwigluOutputs:
    """Fused forward w13: ``swiglu(x @ w13)`` with a dual-orientation MXFP8 epilogue.

    Args:
        x_q: (M, H) float8_e4m3fn tokens, experts contiguous with ``GROUP_PAD``-padded rows.
        x_sf: row-orientation swizzled scales of ``x_q``, any shape with ``_sf_row_bytes(M, H)`` bytes.
        w13i_q: (E, 2I, H) float8_e4m3fn, gate/up interleaved in 32-column blocks along ``2I``,
            quantized along ``H``.
        w13i_sf: per-expert swizzled scales of ``w13i_q``.
        offs: (E,) int32 padded END offsets, each a multiple of ``GROUP_PAD``.

    Returns:
        The interleaved pre-activation, both quantized copies of the SwiGLU output, and their
        scales; see :class:`SwigluOutputs`.
    """
    m, experts, intermediate = _swiglu_problem(x_q, x_sf, w13i_q, w13i_sf, offs)
    call = cutlass_call(
        _swiglu_launcher(**_swiglu_launcher_config(experts)),
        output_shape_dtype=(
            jax.ShapeDtypeStruct((1, m, 2 * intermediate), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, m, intermediate), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, m, intermediate), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((_sf_row_bytes(m, intermediate),), jnp.float8_e8m0fnu),
            jax.ShapeDtypeStruct((_sf_col_bytes(m, intermediate),), jnp.float8_e8m0fnu),
        ),
        input_spec=_SWIGLU_INPUT_SPEC,
        output_spec=(_BATCH_LAST, _BATCH_LAST, _BATCH_LAST, _SPEC, _SPEC),
        use_static_tensors=True,
    )
    c13, h, h_col, sfh_row, sfh_col = call(*_swiglu_inputs(x_q, x_sf, w13i_q, w13i_sf, offs, rows=m, experts=experts))
    return SwigluOutputs(c13[0], h[0], h_col[0], sfh_row, sfh_col)


def mxfp8_fused_swiglu_forward(x_q, x_sf, w13i_q, w13i_sf, offs):
    """:func:`mxfp8_fused_swiglu`'s ``(h, h_scales)``, the down GEMM's operand, bit for bit.

    The same kernel skips the pre-activation and the token-quantized copy, which only the backward
    reads, so a forward pass whose backward recomputes it writes neither. Arguments as for
    :func:`mxfp8_fused_swiglu`.
    """
    m, experts, intermediate = _swiglu_problem(x_q, x_sf, w13i_q, w13i_sf, offs)
    call = cutlass_call(
        _swiglu_forward_launcher(**_swiglu_launcher_config(experts)),
        output_shape_dtype=(
            jax.ShapeDtypeStruct((1, m, intermediate), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((_sf_row_bytes(m, intermediate),), jnp.float8_e8m0fnu),
        ),
        input_spec=_SWIGLU_INPUT_SPEC,
        output_spec=(_BATCH_LAST, _SPEC),
        use_static_tensors=True,
    )
    h, sfh_row = call(*_swiglu_inputs(x_q, x_sf, w13i_q, w13i_sf, offs, rows=m, experts=experts))
    return h[0], sfh_row


# --------------------------------------------------------------------------- #
# plain grouped GEMM (bf16 out): fwd w2 and dgrad w13
# --------------------------------------------------------------------------- #


@cute_launcher_factory
def _gemm_launcher(*, expert_count, mma_tiler_mn, cluster_shape_mn, max_active_clusters):
    """Signature: (stream, a, b, sfa, sfb, offs, alpha, prob, out, workspace)."""

    @cute.jit
    def launcher(stream, mA, mB, mSFA, mSFB, mOffs, mAlpha, mProb, mD, mWorkspace):
        kernel = BlockScaledMoEGroupedGemmQuantKernel(
            sf_vec_size=MXFP8_BLOCK_SIZE,
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=mma_tiler_mn[0] == _TWO_CTA_TILE_M,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            vectorized_f32=False,
            generate_sfd=False,
            discrete_col_sfd=False,
            enable_bias=False,
            expert_cnt=expert_count,
            weight_mode=MoEWeightMode.DENSE,
            use_dynamic_sched=False,
        )
        kernel(
            mA,
            mB,
            mSFB,
            cutlass.Int32(0),  # n: discrete weight mode only
            cutlass.Int32(0),  # k: discrete weight mode only
            cutlass.Int64(0),  # b_stride_size: discrete weight mode only
            cute.nvgpu.OperandMajorMode.K,  # b_major_mode: discrete weight mode only
            mWorkspace.iterator,
            mD,
            mD,  # d_col: unused without generate_sfd, but its TMA atom is built unconditionally
            mSFA,
            None,  # sfd_row
            None,  # sfd_col
            None,  # amax
            None,  # norm_const
            mOffs,
            mAlpha,
            None,  # row_scale
            None,  # bias
            mProb,
            max_active_clusters,
            stream,
        )

    return launcher


def mxfp8_fused_gemm(a_q, a_sf, b_q, b_sf, offs):
    """Grouped GEMM ``a[M, K] @ w[E, K, N] -> [M, N]`` in bf16, without a quantizing epilogue.

    Args:
        a_q: (M, K) float8_e4m3fn, experts contiguous with ``GROUP_PAD``-padded rows.
        a_sf: row-orientation swizzled scales of ``a_q``.
        b_q: (E, N, K) float8_e4m3fn, the transposed weight buffer quantized along ``K``.
        b_sf: per-expert swizzled scales of ``b_q``.
        offs: (E,) int32 padded END offsets, each a multiple of ``GROUP_PAD``.
    """
    _require_sm100()
    m, k = a_q.shape
    experts, n, kb = b_q.shape
    if kb != k:
        raise ValueError(f"contraction mismatch: a_q {a_q.shape}, b_q {b_q.shape}")
    _check_padded_rows(m, offs, experts)
    _check_quantized(a_q, b_q)
    _check_scales(a_sf, b_sf)
    mma_tiler_mn, cluster_shape_mn = _GEMM_TILING
    launcher = _gemm_launcher(
        expert_count=experts,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        max_active_clusters=_max_active_clusters(cluster_shape_mn),
    )
    call = cutlass_call(
        launcher,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((1, m, n), jnp.bfloat16),
            jax.ShapeDtypeStruct((_UNUSED_WORKSPACE_BYTES,), jnp.uint8),
        ),
        input_spec=(_BATCH_LAST, _BATCH_LAST, _SPEC, _SPEC, _SPEC, _SPEC, _SPEC),
        output_spec=(_BATCH_LAST, _SPEC),
        use_static_tensors=True,
    )
    out, _workspace = call(
        _as_batch_last(a_q),
        b_q,
        a_sf.reshape(-1),
        b_sf.reshape(-1),
        offs.astype(jnp.int32),
        _unit_vector(experts),
        jnp.ones((m, 1, 1), jnp.float32),
    )
    return out[0]


# --------------------------------------------------------------------------- #
# bwd: dgrad-w2 GEMM + dSwiGLU + dual-orientation MXFP8 quantize + routing gradient
# --------------------------------------------------------------------------- #


@cute_launcher_factory
def _dswiglu_launcher(*, expert_count, mma_tiler_mn, cluster_shape_mn, vectorized_f32, max_active_clusters):
    """Signature: (stream, g, w2, c13, sfg, sfw2, offs, alpha, beta, prob, norm, dprob_slots, dc, dc_col,
    sfdc_row, sfdc_col, workspace); ``dprob_slots`` is the zeroed input the last output aliases."""

    @cute.jit
    def launcher(
        stream,
        mA,
        mB,
        mC,
        mSFA,
        mSFB,
        mOffs,
        mAlpha,
        mBeta,
        mProb,
        mNorm,
        mDProbSlots,
        mD,
        mDCol,
        mSFDRow,
        mSFDCol,
        mWorkspace,
    ):
        kernel = BlockScaledMoEGroupedGemmDgluDbiasKernel(
            sf_vec_size=MXFP8_BLOCK_SIZE,
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=mma_tiler_mn[0] == _TWO_CTA_TILE_M,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            vectorized_f32=vectorized_f32,
            discrete_col_sfd=True,
            expert_cnt=expert_count,
            weight_mode=MoEWeightMode.DENSE,
            use_dynamic_sched=False,
            act_func="dswiglu",
            deterministic=True,
        )
        kernel(
            mA,
            mB,
            mSFB,
            cutlass.Int32(0),  # n: discrete weight mode only
            cutlass.Int32(0),  # k: discrete weight mode only
            cutlass.Int64(0),  # b_stride_size: discrete weight mode only
            cute.nvgpu.OperandMajorMode.K,  # b_major_mode: discrete weight mode only
            mWorkspace.iterator,
            mC,
            mD,
            mDCol,
            mSFA,
            mSFDRow,
            mSFDCol,
            None,  # amax: only produced on the bf16-output path
            mNorm,
            mOffs,
            mAlpha,
            mBeta,
            mProb,
            mDProbSlots,
            None,  # dbias: the expert MLP has no bias
            max_active_clusters,
            stream,
        )

    return launcher


def mxfp8_fused_dswiglu(g_q, g_sf, w2_q, w2_sf, c13, offs) -> DSwigluOutputs:
    """Fused backward: ``dc13 = dSwiGLU(g @ w2^T, c13)`` with a dual MXFP8 epilogue.

    Args:
        g_q: (M, H) float8_e4m3fn output cotangent, quantized along ``H``.
        g_sf: row-orientation swizzled scales of ``g_q``.
        w2_q: (E, I, H) float8_e4m3fn, ``w2`` in its natural buffer quantized along ``H``.
        w2_sf: per-expert swizzled scales of ``w2_q``.
        c13: (M, 2I) bf16 interleaved pre-activation from :func:`mxfp8_fused_swiglu`.
        offs: (E,) int32 padded END offsets, each a multiple of ``GROUP_PAD``.

    Returns:
        Both quantized copies of the interleaved pre-activation's gradient, their scales, and each
        row's routing-weight gradient; see :class:`DSwigluOutputs`. The kernel adds each N tile's
        routing-gradient partial into that tile's zeroed slot, the only write to the slot, and the
        slots are summed here, so the row dot is deterministic.
    """
    _require_sm100()
    m, hidden = g_q.shape
    experts, intermediate, k = w2_q.shape
    if k != hidden:
        raise ValueError(f"contraction mismatch: g_q {g_q.shape}, w2_q {w2_q.shape}")
    n2 = 2 * intermediate
    if c13.shape != (m, n2) or c13.dtype != jnp.bfloat16:
        raise ValueError(f"c13 must be bf16 of shape ({m}, {n2}), got {c13.dtype} {c13.shape}")
    _check_padded_rows(m, offs, experts)
    _check_quantized(g_q, w2_q)
    _check_scales(g_sf, w2_sf)
    mma_tiler_mn, cluster_shape_mn = _DSWIGLU_TILING
    slots = _dprob_slot_count(intermediate)
    launcher = _dswiglu_launcher(
        expert_count=experts,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        vectorized_f32=_DSWIGLU_VECTORIZED_F32,
        max_active_clusters=_max_active_clusters(cluster_shape_mn),
    )
    call = cutlass_call(
        launcher,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((1, m, n2), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, m, n2), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((_sf_row_bytes(m, n2),), jnp.float8_e8m0fnu),
            jax.ShapeDtypeStruct((_sf_col_bytes(m, n2),), jnp.float8_e8m0fnu),
            jax.ShapeDtypeStruct((_UNUSED_WORKSPACE_BYTES,), jnp.uint8),
            jax.ShapeDtypeStruct((m, slots, 1), jnp.float32),
        ),
        input_spec=(_BATCH_LAST, _BATCH_LAST, _BATCH_LAST, _SPEC, _SPEC, _SPEC, _SPEC, _SPEC, _SPEC, _SPEC, _SPEC),
        output_spec=(_BATCH_LAST, _BATCH_LAST, _SPEC, _SPEC, _SPEC, _SPEC),
        input_output_aliases=((_DPROB_SLOTS_INPUT, _DPROB_SLOTS_OUTPUT),),
        use_static_tensors=True,
    )
    dc, dc_col, sfdc_row, sfdc_col, _workspace, dprob_slots = call(
        _as_batch_last(g_q),
        w2_q,
        _as_batch_last(c13),
        g_sf.reshape(-1),
        w2_sf.reshape(-1),
        offs.astype(jnp.int32),
        _unit_vector(experts),
        _unit_vector(experts),
        jnp.ones((m, 1, 1), jnp.float32),
        _unit_vector(1),
        jnp.zeros((m, slots, 1), jnp.float32),
    )
    return DSwigluOutputs(dc[0], dc_col[0], sfdc_row, sfdc_col, jnp.sum(dprob_slots[:, :, 0], axis=-1))


# --------------------------------------------------------------------------- #
# 2Dx2D wgrad (online scale-factor tensormaps)
# --------------------------------------------------------------------------- #


@cute_launcher_factory
def _wgrad_launcher(*, expert_count, mma_tiler_mn, cluster_shape_mn, max_active_clusters):
    """Signature: (stream, x, g, sfx, sfg, offs, dw, workspace)."""

    @cute.jit
    def launcher(stream, mA, mB, mSFA, mSFB, mOffs, mOut, mWorkspace):
        kernel = BlockScaledMoEGroupedGemmWgradKernel(
            sf_vec_size=MXFP8_BLOCK_SIZE,
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=mma_tiler_mn[0] == _TWO_CTA_TILE_M,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            accumulate_on_output=False,
            expert_cnt=expert_count,
            weight_mode=_WGRAD_WEIGHT_MODE,
            input_order=_WGRAD_INPUT_ORDER,
        )
        kernel(mA, mB, mSFA, mSFB, mOut, mOffs, mWorkspace, max_active_clusters, stream)

    return launcher


def mxfp8_fused_wgrad(x_q, x_sf, g_q, g_sf, offs):
    """Grouped weight gradient ``dw[e] = x[rows_e]^T @ g[rows_e]`` -> (E, K, N) bf16.

    Args:
        x_q: (T, K) float8_e4m3fn in natural token-major storage, quantized along tokens (a
            ``*_col`` output of the fused kernels or of a dual-orientation quantizer).
        x_sf: (round_up(K, 128), padded_token_columns) column-orientation scales of ``x_q``.
        g_q: (T, N) float8_e4m3fn in natural storage, quantized along tokens.
        g_sf: (round_up(N, 128), padded_token_columns) column-orientation scales of ``g_q``.
        offs: (E,) int32 cumulative END offsets of the expert row groups (``offs[-1] == T``).
            An expert with no rows gets an exactly zero gradient.
    """
    _require_sm100()
    t, k = x_q.shape
    tg, n = g_q.shape
    if t != tg:
        raise ValueError(f"token mismatch: x_q {x_q.shape}, g_q {g_q.shape}")
    if offs.ndim != 1:
        raise ValueError(f"offs must be rank 1, got {offs.shape}")
    (experts,) = offs.shape
    _check_quantized(x_q, g_q)
    _check_scales(x_sf, g_sf)
    if x_sf.shape[0] != _round_up(k, SF_TILE_ROWS) or g_sf.shape[0] != _round_up(n, SF_TILE_ROWS):
        raise ValueError(
            f"scale rows must be the {SF_TILE_ROWS}-padded feature counts, got {x_sf.shape} and {g_sf.shape}"
        )
    if x_sf.shape[1] != g_sf.shape[1]:
        raise ValueError(f"x_sf and g_sf must cover the same token columns, got {x_sf.shape} and {g_sf.shape}")
    mma_tiler_mn, cluster_shape_mn = _WGRAD_TILING
    launcher = _wgrad_launcher(
        expert_count=experts,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        max_active_clusters=_max_active_clusters(cluster_shape_mn),
    )
    # The kernel's helper launch writes one online TMA descriptor per expert for each of A, B, SFA
    # and SFB here: the size BlockScaledMoEGroupedGemmWgradKernel.get_workspace_bytes reports.
    workspace_bytes = WgradSfTensormapConstructor.get_workspace_size(_WGRAD_INPUT_ORDER, _WGRAD_WEIGHT_MODE, experts)
    call = cutlass_call(
        launcher,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((experts, k, n), jnp.bfloat16),
            jax.ShapeDtypeStruct((workspace_bytes,), jnp.uint8),
        ),
        input_spec=(_TRANSPOSED, _SPEC, _SPEC, _SPEC, _SPEC),
        output_spec=(_SPEC, _SPEC),
        use_static_tensors=True,
    )
    dw, _workspace = call(x_q, g_q, x_sf, g_sf, offs.astype(jnp.int32))
    return dw
