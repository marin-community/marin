# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX entry points for the cudnn-frontend fused MXFP8 ReLU^2 grouped GEMMs (SM100 only).

Two calls carry the ReLU^2 of an ungated expert MLP, ``y = relu(x @ w_up)^2 @ w_down``:

- :func:`mxfp8_fused_srelu`: ``x_q @ w_up`` followed in the epilogue by ReLU^2 and the row MXFP8
  quantization of the hidden activation ``h``, storing the bf16 pre-activation for the backward.
- :func:`mxfp8_fused_dsrelu`: ``g_q @ w_down^T`` followed by the ReLU^2 backward against the saved
  pre-activation, the MXFP8 quantization of its result in both orientations, ``h`` regenerated
  from the pre-activation and quantized along tokens, and each row's ``<h, dh>``.

Row layout, scale layouts and launcher memoization follow :mod:`.adapter`. The routing-weight
knobs of the kernels (alpha, prob, norm_const) are pinned to 1. The backward runs in the kernel's
deterministic mode, which writes the routing-weight gradient into one slot per N tile instead of
adding across tiles with float32 atomics.
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import jax
import jax.numpy as jnp
from cudnn.gemm.cutedsl.grouped.dsrelu.moe_blockscaled_grouped_gemm_dsrelu_quant import (
    BlockScaledMoEGroupedGemmQuantBwdKernel,
)
from cudnn.gemm.cutedsl.grouped.dsrelu.moe_blockscaled_grouped_gemm_dsrelu_quant import (
    EpilogueType as DsreluEpilogueType,
)
from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode
from cudnn.gemm.cutedsl.grouped.srelu.moe_blockscaled_grouped_gemm_srelu_quant import (
    BlockScaledMoEGroupedGemmQuantKernel as SreluKernel,
)
from cudnn.gemm.cutedsl.grouped.srelu.moe_blockscaled_grouped_gemm_srelu_quant import (
    EpilogueType as SreluEpilogueType,
)

from levanter.cutlass_kernel_cache import cute_launcher_factory, cutlass_call
from levanter.grug._moe.mxfp8.adapter import (
    _BATCH_LAST,
    _SPEC,
    _TWO_CTA_TILE_M,
    _UNUSED_WORKSPACE_BYTES,
    _WEIGHT_SPECS,
    _as_batch_last,
    _check_padded_rows,
    _check_quantized,
    _check_scales,
    _dprob_slot_count,
    _max_active_clusters,
    _require_sm100,
    _sf_col_bytes,
    _sf_row_bytes,
    _unit_vector,
)
from levanter.grug._moe.mxfp8.quantize import MXFP8_BLOCK_SIZE, WeightMajor

# (mma_tiler_mn, cluster_shape_mn): the plain GEMM's. The fused ReLU^2 kernels take only a 256-wide
# MMA N tile, and the 256-row MMA tile runs as a 2-CTA instruction. Not tuned for these kernels.
_SRELU_TILING = ((256, 256), (2, 1))
_DSRELU_TILING = ((256, 256), (2, 1))
# The dsrelu kernel adds each routing-gradient partial into its slot, so the slots go in zeroed as
# the last input and come back as the last output in the same buffer.
_DPROB_SLOTS_INPUT = 9
_DPROB_SLOTS_OUTPUT = 7


class SreluOutputs(NamedTuple):
    """:func:`mxfp8_fused_srelu`'s outputs for ``M`` rows and intermediate size ``I``."""

    c: jax.Array
    """(M, I) bf16 pre-activation ``x @ w_up``."""
    h: jax.Array
    """(M, I) e4m3 ``relu(c)^2`` quantized along features, the down GEMM's operand."""
    h_scales: jax.Array
    """Flat row-orientation scales of ``h``."""


class DSreluOutputs(NamedTuple):
    """:func:`mxfp8_fused_dsrelu`'s outputs for ``M`` rows and intermediate size ``I``."""

    dc: jax.Array
    """(M, I) e4m3 gradient of the pre-activation, quantized along features."""
    dc_scales: jax.Array
    """Flat row-orientation scales of ``dc``."""
    dc_col: jax.Array
    """(M, I) e4m3 gradient of the pre-activation, quantized along tokens."""
    dc_col_scales: jax.Array
    """Flat column-orientation scales of ``dc_col``."""
    h_col: jax.Array
    """(M, I) e4m3 ``relu(c)^2`` quantized along tokens, the down weight gradient's operand."""
    h_col_scales: jax.Array
    """Flat column-orientation scales of ``h_col``."""
    row_dot: jax.Array
    """(M,) float32 routing-weight gradient ``<relu(c)^2, dh>`` with ``dh = g @ w_down^T``."""


# --------------------------------------------------------------------------- #
# fwd up projection: grouped GEMM + ReLU^2 + MXFP8 quantize
# --------------------------------------------------------------------------- #


@cute_launcher_factory
def _srelu_launcher(*, expert_count, mma_tiler_mn, cluster_shape_mn, max_active_clusters):
    """Signature: (stream, x, w_up, sfx, sfw_up, offs, alpha, prob, norm, c, h, h_col, sfh_row, sfh_col, workspace)."""

    @cute.jit
    def launcher(stream, mA, mB, mSFA, mSFB, mOffs, mAlpha, mProb, mNorm, mC, mD, mDCol, mSFDRow, mSFDCol, mWorkspace):
        kernel = SreluKernel(
            sf_vec_size=MXFP8_BLOCK_SIZE,
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=mma_tiler_mn[0] == _TWO_CTA_TILE_M,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            vectorized_f32=False,
            generate_sfd=True,
            discrete_col_sfd=True,
            generate_c=True,
            enable_bias=False,
            expert_cnt=expert_count,
            weight_mode=MoEWeightMode.DENSE,
            use_dynamic_sched=False,
            epilogue_type=SreluEpilogueType.SRELU.value,
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
            None,  # bias: the expert MLP has no bias
            mProb,
            max_active_clusters,
            stream,
        )

    return launcher


def mxfp8_fused_srelu(x_q, x_sf, w_up_q, w_up_sf, offs) -> SreluOutputs:
    """Fused forward up projection: ``relu(x @ w_up)^2`` with a row MXFP8 epilogue.

    The kernel writes ``relu(c)^2`` quantized along tokens together with the row copy; that copy
    goes to a scratch output dropped here, since the backward regenerates it from ``c``.

    Args:
        x_q: (M, H) float8_e4m3fn tokens, experts contiguous with ``GROUP_PAD``-padded rows.
        x_sf: row-orientation swizzled scales of ``x_q``.
        w_up_q: (E, H, I) float8_e4m3fn, ``w_up`` in its stored layout quantized along ``H``.
        w_up_sf: per-expert swizzled scales of ``w_up_q`` over ``(I, H / 32)``.
        offs: (E,) int32 padded END offsets, each a multiple of ``GROUP_PAD``.
    """
    _require_sm100()
    m, hidden = x_q.shape
    experts, k, intermediate = w_up_q.shape
    if k != hidden:
        raise ValueError(f"contraction mismatch: x_q {x_q.shape}, w_up_q {w_up_q.shape}")
    _check_padded_rows(m, offs, experts)
    _check_quantized(x_q, w_up_q)
    _check_scales(x_sf, w_up_sf)
    mma_tiler_mn, cluster_shape_mn = _SRELU_TILING
    launcher = _srelu_launcher(
        expert_count=experts,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        max_active_clusters=_max_active_clusters(cluster_shape_mn),
    )
    call = cutlass_call(
        launcher,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((1, m, intermediate), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, m, intermediate), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, m, intermediate), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((_sf_row_bytes(m, intermediate),), jnp.float8_e8m0fnu),
            jax.ShapeDtypeStruct((_sf_col_bytes(m, intermediate),), jnp.float8_e8m0fnu),
            jax.ShapeDtypeStruct((_UNUSED_WORKSPACE_BYTES,), jnp.uint8),
        ),
        # w_up is stored [E, H, I], (K, N) per expert, and read N-major in place.
        input_spec=(_BATCH_LAST, _WEIGHT_SPECS[WeightMajor.N], _SPEC, _SPEC, _SPEC, _SPEC, _SPEC, _SPEC),
        output_spec=(_BATCH_LAST, _BATCH_LAST, _BATCH_LAST, _SPEC, _SPEC, _SPEC),
        use_static_tensors=True,
    )
    c, h, _h_col, sfh_row, _sfh_col, _workspace = call(
        _as_batch_last(x_q),
        w_up_q,
        x_sf.reshape(-1),
        w_up_sf.reshape(-1),
        offs.astype(jnp.int32),
        _unit_vector(experts),
        jnp.ones((m, 1, 1), jnp.float32),
        _unit_vector(1),
    )
    return SreluOutputs(c[0], h[0], sfh_row)


# --------------------------------------------------------------------------- #
# bwd: dgrad-w_down GEMM + ReLU^2 backward + dual MXFP8 quantize + h regeneration + routing gradient
# --------------------------------------------------------------------------- #


@cute_launcher_factory
def _dsrelu_launcher(*, expert_count, mma_tiler_mn, cluster_shape_mn, max_active_clusters):
    """Signature: (stream, g, w_down, c, sfg, sfw_down, offs, alpha, prob, norm, dprob_slots, dc, dc_col,
    sfdc_row, sfdc_col, h_col, sfh_col, workspace); ``dprob_slots`` is the zeroed input the last output aliases."""

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
        mProb,
        mNorm,
        mDProbSlots,
        mD,
        mDCol,
        mSFDRow,
        mSFDCol,
        mDSrelu,
        mSFDSreluCol,
        mWorkspace,
    ):
        kernel = BlockScaledMoEGroupedGemmQuantBwdKernel(
            sf_vec_size=MXFP8_BLOCK_SIZE,
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=mma_tiler_mn[0] == _TWO_CTA_TILE_M,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            vectorized_f32=False,
            generate_sfd=True,
            discrete_col_sfd=True,
            expert_cnt=expert_count,
            weight_mode=MoEWeightMode.DENSE,
            use_dynamic_sched=False,
            epilogue_type=DsreluEpilogueType.DSRELU.value,
            generate_d_srelu=True,
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
            mProb,
            mDProbSlots,
            None,  # dbias: the expert MLP has no bias
            mDSrelu,
            mSFDSreluCol,
            max_active_clusters,
            stream,
        )

    return launcher


def mxfp8_fused_dsrelu(g_q, g_sf, w_down_q, w_down_sf, c, offs) -> DSreluOutputs:
    """Fused backward: ``dc = 2 relu(c) * (g @ w_down^T)`` with a dual MXFP8 epilogue.

    The epilogue also regenerates ``h = relu(c)^2`` quantized along tokens for the down weight
    gradient. It reads the bf16 ``c``, so ``h_col`` is the quantization of ``relu(c)^2`` from the
    bf16-rounded pre-activation, not bit for bit the token copy of the forward epilogue's
    float32 ``h``.

    Args:
        g_q: (M, H) float8_e4m3fn output cotangent, quantized along ``H``.
        g_sf: row-orientation swizzled scales of ``g_q``.
        w_down_q: (E, I, H) float8_e4m3fn, ``w_down`` in its stored layout quantized along ``H``.
        w_down_sf: per-expert swizzled scales of ``w_down_q`` over ``(I, H / 32)``.
        c: (M, I) bf16 pre-activation from :func:`mxfp8_fused_srelu`.
        offs: (E,) int32 padded END offsets, each a multiple of ``GROUP_PAD``.

    Returns:
        Both quantized copies of the pre-activation's gradient, the token copy of ``h``, their
        scales, and each row's routing-weight gradient; see :class:`DSreluOutputs`. The kernel adds
        each N tile's routing-gradient partial into that tile's zeroed slot, the only write to the
        slot, and the slots are summed here, so the row dot is deterministic.
    """
    _require_sm100()
    m, hidden = g_q.shape
    experts, intermediate, k = w_down_q.shape
    if k != hidden:
        raise ValueError(f"contraction mismatch: g_q {g_q.shape}, w_down_q {w_down_q.shape}")
    if c.shape != (m, intermediate) or c.dtype != jnp.bfloat16:
        raise ValueError(f"c must be bf16 of shape ({m}, {intermediate}), got {c.dtype} {c.shape}")
    _check_padded_rows(m, offs, experts)
    _check_quantized(g_q, w_down_q)
    _check_scales(g_sf, w_down_sf)
    mma_tiler_mn, cluster_shape_mn = _DSRELU_TILING
    slots = _dprob_slot_count(intermediate, _DSRELU_TILING)
    launcher = _dsrelu_launcher(
        expert_count=experts,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        max_active_clusters=_max_active_clusters(cluster_shape_mn),
    )
    col_scales = jax.ShapeDtypeStruct((_sf_col_bytes(m, intermediate),), jnp.float8_e8m0fnu)
    call = cutlass_call(
        launcher,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((1, m, intermediate), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, m, intermediate), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((_sf_row_bytes(m, intermediate),), jnp.float8_e8m0fnu),
            col_scales,
            jax.ShapeDtypeStruct((1, m, intermediate), jnp.float8_e4m3fn),
            col_scales,
            jax.ShapeDtypeStruct((_UNUSED_WORKSPACE_BYTES,), jnp.uint8),
            jax.ShapeDtypeStruct((m, slots, 1), jnp.float32),
        ),
        # w_down is stored [E, I, H], (N, K) per expert.
        input_spec=(
            _BATCH_LAST,
            _WEIGHT_SPECS[WeightMajor.K],
            _BATCH_LAST,
            _SPEC,
            _SPEC,
            _SPEC,
            _SPEC,
            _SPEC,
            _SPEC,
            _SPEC,
        ),
        output_spec=(_BATCH_LAST, _BATCH_LAST, _SPEC, _SPEC, _BATCH_LAST, _SPEC, _SPEC, _SPEC),
        input_output_aliases=((_DPROB_SLOTS_INPUT, _DPROB_SLOTS_OUTPUT),),
        use_static_tensors=True,
    )
    dc, dc_col, sfdc_row, sfdc_col, h_col, sfh_col, _workspace, dprob_slots = call(
        _as_batch_last(g_q),
        w_down_q,
        _as_batch_last(c),
        g_sf.reshape(-1),
        w_down_sf.reshape(-1),
        offs.astype(jnp.int32),
        _unit_vector(experts),
        jnp.ones((m, 1, 1), jnp.float32),
        _unit_vector(1),
        jnp.zeros((m, slots, 1), jnp.float32),
    )
    return DSreluOutputs(
        dc[0], sfdc_row, dc_col[0], sfdc_col, h_col[0], sfh_col, jnp.sum(dprob_slots[:, :, 0], axis=-1)
    )
