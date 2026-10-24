# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX entry points for the cudnn-frontend MXFP8 MoE grouped GEMMs (SM100 only).

The kernels come from the nvidia-cudnn-frontend package (``cudnn.gemm.cutedsl.grouped``). This
module holds the two plain ones and what every wrapper shares; :mod:`.adapter_srelu` holds the
GEMMs with a fused ReLU^2 epilogue.

- :func:`mxfp8_fused_gemm`: a grouped GEMM with a bf16 epilogue.
- :func:`mxfp8_fused_wgrad`: the grouped weight gradient from two token-axis quantized operands.

Row layout: tokens are stored contiguously per expert, and every expert's row count is padded to
a multiple of :data:`GROUP_PAD`. ``offs`` holds the int32 END offset of each expert in that padded
buffer. ``offs[-1]`` may stop short of the buffer's rows: the kernels schedule their tiles from the
offsets alone, so they neither read nor write the rows past it, and those output rows are
unspecified. Padding rows must be finite (zeros), since a non-finite row poisons the 32-token
column scales it shares.

Scale-factor layouts, all float8_e8m0fnu bytes:

- row orientation (scales along features): the whole ``(rows, features / 32)`` scale matrix in the
  SM100 32x4x4 swizzle, flat. Because each expert spans whole 128-row blocks this equals the
  per-expert swizzle concatenation.
- column orientation (scales along tokens): per-expert swizzles of each ``(features,
  tokens_e / 32)`` scale matrix, concatenated in expert order. This is what :func:`mxfp8_fused_wgrad`
  consumes.
- weights: one swizzled ``(N, K / 32)`` block per expert, stacked, whichever way the values are
  stored (:class:`WeightMajor`).

The routing-weight knobs of the kernels (alpha, prob, norm_const) are pinned to 1.

Launchers are memoized through :func:`levanter.cutlass_kernel_cache.cute_launcher_factory`, so
shape-identical calls share one CuTeDSL compile and the persistent kernel cache can serve them.
"""

import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
import jax
import jax.numpy as jnp
from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode, WGradInputOrder, WgradSfTensormapConstructor
from cudnn.gemm.cutedsl.grouped.quant.grouped_gemm_quant import BlockScaledMoEGroupedGemmQuantKernel
from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad import BlockScaledMoEGroupedGemmWgradKernel
from quack.cute_dsl_utils import get_max_active_clusters

from levanter.cutlass_kernel_cache import cute_launcher_factory, cutlass_call, gpu_compute_capability
from levanter.grug._moe.mxfp8.layout import GROUP_PAD
from levanter.grug._moe.mxfp8.quantize import MXFP8_BLOCK_SIZE, SF_TILE_ROWS, WeightMajor

# (mma_tiler_mn, cluster_shape_mn) per kernel: the winners of the research ablations in
# c3cb334f8e:experiments/grug/moe/mxfp8.py. A sweep of tile and
# cluster shapes at the hero expert chunk (H = I = 3072, 3 experts, 302,336 rows) on one B200
# found no other configuration more than 3% faster for any kernel.
_GEMM_TILING = ((256, 256), (2, 1))
_WGRAD_TILING = ((256, 256), (2, 2))
# An MMA tile this many rows tall runs as a 2-CTA instruction over a pair of SMs.
_TWO_CTA_TILE_M = 256

# The wgrad operands: both token-major 2D tensors, one dense (E, K, N) output.
_WGRAD_INPUT_ORDER = WGradInputOrder.Tensor2D
_WGRAD_WEIGHT_MODE = MoEWeightMode.DENSE
# The dense, statically scheduled grouped GEMMs ignore their workspace but still take a pointer.
_UNUSED_WORKSPACE_BYTES = 16

_SPEC = cjax.TensorSpec()
# A row-major (L, M, K) buffer presented as the kernels' logical (M, K, L), with L the batch or
# expert mode. Token operands are passed as (1, M, K) so they carry the same trailing unit mode.
_BATCH_LAST = cjax.TensorSpec(mode=(1, 2, 0))
# A row-major (T, K) buffer presented as (K, T): the wgrad kernel's M-major A operand.
_TRANSPOSED = cjax.TensorSpec(mode=(1, 0))
# The grouped GEMMs' B operand, logical (N, K, L): an (E, N, K) buffer, or an (E, K, N) one read N-major.
_WEIGHT_SPECS = {WeightMajor.K: _BATCH_LAST, WeightMajor.N: cjax.TensorSpec(mode=(2, 1, 0))}


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


def _dprob_slot_count(n: int, tiling: tuple[tuple[int, int], tuple[int, int]]) -> int:
    """Routing-gradient slots per row of an ``n``-column output: one per N tile the scheduler can emit.

    The scheduler counts whole clusters along N, so the count rounds up to a multiple of the
    cluster's N extent (upstream's ``dsrelu.api._dprob_n_slots``).
    """
    (_tile_m, tile_n), (_cluster_m, cluster_n) = tiling
    return _ceil_div(n, tile_n * cluster_n) * cluster_n


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


def _weight_features(b_q: jax.Array, major: WeightMajor) -> tuple[int, int, int]:
    """``(experts, n, k)`` of a grouped weight buffer stored as ``major`` says."""
    experts, first, second = b_q.shape
    return (experts, first, second) if major == WeightMajor.K else (experts, second, first)


# --------------------------------------------------------------------------- #
# plain grouped GEMM (bf16 out)
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


def mxfp8_fused_gemm(a_q, a_sf, b_q, b_sf, offs, *, b_major: WeightMajor):
    """Grouped GEMM ``a[M, K] @ w[E, K, N] -> [M, N]`` in bf16, without a quantizing epilogue.

    Args:
        a_q: (M, K) float8_e4m3fn, experts contiguous with ``GROUP_PAD``-padded rows.
        a_sf: row-orientation swizzled scales of ``a_q``.
        b_q: float8_e4m3fn weights quantized along ``K``: ``(E, N, K)`` for ``WeightMajor.K``,
            ``(E, K, N)`` for ``WeightMajor.N``.
        b_sf: per-expert swizzled scales of ``b_q`` over ``(N, K / 32)``.
        offs: (E,) int32 padded END offsets, each a multiple of ``GROUP_PAD``.
        b_major: how ``b_q`` is stored.
    """
    _require_sm100()
    m, k = a_q.shape
    experts, n, kb = _weight_features(b_q, b_major)
    if kb != k:
        raise ValueError(f"contraction mismatch: a_q {a_q.shape}, b_q {b_q.shape} ({b_major}-major)")
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
        input_spec=(_BATCH_LAST, _WEIGHT_SPECS[b_major], _SPEC, _SPEC, _SPEC, _SPEC, _SPEC),
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
        offs: (E,) int32 cumulative END offsets of the expert row groups (``offs[-1] <= T``; the
            tokens past it, and their scales, are not read). An expert with no rows gets an exactly
            zero gradient.
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
