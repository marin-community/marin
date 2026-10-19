# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""MXFP8 (e4m3 values, one e8m0 scale per 32 contraction elements) grouped GEMMs on QuACK's SM100 kernels.

The expert MLP's activation-path GEMMs run block-scaled. Quantization rides in producers: the ReLU^2 (or SwiGLU)
up-projection writes its postactivation as e4m3 plus scales from its epilogue, the ReLU^2 backward GEMM writes
``d pre`` the same way, and token rows are quantized by the XLA fusion that produces them (before the dispatch
transport). The gated SwiGLU backward runs the SwiGLU derivative in XLA, as the bf16 gated path does, and quantizes
``d gate/up`` with the Pallas row quantizer.

Scale layout. QuACK reads scales in the blocked ``(L, rm, rk, 32, 4, 4)`` layout, one contiguous 512-byte atom per
128 rows x 4 scale columns. For a varlen-m operand (rows grouped by expert) the scales live in one row-padded
buffer, ``L = 1``, where group ``e``'s rows start at tile ``cu[e] // 128 + e``, so every group starts on a tile.
``padded_scale_rows`` builds that layout from per-row scales, and the epilogue quantizers write the same layout.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard

import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
from cutlass import Float32
from quack.activation import act_fn_map, dact_fn_map, gate_fn_map
from quack.epilogue.frontend import gemm_epilogue
from quack.epilogue.math import pack, unpack
from quack.epilogue.ops import ColVecReduce, Scalar, TileStore
from quack.epilogue.quantize_out import BlockScaleFactorStore
from quack.gemm_default_epi import GemmDefaultEpiMixin, GemmDefaultSm100
from quack.gemm_tvm_ffi_utils import make_scheduler_args, make_varlen_args

from levanter.cutlass_kernel_cache import cute_launcher_factory, cutlass_call
from levanter.grug.mxfp8_pallas import (
    quantize_cols,
    quantize_row_blocks,
    quantize_rows,
    quantize_rows_and_row_blocks,
    quantize_rows_cols,
    requantize_row_blocks,
)
from levanter.grug.mxfp8 import MX_BLOCK, mx_dequantize_rows, mx_quantize_cols, mx_quantize_rows, mx_unpack_rows
from levanter.grug._moe.common import _swiglu_gate_up_backward, _unpack_pairs_u32
from levanter.grug._moe.quack_moe_cute import _ACC, _QUACK_USE_PDL, _max_active_clusters, _mint_epilogue_gemm

SCALE_TILE_ROWS = 128
_SCALE_ATOM_COLS = 4
# Block-scaled 2-CTA MMA needs a 256-row tile.
MXFP8_TILE_MN = (256, 256)
MXFP8_CLUSTER = (2, 1, 1)


def blocked_scales(scales: jax.Array) -> jax.Array:
    """``[L, MN, SFK]`` uint8 exponents -> e8m0 ``[L, ceil(MN/128), ceil(SFK/4), 32, 4, 4]``, zero padded."""
    num, mn, sf_k = scales.shape
    rm = -(-mn // SCALE_TILE_ROWS)
    rk = -(-sf_k // _SCALE_ATOM_COLS)
    padded = jnp.pad(scales, ((0, 0), (0, rm * SCALE_TILE_ROWS - mn), (0, rk * _SCALE_ATOM_COLS - sf_k)))
    tiles = padded.reshape(num, rm, SCALE_TILE_ROWS, rk, _SCALE_ATOM_COLS).transpose(0, 1, 3, 2, 4)
    atoms = tiles.reshape(num, rm, rk, 4, 32, _SCALE_ATOM_COLS).transpose(0, 1, 2, 4, 3, 5)
    return jax.lax.bitcast_convert_type(atoms, jnp.float8_e8m0fnu)


def padded_scale_tiles(rows: int, num_groups: int) -> int:
    return -(-rows // SCALE_TILE_ROWS) + num_groups - 1


def padded_scale_rows(scales: jax.Array, cu_seqlens: jax.Array) -> jax.Array:
    """Per-row scales ``[M, SFK]`` -> the varlen-m blocked layout ``[1, P, rk, 32, 4, 4]`` (see module docstring).

    Rows at or past ``cu_seqlens[-1]`` are dropped."""
    rows, sf_k = scales.shape
    num_groups = cu_seqlens.shape[0] - 1
    tiles = padded_scale_tiles(rows, num_groups)
    row = jnp.arange(rows, dtype=jnp.int32)
    group = jnp.searchsorted(cu_seqlens[1:], row, side="right").astype(jnp.int32)
    clamped = jnp.minimum(group, num_groups - 1)
    dest = row + SCALE_TILE_ROWS * clamped - cu_seqlens[clamped] % SCALE_TILE_ROWS
    dest = jnp.where(group < num_groups, dest, tiles * SCALE_TILE_ROWS)
    padded = jnp.zeros((tiles * SCALE_TILE_ROWS, sf_k), scales.dtype).at[dest].set(scales, mode="drop")
    return blocked_scales(padded[None])


def unpadded_scale_rows(blocked: jax.Array, cu_seqlens: jax.Array, rows: int, sf_k: int) -> jax.Array:
    """The inverse of ``padded_scale_rows``: per-row exponents ``[rows, sf_k]`` (uint8) from the varlen-m blocked
    layout. Rows at or past ``cu_seqlens[-1]`` read an arbitrary row's scales."""
    num, rm, rk = blocked.shape[:3]
    atoms = jax.lax.bitcast_convert_type(blocked, jnp.uint8).transpose(0, 1, 2, 4, 3, 5)
    flat = atoms.reshape(num, rm, rk, SCALE_TILE_ROWS, _SCALE_ATOM_COLS).transpose(0, 1, 3, 2, 4)
    flat = flat.reshape(rm * SCALE_TILE_ROWS, rk * _SCALE_ATOM_COLS)[:, :sf_k]
    num_groups = cu_seqlens.shape[0] - 1
    row = jnp.arange(rows, dtype=jnp.int32)
    group = jnp.minimum(jnp.searchsorted(cu_seqlens[1:], row, side="right").astype(jnp.int32), num_groups - 1)
    src = jnp.clip(row + SCALE_TILE_ROWS * group - cu_seqlens[group] % SCALE_TILE_ROWS, 0, flat.shape[0] - 1)
    return flat[src]


def varlen_output_scale_shape(rows: int, cols: int, num_groups: int) -> tuple[int, ...]:
    return (1, padded_scale_tiles(rows, num_groups), -(-(cols // MX_BLOCK) // _SCALE_ATOM_COLS), 32, 4, 4)


def mx_weights(w: jax.Array, b_major: str) -> tuple[jax.Array, jax.Array]:
    """Group weights -> e4m3 in their own layout and blocked e8m0 scales over the logical ``(E, N, K / 32)``.

    ``b_major="k"``: ``w`` is ``[E, N, K]`` (contraction last). ``b_major="n"``: ``w`` is ``[E, K, N]``, quantized
    along ``K`` in place. Quantizing in the stored layout matters inside the layer scan: a transposed copy makes
    XLA's layout assignment transpose the whole stacked weight outside the loop (+24 GiB at the GB200 recipe)."""
    if b_major == "k":
        values, scales = mx_quantize_rows(w)
        return values, blocked_scales(scales)
    if b_major == "n":
        values, scales = mx_quantize_cols(w)
        return values, blocked_scales(jnp.swapaxes(scales, 1, 2))
    raise ValueError(f"b_major must be 'k' or 'n', got {b_major!r}")


_TS = cjax.TensorSpec
_ROWS_SPEC = _TS(divisibility=(1, 16), static=False)
_WEIGHT_SPECS = {
    "k": _TS(divisibility=(1, 1, 16), static=False),
    # [E, K, N] stored; the kernel's logical order is (E, N, K).
    "n": _TS(mode=(0, 2, 1), divisibility=(1, 1, 16), static=False),
}
_SCALE_SPEC = _TS(static=False)
_INDEX_SPEC = _TS(static=False)
_OUT_SPEC = _TS(divisibility=(1, 8), static=False)


def _gemm(gemm_type, a_dtype):
    return gemm_type(
        _ACC,
        a_dtype,
        MXFP8_TILE_MN,
        MXFP8_CLUSTER,
        sf_vec_size=MX_BLOCK,
        gather_A=False,
        use_clc_persistence=False,
        use_pdl=_QUACK_USE_PDL,
    )


@cute_launcher_factory
def _build_plain_launcher(*, max_active_clusters, max_swizzle):
    @cute.jit
    def launcher(stream, mA, mB, mSFA, mSFB, mCuSeqlens, mD):
        gemm = _gemm(GemmDefaultSm100, cutlass.Float8E4M3FN)
        scheduler_args = make_scheduler_args(max_active_clusters, max_swizzle, None)
        varlen_args = make_varlen_args(mCuSeqlens, None, None)
        gemm(
            mA, mB, mD, None, GemmDefaultEpiMixin.EpilogueArguments(), scheduler_args, varlen_args, stream, mSFA, mSFB
        )

    return launcher


def mxfp8_grouped_gemm(a, a_scales, b, b_scales, cu_seqlens, *, b_major, out_dtype=jnp.bfloat16, max_swizzle=8):
    """``a[M, K] @ b[e, N, K]^T`` per row group, e4m3 operands with blocked e8m0 scales; rows past ``cu[-1]``
    unspecified. ``a_scales`` is the varlen-m layout, ``b_scales`` ``[E, rn, rk, 32, 4, 4]``."""
    launcher = _build_plain_launcher(max_active_clusters=_max_active_clusters(MXFP8_CLUSTER), max_swizzle=max_swizzle)
    call = cutlass_call(
        launcher,
        output_shape_dtype=jax.ShapeDtypeStruct((a.shape[0], b.shape[1 if b_major == "k" else 2]), out_dtype),
        input_spec=(_ROWS_SPEC, _WEIGHT_SPECS[b_major], _SCALE_SPEC, _SCALE_SPEC, _INDEX_SPEC),
        output_spec=(_OUT_SPEC,),
        use_static_tensors=False,
    )
    return call(a, b, a_scales, b_scales, cu_seqlens.astype(jnp.int32))


def _relu2_quant_epilogue():
    act = act_fn_map["relu_sq"]

    @gemm_epilogue(
        outputs=(TileStore("postact", quant=BlockScaleFactorStore("postact_sf", output="postact")),),
        extra_ops=(Scalar("sfd_norm_const"),),
    )
    def relu2_quant_epi(acc):
        return {"D": acc, "postact": act(acc)}

    return relu2_quant_epi


@cute_launcher_factory
def _build_relu2_launcher(*, max_active_clusters, max_swizzle):
    gemm_type = _mint_epilogue_gemm(_relu2_quant_epilogue(), kind_sig=(), paired_acc=False)

    @cute.jit
    def launcher(stream, mA, mB, mSFA, mSFB, mCuSeqlens, mD, mPostAct, mPostActSF):
        gemm = _gemm(gemm_type, cutlass.Float8E4M3FN)
        epi_args = gemm_type.EpilogueArguments(postact=mPostAct, postact_sf=mPostActSF, sfd_norm_const=Float32(1.0))
        scheduler_args = make_scheduler_args(max_active_clusters, max_swizzle, None)
        varlen_args = make_varlen_args(mCuSeqlens, None, None)
        gemm(mA, mB, mD, None, epi_args, scheduler_args, varlen_args, stream, mSFA, mSFB)

    return launcher


def mxfp8_relu2_grouped_gemm(a, a_scales, b, b_scales, cu_seqlens, *, b_major, max_swizzle=8):
    """Block-scaled ``pre = a @ b^T`` with ``relu(pre)^2`` quantized in the epilogue.

    Returns ``(pre bf16 [M, N], post e4m3 [M, N], post scales)``, the scales in the varlen-m layout."""
    rows, cols = a.shape[0], b.shape[1 if b_major == "k" else 2]
    launcher = _build_relu2_launcher(max_active_clusters=_max_active_clusters(MXFP8_CLUSTER), max_swizzle=max_swizzle)
    call = cutlass_call(
        launcher,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((rows, cols), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, cols), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct(varlen_output_scale_shape(rows, cols, cu_seqlens.shape[0] - 1), jnp.float8_e8m0fnu),
        ),
        input_spec=(_ROWS_SPEC, _WEIGHT_SPECS[b_major], _SCALE_SPEC, _SCALE_SPEC, _INDEX_SPEC),
        output_spec=(_OUT_SPEC, _ROWS_SPEC, _SCALE_SPEC),
        use_static_tensors=False,
    )
    return call(a, b, a_scales, b_scales, cu_seqlens.astype(jnp.int32))


def _drelu2_quant_epilogue():
    dact = dact_fn_map["relu_sq"]

    @gemm_epilogue(
        outputs=(
            "mAuxOut",
            TileStore("dpre_q", quant=BlockScaleFactorStore("dpre_sf", output="dpre_q")),
        ),
        reduces={"mColVecReduce": ColVecReduce("mColVecReduce", scaled=True)},
        extra_ops=(Scalar("sfd_norm_const"),),
    )
    def drelu2_quant_epi(acc, c):
        dx, out = dact(c, acc)
        return {"D": dx, "mAuxOut": out, "dpre_q": dx, "mColVecReduce": (out, acc)}

    return drelu2_quant_epi


@cute_launcher_factory
def _build_drelu2_launcher(*, max_active_clusters, max_swizzle):
    gemm_type = _mint_epilogue_gemm(_drelu2_quant_epilogue(), kind_sig=(("c", "c"),), paired_acc=False)

    @cute.jit
    def launcher(stream, mA, mB, mSFA, mSFB, mCuSeqlens, mC, mPostAct, mDPreQ, mDPreSF, mDotPartials):
        gemm = _gemm(gemm_type, cutlass.Float8E4M3FN)
        epi_args = gemm_type.EpilogueArguments(
            mAuxOut=mPostAct,
            dpre_q=mDPreQ,
            dpre_sf=mDPreSF,
            sfd_norm_const=Float32(1.0),
            mColVecReduce=mDotPartials,
        )
        scheduler_args = make_scheduler_args(max_active_clusters, max_swizzle, None)
        varlen_args = make_varlen_args(mCuSeqlens, None, None)
        # No D: ``d pre`` is stored only quantized (``dpre_q``), which is all the input-gradient GEMM reads and
        # half the bytes of a bf16 copy; the weight gradient dequantizes it.
        gemm(mA, mB, None, mC, epi_args, scheduler_args, varlen_args, stream, mSFA, mSFB)

    return launcher


def mxfp8_drelu2_grouped_gemm(dy, dy_scales, b, b_scales, pre, cu_seqlens, *, max_swizzle=8):
    """ReLU^2 backward fused into ``d post = dy @ b^T`` (block-scaled).

    Returns ``(post bf16, d_pre e4m3, d_pre scales (varlen-m layout), row_dot fp32 [M])`` with
    ``row_dot = <post, d post>``."""
    rows, cols = dy.shape[0], b.shape[1]
    if pre.shape != (rows, cols):
        raise ValueError(f"pre must be {(rows, cols)}, got {pre.shape}")
    launcher = _build_drelu2_launcher(max_active_clusters=_max_active_clusters(MXFP8_CLUSTER), max_swizzle=max_swizzle)
    n_tiles = -(-cols // MXFP8_TILE_MN[1])
    call = cutlass_call(
        launcher,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((rows, cols), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, cols), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct(varlen_output_scale_shape(rows, cols, cu_seqlens.shape[0] - 1), jnp.float8_e8m0fnu),
            jax.ShapeDtypeStruct((rows, n_tiles), jnp.float32),
        ),
        input_spec=(_ROWS_SPEC, _WEIGHT_SPECS["k"], _SCALE_SPEC, _SCALE_SPEC, _INDEX_SPEC, _OUT_SPEC),
        output_spec=(_OUT_SPEC, _ROWS_SPEC, _SCALE_SPEC, _TS(static=False)),
        use_static_tensors=False,
    )
    post, d_pre_q, d_pre_scales, dot_partials = call(dy, b, dy_scales, b_scales, cu_seqlens.astype(jnp.int32), pre)
    return post, d_pre_q, d_pre_scales, jnp.sum(dot_partials, axis=-1)


def relu2_expert_mlp_mxfp8_fwd(x, w_up, w_down, cu):
    """``relu(x W_up)^2 W_down`` with MXFP8 operands. ``x`` is the dispatched rows, either ``mx_pack_rows`` bytes
    (uint8) or bf16 rows, which are quantized here and saved as bf16 for the weight gradient.

    Same contract as ``_relu2_expert_mlp_quack_fwd``: rows past ``cu[-1]`` are unspecified."""
    if x.dtype == jnp.uint8:
        values, exponents = mx_unpack_rows(x, w_up.shape[1])
    else:
        values, exponents = quantize_rows(x)
    pre, post, post_scales = mxfp8_relu2_grouped_gemm(
        values, padded_scale_rows(exponents, cu), *mx_weights(w_up, "n"), cu, b_major="n"
    )
    y = mxfp8_grouped_gemm(post, post_scales, *mx_weights(w_down, "n"), cu, b_major="n")
    return y, (x, w_up, w_down, pre, cu)


def relu2_expert_mlp_mxfp8_backward_input(res, dy):
    """``(dx, <y, dy>, wgrad_args)`` of ``relu2_expert_mlp_mxfp8_fwd``; ``wgrad_args`` feed the bf16
    ``_relu2_expert_mlp_quack_wgrads``: the bf16 rows and ``dy``, and ``d pre`` dequantized from the MXFP8 copy the
    input-gradient GEMM read (the ReLU^2 backward GEMM stores ``d pre`` only quantized)."""
    x, w_up, w_down, pre, cu = res
    dtype = dy.dtype
    dy_values, dy_exponents = quantize_rows(dy)
    post, d_pre_values, d_pre_scales, output_dot = mxfp8_drelu2_grouped_gemm(
        dy_values, padded_scale_rows(dy_exponents, cu), *mx_weights(w_down, "k"), pre, cu
    )
    dx = mxfp8_grouped_gemm(d_pre_values, d_pre_scales, *mx_weights(w_up, "k"), cu, b_major="k")
    rows, width = d_pre_values.shape
    d_pre = mx_dequantize_rows(d_pre_values, unpadded_scale_rows(d_pre_scales, cu, rows, width // MX_BLOCK), dtype)
    if x.dtype == jnp.uint8:
        x = mx_dequantize_rows(*mx_unpack_rows(x, w_up.shape[1]), dtype)
    return dx, output_dot, (x, d_pre, post, dy, cu)


def _swiglu_quant_epilogue():
    act = gate_fn_map["swiglu"]

    # acc_pair: the accumulator pairs adjacent (gate, up) columns of the interleaved weights, so the postact is half
    # the GEMM's N; its scale vectors span 2 * 32 accumulator columns (``BlockScaleFactorStore(output=...)``).
    @gemm_epilogue(
        outputs=(TileStore("postact", gated=True, quant=BlockScaleFactorStore("postact_sf", output="postact")),),
        extra_ops=(Scalar("sfd_norm_const"),),
        mode="acc_pair",
    )
    def swiglu_quant_epi(acc):
        gate, up = unpack(acc)
        return {"D": pack(gate, up), "postact": act(gate, up)}

    return swiglu_quant_epi


@cute_launcher_factory
def _build_swiglu_launcher(*, max_active_clusters, max_swizzle):
    gemm_type = _mint_epilogue_gemm(_swiglu_quant_epilogue(), kind_sig=(), paired_acc=True)

    @cute.jit
    def launcher(stream, mA, mB, mSFA, mSFB, mCuSeqlens, mD, mPostAct, mPostActSF):
        gemm = _gemm(gemm_type, cutlass.Float8E4M3FN)
        epi_args = gemm_type.EpilogueArguments(postact=mPostAct, postact_sf=mPostActSF, sfd_norm_const=Float32(1.0))
        scheduler_args = make_scheduler_args(max_active_clusters, max_swizzle, None)
        varlen_args = make_varlen_args(mCuSeqlens, None, None)
        gemm(mA, mB, mD, None, epi_args, scheduler_args, varlen_args, stream, mSFA, mSFB)

    return launcher


def mxfp8_swiglu_grouped_gemm(a, a_scales, b, b_scales, cu_seqlens, *, max_swizzle=8):
    """Block-scaled ``gu = a @ b`` over interleaved gate/up weights ``b[E, K, 2N]`` (``[g0, u0, g1, u1, ...]``,
    quantized along ``K`` in place, ``mx_weights(b, "n")``) with ``silu(g) * u`` quantized in the epilogue.

    Returns ``(gu bf16 [M, 2N] interleaved, h e4m3 [M, N], h scales)``, the scales in the varlen-m layout."""
    rows, cols = a.shape[0], b.shape[2]
    if cols % (2 * MX_BLOCK):
        raise ValueError(f"gate/up width {cols} must divide {2 * MX_BLOCK}")
    launcher = _build_swiglu_launcher(max_active_clusters=_max_active_clusters(MXFP8_CLUSTER), max_swizzle=max_swizzle)
    call = cutlass_call(
        launcher,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((rows, cols), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, cols // 2), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct(
                varlen_output_scale_shape(rows, cols // 2, cu_seqlens.shape[0] - 1), jnp.float8_e8m0fnu
            ),
        ),
        input_spec=(_ROWS_SPEC, _WEIGHT_SPECS["n"], _SCALE_SPEC, _SCALE_SPEC, _INDEX_SPEC),
        output_spec=(_OUT_SPEC, _ROWS_SPEC, _SCALE_SPEC),
        use_static_tensors=False,
    )
    return call(a, b, a_scales, b_scales, cu_seqlens.astype(jnp.int32))


def _dispatched_rows(x: jax.Array, width: int) -> tuple[jax.Array, jax.Array]:
    """e4m3 values and per-row exponents of the dispatched rows: ``mx_pack_rows`` bytes or bf16 rows."""
    if x.dtype == jnp.uint8:
        return mx_unpack_rows(x, width)
    return quantize_rows(x)


def swiglu_expert_mlp_mxfp8_fwd(x, w13_il, w_down, cu):
    """``(silu(x W_gate) * x W_up) W_down`` with MXFP8 operands; ``w13_il`` is the interleaved gate/up bank
    ``[E, H, 2I]`` (``_interleave_gate_up``). ``x`` is ``mx_pack_rows`` bytes or bf16 rows.

    Same contract as ``_expert_mlp_quack_wgrad_fwd``: rows past ``cu[-1]`` are unspecified. The residuals keep
    ``gu`` (bf16) for the SwiGLU backward and no copy of ``h``: the weight gradient recomputes it from ``gu``."""
    values, exponents = _dispatched_rows(x, w13_il.shape[1])
    gu, h, h_scales = mxfp8_swiglu_grouped_gemm(values, padded_scale_rows(exponents, cu), *mx_weights(w13_il, "n"), cu)
    y = mxfp8_grouped_gemm(h, h_scales, *mx_weights(w_down, "n"), cu, b_major="n")
    return y, (x, w13_il, w_down, gu, cu)


def swiglu_expert_mlp_mxfp8_backward_input(res, dy):
    """``(dx, <y, dy>, wgrad_args)`` of ``swiglu_expert_mlp_mxfp8_fwd``.

    ``dh = dy W_down^T`` and ``dx = d gu W13^T`` run block-scaled against the weights quantized in their stored
    layout along the contraction axis (no transposed copies). ``wgrad_args`` feed the bf16
    ``swiglu_expert_mlp_mxfp8_wgrads``: the bf16 rows, ``d gu``, ``h`` recomputed from ``gu`` and ``dy``."""
    x, w13_il, w_down, gu, cu = res
    dy_values, dy_exponents = quantize_rows(dy)
    dh = mxfp8_grouped_gemm(dy_values, padded_scale_rows(dy_exponents, cu), *mx_weights(w_down, "k"), cu, b_major="k")
    gate, up = _unpack_pairs_u32(gu)
    h = jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
    output_dot = jnp.sum(dh.astype(jnp.float32) * h, axis=-1)
    d_gu = _swiglu_gate_up_backward(gu, dh)
    d_gu_values, d_gu_exponents = quantize_rows(d_gu)
    dx = mxfp8_grouped_gemm(
        d_gu_values, padded_scale_rows(d_gu_exponents, cu), *mx_weights(w13_il, "k"), cu, b_major="k"
    )
    if x.dtype == jnp.uint8:
        x = mx_dequantize_rows(*mx_unpack_rows(x, w13_il.shape[1]), dy.dtype)
    return dx, output_dot, (x, d_gu, h.astype(dy.dtype), dy, cu)


def token_scale_blocks(cu_seqlens: jax.Array, rows: int) -> tuple[jax.Array, jax.Array]:
    """The token-axis 32-row blocks of a varlen-k operand, in the order of its blocked scale columns.

    QuACK's varlen-k GEMM reads group ``e``'s scales from scale tile ``cu[e] // 128 + e`` on (4 blocks of 32
    contraction rows per tile), so group ``e``'s block ``k`` covers rows ``cu[e] + 32 k`` onward, clipped to the
    group: no block mixes groups or reads rows past ``cu[-1]``. Returns ``(block_starts, block_rows)`` over the
    ``4 * padded_scale_tiles(rows, E)`` scale columns; columns past a group's last block have zero rows."""
    num_groups = cu_seqlens.shape[0] - 1
    num_blocks = _SCALE_ATOM_COLS * padded_scale_tiles(rows, num_groups)
    cu = cu_seqlens.astype(jnp.int32)
    first_block = _SCALE_ATOM_COLS * (cu[:-1] // SCALE_TILE_ROWS + jnp.arange(num_groups, dtype=jnp.int32))
    column = jnp.arange(num_blocks, dtype=jnp.int32)
    group = jnp.clip(jnp.searchsorted(first_block, column, side="right").astype(jnp.int32) - 1, 0, num_groups - 1)
    starts = cu[group] + MX_BLOCK * (column - first_block[group])
    counts = jnp.clip(cu[group + 1] - starts, 0, MX_BLOCK)
    return jnp.minimum(starts, rows), counts


def quantize_token_blocks(x: jax.Array, cu_seqlens: jax.Array) -> tuple[jax.Array, jax.Array]:
    """MXFP8 ``x[T, N]`` along ``T`` for a varlen-k GEMM: e4m3 ``[T, N]`` in place (rows past ``cu[-1]``
    unspecified) and uint8 exponents ``[4 * padded_scale_tiles(T, E), N]`` over ``token_scale_blocks``."""
    return quantize_row_blocks(x, *token_scale_blocks(cu_seqlens, x.shape[0]))


@cute_launcher_factory
def _build_wgrad_launcher(*, tile_mn, cluster_mnk, max_active_clusters, max_swizzle):
    @cute.jit
    def launcher(stream, mA, mB, mSFA, mSFB, mCuSeqlens, mD):
        gemm = GemmDefaultSm100(
            _ACC,
            cutlass.Float8E4M3FN,
            tile_mn,
            cluster_mnk,
            sf_vec_size=MX_BLOCK,
            gather_A=False,
            use_clc_persistence=False,
            use_pdl=_QUACK_USE_PDL,
        )
        scheduler_args = make_scheduler_args(max_active_clusters, max_swizzle, None)
        varlen_args = make_varlen_args(None, mCuSeqlens, None)
        gemm(
            mA, mB, mD, None, GemmDefaultEpiMixin.EpilogueArguments(), scheduler_args, varlen_args, stream, mSFA, mSFB
        )

    return launcher


# varlen-k wants A m-major and B n-major: [T, M] and [T, N] as stored. Logical (M, K) / (N, K) / (L, M, N).
_TOKEN_MAJOR = _TS(mode=(1, 0), divisibility=(1, 16), static=False)
_WGRAD_OUT_SPEC = _TS(divisibility=(1, 1, 8), static=False)
MXFP8_WGRAD_TILE_MN = (256, 256)
# bench_mxfp8_expert_wgrad.py at the hero chunk: (2, 2, 1) beats (2, 1, 1) by 1-5%; 256 x 128 tiles lose 12-25%.
MXFP8_WGRAD_CLUSTER = (2, 2, 1)


def mxfp8_token_grouped_gemm(
    lhs: tuple[jax.Array, jax.Array],
    rhs: tuple[jax.Array, jax.Array],
    cu_seqlens: jax.Array,
    *,
    out_dtype=jnp.bfloat16,
    tile_mn=MXFP8_WGRAD_TILE_MN,
    cluster_mnk=MXFP8_WGRAD_CLUSTER,
    max_swizzle=8,
) -> jax.Array:
    """Per-group ``lhs.T @ rhs`` (``[E, M, N]``) of two operands quantized along the token axis
    (``quantize_token_blocks``: e4m3 ``[T, M]`` and exponents ``[J, M]``), on QuACK's block-scaled varlen-k GEMM:
    fp32 accumulation, rows past ``cu[-1]`` never read, and an empty group yields zeros."""
    (a, a_exponents), (b, b_exponents) = lhs, rhs
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"lhs and rhs row counts must match, got lhs={a.shape}, rhs={b.shape}")
    launcher = _build_wgrad_launcher(
        tile_mn=tile_mn,
        cluster_mnk=cluster_mnk,
        max_active_clusters=_max_active_clusters(cluster_mnk),
        max_swizzle=max_swizzle,
    )
    call = cutlass_call(
        launcher,
        output_shape_dtype=jax.ShapeDtypeStruct((cu_seqlens.shape[0] - 1, a.shape[1], b.shape[1]), out_dtype),
        input_spec=(_TOKEN_MAJOR, _TOKEN_MAJOR, _SCALE_SPEC, _SCALE_SPEC, _INDEX_SPEC),
        output_spec=(_WGRAD_OUT_SPEC,),
        use_static_tensors=False,
    )
    a_scales, b_scales = blocked_scales(a_exponents.T[None]), blocked_scales(b_exponents.T[None])
    return call(a, b, a_scales, b_scales, cu_seqlens.astype(jnp.int32))


def mxfp8_grouped_wgrad(lhs: jax.Array, rhs: jax.Array, cu_seqlens: jax.Array, *, out_dtype=jnp.bfloat16):
    """The block-scaled counterpart of ``quack_grouped_wgrad``: per-group ``lhs.T @ rhs`` with both bf16 operands
    quantized along the token axis per group (``token_scale_blocks``)."""
    return mxfp8_token_grouped_gemm(
        quantize_token_blocks(lhs, cu_seqlens), quantize_token_blocks(rhs, cu_seqlens), cu_seqlens, out_dtype=out_dtype
    )


def expert_mlp_mxfp8_wgrads(x, d_up, act, dy, cu):
    """``(dW_up, dW_down)`` of an MXFP8 expert MLP from its bf16 ``wgrad_args``: ``x^T d_up`` and ``act^T dy`` per
    expert, block-scaled. ``d_up`` is the up-projection's preactivation cotangent (``d gu`` interleaved for SwiGLU,
    ``d pre`` for ReLU^2) and ``act`` its postactivation (``h`` / ``post``); the gradients come out in ``dy``'s
    dtype, as the bf16 wgrads do. The fused backwards (``*_expert_mlp_mxfp8_backward``) quantize the same operands
    with fewer passes."""
    return (
        mxfp8_grouped_wgrad(x, d_up, cu, out_dtype=dy.dtype),
        mxfp8_grouped_wgrad(act, dy, cu, out_dtype=dy.dtype),
    )


def _dispatched_rows_and_tokens(x: jax.Array, width: int, cu: jax.Array):
    """The dispatched rows (``mx_pack_rows`` bytes or bf16) quantized along their columns for the up-projection
    and along the token axis for its weight gradient, the bf16 rows in one pass."""
    blocks = token_scale_blocks(cu, x.shape[0])
    if x.dtype == jnp.uint8:
        values, exponents = mx_unpack_rows(x, width)
        return (values, exponents), requantize_row_blocks(values, exponents, *blocks)
    return quantize_rows_and_row_blocks(x, *blocks)


def swiglu_expert_mlp_mxfp8_wgrad_fwd(x, w13_il, w_down, cu):
    """``swiglu_expert_mlp_mxfp8_fwd`` whose residuals keep the rows quantized along the token axis (for
    ``swiglu_expert_mlp_mxfp8_backward``) instead of ``x``."""
    (values, exponents), x_tokens = _dispatched_rows_and_tokens(x, w13_il.shape[1], cu)
    gu, h, h_scales = mxfp8_swiglu_grouped_gemm(values, padded_scale_rows(exponents, cu), *mx_weights(w13_il, "n"), cu)
    y = mxfp8_grouped_gemm(h, h_scales, *mx_weights(w_down, "n"), cu, b_major="n")
    # The token-axis exponents go last: callers read the pre-activations at index 3, as in the other variants.
    return y, (x_tokens[0], w13_il, w_down, gu, cu, x_tokens[1])


def relu2_expert_mlp_mxfp8_wgrad_fwd(x, w_up, w_down, cu):
    """``relu2_expert_mlp_mxfp8_fwd`` whose residuals keep the rows quantized along the token axis (for
    ``relu2_expert_mlp_mxfp8_backward``) instead of ``x``."""
    (values, exponents), x_tokens = _dispatched_rows_and_tokens(x, w_up.shape[1], cu)
    pre, post, post_scales = mxfp8_relu2_grouped_gemm(
        values, padded_scale_rows(exponents, cu), *mx_weights(w_up, "n"), cu, b_major="n"
    )
    y = mxfp8_grouped_gemm(post, post_scales, *mx_weights(w_down, "n"), cu, b_major="n")
    # The token-axis exponents go last: callers read the pre-activations at index 3, as in the other variants.
    return y, (x_tokens[0], w_up, w_down, pre, cu, x_tokens[1])


def swiglu_expert_mlp_mxfp8_backward(res, dy):
    """``(dx, dw13_il, dw_down, <y, dy>)`` of ``swiglu_expert_mlp_mxfp8_wgrad_fwd``: all six GEMMs block-scaled.

    As ``swiglu_expert_mlp_mxfp8_backward_input`` plus the weight gradients, whose operands are quantized along the
    token axis per expert: ``dy`` and ``d gu`` in the same pass as their row quantization for the input-gradient
    GEMMs (the rows come so from the forward), ``h`` (recomputed from ``gu``, as in the bf16 wgrads) on its own."""
    x_token_values, w13_il, w_down, gu, cu, x_token_exponents = res
    x_tokens = (x_token_values, x_token_exponents)
    rows = dy.shape[0]
    blocks = token_scale_blocks(cu, rows)
    (dy_values, dy_exponents), dy_tokens = quantize_rows_and_row_blocks(dy, *blocks)
    dh = mxfp8_grouped_gemm(dy_values, padded_scale_rows(dy_exponents, cu), *mx_weights(w_down, "k"), cu, b_major="k")
    gate, up = _unpack_pairs_u32(gu)
    h = jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
    output_dot = jnp.sum(dh.astype(jnp.float32) * h, axis=-1)
    d_gu = _swiglu_gate_up_backward(gu, dh)
    (d_gu_values, d_gu_exponents), d_gu_tokens = quantize_rows_and_row_blocks(d_gu, *blocks)
    dx = mxfp8_grouped_gemm(
        d_gu_values, padded_scale_rows(d_gu_exponents, cu), *mx_weights(w13_il, "k"), cu, b_major="k"
    )
    h_tokens = quantize_row_blocks(h.astype(dy.dtype), *blocks)
    dw13_il = mxfp8_token_grouped_gemm(x_tokens, d_gu_tokens, cu, out_dtype=dy.dtype)
    dw_down = mxfp8_token_grouped_gemm(h_tokens, dy_tokens, cu, out_dtype=dy.dtype)
    return dx, dw13_il, dw_down, output_dot


def relu2_expert_mlp_mxfp8_backward(res, dy):
    """``(dx, dw_up, dw_down, <y, dy>)`` of ``relu2_expert_mlp_mxfp8_wgrad_fwd``: all six GEMMs block-scaled.

    ``dy`` is quantized along both axes in one pass; ``d pre``, which the dReLU^2 GEMM stores only as MXFP8 rows, is
    requantized along the token axis without a bf16 copy; ``post`` is quantized on its own (the rows come quantized
    from the forward)."""
    x_token_values, w_up, w_down, pre, cu, x_token_exponents = res
    x_tokens = (x_token_values, x_token_exponents)
    rows = dy.shape[0]
    blocks = token_scale_blocks(cu, rows)
    (dy_values, dy_exponents), dy_tokens = quantize_rows_and_row_blocks(dy, *blocks)
    post, d_pre_values, d_pre_scales, output_dot = mxfp8_drelu2_grouped_gemm(
        dy_values, padded_scale_rows(dy_exponents, cu), *mx_weights(w_down, "k"), pre, cu
    )
    dx = mxfp8_grouped_gemm(d_pre_values, d_pre_scales, *mx_weights(w_up, "k"), cu, b_major="k")
    width = d_pre_values.shape[1]
    d_pre_exponents = unpadded_scale_rows(d_pre_scales, cu, rows, width // MX_BLOCK)
    d_pre_tokens = requantize_row_blocks(d_pre_values, d_pre_exponents, *blocks)
    post_tokens = quantize_row_blocks(post, *blocks)
    dw_up = mxfp8_token_grouped_gemm(x_tokens, d_pre_tokens, cu, out_dtype=dy.dtype)
    dw_down = mxfp8_token_grouped_gemm(post_tokens, dy_tokens, cu, out_dtype=dy.dtype)
    return dx, dw_up, dw_down, output_dot


@cute_launcher_factory
def _build_dense_launcher(*, max_active_clusters, max_swizzle):
    @cute.jit
    def launcher(stream, mA, mB, mSFA, mSFB, mD):
        gemm = _gemm(GemmDefaultSm100, cutlass.Float8E4M3FN)
        scheduler_args = make_scheduler_args(max_active_clusters, max_swizzle, None)
        gemm(mA, mB, mD, None, GemmDefaultEpiMixin.EpilogueArguments(), scheduler_args, None, stream, mSFA, mSFB)

    return launcher


_K_MAJOR = _TS(divisibility=(1, 16), static=False)
_MN_MAJOR = _TS(mode=(1, 0), divisibility=(1, 16), static=False)


def _dense_gemm(a, a_scales, b, b_scales, *, a_spec, b_spec, rows, cols, max_swizzle=8):
    launcher = _build_dense_launcher(max_active_clusters=_max_active_clusters(MXFP8_CLUSTER), max_swizzle=max_swizzle)
    call = cutlass_call(
        launcher,
        output_shape_dtype=jax.ShapeDtypeStruct((rows, cols), jnp.bfloat16),
        input_spec=(a_spec, b_spec, _SCALE_SPEC, _SCALE_SPEC),
        output_spec=(_OUT_SPEC,),
        use_static_tensors=False,
    )
    return call(a, b, a_scales, b_scales)


def _quantize_cols(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """MXFP8 ``x[T, K]`` along ``T`` (32-row blocks): e4m3 ``[T, K]`` in place and blocked scales over ``(K, T / 32)``."""
    values, exponents = quantize_cols(x)
    return values, blocked_scales(exponents.T[None])


def _quantize_rows(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    values, exponents = quantize_rows(x)
    return values, blocked_scales(exponents[None])


def mxfp8_matmul(x: jax.Array, w_values: jax.Array, w_exponents: jax.Array) -> jax.Array:
    """Local ``x[T, K] @ w[K, N]`` with both operands MXFP8 along ``K``; ``w`` comes quantized along ``K`` in its
    stored layout (``mx_quantize_cols``: e4m3 ``[K, N]``, exponents ``[K / 32, N]``)."""
    qx, sx = _quantize_rows(x)
    sw = blocked_scales(w_exponents.T[None])
    return _dense_gemm(
        qx, sx, w_values, sw, a_spec=_K_MAJOR, b_spec=_MN_MAJOR, rows=x.shape[0], cols=w_values.shape[1]
    )


def mxfp8_matmul_dgrad(g_rows: tuple[jax.Array, jax.Array], w_values: jax.Array, w_exponents: jax.Array) -> jax.Array:
    """Local ``g[T, N] @ w[K, N]^T`` with both operands MXFP8 along ``N``: ``g_rows`` is ``quantize_rows(g)`` and
    ``w`` comes quantized along ``N`` (``mx_quantize_rows``: e4m3 ``[K, N]``, exponents ``[K, N / 32]``)."""
    qg, sg = g_rows[0], blocked_scales(g_rows[1][None])
    sw = blocked_scales(w_exponents[None])
    return _dense_gemm(
        qg, sg, w_values, sw, a_spec=_K_MAJOR, b_spec=_K_MAJOR, rows=qg.shape[0], cols=w_values.shape[0]
    )


def mxfp8_matmul_wgrad(x: jax.Array, g_cols: tuple[jax.Array, jax.Array]) -> jax.Array:
    """Local ``x[T, K]^T @ g[T, N]`` with both operands MXFP8 along ``T``; ``g_cols`` is ``quantize_cols(g)``."""
    qx, sx = _quantize_cols(x)
    qg, sg = g_cols[0], blocked_scales(g_cols[1].T[None])
    return _dense_gemm(qx, sx, qg, sg, a_spec=_MN_MAJOR, b_spec=_MN_MAJOR, rows=x.shape[1], cols=qg.shape[1])


def _axes(entry) -> tuple[str, ...]:
    if entry is None:
        return ()
    return (entry,) if isinstance(entry, str) else tuple(entry)


def _token_axes_only(w_spec: P, batch_axes: tuple[str, ...]) -> P:
    """``w_spec`` with every mesh axis that does not shard the tokens dropped (the gradient is resharded to it after)."""
    entries = []
    for entry in w_spec:
        axes = tuple(axis for axis in _axes(entry) if axis in batch_axes)
        entries.append(axes if axes else None)
    return P(*entries)


def _reduce_partial_to_spec(dw: jax.Array, w_spec: P, batch_axes: tuple[str, ...]) -> jax.Array:
    """Sum a per-shard partial weight gradient over the token axes, leaving it sharded as ``w_spec`` (token axes
    only, see ``_token_axes_only``)."""
    reduced: list[str] = []
    for dim, entry in enumerate(w_spec):
        axes = _axes(entry)
        if not axes:
            continue
        dw = jax.lax.psum_scatter(dw, axes, scatter_dimension=dim, tiled=True)
        reduced.extend(axes)
    rest = tuple(axis for axis in batch_axes if axis not in reduced)
    return jax.lax.psum(dw, rest) if rest else dw


def _local_size(size: int, entry) -> int:
    mesh = get_abstract_mesh()
    return size // math.prod(mesh.shape[axis] for axis in _axes(entry))


def _gathered_mx_weight(w: jax.Array, along: str) -> tuple[jax.Array, jax.Array]:
    """``w[K, N]`` quantized to MXFP8 along ``K`` (``along="k"``) or ``N`` (``"n"``) and gathered to every device:
    e4m3 values ``[K, N]`` and uint8 exponents. Each device quantizes its own FSDP shard and the gather moves e4m3
    bytes, half the bf16 weight's. A shard that does not hold whole 32-element blocks is gathered first instead."""
    spec = tuple(jax.typeof(w).sharding.spec) + (None,) * (2 - len(tuple(jax.typeof(w).sharding.spec)))
    quantize = mx_quantize_cols if along == "k" else mx_quantize_rows
    axis = 0 if along == "k" else 1
    if _local_size(w.shape[axis], spec[axis]) % MX_BLOCK:
        values, exponents = quantize(reshard(w, P(None, None)))
        return values, exponents

    def local(wl):
        values, exponents = quantize(wl)
        return jax.lax.bitcast_convert_type(values, jnp.uint8), exponents

    values, exponents = jax.shard_map(local, in_specs=P(*spec), out_specs=(P(*spec), P(*spec)), check_vma=False)(w)
    values, exponents = reshard(values, P(None, None)), reshard(exponents, P(None, None))
    return jax.lax.bitcast_convert_type(values, jnp.float8_e4m3fn), exponents


def _token_spec(x_spec: P, ndim: int) -> P:
    """``x``'s spec padded to ``ndim`` with the feature dim unsharded (the kernels need whole rows)."""
    entries = tuple(x_spec) + (None,) * (ndim - len(tuple(x_spec)))
    return P(*entries[:-1], None)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def mx_linear(x: jax.Array, w: jax.Array, out_sharding: P | None = None, fused_grad_quant: bool = False) -> jax.Array:
    """``x[..., K] @ w[K, N]`` with MXFP8 operands in all three GEMMs, on QuACK's block-scaled SM100 kernel.

    Tokens may be sharded on any leading axes; a sharded feature dim of ``x`` is gathered first. ``w`` is gathered like the
    einsum it replaces would gather it, and its gradient comes back sharded as ``w``. Every GEMM quantizes both of
    its operands along its own contraction axis (the weight gradient along tokens), as ``levanter.grug.mxfp8``
    simulates. ``fused_grad_quant`` quantizes the output gradient along rows and columns in one pass (1.22x faster,
    but both quantized copies are live at once)."""
    return _mx_linear_fwd(x, w, out_sharding, fused_grad_quant)[0]


def _mx_linear_fwd(x, w, out_sharding, fused_grad_quant):
    del fused_grad_quant
    x_spec = _token_spec(jax.typeof(x).sharding.spec, x.ndim)
    k, n = w.shape

    def local(xl, wv, we):
        return mxfp8_matmul(xl.reshape(-1, k), wv, we).reshape(*xl.shape[:-1], n)

    # The Pallas quantizers' outputs are not vma-annotated, so skip the varying-axes check.
    replicated = P(None, None)
    y = jax.shard_map(local, in_specs=(x_spec, replicated, replicated), out_specs=x_spec, check_vma=False)(
        reshard(x, x_spec), *_gathered_mx_weight(w, "k")
    )
    if out_sharding is not None:
        y = reshard(y, out_sharding)
    return y, (x, w)


def _mx_linear_bwd(out_sharding, fused_grad_quant, res, g):
    del out_sharding
    x, w = res
    x_orig_spec = jax.typeof(x).sharding.spec
    x_spec = _token_spec(x_orig_spec, x.ndim)
    w_spec = jax.typeof(w).sharding.spec
    batch_axes = tuple(axis for entry in x_spec for axis in _axes(entry))
    dw_spec = _token_axes_only(w_spec, batch_axes)
    k, n = w.shape
    g = reshard(g, x_spec)

    def local(xl, gl, wv, we):
        x2, g2 = xl.reshape(-1, k), gl.reshape(-1, n)
        # The separate passes keep one quantized copy live at a time; the fused pass pushed the GB200 recipe with the
        # bigram table over the NCCL memory edge (lc1-gb200-profrc).
        g_rows, g_cols = quantize_rows_cols(g2) if fused_grad_quant else (quantize_rows(g2), None)
        dx = mxfp8_matmul_dgrad(g_rows, wv, we).reshape(xl.shape)
        dw = mxfp8_matmul_wgrad(x2, quantize_cols(g2) if g_cols is None else g_cols)
        dw = _reduce_partial_to_spec(dw, dw_spec, batch_axes)
        return dx, dw

    replicated = P(None, None)
    dx, dw = jax.shard_map(
        local, in_specs=(x_spec, x_spec, replicated, replicated), out_specs=(x_spec, dw_spec), check_vma=False
    )(reshard(x, x_spec), g, *_gathered_mx_weight(w, "n"))
    dx, dw = reshard(dx, x_orig_spec), reshard(dw, w_spec)
    return dx.astype(x.dtype), dw.astype(w.dtype)


mx_linear.defvjp(_mx_linear_fwd, _mx_linear_bwd)
