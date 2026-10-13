# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""MXFP8 (e4m3 values, one e8m0 scale per 32 contraction elements) grouped GEMMs on QuACK's SM100 kernels.

The expert MLP's activation-path GEMMs run block-scaled. Quantization rides in producers: the ReLU^2 up-projection
writes its postactivation as e4m3 plus scales from its epilogue, the ReLU^2 backward GEMM writes ``d pre`` the same
way, and token rows are quantized by the XLA fusion that produces them (before the dispatch transport).

Scale layout. QuACK reads scales in the blocked ``(L, rm, rk, 32, 4, 4)`` layout, one contiguous 512-byte atom per
128 rows x 4 scale columns. For a varlen-m operand (rows grouped by expert) the scales live in one row-padded
buffer, ``L = 1``, where group ``e``'s rows start at tile ``cu[e] // 128 + e``, so every group starts on a tile.
``padded_scale_rows`` builds that layout from per-row scales, and the epilogue quantizers write the same layout.
"""

import functools

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
from cutlass import Float32
from quack.activation import act_fn_map, dact_fn_map
from quack.epilogue.frontend import gemm_epilogue
from quack.epilogue.ops import ColVecReduce, Scalar, TileStore
from quack.epilogue.quantize_out import BlockScaleFactorStore
from quack.gemm_default_epi import GemmDefaultEpiMixin, GemmDefaultSm100
from quack.gemm_tvm_ffi_utils import make_scheduler_args, make_varlen_args

from levanter.cutlass_kernel_cache import cute_launcher_factory, cutlass_call
from levanter.grug.mxfp8 import MX_BLOCK, mx_dequantize_rows, mx_quantize_cols, mx_quantize_rows, mx_unpack_rows
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


def mx_weights(w: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Group weights ``[E, N, K]`` (contraction last) -> e4m3 ``[E, N, K]`` and blocked e8m0 scales."""
    values, scales = mx_quantize_rows(w)
    return values, blocked_scales(scales)


_TS = cjax.TensorSpec
_ROWS_SPEC = _TS(divisibility=(1, 16), static=False)
_WEIGHT_SPEC = _TS(divisibility=(1, 1, 16), static=False)
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


def mxfp8_grouped_gemm(a, a_scales, b, b_scales, cu_seqlens, *, out_dtype=jnp.bfloat16, max_swizzle=8):
    """``a[M, K] @ b[e, N, K]^T`` per row group, e4m3 operands with blocked e8m0 scales; rows past ``cu[-1]``
    unspecified. ``a_scales`` is the varlen-m layout, ``b_scales`` ``[E, rn, rk, 32, 4, 4]``."""
    launcher = _build_plain_launcher(max_active_clusters=_max_active_clusters(MXFP8_CLUSTER), max_swizzle=max_swizzle)
    call = cutlass_call(
        launcher,
        output_shape_dtype=jax.ShapeDtypeStruct((a.shape[0], b.shape[1]), out_dtype),
        input_spec=(_ROWS_SPEC, _WEIGHT_SPEC, _SCALE_SPEC, _SCALE_SPEC, _INDEX_SPEC),
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


def mxfp8_relu2_grouped_gemm(a, a_scales, b, b_scales, cu_seqlens, *, max_swizzle=8):
    """Block-scaled ``pre = a @ b^T`` with ``relu(pre)^2`` quantized in the epilogue.

    Returns ``(pre bf16 [M, N], post e4m3 [M, N], post scales)``, the scales in the varlen-m layout."""
    rows, cols = a.shape[0], b.shape[1]
    launcher = _build_relu2_launcher(max_active_clusters=_max_active_clusters(MXFP8_CLUSTER), max_swizzle=max_swizzle)
    call = cutlass_call(
        launcher,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((rows, cols), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, cols), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct(varlen_output_scale_shape(rows, cols, cu_seqlens.shape[0] - 1), jnp.float8_e8m0fnu),
        ),
        input_spec=(_ROWS_SPEC, _WEIGHT_SPEC, _SCALE_SPEC, _SCALE_SPEC, _INDEX_SPEC),
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
        input_spec=(_ROWS_SPEC, _WEIGHT_SPEC, _SCALE_SPEC, _SCALE_SPEC, _INDEX_SPEC, _OUT_SPEC),
        output_spec=(_OUT_SPEC, _ROWS_SPEC, _SCALE_SPEC, _TS(static=False)),
        use_static_tensors=False,
    )
    post, d_pre_q, d_pre_scales, dot_partials = call(dy, b, dy_scales, b_scales, cu_seqlens.astype(jnp.int32), pre)
    return post, d_pre_q, d_pre_scales, jnp.sum(dot_partials, axis=-1)


def relu2_expert_mlp_mxfp8_fwd(x_packed, w_up, w_down, cu):
    """``relu(x W_up)^2 W_down`` with MXFP8 operands; ``x_packed`` is ``mx_pack_rows`` of the rows, as dispatched.

    Same contract as ``_relu2_expert_mlp_quack_fwd``: rows past ``cu[-1]`` are unspecified."""
    values, exponents = mx_unpack_rows(x_packed, w_up.shape[1])
    pre, post, post_scales = mxfp8_relu2_grouped_gemm(
        values, padded_scale_rows(exponents, cu), *mx_weights(jnp.swapaxes(w_up, 1, 2)), cu
    )
    y = mxfp8_grouped_gemm(post, post_scales, *mx_weights(jnp.swapaxes(w_down, 1, 2)), cu)
    return y, (x_packed, w_up, w_down, pre, cu)


def relu2_expert_mlp_mxfp8_backward_input(res, dy):
    """``(dx, <y, dy>, wgrad_args)`` of ``relu2_expert_mlp_mxfp8_fwd``; ``wgrad_args`` feed the bf16
    ``_relu2_expert_mlp_quack_wgrads`` with the dequantized rows the forward multiplied."""
    x_packed, w_up, w_down, pre, cu = res
    dtype = dy.dtype
    # Only MXFP8 copies of the row tensors stay live through the backward (the bf16 ``dy`` dies here); the bf16
    # weight-gradient GEMMs read short-lived dequantized copies, the values the other GEMMs multiplied.
    dy_values, dy_exponents = mx_quantize_rows(dy)
    post, d_pre_values, d_pre_scales, output_dot = mxfp8_drelu2_grouped_gemm(
        dy_values, padded_scale_rows(dy_exponents, cu), *mx_weights(w_down), pre, cu
    )
    dx = mxfp8_grouped_gemm(d_pre_values, d_pre_scales, *mx_weights(w_up), cu)
    rows, width = d_pre_values.shape
    d_pre = mx_dequantize_rows(d_pre_values, unpadded_scale_rows(d_pre_scales, cu, rows, width // MX_BLOCK), dtype)
    x = mx_dequantize_rows(*mx_unpack_rows(x_packed, w_up.shape[1]), dtype)
    return dx, output_dot, (x, d_pre, post, mx_dequantize_rows(dy_values, dy_exponents, dtype), cu)


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
    values, exponents = mx_quantize_cols(x)
    return values, blocked_scales(exponents.T[None])


def _quantize_rows(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    values, exponents = mx_quantize_rows(x)
    return values, blocked_scales(exponents[None])


def mxfp8_matmul(x: jax.Array, w: jax.Array) -> jax.Array:
    """Local ``x[T, K] @ w[K, N]`` with both operands MXFP8 along ``K``."""
    qx, sx = _quantize_rows(x)
    qw, sw = _quantize_rows(w.T)
    return _dense_gemm(qx, sx, qw, sw, a_spec=_K_MAJOR, b_spec=_K_MAJOR, rows=x.shape[0], cols=w.shape[1])


def mxfp8_matmul_dgrad(g: jax.Array, w: jax.Array) -> jax.Array:
    """Local ``g[T, N] @ w[K, N]^T`` with both operands MXFP8 along ``N``."""
    qg, sg = _quantize_rows(g)
    qw, sw = _quantize_rows(w)
    return _dense_gemm(qg, sg, qw, sw, a_spec=_K_MAJOR, b_spec=_K_MAJOR, rows=g.shape[0], cols=w.shape[0])


def mxfp8_matmul_wgrad(x: jax.Array, g: jax.Array) -> jax.Array:
    """Local ``x[T, K]^T @ g[T, N]`` with both operands MXFP8 along ``T``."""
    qx, sx = _quantize_cols(x)
    qg, sg = _quantize_cols(g)
    return _dense_gemm(qx, sx, qg, sg, a_spec=_MN_MAJOR, b_spec=_MN_MAJOR, rows=x.shape[1], cols=g.shape[1])


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


def _token_spec(x_spec: P, ndim: int) -> P:
    """``x``'s spec padded to ``ndim`` with the feature dim unsharded (the kernels need whole rows)."""
    entries = tuple(x_spec) + (None,) * (ndim - len(tuple(x_spec)))
    return P(*entries[:-1], None)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def mx_linear(x: jax.Array, w: jax.Array, out_sharding: P | None = None) -> jax.Array:
    """``x[..., K] @ w[K, N]`` with MXFP8 operands in all three GEMMs, on QuACK's block-scaled SM100 kernel.

    Tokens may be sharded on any leading axes; a sharded feature dim of ``x`` is gathered first. ``w`` is gathered like the
    einsum it replaces would gather it, and its gradient comes back sharded as ``w``. Every GEMM quantizes both of
    its operands along its own contraction axis (the weight gradient along tokens), as ``levanter.grug.mxfp8``
    simulates."""
    return _mx_linear_fwd(x, w, out_sharding)[0]


def _mx_linear_fwd(x, w, out_sharding):
    x_spec = _token_spec(jax.typeof(x).sharding.spec, x.ndim)
    k, n = w.shape

    def local(xl, wl):
        return mxfp8_matmul(xl.reshape(-1, k), wl).reshape(*xl.shape[:-1], n)

    y = jax.shard_map(local, in_specs=(x_spec, P(None, None)), out_specs=x_spec)(
        reshard(x, x_spec), reshard(w, P(None, None))
    )
    if out_sharding is not None:
        y = reshard(y, out_sharding)
    return y, (x, w)


def _mx_linear_bwd(out_sharding, res, g):
    del out_sharding
    x, w = res
    x_orig_spec = jax.typeof(x).sharding.spec
    x_spec = _token_spec(x_orig_spec, x.ndim)
    w_spec = jax.typeof(w).sharding.spec
    batch_axes = tuple(axis for entry in x_spec for axis in _axes(entry))
    dw_spec = _token_axes_only(w_spec, batch_axes)
    k, n = w.shape
    g = reshard(g, x_spec)

    def local(xl, gl, wl):
        x2, g2 = xl.reshape(-1, k), gl.reshape(-1, n)
        dx = mxfp8_matmul_dgrad(g2, wl).reshape(xl.shape)
        dw = _reduce_partial_to_spec(mxfp8_matmul_wgrad(x2, g2), dw_spec, batch_axes)
        return dx, dw

    dx, dw = jax.shard_map(local, in_specs=(x_spec, x_spec, P(None, None)), out_specs=(x_spec, dw_spec))(
        reshard(x, x_spec), g, reshard(w, P(None, None))
    )
    dx, dw = reshard(dx, x_orig_spec), reshard(dw, w_spec)
    return dx.astype(x.dtype), dw.astype(w.dtype)


mx_linear.defvjp(_mx_linear_fwd, _mx_linear_bwd)
