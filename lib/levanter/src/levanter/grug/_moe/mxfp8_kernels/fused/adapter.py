# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX adapters for the vendored cudnn-frontend fused MXFP8 MoE grouped kernels.

Wraps the four Blackwell (sm100) CuTeDSL kernel classes in this package as JAX
calls via ``cutlass.jax.cutlass_call``, following the proven pattern from
``../mxfp8_grouped/adapter.py`` (gmem pointer rebuild at the launcher boundary,
``CUTE_DSL_ARCH=sm_100a``, hardcoded SM count).

The fused layer pipeline (grug row-13 shapes: D=2560 model dim, F=1280
intermediate, N2=2F=2560 interleaved gate/up):

- ``mxfp8_fused_swiglu``  fwd w13: x_q[M,D] @ w13i_q[E,N2,D] -> SwiGLU ->
  c[M,N2] bf16 + h quantized in BOTH orientations (+ swizzled SFs).
- ``mxfp8_fused_gemm``    plain grouped GEMM, bf16 out (fwd w2, dgrad w13).
- ``mxfp8_fused_dswiglu`` bwd: g2_q[M,D] @ w2_q^T -> dSwiGLU vs the fwd
  pre-activation c -> dc[M,N2] quantized in BOTH orientations (+ SFs).
- ``mxfp8_fused_wgrad``   2Dx2D grouped wgrad from the column-quantized copies.

Contiguous padded-offset contract (swiglu / dswiglu / gemm): tokens are laid
out per expert in one contiguous buffer, each expert's row count padded to a
multiple of 256 (``FIX_PAD_SIZE``); ``offs`` are int32 END offsets in that
padded buffer (offs[-1] == padded M). The wgrad kernel instead takes RAW
cumsum offsets (any group sizes) with online SF tensormaps.

Weight interleave contract: the fused SwiGLU pairs gate/up columns in
alternating 32-column blocks along the N2 axis (blocks 0,2,4,.. = gate,
1,3,5,.. = up); w13 must be relaid out accordingly (see the bench's
``interleave_w13``). All alpha/beta/prob/norm_const kernel knobs are pinned to
their neutral values (1.0) here; per-expert amax outputs are disabled (fp8
output path emits SFD tensors instead).

SF layouts (byte-compatible with the ``mxfp8_grouped`` builders):

- row orientation: whole-matrix 32x4x4 swizzle of the (rows, K/32) scale
  matrix over the padded token buffer (== ``build_sfa_fast`` when every group
  is a multiple of 128 rows -- always true under 256-padding).
- column orientation (``discrete_col_sfd=True``): per-expert whole swizzles of
  the (features, tokens_g/32) scale matrices, concatenated in expert order
  (== ``build_sf_wgrad_fast`` / what ``mxfp8_fused_wgrad`` consumes).

Each ``*_plan`` function returns the (launcher, input_spec, output_spec,
out_shapes) quadruple; the public wrappers feed it to ``cutlass_call``, and
the CPU-only compile smoke feeds it to ``build_function_spec`` +
``get_or_compile_kernel`` directly (no GPU needed with CUTE_DSL_ARCH=sm_100a).
"""

import os
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
import jax
import jax.numpy as jnp

from .grouped_gemm_dswiglu_quant import (  # noqa: E402
    BlockScaledContiguousGroupedGemmKernel as DSwigluKernel,
)
from .grouped_gemm_quant import BlockScaledMoEGroupedGemmQuantKernel  # noqa: E402
from .grouped_gemm_swiglu_quant import (  # noqa: E402
    BlockScaledContiguousGroupedGemmKernel as SwigluKernel,
)
from .moe_blockscaled_grouped_gemm_wgrad import BlockScaledMoEGroupedGemmWgradKernel  # noqa: E402
from .moe_utils import MoEWeightMode, WGradInputOrder  # noqa: E402

_BLACKWELL_CUTE_ARCH = "sm_100a"
_B200_SMS = 148
SF_VEC_SIZE = 32
FIX_PAD_SIZE = 256
_WGRAD_DESC_SLOTS = 2  # sfa + sfb expert-wise TMA descriptors (dense Tensor2D)
_TMA_DESC_BYTES = 128

_TS = cjax.TensorSpec
_FLAT = _TS(mode=(0,), static=True)
_2D = _TS(mode=(0, 1), static=True)
_3D = _TS(mode=(0, 1, 2), static=True)


def ensure_blackwell_arch() -> None:
    """Fail fast on non-Blackwell GPUs and pin the CuTe trace architecture."""
    device = jax.devices()[0]
    compute_capability = str(device.compute_capability)
    if not compute_capability.startswith("10"):
        raise RuntimeError(
            f"MXFP8 grouped GEMMs require a Blackwell (sm100) GPU; got {device} "
            f"with compute_capability={compute_capability!r}"
        )
    os.environ.setdefault("CUTE_DSL_ARCH", _BLACKWELL_CUTE_ARCH)


def _as_gmem_tensor(t):
    """Rebuild an FFI tensor with a global-memory pointer."""
    if t.iterator.memspace == cute.AddressSpace.gmem:
        return t
    pointer = cute.make_ptr(
        t.element_type,
        t.iterator.toint(),
        cute.AddressSpace.gmem,
        assumed_align=t.iterator.alignment,
    )
    return cute.make_tensor(pointer, t.layout)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _round_up(a: int, b: int) -> int:
    return _ceil_div(a, b) * b


def sf_row_bytes(m: int, k: int) -> int:
    """Byte count of a row-orientation swizzled SF tensor for a (m, k) operand."""
    return _round_up(m, 128) * _round_up(_ceil_div(k, SF_VEC_SIZE), 4)


def sf_col_bytes(m: int, k: int) -> int:
    """Byte count of the column-orientation (2Dx2D) SF tensor: same count, swapped roles."""
    return _round_up(k, 128) * _round_up(_ceil_div(m, SF_VEC_SIZE), 4)


def _relayout(t, shape, stride):
    """Rebuild ``t`` as a gmem tensor with an exact static (shape, stride) layout."""
    g = _as_gmem_tensor(t)
    return cute.make_tensor(g.iterator, cute.make_layout(shape, stride=stride))


@dataclass(frozen=True)
class CallPlan:
    """Everything cutlass_call (or the CPU compile smoke) needs for one kernel."""

    launcher: object
    input_spec: tuple
    output_spec: tuple
    out_shapes: tuple

    def build(self):
        return cjax.cutlass_call(
            self.launcher,
            output_shape_dtype=self.out_shapes,
            input_spec=self.input_spec,
            output_spec=self.output_spec,
            use_static_tensors=True,
            compile_options=(cute.GPUArch(_BLACKWELL_CUTE_ARCH),),
        )


def _max_active_clusters(cluster_shape_mn: tuple[int, int]) -> int:
    return _B200_SMS // (cluster_shape_mn[0] * cluster_shape_mn[1])


# --------------------------------------------------------------------------- #
# fwd w13: grouped GEMM + SwiGLU + dual-orientation MXFP8 quantize
# --------------------------------------------------------------------------- #


def swiglu_plan(
    m: int,
    d: int,
    n2: int,
    e: int,
    *,
    c_dtype=jnp.bfloat16,
    mma_tiler_mn: tuple[int, int] = (256, 256),
    cluster_shape_mn: tuple[int, int] = (2, 1),
    vector_f32: bool = False,
) -> CallPlan:
    n_out = n2 // 2
    use_2cta = mma_tiler_mn[0] == 256
    mac = _max_active_clusters(cluster_shape_mn)

    class Launcher:
        @cute.jit
        def __call__(self, stream, mA, mB, mSFA, mSFB, mOffs, mAlpha, mProb, mNorm, mC, mD, mDCol, mSFDRow, mSFDCol):
            a3 = _relayout(mA, (m, d, 1), (d, 1, m * d))
            b3 = _relayout(mB, (n2, d, e), (d, 1, n2 * d))
            c3 = _relayout(mC, (m, n2, 1), (n2, 1, m * n2))
            d3 = _relayout(mD, (m, n_out, 1), (n_out, 1, m * n_out))
            dcol3 = _relayout(mDCol, (m, n_out, 1), (n_out, 1, m * n_out))
            prob3 = _relayout(mProb, (m, 1, 1), (1, 1, 1))
            kernel = SwigluKernel(
                sf_vec_size=SF_VEC_SIZE,
                acc_dtype=cutlass.Float32,
                use_2cta_instrs=use_2cta,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                vector_f32=vector_f32,
                generate_sfd=True,
                discrete_col_sfd=True,
                expert_cnt=e,
            )
            kernel(
                a3,
                b3,
                c3,
                d3,
                dcol3,
                _as_gmem_tensor(mSFA),
                _as_gmem_tensor(mSFB),
                _as_gmem_tensor(mSFDRow),
                _as_gmem_tensor(mSFDCol),
                None,  # amax (bf16-output path only)
                _as_gmem_tensor(mNorm),
                _as_gmem_tensor(mOffs),
                _as_gmem_tensor(mAlpha),
                prob3,
                mac,
                stream,
            )

    out_shapes = (
        jax.ShapeDtypeStruct((m, n2), c_dtype),
        jax.ShapeDtypeStruct((m, n_out), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((m, n_out), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((sf_row_bytes(m, n_out),), jnp.float8_e8m0fnu),
        jax.ShapeDtypeStruct((sf_col_bytes(m, n_out),), jnp.float8_e8m0fnu),
    )
    return CallPlan(
        launcher=Launcher(),
        input_spec=(_2D, _3D, _FLAT, _FLAT, _FLAT, _FLAT, _FLAT, _FLAT),
        output_spec=(_2D, _2D, _2D, _FLAT, _FLAT),
        out_shapes=out_shapes,
    )


def mxfp8_fused_swiglu(
    x_q,
    x_sf,
    w13i_q,
    w13i_sf,
    offs,
    *,
    c_dtype=jnp.bfloat16,
    mma_tiler_mn: tuple[int, int] = (256, 256),
    cluster_shape_mn: tuple[int, int] = (2, 1),
    vector_f32: bool = False,
):
    """Fused fwd w13: ``swiglu(x @ w13i)`` with a dual-orientation MXFP8 epilogue.

    Args:
        x_q: (M, D) float8_e4m3fn, K-contiguous, experts contiguous with
            256-padded rows.
        x_sf: row-orientation swizzled scales of x (flat or 2D, bytes only).
        w13i_q: (E, N2, D) float8_e4m3fn, D contiguous; the gate/up columns of
            the N2 axis interleaved in 32-column blocks (gate even, up odd),
            quantized along D.
        w13i_sf: swizzled per-expert scales (``build_sfb`` layout on the
            (E, N2, D//32) scale tensor).
        offs: (E,) int32 padded END offsets, each a multiple of 256.

    Returns:
        (c, h, h_col, sfh_row, sfh_col): c (M, N2) bf16 pre-activation;
        h/h_col (M, N2//2) e4m3 SwiGLU output quantized along features /
        tokens; sfh_row/sfh_col swizzled e8m0 scales (row: whole-matrix
        swizzle; col: per-expert 2Dx2D chunks, wgrad-ready).
    """
    ensure_blackwell_arch()
    m, d = x_q.shape
    e, n2, dk = w13i_q.shape
    assert dk == d, f"K mismatch: x_q D={d}, w13i_q D={dk}"
    assert n2 % 64 == 0, f"N2={n2} must be a multiple of 64 (32-col gate/up block pairs)"
    assert m % FIX_PAD_SIZE == 0
    assert x_sf.dtype == jnp.float8_e8m0fnu and w13i_sf.dtype == jnp.float8_e8m0fnu
    assert offs.shape == (e,)

    plan = swiglu_plan(
        m,
        d,
        n2,
        e,
        c_dtype=c_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        vector_f32=vector_f32,
    )
    alpha = jnp.ones((e,), jnp.float32)
    prob = jnp.ones((m,), jnp.float32)
    norm = jnp.ones((1,), jnp.float32)
    return plan.build()(x_q, w13i_q, x_sf.reshape(-1), w13i_sf.reshape(-1), offs.astype(jnp.int32), alpha, prob, norm)


# --------------------------------------------------------------------------- #
# plain grouped GEMM (bf16 out): fwd w2 and dgrad w13
# --------------------------------------------------------------------------- #


def gemm_plan(
    m: int,
    k: int,
    n: int,
    e: int,
    *,
    out_dtype=jnp.bfloat16,
    mma_tiler_mn: tuple[int, int] = (128, 256),
    cluster_shape_mn: tuple[int, int] = (1, 1),
) -> CallPlan:
    use_2cta = mma_tiler_mn[0] == 256
    assert mma_tiler_mn[1] == 256, "quant kernel requires mma tiler N == 256"
    mac = _max_active_clusters(cluster_shape_mn)

    class Launcher:
        @cute.jit
        def __call__(self, stream, mA, mB, mSFA, mSFB, mOffs, mAlpha, mProb, mD, mWs):
            a3 = _relayout(mA, (m, k, 1), (k, 1, m * k))
            b3 = _relayout(mB, (n, k, e), (k, 1, n * k))
            d3 = _relayout(mD, (m, n, 1), (n, 1, m * n))
            prob3 = _relayout(mProb, (m, 1, 1), (1, 1, 1))
            kernel = BlockScaledMoEGroupedGemmQuantKernel(
                sf_vec_size=SF_VEC_SIZE,
                acc_dtype=cutlass.Float32,
                use_2cta_instrs=use_2cta,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                vectorized_f32=False,
                generate_sfd=False,
                discrete_col_sfd=False,
                enable_bias=False,
                expert_cnt=e,
                weight_mode=MoEWeightMode.DENSE,
                use_dynamic_sched=False,
            )
            kernel(
                a3,
                b3,
                _as_gmem_tensor(mSFB),
                cutlass.Int32(0),  # n (discrete mode only)
                cutlass.Int32(0),  # k (discrete mode only)
                cutlass.Int64(0),  # b_stride_size (discrete mode only)
                cute.nvgpu.OperandMajorMode.K,  # b_major_mode (discrete mode only)
                _as_gmem_tensor(mWs).iterator,
                d3,
                d3,  # d_col: unused without generate_sfd, but a TMA atom is built unconditionally
                _as_gmem_tensor(mSFA),
                None,  # sfd_row
                None,  # sfd_col
                None,  # amax
                None,  # norm_const
                _as_gmem_tensor(mOffs),
                _as_gmem_tensor(mAlpha),
                None,  # row_scale
                None,  # bias
                prob3,
                mac,
                stream,
            )

    out_shapes = (
        jax.ShapeDtypeStruct((m, n), out_dtype),
        jax.ShapeDtypeStruct((16,), jnp.uint8),  # dummy workspace (dense static-sched mode ignores it)
    )
    return CallPlan(
        launcher=Launcher(),
        input_spec=(_2D, _3D, _FLAT, _FLAT, _FLAT, _FLAT, _FLAT),
        output_spec=(_2D, _FLAT),
        out_shapes=out_shapes,
    )


def mxfp8_fused_gemm(
    a_q,
    a_sf,
    b_q,
    b_sf,
    offs,
    *,
    out_dtype=jnp.bfloat16,
    mma_tiler_mn: tuple[int, int] = (128, 256),
    cluster_shape_mn: tuple[int, int] = (1, 1),
):
    """Grouped GEMM ``a[M,K] @ b[E,K,N] -> [M,N]`` (bf16 epilogue, no quantize).

    Same product as ``mxfp8_grouped_mm`` but on the cudnn-frontend
    ``BlockScaledMoEGroupedGemmQuantKernel`` with the contiguous padded-offset
    scheduler, so it consumes the fused kernels' outputs directly.

    Args:
        a_q: (M, K) float8_e4m3fn, K-contiguous, 256-padded rows per expert.
        a_sf: row-orientation swizzled scales (flat bytes).
        b_q: (E, N, K) float8_e4m3fn, K contiguous (the transposed-weight
            buffer; logical operand is w[E, K, N]).
        b_sf: swizzled per-expert scales (``build_sfb`` layout).
        offs: (E,) int32 padded END offsets (multiples of 256).
    """
    ensure_blackwell_arch()
    m, k = a_q.shape
    e, n, kb = b_q.shape
    assert kb == k, f"K mismatch: a_q K={k}, b_q K={kb}"
    assert m % FIX_PAD_SIZE == 0
    assert a_sf.dtype == jnp.float8_e8m0fnu and b_sf.dtype == jnp.float8_e8m0fnu
    assert offs.shape == (e,)

    plan = gemm_plan(m, k, n, e, out_dtype=out_dtype, mma_tiler_mn=mma_tiler_mn, cluster_shape_mn=cluster_shape_mn)
    alpha = jnp.ones((e,), jnp.float32)
    prob = jnp.ones((m,), jnp.float32)
    out = plan.build()(a_q, b_q, a_sf.reshape(-1), b_sf.reshape(-1), offs.astype(jnp.int32), alpha, prob)
    return out[0]


# --------------------------------------------------------------------------- #
# bwd: dgrad-w2 GEMM + dSwiGLU + dual-orientation MXFP8 quantize
# --------------------------------------------------------------------------- #


def dswiglu_plan(
    m: int,
    d: int,
    f: int,
    e: int,
    *,
    c_dtype=jnp.bfloat16,
    mma_tiler_mn: tuple[int, int] = (256, 256),
    cluster_shape_mn: tuple[int, int] = (2, 1),
    vectorized_f32: bool = False,
) -> CallPlan:
    n2 = 2 * f
    use_2cta = mma_tiler_mn[0] == 256
    mac = _max_active_clusters(cluster_shape_mn)
    del c_dtype  # c dtype comes from the input array

    class Launcher:
        @cute.jit
        def __call__(
            self,
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
            mD,
            mDCol,
            mSFDRow,
            mSFDCol,
            mDProb,
        ):
            a3 = _relayout(mA, (m, d, 1), (d, 1, m * d))
            b3 = _relayout(mB, (f, d, e), (d, 1, f * d))
            c3 = _relayout(mC, (m, n2, 1), (n2, 1, m * n2))
            d3 = _relayout(mD, (m, n2, 1), (n2, 1, m * n2))
            dcol3 = _relayout(mDCol, (m, n2, 1), (n2, 1, m * n2))
            prob3 = _relayout(mProb, (m, 1, 1), (1, 1, 1))
            dprob3 = _relayout(mDProb, (m, 1, 1), (1, 1, 1))
            kernel = DSwigluKernel(
                sf_vec_size=SF_VEC_SIZE,
                acc_dtype=cutlass.Float32,
                use_2cta_instrs=use_2cta,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                vectorized_f32=vectorized_f32,
                discrete_col_sfd=True,
                expert_cnt=e,
            )
            kernel(
                a3,
                b3,
                c3,
                d3,
                dcol3,
                _as_gmem_tensor(mSFA),
                _as_gmem_tensor(mSFB),
                _as_gmem_tensor(mSFDRow),
                _as_gmem_tensor(mSFDCol),
                None,  # amax (bf16-output path only)
                _as_gmem_tensor(mNorm),
                _as_gmem_tensor(mOffs),
                _as_gmem_tensor(mAlpha),
                _as_gmem_tensor(mBeta),
                prob3,
                dprob3,
                mac,
                stream,
            )

    out_shapes = (
        jax.ShapeDtypeStruct((m, n2), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((m, n2), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((sf_row_bytes(m, n2),), jnp.float8_e8m0fnu),
        jax.ShapeDtypeStruct((sf_col_bytes(m, n2),), jnp.float8_e8m0fnu),
        jax.ShapeDtypeStruct((m,), jnp.float32),  # dprob (routing-prob grad; unused, contents garbage)
    )
    return CallPlan(
        launcher=Launcher(),
        input_spec=(_2D, _3D, _2D, _FLAT, _FLAT, _FLAT, _FLAT, _FLAT, _FLAT, _FLAT),
        output_spec=(_2D, _2D, _FLAT, _FLAT, _FLAT),
        out_shapes=out_shapes,
    )


def mxfp8_fused_dswiglu(
    g_q,
    g_sf,
    w2_q,
    w2_sf,
    c13,
    offs,
    *,
    mma_tiler_mn: tuple[int, int] = (256, 256),
    cluster_shape_mn: tuple[int, int] = (2, 1),
    vectorized_f32: bool = False,
):
    """Fused bwd: ``dc = dSwiGLU(g2 @ w2^T, c13)`` with dual MXFP8 epilogue.

    Args:
        g_q: (M, D) float8_e4m3fn cotangent of the layer output, quantized
            along D (the contraction axis of the dgrad-w2 product).
        g_sf: row-orientation swizzled scales of g.
        w2_q: (E, F, D) float8_e4m3fn -- w2 in its NATURAL buffer, quantized
            along D (the dgrad orientation copy).
        w2_sf: swizzled per-expert scales (``build_sfb`` on (E, F, D//32)).
        c13: (M, N2) pre-activation from ``mxfp8_fused_swiglu`` (bf16).
        offs: (E,) int32 padded END offsets (multiples of 256).

    Returns:
        (dc, dc_col, sfdc_row, sfdc_col): dc/dc_col (M, N2) e4m3 gradient wrt
        the interleaved pre-activation, quantized along features / tokens;
        swizzled e8m0 scales (col in per-expert 2Dx2D chunks, wgrad-ready).
    """
    ensure_blackwell_arch()
    m, d = g_q.shape
    e, f, dk = w2_q.shape
    assert dk == d, f"K mismatch: g_q D={d}, w2_q D={dk}"
    mc, n2 = c13.shape
    assert mc == m and n2 == 2 * f, f"c13 {c13.shape} vs (M={m}, 2F={2 * f})"
    assert m % FIX_PAD_SIZE == 0
    assert g_sf.dtype == jnp.float8_e8m0fnu and w2_sf.dtype == jnp.float8_e8m0fnu
    assert offs.shape == (e,)

    plan = dswiglu_plan(
        m, d, f, e, mma_tiler_mn=mma_tiler_mn, cluster_shape_mn=cluster_shape_mn, vectorized_f32=vectorized_f32
    )
    alpha = jnp.ones((e,), jnp.float32)
    beta = jnp.ones((e,), jnp.float32)
    prob = jnp.ones((m,), jnp.float32)
    norm = jnp.ones((1,), jnp.float32)
    dc, dc_col, sf_row, sf_col, _dprob = plan.build()(
        g_q, w2_q, c13, g_sf.reshape(-1), w2_sf.reshape(-1), offs.astype(jnp.int32), alpha, beta, prob, norm
    )
    return dc, dc_col, sf_row, sf_col


# --------------------------------------------------------------------------- #
# 2Dx2D wgrad (raw offsets, online SF tensormaps)
# --------------------------------------------------------------------------- #


def wgrad_plan(
    t: int,
    k: int,
    n: int,
    e: int,
    sf_cols: int,
    *,
    out_dtype=jnp.bfloat16,
    mma_tiler_mn: tuple[int, int] = (128, 256),
    cluster_shape_mn: tuple[int, int] = (1, 1),
) -> CallPlan:
    use_2cta = mma_tiler_mn[0] == 256
    mac = _max_active_clusters(cluster_shape_mn)
    ws_bytes = _WGRAD_DESC_SLOTS * e * _TMA_DESC_BYTES
    k_pad = _round_up(k, 128)
    n_pad = _round_up(n, 128)

    class Launcher:
        @cute.jit
        def __call__(self, stream, mA, mB, mSFA, mSFB, mOffs, mOut, mWs):
            # x_q buffer (T, K) row-major -> mat_a (hidden=K, tokens=T), m-major.
            a2 = _relayout(mA, (k, t), (1, k))
            # g_q buffer (T, N) row-major -> mat_b (tokens=T, N), n-major.
            b2 = _relayout(mB, (t, n), (n, 1))
            sfa2 = _relayout(mSFA, (k_pad, sf_cols), (sf_cols, 1))
            sfb2 = _relayout(mSFB, (n_pad, sf_cols), (sf_cols, 1))
            out3 = _relayout(mOut, (e, k, n), (k * n, n, 1))
            kernel = BlockScaledMoEGroupedGemmWgradKernel(
                sf_vec_size=SF_VEC_SIZE,
                acc_dtype=cutlass.Float32,
                use_2cta_instrs=use_2cta,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                accumulate_on_output=False,
                expert_cnt=e,
                weight_mode=MoEWeightMode.DENSE,
                input_order=WGradInputOrder.Tensor2D,
            )
            kernel(
                a2,
                b2,
                sfa2,
                sfb2,
                out3,
                _as_gmem_tensor(mOffs),
                _as_gmem_tensor(mWs),
                mac,
                stream,
            )

    out_shapes = (
        jax.ShapeDtypeStruct((e, k, n), out_dtype),
        jax.ShapeDtypeStruct((ws_bytes,), jnp.uint8),
    )
    return CallPlan(
        launcher=Launcher(),
        input_spec=(_2D, _2D, _2D, _2D, _FLAT),
        output_spec=(_3D, _FLAT),
        out_shapes=out_shapes,
    )


def mxfp8_fused_wgrad(
    x_q,
    x_sf,
    g_q,
    g_sf,
    offs,
    *,
    out_dtype=jnp.bfloat16,
    mma_tiler_mn: tuple[int, int] = (128, 256),
    cluster_shape_mn: tuple[int, int] = (1, 1),
):
    """Grouped wgrad ``dw[e] = x[s_e].T @ g[s_e]`` -> (E, K, N) on sm100.

    Args:
        x_q: (T, K) float8_e4m3fn, natural token-major storage, token-axis
            quantized (a ``*_col`` output of the fused kernels or
            ``dual_quantize_activation``).
        x_sf: (round_up(K,128), total_padded_token_cols) e8m0, per-expert
            2Dx2D chunk layout (``build_sf_wgrad`` == fused ``sf*_col``).
        g_q: (T, N) float8_e4m3fn, natural storage, token-axis quantized.
        g_sf: (round_up(N,128), total_padded_token_cols) e8m0.
        offs: (E,) int32 raw cumsum END offsets (offs[-1] == T).
    """
    ensure_blackwell_arch()
    t, k = x_q.shape
    tg, n = g_q.shape
    assert t == tg, f"token mismatch: x_q T={t}, g_q T={tg}"
    (e,) = offs.shape
    assert x_sf.dtype == jnp.float8_e8m0fnu and g_sf.dtype == jnp.float8_e8m0fnu
    assert x_sf.shape[0] == _round_up(k, 128) and g_sf.shape[0] == _round_up(n, 128)
    assert x_sf.shape[1] == g_sf.shape[1], "SFA/SFB padded token columns must match"

    plan = wgrad_plan(
        t,
        k,
        n,
        e,
        int(x_sf.shape[1]),
        out_dtype=out_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
    )
    out = plan.build()(x_q, g_q, x_sf, g_sf, offs.astype(jnp.int32))
    return out[0]
