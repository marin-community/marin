# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX adapter for the vendored NVIDIA Blackwell MXFP8 scaled grouped GEMM.

Wraps ``ScaledGroupedGemmKernel`` (vendored from cutlass v4.5.2
``examples/python/CuTeDSL/cute/blackwell/kernel/moe/torch_scaled_grouped_mm.py``)
as a JAX call via ``cutlass.jax.cutlass_call``, following the proven Hopper
``_tma_grouped_adapter`` pattern (branch ``fp8-ragged-cute``).

All three MoE ragged_dot products are covered:

- fwd (2Dx3D):   ``x[M, K] @ w[E, K, N] -> out[M, N]`` (``mxfp8_grouped_mm``)
- dgrad (2Dx3D): ``g[M, N] @ w^T[E, N, K] -> dx[M, K]`` -- the same call with
  the weight quantized along N in its natural ``(E, K, N)`` buffer
  (``mxfp8_grouped_dgrad``)
- wgrad (2Dx2D): ``x^T[K, M] grouped-outer g[M, N] -> dw[E, K, N]``
  (``mxfp8_grouped_wgrad``); operands stay in natural token-major storage
  (A m-major / B n-major, both supported by the vendored kernel for fp8)

Rows of the 2D operand are grouped per expert by an ``offs`` cumsum tensor.
Operands are MXFP8: e4m3 data + e8m0 scale factors, one scale per 32 contiguous
elements ALONG THE CONTRACTION AXIS of each product, with the scale tensors
pre-swizzled into the sm100 32x4x4 block-scaled atom layout (``to_blocked``
below reproduces the torch harness layout exactly). Each product therefore
needs a fresh quantization of its operands from the high-precision original.

Also re-exports the JAX quantization + scale-layout helpers used by the bench
(now defined in the cutlass-free ``quantize`` module):
``quantize_mxfp8`` / ``quantize_mxfp8_tokens`` (round-up e8m0, matching jax's
scaled_matmul_stablehlo), naive per-group swizzle assembly (``build_sfa`` /
``build_sfb`` / ``build_sf_wgrad``), vectorized bit-exact equivalents
(``*_fast``), and the fused dual-orientation producers
(``dual_quantize_activation`` / ``dual_quantize_weight``).
"""

import os

import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
import jax
import jax.numpy as jnp

from .moe_utils import MoEScaledGroupedGemmTensormapConstructor

# Quantization + scale-factor layout helpers live in the cutlass-free
# ``quantize`` module (split out in MXFP8-004c so the integration op and CPU
# tests can import them without the CuTe DSL); re-exported here so existing
# call sites (benches, quantize_cute) keep importing from the adapter.
from .quantize import (
    E4M3_MAX,  # noqa: F401
    SF_VEC_SIZE,
    _ceil_div,
    _round_up,
    _to_blocked_block_grid,  # noqa: F401
    build_sf_wgrad,  # noqa: F401
    build_sf_wgrad_fast,  # noqa: F401
    build_sfa,  # noqa: F401
    build_sfa_fast,  # noqa: F401
    build_sfb,  # noqa: F401
    build_sfb_fast,  # noqa: F401
    cast_to_e8m0_with_rounding_up,  # noqa: F401
    dequantize_mxfp8,  # noqa: F401
    dequantize_mxfp8_tokens,  # noqa: F401
    dual_quantize_activation,  # noqa: F401
    dual_quantize_weight,  # noqa: F401
    e8m0_to_f32,  # noqa: F401
    quantize_mxfp8,  # noqa: F401
    quantize_mxfp8_tokens,  # noqa: F401
    sf_wgrad_col_layout,  # noqa: F401
    sfa_row_gather_indices,  # noqa: F401
    to_blocked,  # noqa: F401
)
from .torch_scaled_grouped_mm import ScaledGroupedGemmKernel

# CuTe DSL trace arch for Blackwell (GB200/B200). GPU detection on the FFI
# compile thread can fail; the env var is the only reliable control (compile
# options do NOT set the trace arch -- see NOTES_fp8_cute.md gotcha #3).
_BLACKWELL_CUTE_ARCH = "sm_100a"
# B200 has 148 SMs; the persistent grid launches max_active_clusters CTAs.
# HardwareInfo() needs a live CUDA context on the compile thread (unavailable
# on the FFI path), so this is hardcoded, as the Hopper adapter does for H100.
_B200_SMS = 148

# 128-bit contiguous alignment for fp8 TMA operands (16 fp8 elems).
_FP8_TMA_VEC = 16


def ensure_blackwell_arch() -> None:
    """Fail fast on non-Blackwell GPUs and pin the CuTe trace arch."""
    dev = jax.devices()[0]
    cc = str(dev.compute_capability)
    if not cc.startswith("10"):
        raise RuntimeError(
            f"mxfp8_grouped_mm requires a Blackwell (sm_100) GPU; got {dev} with compute_capability={cc!r}"
        )
    os.environ.setdefault("CUTE_DSL_ARCH", _BLACKWELL_CUTE_ARCH)


# --------------------------------------------------------------------------- #
# cutlass_call launcher
# --------------------------------------------------------------------------- #


def _as_gmem_tensor(t):
    """Rebuild ``t`` with a gmem-space pointer (same address, same layout).

    Only used inside the ``@cute.jit`` launcher trace. If the pointer is
    already gmem this is the identity.
    """
    if t.iterator.memspace == cute.AddressSpace.gmem:
        return t
    ptr = cute.make_ptr(
        t.element_type,
        t.iterator.toint(),
        cute.AddressSpace.gmem,
        assumed_align=t.iterator.alignment,
    )
    return cute.make_tensor(ptr, t.layout)


def _build_launcher(
    *,
    e: int,
    scenario: str = "2Dx3D",
    mma_tiler_mnk: tuple[int, int, int],
    cluster_shape_mnk: tuple[int, int, int],
    max_active_clusters: int,
):
    """Build the stream-first ``@cute.jit`` launcher consumed by ``cutlass_call``.

    Signature: (stream, x_q, w_q, x_sf, w_sf, offs, out, workspace). All
    tensors pass through untouched: cutlass_call builds them as genuine
    gmem-space pointers, and any recast (``recast_tensor``/``recast_ptr``)
    would drop the gmem address space -- the tensormap constructor then fails
    with ``gmem_ptr_to_generic requires pointer in gmem address space``
    (observed on job mxfp8-002-g3). Logical layout permutations (transposed
    weight views, token-major wgrad operands) are expressed via TensorSpec
    ``mode`` instead.
    """

    class MxFp8GroupedLauncher:
        @cute.jit
        def __call__(
            self,
            stream,
            mA: cute.Tensor,
            mB: cute.Tensor,
            mSFA: cute.Tensor,
            mSFB: cute.Tensor,
            mOffs: cute.Tensor,
            mOut: cute.Tensor,
            mWs: cute.Tensor,
        ):
            # Normalize every FFI tensor to a gmem-space pointer at the
            # boundary. The cu13 DSL wheel's make_ptr derives the address
            # space from the LLVM pointer type of the FFI buffer (addrspace
            # 0 = generic), so cutlass.jax hands us generic-space tensors on
            # that stack; the vendored kernel requires gmem (copy_tensormap
            # verifier, gmem_ptr_to_generic). Rebuilding the pointer from the
            # integer address (make_ptr honors mem_space for integer values)
            # matches what the torch/from_dlpack flow produces.
            mA = _as_gmem_tensor(mA)
            mB = _as_gmem_tensor(mB)
            mSFA = _as_gmem_tensor(mSFA)
            mSFB = _as_gmem_tensor(mSFB)
            mOffs = _as_gmem_tensor(mOffs)
            mOut = _as_gmem_tensor(mOut)
            mWs = _as_gmem_tensor(mWs)
            kernel = ScaledGroupedGemmKernel(
                scenario=scenario,
                sf_vec_size=SF_VEC_SIZE,
                accumulate_on_output=False,
                separate_tensormap_init=True,
                consistent_token_padding=False,
                acc_dtype=cutlass.Float32,
                mma_tiler_mnk=mma_tiler_mnk,
                cluster_shape_mnk=cluster_shape_mnk,
                use_2cta_instrs=False,
                fixed_expert_cnt=e,
            )
            kernel(
                mA,
                mB,
                mSFA,
                mSFB,
                mOut,
                mOffs,
                mWs,
                max_active_clusters,
                stream,
            )

    return MxFp8GroupedLauncher()


def mxfp8_grouped_mm(
    x_q,
    x_sf,
    w_q,
    w_sf,
    offs,
    *,
    out_dtype=jnp.bfloat16,
    # (128, 256, 128) measured ~8% faster than (128, 128, 128) at the row-13
    # shapes on GB200 (2200 vs 2027 TF/s w13; 2050 vs 1892 TF/s w2), same
    # numerics (jobs mxfp8-002-g8/g9). N=256 uses the overlapping-accumulator
    # TMEM path. K must stay 128 (sf_vec_size * 4).
    mma_tiler_mnk: tuple[int, int, int] = (128, 256, 128),
    cluster_shape_mnk: tuple[int, int, int] = (1, 1, 1),
):
    """MXFP8 scaled grouped GEMM ``x[M,K] @ w[E,K,N] -> [M,N]`` on sm100.

    Args:
        x_q: (M, K) float8_e4m3fn, K contiguous. Rows grouped per expert.
        x_sf: (sum(round_up(M_g,128)), round_up(K//32,4)) float8_e8m0fnu,
            swizzled scales from ``build_sfa``.
        w_q: (E, N, K) float8_e4m3fn row-major (K contiguous) -- the transposed
            weight layout; logical operand is w[E, K, N].
        w_sf: (E, round_up(N,128) * round_up(K//32,4)) float8_e8m0fnu from
            ``build_sfb``.
        offs: (E,) int32 END offsets (cumsum of group sizes; offs[-1] == M).
        out_dtype: output dtype (bf16 default; f32 accumulate).
    """
    ensure_blackwell_arch()
    m, kx = x_q.shape
    e, n, k = w_q.shape
    assert kx == k, f"K mismatch: x_q K={kx}, w_q K={k}"
    assert k % 128 == 0, f"K={k} must be divisible by 128 (sf_vec_size * 4)"
    assert offs.shape == (e,)
    sfa_cols = x_sf.shape[1]
    sfb_cols = w_sf.shape[1]
    assert sfa_cols == _round_up(_ceil_div(k, SF_VEC_SIZE), 4)
    assert sfb_cols == _round_up(n, 128) * sfa_cols
    assert x_sf.dtype == jnp.float8_e8m0fnu and w_sf.dtype == jnp.float8_e8m0fnu

    # Workspace: expert-wise TMA descriptors + padded scale offsets.
    desc_bytes = MoEScaledGroupedGemmTensormapConstructor.get_workspace_size("2Dx3D", e)
    ws_bytes = desc_bytes + e * 4  # + padded offs (consistent_token_padding=False)

    cluster_size = cluster_shape_mnk[0] * cluster_shape_mnk[1]
    max_active_clusters = _B200_SMS // cluster_size

    launcher = _build_launcher(
        e=e,
        mma_tiler_mnk=mma_tiler_mnk,
        cluster_shape_mnk=cluster_shape_mnk,
        max_active_clusters=max_active_clusters,
    )

    ts = cjax.TensorSpec
    a_spec = ts(mode=(0, 1), divisibility=(1, _FP8_TMA_VEC), static=True)
    # w_q buffer is row-major (E, N, K); the kernel's torch-facing __call__
    # expects mat_b logical (E, K, N) with K stride-1. mode reorders the
    # logical view without materializing a transpose; divisibility is in
    # input-dimension order (before mode).
    b_spec = ts(mode=(0, 2, 1), divisibility=(1, 1, _FP8_TMA_VEC), static=True)
    sfa_spec = ts(mode=(0, 1), static=True)
    sfb_spec = ts(mode=(0, 1), static=True)
    offs_spec = ts(mode=(0,), static=True)
    out_spec = ts(mode=(0, 1), static=True)
    ws_spec = ts(mode=(0,), static=True)

    out_shapes = (
        jax.ShapeDtypeStruct((m, n), out_dtype),
        jax.ShapeDtypeStruct((ws_bytes,), jnp.uint8),
    )

    call = cjax.cutlass_call(
        launcher,
        output_shape_dtype=out_shapes,
        input_spec=(a_spec, b_spec, sfa_spec, sfb_spec, offs_spec),
        output_spec=(out_spec, ws_spec),
        use_static_tensors=True,
        compile_options=(cute.GPUArch(_BLACKWELL_CUTE_ARCH),),
    )
    out = call(x_q, w_q, x_sf, w_sf, offs.astype(jnp.int32))
    return out[0]


def mxfp8_grouped_dgrad(
    g_q,
    g_sf,
    w_q,
    w_sf,
    offs,
    *,
    out_dtype=jnp.bfloat16,
    mma_tiler_mnk: tuple[int, int, int] = (128, 256, 128),
    cluster_shape_mnk: tuple[int, int, int] = (1, 1, 1),
):
    """MXFP8 grouped dgrad ``g[M, N] @ w^T[E, N, K] -> dx[M, K]`` on sm100.

    Structurally the SAME 2Dx3D grouped product as the forward: the transposed
    weight view is what ``mxfp8_grouped_mm`` already expresses via TensorSpec
    mode, so dgrad is the forward call with the OTHER quantized copy of the
    weight. The orientation contract (this is the whole point of the wrapper):

    Args:
        g_q: (M, N) float8_e4m3fn cotangent, quantized along N (its natural
            last axis -- N is the contraction dim here).
        g_sf: swizzled scales from ``build_sfa``/``build_sfa_fast`` on g's
            (M, N//32) scale matrix.
        w_q: (E, K, N) float8_e4m3fn -- the weight quantized along N in its
            NATURAL (E, K, N) row-major buffer (fresh quantization from the
            high-precision original; never transpose-requantize the fwd copy).
        w_sf: (E, round_up(K,128) * round_up(N//32,4)) float8_e8m0fnu from
            ``build_sfb``/``build_sfb_fast`` on the (E, K, N//32) scales.
        offs: (E,) int32 END offsets (cumsum of group sizes; offs[-1] == M).
    """
    return mxfp8_grouped_mm(
        g_q,
        g_sf,
        w_q,
        w_sf,
        offs,
        out_dtype=out_dtype,
        mma_tiler_mnk=mma_tiler_mnk,
        cluster_shape_mnk=cluster_shape_mnk,
    )


def mxfp8_grouped_wgrad(
    x_q,
    x_sf,
    g_q,
    g_sf,
    offs,
    *,
    out_dtype=jnp.bfloat16,
    mma_tiler_mnk: tuple[int, int, int] = (128, 256, 128),
    cluster_shape_mnk: tuple[int, int, int] = (1, 1, 1),
):
    """MXFP8 grouped wgrad ``x^T grouped-outer g -> dw[E, K, N]`` on sm100 (2Dx2D).

    Per expert e with token slice s_e: ``dw[e] = x[s_e].T @ g[s_e]``. The
    contraction axis is TOKENS, so both operands are quantized along the token
    axis (``quantize_mxfp8_tokens``) and stay in their NATURAL (T, D) storage:
    the kernel supports m-major A / n-major B for fp8, expressed via TensorSpec
    mode -- no materialized fp8 transpose. Zero-token experts get zero output
    tiles (the kernel epilogue stores zeros when k_tile_cnt == 0).

    Args:
        x_q: (T, K) float8_e4m3fn, natural row-major, token-axis quantized.
        x_sf: (K, sum(round_up(T_g,128))//32) float8_e8m0fnu from
            ``build_sf_wgrad``/``build_sf_wgrad_fast`` on x's transposed
            (K, T//32) scale matrix. K % 128 == 0 required.
        g_q: (T, N) float8_e4m3fn, natural row-major, token-axis quantized.
        g_sf: (N, sum(round_up(T_g,128))//32) float8_e8m0fnu. N % 128 == 0.
        offs: (E,) int32 END offsets (cumsum of group sizes; offs[-1] == T).
        out_dtype: output dtype for dw (E, K, N); f32 accumulate.
    """
    ensure_blackwell_arch()
    t, kdim = x_q.shape
    tg, n = g_q.shape
    assert t == tg, f"token mismatch: x_q T={t}, g_q T={tg}"
    assert kdim % 128 == 0, f"K={kdim} must be a multiple of 128"
    assert n % 128 == 0, f"N={n} must be a multiple of 128"
    (e,) = offs.shape
    assert x_sf.shape[0] == kdim and g_sf.shape[0] == n
    assert x_sf.shape[1] == g_sf.shape[1], "SFA/SFB padded token columns must match"
    assert x_sf.dtype == jnp.float8_e8m0fnu and g_sf.dtype == jnp.float8_e8m0fnu

    desc_bytes = MoEScaledGroupedGemmTensormapConstructor.get_workspace_size("2Dx2D", e)
    ws_bytes = desc_bytes + e * 4  # + padded offs (consistent_token_padding=False)

    cluster_size = cluster_shape_mnk[0] * cluster_shape_mnk[1]
    max_active_clusters = _B200_SMS // cluster_size

    launcher = _build_launcher(
        e=e,
        scenario="2Dx2D",
        mma_tiler_mnk=mma_tiler_mnk,
        cluster_shape_mnk=cluster_shape_mnk,
        max_active_clusters=max_active_clusters,
    )

    ts = cjax.TensorSpec
    # x_q buffer (T, K): logical mat_a (hidden=K, tokens=T) via mode; A is
    # m-major (hidden stride-1), so per-expert token slices are always
    # 16-elem aligned regardless of group sizes.
    a_spec = ts(mode=(1, 0), divisibility=(1, _FP8_TMA_VEC), static=True)
    # g_q buffer (T, N): logical mat_b (tokens=T, N) is the identity view;
    # B is n-major (N stride-1).
    b_spec = ts(mode=(0, 1), divisibility=(1, _FP8_TMA_VEC), static=True)
    sfa_spec = ts(mode=(0, 1), static=True)
    sfb_spec = ts(mode=(0, 1), static=True)
    offs_spec = ts(mode=(0,), static=True)
    out_spec = ts(mode=(0, 1, 2), static=True)
    ws_spec = ts(mode=(0,), static=True)

    out_shapes = (
        jax.ShapeDtypeStruct((e, kdim, n), out_dtype),
        jax.ShapeDtypeStruct((ws_bytes,), jnp.uint8),
    )

    call = cjax.cutlass_call(
        launcher,
        output_shape_dtype=out_shapes,
        input_spec=(a_spec, b_spec, sfa_spec, sfb_spec, offs_spec),
        output_spec=(out_spec, ws_spec),
        use_static_tensors=True,
        compile_options=(cute.GPUArch(_BLACKWELL_CUTE_ARCH),),
    )
    out = call(x_q, g_q, x_sf, g_sf, offs.astype(jnp.int32))
    return out[0]
