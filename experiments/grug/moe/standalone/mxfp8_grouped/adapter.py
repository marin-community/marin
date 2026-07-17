# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX adapter for the vendored NVIDIA Blackwell MXFP8 scaled grouped GEMM.

Wraps ``ScaledGroupedGemmKernel`` (vendored from cutlass v4.5.2
``examples/python/CuTeDSL/cute/blackwell/kernel/moe/torch_scaled_grouped_mm.py``)
as a JAX call via ``cutlass.jax.cutlass_call``, following the proven Hopper
``_tma_grouped_adapter`` pattern (branch ``fp8-ragged-cute``).

Forward 2Dx3D scenario only: ``x[M, K] @ w[E, K, N] -> out[M, N]`` where rows of
``x`` are grouped per expert by an ``offs`` cumsum tensor. Operands are MXFP8:
e4m3 data + e8m0 scale factors, one scale per 32 contiguous K elements, with the
scale tensors pre-swizzled into the sm100 32x4x4 block-scaled atom layout
(``to_blocked`` below reproduces the torch harness layout exactly).

Also hosts the JAX quantization + scale-layout helpers used by the bench:
``quantize_mxfp8`` (round-up e8m0, matching jax's scaled_matmul_stablehlo),
``build_sfa`` / ``build_sfb`` (per-group 128-row padded swizzled assembly).
"""

import os

import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
import jax
import jax.numpy as jnp

from .moe_utils import MoEScaledGroupedGemmTensormapConstructor
from .torch_scaled_grouped_mm import ScaledGroupedGemmKernel

# CuTe DSL trace arch for Blackwell (GB200/B200). GPU detection on the FFI
# compile thread can fail; the env var is the only reliable control (compile
# options do NOT set the trace arch -- see NOTES_fp8_cute.md gotcha #3).
_BLACKWELL_CUTE_ARCH = "sm_100a"
# B200 has 148 SMs; the persistent grid launches max_active_clusters CTAs.
# HardwareInfo() needs a live CUDA context on the compile thread (unavailable
# on the FFI path), so this is hardcoded, as the Hopper adapter does for H100.
_B200_SMS = 148

SF_VEC_SIZE = 32  # MXFP8 block size along K
E4M3_MAX = 448.0
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
# Quantization (host/device JAX code)
# --------------------------------------------------------------------------- #


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _round_up(a: int, b: int) -> int:
    return _ceil_div(a, b) * b


def cast_to_e8m0_with_rounding_up(x):
    """f32 -> e8m0 exponent byte with round-UP (mirrors jax scaled_matmul_stablehlo)."""
    temp = x.astype(jnp.float32).view(jnp.uint32)
    exp = temp >> 23
    mant = temp & 0x7FFFFF
    is_ru = jnp.logical_and(
        jnp.logical_and(mant > 0, exp != 0xFE),
        ~jnp.logical_and(exp == 0, mant <= 0x400000),
    )
    exp = jnp.where(is_ru, exp + 1, exp)
    return exp.astype(jnp.uint8)


def e8m0_to_f32(scale_u8):
    return jnp.exp2(scale_u8.astype(jnp.float32) - 127.0)


def quantize_mxfp8(x):
    """Quantize ``x[..., K]`` to MXFP8 along the last axis (K % 32 == 0).

    Returns ``(q, scales)`` with ``q`` float8_e4m3fn of x's shape and ``scales``
    uint8 e8m0 bit patterns of shape ``(..., K // 32)``. Scales are round-up
    ``2^ceil(log2(blockamax / 448))`` per 32-block.
    """
    k = x.shape[-1]
    assert k % SF_VEC_SIZE == 0, f"K={k} not divisible by {SF_VEC_SIZE}"
    xb = x.astype(jnp.float32).reshape(*x.shape[:-1], k // SF_VEC_SIZE, SF_VEC_SIZE)
    amax = jnp.max(jnp.abs(xb), axis=-1, keepdims=True)
    scales = cast_to_e8m0_with_rounding_up(amax / E4M3_MAX)
    scaled = xb / e8m0_to_f32(scales)
    q = jnp.clip(scaled, -E4M3_MAX, E4M3_MAX).astype(jnp.float8_e4m3fn)
    return q.reshape(x.shape), scales.squeeze(-1)


def dequantize_mxfp8(q, scales):
    """Inverse of ``quantize_mxfp8`` (f32 output)."""
    k = q.shape[-1]
    qb = q.astype(jnp.float32).reshape(*q.shape[:-1], k // SF_VEC_SIZE, SF_VEC_SIZE)
    deq = qb * e8m0_to_f32(scales)[..., None]
    return deq.reshape(q.shape)


# --------------------------------------------------------------------------- #
# Scale-factor layout (sm100 32x4x4 block-scaled atom, torch harness parity)
# --------------------------------------------------------------------------- #


def to_blocked(scale_2d):
    """Pad + apply the Blackwell 32_4_4 scale swizzle; returns a flat uint8 array.

    Exact JAX port of the vendored torch harness ``to_blocked``: pad to
    (round_up(rows,128), round_up(cols,4)), then
    view(rb,128,cb,4) -> permute(0,2,1,3) -> reshape(-1,4,32,4) -> swap ->
    reshape(-1,32,16) -> flatten.
    """
    rows, cols = scale_2d.shape
    rb, cb = _ceil_div(rows, 128), _ceil_div(cols, 4)
    padded = scale_2d
    if (rows, cols) != (rb * 128, cb * 4):
        padded = jnp.zeros((rb * 128, cb * 4), jnp.uint8).at[:rows, :cols].set(scale_2d)
    blocks = padded.reshape(rb, 128, cb, 4).transpose(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).swapaxes(1, 2).reshape(-1, 32, 16)
    return rearranged.reshape(-1)


def build_sfa(x_scales, group_sizes):
    """Assemble the swizzled SFA tensor for the 2Dx3D A operand.

    ``x_scales``: uint8 (M, K//32) raw per-token scales; ``group_sizes``: host
    ints summing to M. Each group's block is swizzled independently (rows padded
    to 128) and concatenated: float8_e8m0fnu of shape
    ``(sum(round_up(M_g,128)), round_up(K//32,4))``.
    """
    kcols = x_scales.shape[1]
    parts = []
    start = 0
    for g in group_sizes:
        g = int(g)
        if g > 0:
            parts.append(to_blocked(x_scales[start : start + g]))
        start += g
    flat = jnp.concatenate(parts)
    total_rows = sum(_round_up(int(g), 128) for g in group_sizes)
    sfa = flat.reshape(total_rows, _round_up(kcols, 4))
    return jax.lax.bitcast_convert_type(sfa, jnp.float8_e8m0fnu)


def build_sfb(w_scales):
    """Assemble the swizzled SFB tensor for the 2Dx3D B operand.

    ``w_scales``: uint8 (E, N, K//32) raw per-output-row scales. Per-expert
    swizzle, stacked: float8_e8m0fnu of shape
    ``(E, round_up(N,128) * round_up(K//32,4))``.
    """
    e = w_scales.shape[0]
    parts = [to_blocked(w_scales[i]) for i in range(e)]
    return jax.lax.bitcast_convert_type(jnp.stack(parts), jnp.float8_e8m0fnu)


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
    (observed on job mxfp8-002-g3). The (E,K,N) logical view of the row-major
    (E,N,K) weight buffer is expressed via TensorSpec ``mode`` instead.
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
                scenario="2Dx3D",
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
