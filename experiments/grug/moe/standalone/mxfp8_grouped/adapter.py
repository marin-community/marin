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

Also hosts the JAX quantization + scale-layout helpers used by the bench:
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
import numpy as np

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
    """Exact e8m0 decode: bit-construct ``2^(e - 127)`` (byte 0 -> subnormal 2^-127).

    NOT ``jnp.exp2``: on GPU that lowers to the approximate ``ex2`` path, which
    can be 1 ulp off the exact power of two at extreme exponents. An inexact
    scale silently perturbs tie-rounding in ``quantize_mxfp8`` (observed on
    GB200: adversarial denormal/huge blocks rounded ties away instead of RTNE,
    job mxfp8-002c-g10), while the grouped GEMM hardware applies the EXACT
    e8m0 scale -- the bit-exact decode is the semantically correct reference.
    """
    e = scale_u8.astype(jnp.uint32)
    bits = jnp.where(e == 0, jnp.uint32(0x00400000), e << 23)
    return jax.lax.bitcast_convert_type(bits, jnp.float32)


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


def quantize_mxfp8_tokens(x):
    """Quantize ``x[T, ...]`` to MXFP8 along the FIRST (token) axis (T % 32 == 0).

    Token-axis quantization for the wgrad (2Dx2D) operands: the contraction
    runs over tokens, so scale blocks span 32 consecutive tokens per feature.
    Returns ``(q, scales)`` with ``q`` float8_e4m3fn of x's shape (natural
    storage, no transpose) and ``scales`` uint8 e8m0 of shape ``(T // 32, ...)``.
    """
    t = x.shape[0]
    assert t % SF_VEC_SIZE == 0, f"T={t} not divisible by {SF_VEC_SIZE}"
    xb = x.astype(jnp.float32).reshape(t // SF_VEC_SIZE, SF_VEC_SIZE, *x.shape[1:])
    amax = jnp.max(jnp.abs(xb), axis=1, keepdims=True)
    scales = cast_to_e8m0_with_rounding_up(amax / E4M3_MAX)
    scaled = xb / e8m0_to_f32(scales)
    q = jnp.clip(scaled, -E4M3_MAX, E4M3_MAX).astype(jnp.float8_e4m3fn)
    return q.reshape(x.shape), scales.squeeze(1)


def dequantize_mxfp8_tokens(q, scales):
    """Inverse of ``quantize_mxfp8_tokens`` (f32 output)."""
    t = q.shape[0]
    qb = q.astype(jnp.float32).reshape(t // SF_VEC_SIZE, SF_VEC_SIZE, *q.shape[1:])
    return (qb * e8m0_to_f32(scales)[:, None]).reshape(q.shape)


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


def build_sf_wgrad(scales_2d, group_sizes):
    """Assemble the swizzled SF tensor for a 2Dx2D (wgrad) operand -- naive.

    ``scales_2d``: uint8 (R, T//32) raw token-axis scales laid out with the
    fixed non-token dim R first (R % 128 == 0; R is M=hidden for the A operand,
    N=intermediate for B). Groups run along the columns (contraction = tokens).
    Per-group column slice swizzled independently (cols padded to 4 = 128
    tokens) and concatenated: float8_e8m0fnu of shape
    ``(R, sum(round_up(tok_g, 128)) // 32)``.
    """
    r = scales_2d.shape[0]
    assert r % 128 == 0, f"non-token dim {r} must be a multiple of 128"
    parts = []
    start = 0
    for g in group_sizes:
        cols = int(g) // SF_VEC_SIZE
        if cols > 0:
            parts.append(to_blocked(scales_2d[:, start : start + cols]))
        start += cols
    flat = jnp.concatenate(parts)
    total_cols = sum(_round_up(int(g) // SF_VEC_SIZE, 4) for g in group_sizes)
    return jax.lax.bitcast_convert_type(flat.reshape(r, total_cols), jnp.float8_e8m0fnu)


# --------------------------------------------------------------------------- #
# Vectorized swizzle (bit-exact vs the naive per-group builders above)
# --------------------------------------------------------------------------- #


def _to_blocked_block_grid(padded):
    """(rows % 128 == 0, cols % 4 == 0) uint8 -> (rb*cb, 512) swizzled atom blocks.

    Each 128x4 tile becomes 512 contiguous output bytes; tile (r, c) lands at
    block index ``r * cb + c``. ``to_blocked(padded)`` equals
    ``_to_blocked_block_grid(padded).reshape(-1)`` when no padding is needed.
    """
    rows, cols = padded.shape
    rb, cb = rows // 128, cols // 4
    blocks = padded.reshape(rb, 128, cb, 4).transpose(0, 2, 1, 3)
    return blocks.reshape(rb * cb, 4, 32, 4).swapaxes(1, 2).reshape(rb * cb, 512)


def sfa_row_gather_indices(group_sizes) -> np.ndarray:
    """Host-side padded-row -> source-row map for ``build_sfa_fast`` (-1 = zero pad).

    Because every group is padded to a whole number of 128-row blocks, the
    whole-matrix swizzle of the row-gathered scales equals the concatenation of
    per-group swizzles, so one gather + one ``_to_blocked_block_grid`` replaces
    the per-group loop.
    """
    idx: list[int] = []
    start = 0
    for g in group_sizes:
        g = int(g)
        idx.extend(range(start, start + g))
        idx.extend([-1] * (_round_up(g, 128) - g))
        start += g
    return np.asarray(idx, dtype=np.int32)


def build_sfa_fast(x_scales, row_idx):
    """Vectorized ``build_sfa``: one row gather + whole-matrix swizzle.

    ``x_scales``: uint8 (M, K//32); ``row_idx``: int32 device array from
    ``sfa_row_gather_indices``. Bit-exact vs ``build_sfa``.
    """
    m, kb = x_scales.shape
    kb4 = _round_up(kb, 4)
    if kb4 != kb:
        x_scales = jnp.pad(x_scales, ((0, 0), (0, kb4 - kb)))
    gathered = jnp.where((row_idx >= 0)[:, None], x_scales[jnp.clip(row_idx, 0, m - 1)], 0)
    flat = _to_blocked_block_grid(gathered).reshape(-1)
    return jax.lax.bitcast_convert_type(flat.reshape(row_idx.shape[0], kb4), jnp.float8_e8m0fnu)


def build_sfb_fast(w_scales):
    """Vectorized ``build_sfb``: batched whole-tensor swizzle (bit-exact)."""
    e, n, kb = w_scales.shape
    assert n % 128 == 0, f"N={n} must be a multiple of 128"
    kb4 = _round_up(kb, 4)
    if kb4 != kb:
        w_scales = jnp.pad(w_scales, ((0, 0), (0, 0), (0, kb4 - kb)))
    rb, cb = n // 128, kb4 // 4
    blocks = w_scales.reshape(e, rb, 128, cb, 4).transpose(0, 1, 3, 2, 4)
    flat = blocks.reshape(e, rb * cb, 4, 32, 4).swapaxes(2, 3).reshape(e, rb * cb * 512)
    return jax.lax.bitcast_convert_type(flat, jnp.float8_e8m0fnu)


def sf_wgrad_col_layout(group_sizes, rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Host-side (column gather map, atom-block permutation) for ``build_sf_wgrad_fast``.

    ``col_idx`` maps padded scale-column -> source column (-1 = zero pad; each
    group's columns are padded to a multiple of 4 = 128 tokens). ``block_perm``
    reorders the whole-matrix 128x4 atom-block grid (row-block-major) into the
    kernel's per-expert-chunk order [expert][row_block][col_block] -- unlike the
    row-grouped case the chunk order is NOT a plain reshape because each
    expert's flat chunk spans all row blocks of only its own columns.
    """
    col_idx: list[int] = []
    group_cbs: list[int] = []
    start = 0
    for g in group_sizes:
        cols = int(g) // SF_VEC_SIZE
        padded = _round_up(cols, 4)
        col_idx.extend(range(start, start + cols))
        col_idx.extend([-1] * (padded - cols))
        group_cbs.append(padded // 4)
        start += cols
    rb = _round_up(rows, 128) // 128
    cb_total = sum(group_cbs)
    perm: list[int] = []
    cb_start = 0
    for gcb in group_cbs:
        for r in range(rb):
            perm.extend(r * cb_total + c for c in range(cb_start, cb_start + gcb))
        cb_start += gcb
    return np.asarray(col_idx, dtype=np.int32), np.asarray(perm, dtype=np.int32)


def build_sf_wgrad_fast(scales_2d, col_idx, block_perm):
    """Vectorized ``build_sf_wgrad``: column gather + swizzle + block permute.

    ``scales_2d``: uint8 (R, T//32) with R % 128 == 0; ``col_idx`` /
    ``block_perm``: int32 device arrays from ``sf_wgrad_col_layout``. Bit-exact
    vs ``build_sf_wgrad``.
    """
    r, tb = scales_2d.shape
    gathered = jnp.where((col_idx >= 0)[None, :], scales_2d[:, jnp.clip(col_idx, 0, tb - 1)], 0)
    flat = _to_blocked_block_grid(gathered)[block_perm].reshape(-1)
    return jax.lax.bitcast_convert_type(flat.reshape(r, col_idx.shape[0]), jnp.float8_e8m0fnu)


# --------------------------------------------------------------------------- #
# Fused dual-orientation producers (one jitted pass per tensor)
# --------------------------------------------------------------------------- #


def dual_quantize_activation(t, row_idx, col_idx, block_perm):
    """Produce BOTH MXFP8 orientations of an activation/cotangent ``t[T, D]``.

    Returns ``(q_row, sf_row, q_col, sf_col)``: ``q_row`` quantized along the
    feature axis with fwd/dgrad-A swizzled scales (``build_sfa`` layout);
    ``q_col`` quantized along the token axis in NATURAL storage (the 2Dx2D
    wgrad operand -- no data transpose) with 2Dx2D swizzled scales. Index
    arrays come from ``sfa_row_gather_indices`` / ``sf_wgrad_col_layout``.
    Designed to be wrapped in a single ``jax.jit``.
    """
    q_row, s_row = quantize_mxfp8(t)
    sf_row = build_sfa_fast(s_row, row_idx)
    q_col, s_col = quantize_mxfp8_tokens(t)
    sf_col = build_sf_wgrad_fast(s_col.T, col_idx, block_perm)
    return q_row, sf_row, q_col, sf_col


def dual_quantize_weight(w):
    """Produce both MXFP8 orientations of a weight ``w[E, K, N]``.

    Returns ``(wq_fwd, sfb_fwd, wq_dgrad, sfb_dgrad)``: the fwd copy quantized
    along K (buffer (E, N, K), for ``mxfp8_grouped_mm``), the dgrad copy
    quantized along N (natural buffer (E, K, N), for ``mxfp8_grouped_dgrad``).
    """
    wq_fwd, s_fwd = quantize_mxfp8(jnp.swapaxes(w, 1, 2))
    wq_dgrad, s_dgrad = quantize_mxfp8(w)
    return wq_fwd, build_sfb_fast(s_fwd), wq_dgrad, build_sfb_fast(s_dgrad)


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
