# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure-JAX MXFP8 quantization + sm100 scale-factor layout helpers.

Extracted verbatim from ``adapter.py`` (MXFP8-004c) so the integration op and
CPU-side tests can import the quantize/swizzle math without pulling in the
CuTe DSL (``cutlass``) dependency. ``adapter.py`` re-exports everything here,
so all existing call sites keep working.

Contents: round-up e8m0 quantizers (``quantize_mxfp8`` /
``quantize_mxfp8_tokens``, matching jax's scaled_matmul_stablehlo), the exact
e8m0 decode, naive per-group swizzle assembly (``build_sfa`` / ``build_sfb`` /
``build_sf_wgrad``), vectorized bit-exact equivalents (``*_fast``), and the
fused dual-orientation producers (``dual_quantize_activation`` /
``dual_quantize_weight``).
"""

import jax
import jax.numpy as jnp
import numpy as np

SF_VEC_SIZE = 32  # MXFP8 block size along K
E4M3_MAX = 448.0


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

    NOT ``jnp.exp2``: on GPU that lowers to the approximate ``ex2`` path,
    which misses the exact power of two on 217 of the 256 exponent bytes (up
    to tens of ulps at the extremes; byte 0 gives 0x003ffff9 instead of
    0x00400000 -- job mxfp8-002c-g11). An inexact scale silently perturbs
    tie-rounding in ``quantize_mxfp8`` (g10: adversarial denormal/huge blocks
    rounded ties away instead of RTNE), while the grouped GEMM hardware
    applies the EXACT e8m0 scale -- the bit-exact decode is the semantically
    correct reference.
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
