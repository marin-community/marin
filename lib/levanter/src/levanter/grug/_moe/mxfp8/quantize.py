# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure-JAX reference MXFP8 quantizer and the SM100 scale-factor layouts.

MXFP8 stores E4M3 data with one E8M0 power-of-two scale per block of 32 values. A block's scale is
``2^ceil(log2(max(amax, 1e-4) / 448))``, the smallest power of two that keeps the block inside the
E4M3 range, and the data is cast with saturating round-to-nearest-even. This is TileKernels'
``tile_kernels/torch/cast.py`` with ``round_sf=True``. Scales are returned as raw E8M0 bytes: the
biased exponent, so byte ``b`` decodes to ``2^(b - 127)``.

Row quantization takes blocks of 32 along the last axis, the operand layout of the forward and
data-gradient GEMMs. Token quantization takes blocks of 32 along the first axis, the layout of the
weight-gradient GEMMs, which contract over tokens; its data keeps the natural row-major storage.

The ``build_*`` functions lay raw scale bytes out as the Blackwell block-scaled GEMMs read them,
in tiles of 128 rows by 4 scale columns, 512 bytes each.

Ported from c3cb334f8e:experiments/grug/moe/standalone/mxfp8_grouped/quantize.py.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, UInt8

MXFP8_BLOCK_SIZE = 32
E4M3_MAX = 448.0
# TileKernels' amax clamp for E4M3 with E8M0 scales. It floors every scale at 2^-22, so all-zero
# blocks get a finite scale and E8M0 byte 0 is never produced.
_AMAX_FLOOR = 1e-4
_F32_MANTISSA_BITS = 23
# One swizzle tile of the SM100 scale-factor layout: 128 rows by 4 scale columns, 512 bytes. Every
# dimension a scale matrix tiles must be a multiple of the 128 rows.
SF_TILE_ROWS = 128
_TILE_COLS = 4


def saturating_cast_e4m3(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Round to E4M3 to nearest-even, clamping magnitudes above 448 to 448 instead of NaN."""
    return jnp.clip(x.astype(jnp.float32), -E4M3_MAX, E4M3_MAX).astype(jnp.float8_e4m3fn)


def e8m0_to_f32(scales: UInt8[Array, "..."]) -> Float[Array, "..."]:
    """Decode E8M0 bytes to the exact power of two, built from bits.

    Byte 0 decodes to the subnormal 2^-127. ``jnp.exp2`` is not used because GPU lowers it to an
    approximate instruction that misses the exact power of two for most exponents.
    """
    exponent = scales.astype(jnp.uint32)
    bits = jnp.where(exponent == 0, jnp.uint32(1 << (_F32_MANTISSA_BITS - 1)), exponent << _F32_MANTISSA_BITS)
    return jax.lax.bitcast_convert_type(bits, jnp.float32)


def _e8m0_scales(amax: Float[Array, "..."]) -> UInt8[Array, "..."]:
    """E8M0 bytes of ``2^ceil(log2(max(amax, 1e-4) / 448))`` for finite ``amax``."""
    scale = jnp.maximum(amax, _AMAX_FLOOR) / E4M3_MAX
    bits = jax.lax.bitcast_convert_type(scale, jnp.uint32)
    # The floor keeps `scale` positive and normal, where this rounds the exponent field up whenever
    # the mantissa is nonzero.
    return (((bits - 1) >> _F32_MANTISSA_BITS) + 1).astype(jnp.uint8)


def quantize_mxfp8(x: Float[Array, "... K"]) -> tuple[Float[Array, "... K"], UInt8[Array, "... K/32"]]:
    """Quantize along the last axis in blocks of 32. Returns E4M3 data and raw E8M0 scale bytes.

    Inputs must be finite. Any leading shape works, so weights ``[E, N, K]`` quantize along K.
    """
    k = x.shape[-1]
    assert k % MXFP8_BLOCK_SIZE == 0, f"last axis {k} is not a multiple of {MXFP8_BLOCK_SIZE}"
    blocks = x.astype(jnp.float32).reshape(*x.shape[:-1], k // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
    scales = _e8m0_scales(jnp.max(jnp.abs(blocks), axis=-1))
    q = saturating_cast_e4m3(blocks / e8m0_to_f32(scales)[..., None])
    return q.reshape(x.shape), scales


def dequantize_mxfp8(q: Float[Array, "... K"], scales: UInt8[Array, "... K/32"]) -> Float[Array, "... K"]:
    """Exact f32 inverse of `quantize_mxfp8`'s encoding."""
    k = q.shape[-1]
    blocks = q.astype(jnp.float32).reshape(*q.shape[:-1], k // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
    return (blocks * e8m0_to_f32(scales)[..., None]).reshape(q.shape)


def quantize_mxfp8_tokens(x: Float[Array, "T ..."]) -> tuple[Float[Array, "T ..."], UInt8[Array, "T/32 ..."]]:
    """Quantize along the first (token) axis in blocks of 32 tokens.

    Returns E4M3 data in ``x``'s own layout and raw E8M0 scales of shape ``[T // 32, ...]``.
    Inputs must be finite.
    """
    t = x.shape[0]
    assert t % MXFP8_BLOCK_SIZE == 0, f"token axis {t} is not a multiple of {MXFP8_BLOCK_SIZE}"
    blocks = x.astype(jnp.float32).reshape(t // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE, *x.shape[1:])
    scales = _e8m0_scales(jnp.max(jnp.abs(blocks), axis=1))
    q = saturating_cast_e4m3(blocks / e8m0_to_f32(scales)[:, None])
    return q.reshape(x.shape), scales


def dequantize_mxfp8_tokens(q: Float[Array, "T ..."], scales: UInt8[Array, "T/32 ..."]) -> Float[Array, "T ..."]:
    """Exact f32 inverse of `quantize_mxfp8_tokens`'s encoding."""
    t = q.shape[0]
    blocks = q.astype(jnp.float32).reshape(t // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE, *q.shape[1:])
    return (blocks * e8m0_to_f32(scales)[:, None]).reshape(q.shape)


def _swizzle_tiles(scales: UInt8[Array, "... R C"]) -> UInt8[Array, "... tiles 512"]:
    """Swizzle each 128x4 tile into its 512-byte SM100 atom; tile (i, j) lands at ``i * C/4 + j``.

    Row ``r`` and column ``c`` of a tile go to byte ``(r % 32) * 16 + (r // 32) * 4 + c``.
    """
    *lead, rows, cols = scales.shape
    assert rows % SF_TILE_ROWS == 0, f"scale rows {rows} are not a multiple of {SF_TILE_ROWS}"
    assert cols % _TILE_COLS == 0, f"scale columns {cols} are not a multiple of {_TILE_COLS}"
    tile_grid = (rows // SF_TILE_ROWS) * (cols // _TILE_COLS)
    tiles = scales.reshape(*lead, rows // SF_TILE_ROWS, SF_TILE_ROWS, cols // _TILE_COLS, _TILE_COLS)
    tiles = jnp.swapaxes(tiles, -3, -2).reshape(*lead, tile_grid, SF_TILE_ROWS // 32, 32, _TILE_COLS)
    return jnp.swapaxes(tiles, -3, -2).reshape(*lead, tile_grid, SF_TILE_ROWS * _TILE_COLS)


def build_sfa(row_scales: UInt8[Array, "M K/32"]) -> Array:
    """Swizzled scales of the grouped GEMMs' A operand, from `quantize_mxfp8` row scales.

    The kernels swizzle each expert's rows separately. Under the GROUP_PAD-aligned dispatch layout
    every expert starts on a 128-row tile, so the whole-matrix swizzle is the same bytes. Returns
    ``float8_e8m0fnu`` of shape ``[M, K/32]``.
    """
    return jax.lax.bitcast_convert_type(_swizzle_tiles(row_scales).reshape(row_scales.shape), jnp.float8_e8m0fnu)


def build_sfb(weight_scales: UInt8[Array, "E N K/32"]) -> Array:
    """Swizzled per-expert scales of the grouped GEMMs' B operand, as ``float8_e8m0fnu`` ``[E, N*K/32]``."""
    e = weight_scales.shape[0]
    return jax.lax.bitcast_convert_type(_swizzle_tiles(weight_scales).reshape(e, -1), jnp.float8_e8m0fnu)


def build_sf_wgrad(token_scales: UInt8[Array, "T/32 R"], block_perm: Int[Array, "tiles"]) -> Array:
    """Swizzled scales of a weight-gradient operand, from `quantize_mxfp8_tokens` scales.

    The weight-gradient kernel reads ``[R, T/32]`` scales one expert at a time: each expert's token
    columns are swizzled on their own, and the experts' chunks are concatenated. ``block_perm``
    from `layout.wgrad_block_perm` reorders the whole-matrix tiles into that order. Returns
    ``float8_e8m0fnu`` of shape ``[R, T/32]``.
    """
    scales = token_scales.T
    swizzled = _swizzle_tiles(scales)[block_perm]
    return jax.lax.bitcast_convert_type(swizzled.reshape(scales.shape), jnp.float8_e8m0fnu)


class DualQuantized(NamedTuple):
    """Both MXFP8 copies of an ``[M, D]`` activation or cotangent, with their swizzled scales."""

    q_rows: Array
    """E4M3 values quantized along features, the forward and data-gradient GEMMs' operand."""
    sfa: Array
    """E8M0 `build_sfa` scales of ``q_rows``."""
    q_tokens: Array
    """E4M3 values quantized along tokens, the weight-gradient GEMMs' operand."""
    sf_wgrad: Array
    """E8M0 `build_sf_wgrad` scales of ``q_tokens``."""


def dual_quantize_activation(t: Float[Array, "M D"], block_perm: Int[Array, "tiles"]) -> DualQuantized:
    """Both MXFP8 copies of an activation or cotangent in the padded dispatch layout."""
    q_rows, row_scales = quantize_mxfp8(t)
    q_tokens, token_scales = quantize_mxfp8_tokens(t)
    return DualQuantized(q_rows, build_sfa(row_scales), q_tokens, build_sf_wgrad(token_scales, block_perm))
