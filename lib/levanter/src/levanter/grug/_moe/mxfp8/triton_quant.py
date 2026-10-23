# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Both MXFP8 copies of an aligned receiver buffer in one Triton pass.

`dual_quantize_padded` returns exactly
``dual_quantize_activation(jnp.where(layout.source_valid[:, None], x, 0), block_perm)`` in every
row before ``layout.group_ends[-1]``, the end of the last expert's received rows, and in those rows'
scales, but reads each received row of ``x`` once and writes each output once. The XLA composition
writes the zeroed buffer to HBM, then reads it twice more, once per orientation. The grouped kernels
stop at that end, so the programs past it write nothing, and the outputs there are unspecified.

Each program owns ``block_m`` padded rows and walks a run of 64-column tiles across them. Every
row that was not received reads as zero, so the non-finite rows in the gaps between experts and
past the last are never read. Programs start on 32-row boundaries, so a token block never spans
two programs.

A tile is indexed ``[e, c, r, g, k, d]`` for padded row ``32g + 8r + k`` and column
``32c + 8e + d``. Triton lays a tensor out from its contiguous axis ``d``, then the others in
ascending order: lanes take ``e``, ``c`` and ``r``, warps take ``g``, and each thread keeps all of
``k`` and ``d``, an 8x8 sub-tile read as 8 rows of 16 bytes. So both block amaxes stay inside a
warp: a row block (32 columns) reduces 8 values in the thread and then across the 4 ``e`` lanes, and
a token block (32 rows) reduces 8 values in the thread and then across the 4 ``r`` lanes. A
``[row blocks, 32, column blocks, 32]`` tile spreads each token block over warps and reduces it
through shared memory: 1.46 ms against 1.09 ms at the hero chunk, both one tile per program.

The rules follow `quantize`: the scale is the E8M0 byte of ``amax * (1 / 448)`` rounded up to a
power of two and clipped to ``[2^-126, 2^127]``, with no floor on the amax, and the data is
multiplied by the exact power-of-two reciprocal and cast to E4M3 with saturating
round-to-nearest-even (``cvt.rn.satfinite``). For bf16 inputs the multiply by ``1 / 448`` lands in
the same power-of-two bin as the reference's division: the only bf16 amax whose quotient is a power
of two is ``448 * 2^k``, and its rounded product is exactly ``2^k``; every quotient below 2^-126,
zero included, clips to the smallest scale either way. That is why the kernel takes only bf16, and
why it never uses Triton's approximate f32 division.

Scales are stored straight into their swizzled layouts: the row scales at their `build_sfa`
bytes, and the token scales at their `build_sf_wgrad` bytes, in the tile the inverse of
``block_perm`` assigns.

Written against TileKernels' ``quant/common.py::get_sf_and_inv`` and the dual-write CuTe quantizer
of c3cb334f8e:experiments/grug/moe/standalone/mxfp8_grouped/quantize_cute.py as references.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from levanter.grug._moe.mxfp8.layout import PaddedDispatch
from levanter.grug._moe.mxfp8.quantize import (
    E4M3_MAX,
    E8M0_MAX_EXPONENT,
    E8M0_MIN_EXPONENT,
    MXFP8_BLOCK_SIZE,
    SF_TILE_ROWS,
    DualQuantized,
)

try:
    import jax_triton as jt
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    jt = None
    triton = None
    tl = None

# Columns of one tile: two row blocks, the widest tile whose lanes keep each token block in a warp.
_BLOCK_H = 64


class _TileConfig(NamedTuple):
    block_m: int
    """Padded rows per program, a multiple of ``32 * num_warps``: each warp owns 32-row groups."""
    num_warps: int
    max_tiles_per_program: int
    """Most 64-column tiles one program walks; it takes the largest count that divides ``H / 64``."""


# Best of a sweep over block_m, num_warps and tiles per program on one B200 at the hero chunk
# (C = 302,336, H = 3072): 0.82 ms end to end, against 0.83 ms for 4 tiles per program, 1.17 ms for
# one, and 5.1 ms for the XLA composition.
_DEFAULT_TILE = _TileConfig(block_m=128, num_warps=4, max_tiles_per_program=8)


if triton is not None and tl is not None:

    @triton.jit
    def _e8m0_exponent(amax, inv_e4m3_max: tl.constexpr, min_exponent: tl.constexpr, max_exponent: tl.constexpr):
        # Biased exponent of the smallest power of two >= amax / 448: the exponent field, plus one
        # when the mantissa is nonzero, clipped to the E8M0 range. A zero or subnormal quotient has a
        # zero field and clips up to the smallest exponent, so a flushed subnormal gives the same scale.
        bits = (amax * inv_e4m3_max).to(tl.int32, bitcast=True)
        exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        return tl.minimum(tl.maximum(exponent, min_exponent), max_exponent)

    @triton.jit
    def _e8m0_reciprocal(exponent):
        # 2^(127 - e). A finite bf16 block reaches e in [1, 247], where this is a normal float32
        # between 2^-120 and 2^126, so multiplying by it is exact wherever dividing by 2^(e - 127) is.
        return ((254 - exponent) << 23).to(tl.float32, bitcast=True)

    @triton.jit
    def _dual_quantize_padded_kernel(
        x_ptr,  # (Mp, H) bf16
        source_row_ptr,  # (Mp,) int32: the row of x each output row reads, -1 for a zero row
        tile_dest_ptr,  # (H/128 * Mp/128,) int32: position of each wgrad scale tile, inverse of block_perm
        rows_end_ptr,  # (1,) int32: the last expert's end, a multiple of block_m; later programs write nothing
        q_rows_ptr,  # (Mp, H) fp8e4nv
        sfa_ptr,  # (Mp, H/32) uint8, swizzled
        q_tokens_ptr,  # (Mp, H) fp8e4nv
        sf_wgrad_ptr,  # (H, Mp/32) uint8, swizzled and tile-permuted
        h: tl.constexpr,
        padded_rows: tl.constexpr,
        block_m: tl.constexpr,
        tiles_per_program: tl.constexpr,
        inv_e4m3_max: tl.constexpr,
        min_exponent: tl.constexpr,
        max_exponent: tl.constexpr,
    ):
        groups: tl.constexpr = block_m // 32
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)

        e = tl.arange(0, 4)[:, None, None, None, None, None]
        c = tl.arange(0, 2)[None, :, None, None, None, None]
        r = tl.arange(0, 4)[None, None, :, None, None, None]
        g = tl.arange(0, groups)[None, None, None, :, None, None]
        k = tl.arange(0, 8)[None, None, None, None, :, None]
        d = tl.arange(0, 8)[None, None, None, None, None, :]
        row = pid_m * block_m + g * 32 + r * 8 + k
        column = c * 32 + e * 8 + d
        source = tl.load(source_row_ptr + row)
        x_ptrs = x_ptr + source.to(tl.int64) * h + column
        out_offsets = row.to(tl.int64) * h + column

        # Row scales are indexed [c, r, g, k] and token scales [e, c, g, d], as the reductions leave them.
        sf_row = (
            pid_m * block_m
            + tl.arange(0, 4)[None, :, None, None] * 8
            + tl.arange(0, groups)[None, None, :, None] * 32
            + tl.arange(0, 8)[None, None, None, :]
        )
        sf_feature = (
            tl.arange(0, 4)[:, None, None, None] * 8
            + tl.arange(0, 2)[None, :, None, None] * 32
            + tl.arange(0, 8)[None, None, None, :]
        )
        token_block = pid_m * groups + tl.arange(0, groups)[None, None, :, None]

        # The grouped kernels stop at the last expert's end, so the rows past it are never read.
        if pid_m * block_m < tl.load(rows_end_ptr):
            # One program walks several column tiles of the same rows, loading the row map once. Triton's
            # software pipelining (num_stages > 1) stages the loads through shared memory, and runs 2x
            # slower here than this plain loop.
            for t in tl.range(0, tiles_per_program, num_stages=1):
                tile = pid_h * tiles_per_program + t
                x = tl.load(x_ptrs + tile * 64, mask=source >= 0, other=0.0).to(tl.float32)
                magnitude = tl.abs(x)
                # Reduce the in-thread axis first, so the lanes exchange one value per row or column.
                row_amax = tl.max(tl.max(magnitude, axis=5), axis=0)  # [c, r, g, k]
                row_exponent = _e8m0_exponent(row_amax, inv_e4m3_max, min_exponent, max_exponent)
                q_rows = x * _e8m0_reciprocal(row_exponent)[None, :, :, :, :, None]
                tl.store(q_rows_ptr + out_offsets + tile * 64, q_rows.to(tl.float8e4nv, fp_downcast_rounding="rtne"))

                # The [Mp, H/32] row-scale matrix swizzled whole, as build_sfa does.
                sf_col = tile * 2 + tl.arange(0, 2)[:, None, None, None]
                sfa_tile = (sf_row // 128) * (h // 128) + sf_col // 4
                sfa_offset = sfa_tile * 512 + (sf_row % 32) * 16 + ((sf_row % 128) // 32) * 4 + sf_col % 4
                tl.store(sfa_ptr + sfa_offset, row_exponent.to(tl.uint8))

                token_amax = tl.max(tl.max(magnitude, axis=4), axis=2)  # [e, c, g, d]
                token_exponent = _e8m0_exponent(token_amax, inv_e4m3_max, min_exponent, max_exponent)
                q_tokens = x * _e8m0_reciprocal(token_exponent)[:, :, None, :, None, :]
                tl.store(
                    q_tokens_ptr + out_offsets + tile * 64, q_tokens.to(tl.float8e4nv, fp_downcast_rounding="rtne")
                )

                # The [H, Mp/32] (feature, token block) token-scale matrix swizzled whole, then tile p
                # of the result is source tile block_perm[p]. A tile's 64 features lie in one 128-row tile.
                feature = tile * 64 + sf_feature
                dest_tile = tl.load(tile_dest_ptr + (tile // 2) * (padded_rows // 128) + token_block // 4)
                sf_wgrad_offset = dest_tile * 512 + (feature % 32) * 16 + ((feature % 128) // 32) * 4 + token_block % 4
                tl.store(sf_wgrad_ptr + sf_wgrad_offset, token_exponent.to(tl.uint8))

else:
    _dual_quantize_padded_kernel = None


def mxfp8_triton_quant_available() -> bool:
    """Whether the Triton producer can run in this process: Triton imports and a GPU default device."""
    return jt is not None and _dual_quantize_padded_kernel is not None and jax.devices()[0].platform == "gpu"


def dual_quantize_padded(
    x: Float[Array, "Mp H"], layout: PaddedDispatch, block_perm: Int[Array, "tiles"]
) -> DualQuantized:
    """``dual_quantize_activation`` of ``x`` with its unreceived rows zeroed, bit for bit, in one kernel.

    ``x`` is the bf16 aligned receiver buffer ``layout`` maps; rows it does not mark as received
    may hold anything, including NaN. Returns both copies and their scales with the reference's
    shapes and dtypes; the rows from ``layout.group_ends[-1]`` on, and their scales, are unspecified.
    Raises unless Triton and a GPU are available. The kernel has no partitioning rule, so under a
    sharded mesh call it inside ``shard_map``.
    """
    if not mxfp8_triton_quant_available():
        raise RuntimeError(
            "the MXFP8 Triton quantizer needs jax-triton, triton and a GPU; install the gpu extra for "
            "marin-levanter"
        )
    if x.ndim != 2 or x.dtype != jnp.bfloat16:
        raise ValueError(f"expected a bf16 [Mp, H] buffer, got {x.dtype} {x.shape}")
    rows, hidden = x.shape
    padded_rows = layout.padded_rows
    if rows != padded_rows:
        raise ValueError(f"layout is for {padded_rows} receiver rows, x has {rows}")
    # Both scale layouts tile the hidden axis by 128: 4 scale columns of sfa, 128 rows of sf_wgrad.
    if hidden % SF_TILE_ROWS != 0:
        raise ValueError(f"hidden size {hidden} is not a multiple of {SF_TILE_ROWS}")
    tile = _DEFAULT_TILE
    if tile.block_m % (32 * tile.num_warps) != 0 or padded_rows % tile.block_m != 0:
        raise ValueError(f"tile {tile} does not split the {padded_rows} padded rows into whole warps")
    tiles = (hidden // SF_TILE_ROWS) * (padded_rows // SF_TILE_ROWS)
    if block_perm.shape != (tiles,):
        raise ValueError(f"block_perm has shape {block_perm.shape}, expected ({tiles},)")
    column_tiles = hidden // _BLOCK_H
    tiles_per_program = max(n for n in range(1, tile.max_tiles_per_program + 1) if column_tiles % n == 0)

    source_row = jnp.where(layout.source_valid, jnp.arange(padded_rows, dtype=jnp.int32), -1)
    tile_dest = (
        jnp.zeros((tiles,), jnp.int32).at[block_perm].set(jnp.arange(tiles, dtype=jnp.int32), unique_indices=True)
    )
    q_rows, sfa, q_tokens, sf_wgrad = jt.triton_call(
        x,
        source_row,
        tile_dest,
        layout.group_ends[-1:].astype(jnp.int32),
        kernel=_dual_quantize_padded_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((padded_rows, hidden), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((padded_rows, hidden // MXFP8_BLOCK_SIZE), jnp.uint8),
            jax.ShapeDtypeStruct((padded_rows, hidden), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((hidden, padded_rows // MXFP8_BLOCK_SIZE), jnp.uint8),
        ),
        grid=(padded_rows // tile.block_m, column_tiles // tiles_per_program),
        num_warps=tile.num_warps,
        num_stages=1,
        h=hidden,
        padded_rows=padded_rows,
        block_m=tile.block_m,
        tiles_per_program=tiles_per_program,
        inv_e4m3_max=1.0 / E4M3_MAX,
        min_exponent=E8M0_MIN_EXPONENT,
        max_exponent=E8M0_MAX_EXPONENT,
    )
    e8m0 = jnp.float8_e8m0fnu
    return DualQuantized(
        q_rows,
        jax.lax.bitcast_convert_type(sfa, e8m0),
        q_tokens,
        jax.lax.bitcast_convert_type(sf_wgrad, e8m0),
    )
