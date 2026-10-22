# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Row and column layouts the fused MXFP8 grouped kernels take.

The kernels need every expert's rows to start on a GROUP_PAD-row boundary and the last expert to
end at the end of the buffer. The ragged backend lays the receiver buffer out that way for the
MXFP8 expert MLP, so `aligned_dispatch_layout` maps it in place: only each expert's first rows were
received, and the rows from there to the next expert's start, and the buffer's trailing rows, are
unspecified and may be non-finite. The kernels see every row that was not received as zero, so it
adds nothing to any GEMM or weight gradient, and a non-finite row cannot reach a 32-token scale
block.

Ported from c3cb334f8e:experiments/grug/moe/mxfp8.py.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.mxfp8.quantize import SF_TILE_ROWS

# Every expert's rows start on a multiple of this in the kernel-facing buffer (the fused kernels'
# FIX_PAD_SIZE).
GROUP_PAD = 256
# Width of the gate and up column blocks the fused SwiGLU epilogue pairs.
_GATE_UP_BLOCK = 32


class PaddedDispatch(NamedTuple):
    """The kernels' grouping of a ``[Mp]``-row receiver buffer whose experts start on GROUP_PAD rows."""

    group_ends: Int[Array, "E"]
    """Padded end offset of each expert. The last is ``padded_rows``, as the kernels require."""
    padded_group_sizes: Int[Array, "E"]
    """Rows of each expert's span, each a multiple of GROUP_PAD."""
    source_valid: Bool[Array, "Mp"]
    """Whether the row was received. The kernels read every other row as zero."""

    @property
    def padded_rows(self) -> int:
        return self.source_valid.shape[0]


def aligned_dispatch_layout(
    physical_group_sizes: Int[Array, "E"], active_group_sizes: Int[Array, "E"], *, rows: int
) -> PaddedDispatch:
    """The layout of a ``rows``-row receiver buffer whose experts start on GROUP_PAD-row boundaries.

    Expert ``e`` spans ``physical_group_sizes[e]`` rows, a multiple of GROUP_PAD, and the spans fill
    the buffer. Its first ``active_group_sizes[e]`` rows were received, and its other rows are
    unspecified. The buffer is already in the kernels' row layout, so the kernels read and write it
    in place, and ``source_valid`` marks the received rows.
    """
    physical = physical_group_sizes.astype(jnp.int32)
    group_ends = jnp.cumsum(physical)
    row = jnp.arange(rows, dtype=jnp.int32)
    expert = jnp.searchsorted(group_ends, row, side="right")
    offset = row - (group_ends - physical)[expert]
    return PaddedDispatch(
        group_ends=group_ends,
        padded_group_sizes=physical,
        source_valid=offset < active_group_sizes.astype(jnp.int32)[expert],
    )


def wgrad_block_perm(padded_group_sizes: Int[Array, "E"], *, rows: int, total_tokens: int) -> Int[Array, "tiles"]:
    """Tile order that turns a whole-matrix scale swizzle into the weight-gradient kernel's chunks.

    The swizzled ``[rows, total_tokens / 32]`` weight-gradient scales form a grid of tiles, 128 rows
    by 128 tokens, stored row-block-major. The kernel reads one expert's chunk at a time, in
    ``[expert][row block][token block]`` order. Entry ``p`` is the row-block-major index of the
    tile at position ``p`` in the kernel's order. ``padded_group_sizes`` must be multiples of 128
    that sum to ``total_tokens``, as a `PaddedDispatch` holds them.
    """
    assert rows % SF_TILE_ROWS == 0, f"rows {rows} are not a multiple of {SF_TILE_ROWS}"
    assert total_tokens % SF_TILE_ROWS == 0, f"tokens {total_tokens} are not a multiple of {SF_TILE_ROWS}"
    row_blocks = rows // SF_TILE_ROWS
    token_blocks = total_tokens // SF_TILE_ROWS
    expert_token_blocks = padded_group_sizes.astype(jnp.int32) // SF_TILE_ROWS
    expert_tiles = expert_token_blocks * row_blocks
    tile_ends = jnp.cumsum(expert_tiles)
    token_block_starts = jnp.cumsum(expert_token_blocks) - expert_token_blocks

    position = jnp.arange(row_blocks * token_blocks, dtype=jnp.int32)
    # Experts without tiles are never selected, so the division below never sees a zero width.
    expert = jnp.searchsorted(tile_ends, position, side="right")
    within_expert = position - (tile_ends - expert_tiles)[expert]
    width = expert_token_blocks[expert]
    tile = (within_expert // width) * token_blocks + token_block_starts[expert] + within_expert % width
    return tile.astype(jnp.int32)


def interleave_gate_up_blocks(w13: Float[Array, "... I2"]) -> Float[Array, "... I2"]:
    """Interleave gate and up columns in blocks of 32: ``[g[0:32], u[0:32], g[32:64], u[32:64], ...]``.

    ``w13`` holds the gate columns ``[:I]`` and then the up columns ``[I:]``, as grug stores them.
    The fused SwiGLU kernels take this order, so each 32-column epilogue block pairs a gate block
    with its up block.
    """
    *lead, i2 = w13.shape
    assert i2 % (2 * _GATE_UP_BLOCK) == 0, f"gate/up width {i2} is not a multiple of {2 * _GATE_UP_BLOCK}"
    halves = w13.reshape(*lead, 2, i2 // (2 * _GATE_UP_BLOCK), _GATE_UP_BLOCK)
    return jnp.swapaxes(halves, -3, -2).reshape(w13.shape)


def deinterleave_gate_up_blocks(w13_interleaved: Float[Array, "... I2"]) -> Float[Array, "... I2"]:
    """Inverse of `interleave_gate_up_blocks`, e.g. for a weight gradient over interleaved columns."""
    *lead, i2 = w13_interleaved.shape
    assert i2 % (2 * _GATE_UP_BLOCK) == 0, f"gate/up width {i2} is not a multiple of {2 * _GATE_UP_BLOCK}"
    blocks = w13_interleaved.reshape(*lead, i2 // (2 * _GATE_UP_BLOCK), 2, _GATE_UP_BLOCK)
    return jnp.swapaxes(blocks, -3, -2).reshape(w13_interleaved.shape)
