# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Row and column layouts the fused MXFP8 grouped kernels take.

The kernels need every expert's rows to start and end on GROUP_PAD-row boundaries, the experts
back to back from the buffer's first row. The ragged backend lays the receiver buffer out that way
for the MXFP8 expert MLP, so `aligned_dispatch_layout` maps it in place: only each expert's first
rows were received, and the rows from there to the next expert's start, and the buffer's trailing
rows, are unspecified and may be non-finite. The kernels see every row that was not received as
zero, so it adds nothing to any GEMM or weight gradient, and a non-finite row cannot reach a
32-token scale block. The last expert's group ends at its last received row block, so the kernels
neither read nor write the buffer's unused capacity past it.

Ported from c3cb334f8e:experiments/grug/moe/mxfp8.py.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Bool, Int

from levanter.grug._moe.mxfp8.quantize import SF_TILE_ROWS

# Every expert's rows start on a multiple of this in the kernel-facing buffer (the fused kernels'
# FIX_PAD_SIZE).
GROUP_PAD = 256


class PaddedDispatch(NamedTuple):
    """The kernels' grouping of a ``[Mp]``-row receiver buffer whose experts start on GROUP_PAD rows."""

    group_ends: Int[Array, "E"]
    """Padded end offset of each expert's group. The last is at most ``padded_rows``; the kernels
    neither read nor write the rows past it."""
    padded_group_sizes: Int[Array, "E"]
    """Rows of each expert's group, each a multiple of GROUP_PAD."""
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
    in place, and ``source_valid`` marks the received rows. Every expert's group is its span, except
    the last's, which stops at its received rows rounded up to GROUP_PAD.
    """
    physical = physical_group_sizes.astype(jnp.int32)
    active = active_group_sizes.astype(jnp.int32)
    span_ends = jnp.cumsum(physical)
    row = jnp.arange(rows, dtype=jnp.int32)
    expert = jnp.searchsorted(span_ends, row, side="right")
    offset = row - (span_ends - physical)[expert]
    # The last span also holds the buffer's unused capacity, which no kernel needs to visit.
    grouped = physical.at[-1].set((active[-1] + GROUP_PAD - 1) // GROUP_PAD * GROUP_PAD)
    return PaddedDispatch(
        group_ends=jnp.cumsum(grouped),
        padded_group_sizes=grouped,
        source_valid=offset < active[expert],
    )


def wgrad_block_perm(padded_group_sizes: Int[Array, "E"], *, rows: int, total_tokens: int) -> Int[Array, "tiles"]:
    """Tile order that turns a whole-matrix scale swizzle into the weight-gradient kernel's chunks.

    The swizzled ``[rows, total_tokens / 32]`` weight-gradient scales form a grid of tiles, 128 rows
    by 128 tokens, stored row-block-major. The kernel reads one expert's chunk at a time, in
    ``[expert][row block][token block]`` order. Entry ``p`` is the row-block-major index of the
    tile at position ``p`` in the kernel's order. ``padded_group_sizes`` must be multiples of 128
    that sum to at most ``total_tokens``, as a `PaddedDispatch` holds them. The tokens past the last
    group, which the kernel never reads, take the positions after every expert's chunk, so the order
    stays a permutation.
    """
    assert rows % SF_TILE_ROWS == 0, f"rows {rows} are not a multiple of {SF_TILE_ROWS}"
    assert total_tokens % SF_TILE_ROWS == 0, f"tokens {total_tokens} are not a multiple of {SF_TILE_ROWS}"
    row_blocks = rows // SF_TILE_ROWS
    token_blocks = total_tokens // SF_TILE_ROWS
    group_token_blocks = padded_group_sizes.astype(jnp.int32) // SF_TILE_ROWS
    # The unread tail as one more group after the experts.
    expert_token_blocks = jnp.append(group_token_blocks, token_blocks - jnp.sum(group_token_blocks))
    expert_tiles = expert_token_blocks * row_blocks
    tile_ends = jnp.cumsum(expert_tiles)
    token_block_starts = jnp.cumsum(expert_token_blocks) - expert_token_blocks

    position = jnp.arange(row_blocks * token_blocks, dtype=jnp.int32)
    # Groups without tiles are never selected, so the division below never sees a zero width.
    expert = jnp.searchsorted(tile_ends, position, side="right")
    within_expert = position - (tile_ends - expert_tiles)[expert]
    width = expert_token_blocks[expert]
    tile = (within_expert // width) * token_blocks + token_block_starts[expert] + within_expert % width
    return tile.astype(jnp.int32)
