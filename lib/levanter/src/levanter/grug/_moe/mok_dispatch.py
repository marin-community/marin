# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Remote token gather for the MoK expert-parallel backend.

Dispatch reads each row of the destination-order schedule directly out of the
owning peer's memory with a contiguous bulk copy. A block issues one copy per
destination row and points every copy at the same barrier, so an irregular
gather spanning many peers costs a single wait rather than one wait per peer.

Both the peer and token row come from the schedule at runtime. Declaring the
valid rows in-bounds selects Mosaic's device-side remote-pointer bulk path,
which matches MoK's dynamic ``peer_buf[peer_rank]`` dispatch. A global semaphore
rendezvous guarantees every peer has reached the kernel before its rows are
read.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.mosaic import gpu as mgpu
from jax.experimental.pallas import mosaic_gpu as plgpu
from jaxtyping import Array, Float, Int

# Mosaic vectorizes SMEM reads and TMA transfers over a warpgroup, so a block's
# schedule slice must cover a whole multiple of this many elements and its row
# payload a whole multiple of this many bytes. MoK's dispatch tile is 128 rows
# for the same reason.
_WARPGROUP = 128
_SMEM_BUDGET = 192 * 1024
_MAX_DISPATCH_COLS = 512
_MAX_COMBINE_COLS = 1024
_COMBINE_PIPE_DEPTH = 7


@plgpu.inline_mgpu(
    return_type=plgpu.ShapeDtypeStruct((), jnp.int32, layout=plgpu.Layout.WG_SPLAT),
)
def _thread_index_mgpu(_):
    return mgpu.FragmentedArray.splat(mgpu.thread_idx(), (), is_signed=True)


def _pick_block_cols(hidden: int, dtype, block_rows: int, max_block_cols: int) -> int:
    itemsize = jnp.dtype(dtype).itemsize
    cols = min(max_block_cols, 1 << int(math.log2(max(_SMEM_BUDGET // (block_rows * itemsize), 1))))
    while cols > 1 and (hidden % cols or (cols * itemsize) % 16):
        cols //= 2
    if hidden % cols or (cols * itemsize) % 16:
        raise ValueError(f"cannot chunk {hidden=} for {dtype} at {block_rows=}")
    return cols


def _dispatch_kernel(
    x_ref,
    peer_rank_ref,
    peer_token_ref,
    num_tokens_ref,
    out_ref,
    staging_ref,
    owner_ref,
    token_ref,
    schedule_barrier,
    gather_barrier,
    *,
    block_rows: int,
    block_cols: int,
    row_divisor: int,
    col_blocks: int,
    collective_axis: str,
    axis_size: int,
    grid_blocks: int,
):
    block = lax.axis_index("tiles")
    ready = pl.get_global(plgpu.SemaphoreType.REGULAR((grid_blocks,)))
    plgpu.semaphore_signal_multicast(ready.at[block], collective_axes=collective_axis)
    pl.semaphore_wait(ready.at[block], value=axis_size, decrement=False)

    active_row_blocks = lax.div(num_tokens_ref[()] + block_rows - 1, block_rows)
    row_block = block // col_blocks

    @pl.when(row_block < active_row_blocks)
    def _tile():
        base = row_block * block_rows

        plgpu.copy_gmem_to_smem(peer_rank_ref.at[pl.ds(base, block_rows)], owner_ref, schedule_barrier)
        plgpu.copy_gmem_to_smem(peer_token_ref.at[pl.ds(base, block_rows)], token_ref, schedule_barrier)
        plgpu.barrier_wait(schedule_barrier)

        row = _thread_index_mgpu()
        valid = owner_ref[row] >= 0
        source = plgpu.remote_ref(x_ref, jnp.maximum(owner_ref[row], 0))
        plgpu.copy_gmem_to_smem(
            source.at[
                pl.ds(jnp.maximum(token_ref[row], 0) // row_divisor, 1),
                pl.ds(lax.rem(block, col_blocks) * block_cols, block_cols),
            ],
            staging_ref,
            gather_barrier,
            predicate=valid,
            oob_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS,
            thread_parallel=True,
        )

        plgpu.barrier_wait(gather_barrier)

        col_base = lax.rem(block, col_blocks) * block_cols
        plgpu.copy_smem_to_gmem(
            staging_ref,
            out_ref.at[pl.ds(base + row, 1), pl.ds(col_base, block_cols)],
            predicate=valid,
            thread_parallel=True,
        )
        plgpu.wait_smem_to_gmem(0)


def _scatter_kernel(
    y_ref,
    peer_rank_ref,
    peer_token_ref,
    num_tokens_ref,
    recv_ref,
    staging_ref,
    load_barrier,
    *,
    block_rows: int,
    block_cols: int,
    col_blocks: int,
    pipeline_depth: int,
    dump_slot: int,
    collective_axis: str,
    axis_size: int,
    grid_blocks: int,
):
    active_row_blocks = lax.div(num_tokens_ref[()] + block_rows - 1, block_rows)
    block = lax.axis_index("tiles")
    row = _thread_index_mgpu()
    safe_row = jnp.minimum(row, block_rows - 1)

    for stage in range(pipeline_depth):
        tile = block * pipeline_depth + stage
        row_block = tile // col_blocks
        col_block = lax.rem(tile, col_blocks)

        @pl.when(row_block < active_row_blocks)
        def _load_stage():
            base = row_block * block_rows
            owner = peer_rank_ref[base + safe_row]
            valid = (row < block_rows) & (owner >= 0)
            plgpu.copy_gmem_to_smem(
                y_ref.at[pl.ds(base + row, 1), pl.ds(col_block * block_cols, block_cols)],
                staging_ref.at[stage],
                load_barrier.at[stage],
                predicate=valid,
                oob_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS,
                thread_parallel=True,
            )

    for stage in range(pipeline_depth):
        tile = block * pipeline_depth + stage
        row_block = tile // col_blocks
        col_block = lax.rem(tile, col_blocks)

        @pl.when(row_block < active_row_blocks)
        def _store_stage():
            plgpu.barrier_wait(load_barrier.at[stage])
            base = row_block * block_rows
            owner = peer_rank_ref[base + safe_row]
            valid = (row < block_rows) & (owner >= 0)
            destination = plgpu.remote_ref(recv_ref, jnp.maximum(owner, 0))
            destination_row = jnp.where(valid, peer_token_ref[base + safe_row], dump_slot)
            plgpu.copy_smem_to_gmem(
                staging_ref.at[stage],
                destination.at[pl.ds(destination_row, 1), pl.ds(col_block * block_cols, block_cols)],
                predicate=valid,
                thread_parallel=True,
            )

    plgpu.wait_smem_to_gmem(0)

    ready = pl.get_global(plgpu.SemaphoreType.REGULAR((grid_blocks,)))
    plgpu.semaphore_signal_multicast(ready.at[block], collective_axes=collective_axis)
    pl.semaphore_wait(ready.at[block], value=axis_size, decrement=False)


def gather_routed_tokens(
    x: Float[Array, "T H"],
    peer_rank: Int[Array, "C"],
    peer_token_idx: Int[Array, "C"],
    num_routed_tokens: Int[Array, ""],
    *,
    axis_name: str,
    row_divisor: int,
    block_rows: int = 128,
) -> Float[Array, "C H"]:
    """Gathers each scheduled row from the peer that owns it.

    Must be called inside a ``shard_map`` whose ``axis_name`` spans every device.

    Args:
        x: This rank's token shard.
        peer_rank: Owning rank per destination row, ``-1`` on padding rows.
        peer_token_idx: ``token * topk + k`` within the owner's shard.
        axis_name: Expert-parallel mesh axis.
        row_divisor: ``topk``, recovering the token row from a routed index.
        block_rows: Destination rows gathered per block against one barrier.

    Returns:
        The expert-major receive buffer, zero on padding rows.
    """
    _, hidden = x.shape
    (capacity,) = peer_rank.shape
    if capacity % block_rows:
        raise ValueError(f"{capacity=} must be a multiple of {block_rows=}")

    if block_rows % _WARPGROUP:
        raise ValueError(f"{block_rows=} must be a multiple of {_WARPGROUP}")

    block_cols = _pick_block_cols(hidden, x.dtype, block_rows, _MAX_DISPATCH_COLS)
    col_blocks = hidden // block_cols
    grid_blocks = capacity // block_rows * col_blocks
    axis_size = lax.axis_size(axis_name)
    body = functools.partial(
        _dispatch_kernel,
        block_rows=block_rows,
        block_cols=block_cols,
        row_divisor=row_divisor,
        col_blocks=col_blocks,
        collective_axis=axis_name,
        axis_size=axis_size,
        grid_blocks=grid_blocks,
    )

    out = plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((capacity, hidden), x.dtype),
        scratch_types=[
            plgpu.SMEM((block_rows, block_cols), x.dtype),
            plgpu.SMEM((block_rows,), jnp.int32),
            plgpu.SMEM((block_rows,), jnp.int32),
            plgpu.Barrier(num_arrivals=2),
            plgpu.Barrier(num_arrivals=1),
        ],
        grid=(grid_blocks,),
        grid_names=("tiles",),
        # Remote loads only lower under Lane semantics; warpgroup lowering rejects
        # GMEM refs carrying a peer id.
        compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane),
    )(x, peer_rank, peer_token_idx, num_routed_tokens)

    # Padding rows gather a defined but meaningless row, so they are zeroed here.
    # MoK does this inside dispatch; Mosaic cannot yet broadcast a fragmented
    # vector across the row axis, so it stays outside until the stage is fused.
    return jnp.where((peer_rank >= 0)[:, None], out, 0)


def scatter_routed_tokens(
    y: Float[Array, "C H"],
    peer_rank: Int[Array, "C"],
    peer_token_idx: Int[Array, "C"],
    num_routed_tokens: Int[Array, ""],
    *,
    axis_name: str,
    num_slots: int,
    block_rows: int = 16,
) -> Float[Array, "N H"]:
    """Scatters expert-major rows back to their owning peer slots."""
    capacity, hidden = y.shape
    if peer_rank.shape != (capacity,) or peer_token_idx.shape != (capacity,):
        raise ValueError(f"schedule shapes must match {capacity=}")
    if capacity % block_rows or block_rows > _WARPGROUP:
        raise ValueError(f"{capacity=} must be a multiple of {block_rows=} <= {_WARPGROUP}")

    block_cols = _pick_block_cols(hidden, y.dtype, block_rows, _MAX_COMBINE_COLS)
    col_blocks = hidden // block_cols
    pipeline_depth = min(_COMBINE_PIPE_DEPTH, col_blocks)
    num_tiles = capacity // block_rows * col_blocks
    grid_blocks = (num_tiles + pipeline_depth - 1) // pipeline_depth
    axis_size = lax.axis_size(axis_name)
    body = functools.partial(
        _scatter_kernel,
        block_rows=block_rows,
        block_cols=block_cols,
        col_blocks=col_blocks,
        pipeline_depth=pipeline_depth,
        dump_slot=num_slots,
        collective_axis=axis_name,
        axis_size=axis_size,
        grid_blocks=grid_blocks,
    )
    recv = plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((num_slots + 1, hidden), y.dtype),
        scratch_types=[
            plgpu.SMEM((pipeline_depth, block_rows, block_cols), y.dtype),
            plgpu.Barrier(num_arrivals=1, num_barriers=pipeline_depth),
        ],
        grid=(grid_blocks,),
        grid_names=("tiles",),
        compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane),
    )(y, peer_rank, peer_token_idx, num_routed_tokens)
    return recv[:num_slots]


def reference_gather(
    x_all: Float[Array, "P T H"],
    peer_rank: Int[Array, "C"],
    peer_token_idx: Int[Array, "C"],
    *,
    row_divisor: int,
) -> Float[Array, "C H"]:
    """Gathers the same rows with plain indexing, for parity checks."""
    rows = x_all[jnp.maximum(peer_rank, 0), peer_token_idx // row_divisor]
    return jnp.where((peer_rank >= 0)[:, None], rows, 0)
