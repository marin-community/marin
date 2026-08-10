# Copyright 2025 The JAX Authors.
# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py
# (Apache 2.0) so the expert GEMM can be called from inside our own kernel.
#
# Changes from upstream:
#   * `GroupInfo` is inlined from ragged_dot_mgpu rather than imported, so this
#     module has no dependency on a benchmark file.
#   * The kernel body reads the cluster axis unconditionally while the launch
#     only declares it under `collective`, so upstream's non-collective path
#     raises "unbound axis name: x". Fixed here: the axis is read only when
#     collective, since a fused megakernel wants the non-collective path.
#   * Lowering semantics are a parameter rather than hardcoded to Warpgroup.
#     Peer transport requires Lane, and a fused kernel must use one setting for
#     both; `tcgen05`/TMEM/WarpMesh are measured to run correctly under Lane.
#
# `do_matmul` is the reusable fragment: it emits one output block's matmul given
# refs, grid indices and scratch, so a megakernel can call it for its compute
# CTAs while other CTAs run transport.
"""Ragged/Grouped Matrix Multiplication kernel for Blackwell GPUs."""

import dataclasses
import functools
import math
from collections.abc import Sequence

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jax import lax
from jax.experimental.pallas.ops.gpu import blackwell_matmul_mgpu


@dataclasses.dataclass(frozen=True)
class TuningConfig:
    tile_m: int
    tile_n: int
    tile_k: int
    max_concurrent_steps: int
    collective: bool
    grid_tile_width: int
    grid_minor_dim: blackwell_matmul_mgpu.MatmulDimension
    epilogue_tile_n: int = 64

    def __str__(self):
        return "_".join(f"{k}={v}" for k, v in dataclasses.asdict(self).items())


# TODO(justinfu): Merge with blackwell_matmul_mgpu.py
@dataclasses.dataclass(frozen=True)
class GroupInfo:
    """Information regarding the group being processed in a block."""

    group_id: jax.Array
    block: jax.Array
    block_start: jax.Array
    actual_start: jax.Array
    actual_end: jax.Array
    start_within_block: jax.Array
    actual_size: jax.Array

    @classmethod
    def create(cls, group_lengths, tile, tid):
        """Get the group info for the current block."""

        tile = jnp.int32(tile)
        group_boundaries = [group_lengths[i] for i in range(len(group_lengths))]

        # We usually only have very few groups, so we unroll the loop processing
        # them. Normally we'd break out of the loop early, once we'd have found our
        # boundary, but we can't do that when unrolling, so we rely on many selects
        # to mask out the epilogue of the loop.
        group_end = group_start = block = group = end = jnp.array(0, dtype=jnp.int32)

        for i, b in enumerate(group_boundaries):
            # Start/end are inclusive
            start = end
            end = start + b
            final = end - 1
            start_block = lax.div(start, tile)
            final_block = lax.div(final, tile)
            block_end = final_block + 1
            tid_begin = start_block + i
            tid_end = block_end + i
            # How many blocks after is our block?
            this_is_group = (tid_begin <= tid) & (tid < tid_end)
            block = lax.select(this_is_group, tid - tid_begin + start_block, block)
            group = lax.select(this_is_group, jnp.int32(i), group)
            group_start = lax.select(this_is_group, start, group_start)
            group_end = lax.select(this_is_group, end, group_end)

        block_start = block * tile
        actual_start = jnp.maximum(group_start, block_start)
        actual_end = jnp.minimum(group_end, block_start + tile)
        start_within_block = actual_start - block_start
        actual_size = actual_end - actual_start
        return cls(
            group_id=group,
            block=block,
            block_start=block_start,
            actual_start=actual_start,
            actual_end=actual_end,
            start_within_block=start_within_block,
            actual_size=actual_size,
        )

    @classmethod
    def from_block(cls, group_lengths, tile, block):
        tile = jnp.int32(tile)
        block_start = block * tile
        group_start = group_end = group = jnp.array(0, dtype=jnp.int32)
        end = jnp.array(0, dtype=jnp.int32)

        for i, length in enumerate(group_lengths):
            start = end
            end = start + length
            this_is_group = (start <= block_start) & (block_start < end)
            group = lax.select(this_is_group, jnp.int32(i), group)
            group_start = lax.select(this_is_group, start, group_start)
            group_end = lax.select(this_is_group, end, group_end)

        actual_start = jnp.maximum(group_start, block_start)
        actual_end = jnp.minimum(group_end, block_start + tile)
        return cls(
            group_id=group,
            block=block,
            block_start=block_start,
            actual_start=actual_start,
            actual_end=actual_end,
            start_within_block=actual_start - block_start,
            actual_size=actual_end - actual_start,
        )


def do_matmul(
    a_gmem,
    b_gmem,
    out_gmem,
    grid_indices: Sequence[jax.Array],
    wg_axis: str,
    collective_axes: tuple[str, ...],
    local_index: jax.Array | int,
    previous_total_k_iters,
    config: TuningConfig,
    group_info: GroupInfo,
    a_smem,
    b_smem,
    acc_tmem,
    acc_smem,
    a_tma_barrier,
    b_tma_barrier,
    store_done_barrier,
    mma_done_barrier,
    consumed_barrier,
    alternate_b_gmem=None,
    alternate_out_gmem=None,
    output_stage=0,
    output_ready=None,
    output_ready_index=None,
    transpose_a=False,
    transpose_b=False,
    second_a_gmem=None,
    second_b_gmem=None,
    k_start=0,
    k_end=None,
    a_smem_transposed=False,
    b_smem_transposed=False,
):
    """Compute a non-ragged matmul for a single output block."""
    dtype = out_gmem.dtype
    if transpose_a:
        k, m = a_gmem.shape
    else:
        m, k = a_gmem.shape
    if transpose_b:
        n, b_k = b_gmem.shape
    else:
        b_k, n = b_gmem.shape
    if k != b_k:
        raise ValueError(f"matmul operands have incompatible K dimensions: {k} and {b_k}")
    collective = config.collective
    tile_m, tile_n, tile_k = (config.tile_m, config.tile_n, config.tile_k)
    epilogue_tile_n = config.epilogue_tile_n
    max_concurrent_steps = config.max_concurrent_steps
    block_tile_m = tile_m
    if collective:
        tile_m *= 2
        tile_n *= 2
    if (second_a_gmem is None) != (second_b_gmem is None):
        raise ValueError("second_a_gmem and second_b_gmem must be provided together")
    if second_a_gmem is not None and (second_a_gmem.shape != a_gmem.shape or second_b_gmem.shape != b_gmem.shape):
        raise ValueError("second GEMM operands must match the primary operand shapes")
    k_end = k if k_end is None else k_end
    k_iters = lax.div(k_end - k_start, tile_k)
    total_k_iters = k_iters * (2 if second_a_gmem is not None else 1)

    if collective:
        m_index, n_index, cluster_idx = grid_indices
        block_m_index = m_index * 2 + cluster_idx
        is_lead_block = cluster_idx == 0
    else:
        m_index, n_index = grid_indices
        cluster_idx = 0
        block_m_index = m_index
        is_lead_block = True
    wg_idx = lax.axis_index(wg_axis)
    collective_axis = collective_axes[0] if collective else None

    TMA_WARP = 0
    MMA_WARP = 1
    COMPUTE_WG = 0
    STORE_WG = 1

    block_slice_m = pl.ds(block_m_index * block_tile_m, block_tile_m)
    slice_m = pl.ds(m_index * tile_m, tile_m)
    slice_n = pl.ds(n_index * tile_n, tile_n)
    acc_slot = jnp.int32(0)
    regs_layout = plgpu.Layout.TCGEN05

    @pl.when(wg_idx == COMPUTE_WG)
    @jax.named_scope("compute_wg")
    def _():
        @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
        def _per_warp():
            warp_id = lax.axis_index("warp")

            @pl.when(warp_id == TMA_WARP)
            def _memory():
                def _loop_body(ki, _):
                    operand = lax.div(ki, k_iters)
                    operand_ki = lax.rem(ki, k_iters)
                    slice_k = pl.ds(k_start + operand_ki * tile_k, tile_k)
                    slot = lax.rem(ki, max_concurrent_steps)

                    @pl.when(
                        jnp.logical_or(
                            ki >= max_concurrent_steps,
                            lax.rem(ki, max_concurrent_steps) < previous_total_k_iters,
                        )
                    )
                    def _():
                        plgpu.barrier_wait(consumed_barrier.at[slot])

                    def copy_a(source):
                        source_slice = source.at[slice_k, slice_m] if transpose_a else source.at[slice_m, slice_k]
                        destination = (
                            plgpu.transpose_ref(a_smem.at[slot], (1, 0))
                            if transpose_a != a_smem_transposed
                            else a_smem.at[slot]
                        )
                        plgpu.copy_gmem_to_smem(
                            source_slice,
                            destination,
                            a_tma_barrier.at[slot],
                            leader_tracked=(
                                plgpu.CopyPartition.PARTITIONED(1 if transpose_a else 0) if collective else None
                            ),
                            collective_axes=collective_axis,
                        )

                    def copy_b(source):
                        source_slice = source.at[slice_n, slice_k] if transpose_b else source.at[slice_k, slice_n]
                        destination = (
                            plgpu.transpose_ref(b_smem.at[slot], (1, 0))
                            if transpose_b != b_smem_transposed
                            else b_smem.at[slot]
                        )
                        plgpu.copy_gmem_to_smem(
                            source_slice,
                            destination,
                            b_tma_barrier.at[slot],
                            leader_tracked=(
                                plgpu.CopyPartition.PARTITIONED(0 if transpose_b else 1) if collective else None
                            ),
                            collective_axes=collective_axis,
                        )

                    if second_a_gmem is None:
                        copy_a(a_gmem)
                        if alternate_b_gmem is None:
                            copy_b(b_gmem)
                        else:

                            @pl.when(output_stage == 0)
                            def _copy_primary_b():
                                copy_b(b_gmem)

                            @pl.when(output_stage == 1)
                            def _copy_alternate_b():
                                copy_b(alternate_b_gmem)

                    else:

                        @pl.when(operand == 0)
                        def _copy_primary():
                            copy_a(a_gmem)
                            copy_b(b_gmem)

                        @pl.when(operand == 1)
                        def _copy_second():
                            copy_a(second_a_gmem)
                            copy_b(second_b_gmem)

                lax.fori_loop(0, total_k_iters, _loop_body, None)

            @pl.when(jnp.logical_and(warp_id == MMA_WARP, (local_index > 0) & is_lead_block))
            def _wait_store():
                plgpu.barrier_wait(store_done_barrier.at[acc_slot])

            @pl.when(jnp.logical_and(warp_id == MMA_WARP, is_lead_block))
            def _compute():
                def _loop_body(ki, _):
                    slot = lax.rem(ki, max_concurrent_steps)
                    plgpu.barrier_wait(a_tma_barrier.at[slot])
                    plgpu.barrier_wait(b_tma_barrier.at[slot])

                    is_last_iter = ki >= total_k_iters - 1
                    acc_tmem_slice = acc_tmem.at[:, pl.ds(acc_slot * tile_n, tile_n)]
                    plgpu.tcgen05_mma(
                        acc_tmem_slice,
                        (plgpu.transpose_ref(a_smem.at[slot], (1, 0)) if a_smem_transposed else a_smem.at[slot]),
                        (plgpu.transpose_ref(b_smem.at[slot], (1, 0)) if b_smem_transposed else b_smem.at[slot]),
                        consumed_barrier.at[slot],
                        accumulate=(ki > 0),
                        collective_axis=collective_axis,
                    )

                    @pl.when(is_last_iter)
                    def _():
                        plgpu.tcgen05_commit_arrive(
                            mma_done_barrier.at[acc_slot],
                            collective_axis=collective_axis,
                        )

                lax.fori_loop(0, total_k_iters, _loop_body, None)

    @pl.when(wg_idx == STORE_WG)
    @jax.named_scope("store_wg")
    def _():
        plgpu.barrier_wait(mma_done_barrier.at[acc_slot])
        acc_tmem_slot = acc_tmem.at[:, pl.ds(acc_slot * tile_n, tile_n)]
        step_out_gmem = out_gmem.at[block_slice_m, slice_n]
        # group_info contains start/size info relative to the logical
        # tiling (tile_m) but because for collective matmuls we use 2 CTAs per
        # logical block, but we need to compute the start/size relative to the
        # current block.
        # For example, for the following parameters:
        #     block_tile_m=64 (tile_m=128)
        #     group_info.start_within_block=60
        #     group_info.actual_size=37
        # The requested copy will be split across both blocks
        # Memory:         | Block 0  |  Block 1 |
        #                 |--- 64 ---|--- 64 ---|
        # Copy:                    |-- 37 --|
        # Where block 0 copies rows 60-64 (4 rows total) and block 1 copies
        # the remaining rows 64-97 (33 rows total).
        smem_start = group_info.start_within_block - cluster_idx * block_tile_m
        smem_start = lax.max(smem_start, jnp.int32(0))

        def _clamp(min, x, max):
            return lax.max(lax.min(x, max), min)

        block0_copy_size = _clamp(jnp.int32(0), block_tile_m - group_info.start_within_block, group_info.actual_size)
        block_local_size = lax.select(
            is_lead_block,
            # block 0 copies up to end of the first block or actual_size,
            # whichever comes first.
            block0_copy_size,
            # block 1 copies the remaining rows that block 0 did not copy.
            group_info.actual_size - block0_copy_size,
        )
        for ni in range(tile_n // epilogue_tile_n):
            acc_smem[...] = plgpu.async_load_tmem(
                acc_tmem_slot.at[:, pl.ds(ni * epilogue_tile_n, epilogue_tile_n)], layout=regs_layout
            ).astype(dtype)
            plgpu.commit_smem()
            cur_smem_idx = smem_start
            remaining_rows = min(block_tile_m, m)
            while remaining_rows > 0:
                const_rows_len = 1 << int(math.log2(remaining_rows))
                remaining_rows //= 2

                @pl.when(block_local_size & const_rows_len != 0)
                def _():
                    o_smem_slice = acc_smem.at[pl.ds(cur_smem_idx, const_rows_len)]
                    o_gref_slice = step_out_gmem.at[
                        pl.ds(cur_smem_idx, const_rows_len),
                        pl.ds(ni * epilogue_tile_n, epilogue_tile_n),
                    ]
                    if alternate_out_gmem is None:
                        plgpu.copy_smem_to_gmem(o_smem_slice, o_gref_slice)
                    else:
                        alternate_step_out_gmem = alternate_out_gmem.at[block_slice_m, slice_n]
                        alternate_o_gref_slice = alternate_step_out_gmem.at[
                            pl.ds(cur_smem_idx, const_rows_len),
                            pl.ds(ni * epilogue_tile_n, epilogue_tile_n),
                        ]

                        @pl.when(output_stage == 0)
                        def _store_primary():
                            plgpu.copy_smem_to_gmem(o_smem_slice, o_gref_slice)

                        @pl.when(output_stage == 1)
                        def _store_alternate():
                            plgpu.copy_smem_to_gmem(o_smem_slice, alternate_o_gref_slice)

                cur_smem_idx += block_local_size & const_rows_len
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        plgpu.wait_load_tmem()  # Load must complete before we continue.
        if output_ready is not None:
            plgpu.wait_smem_to_gmem(0)
            pl.semaphore_signal(output_ready.at[output_ready_index])
        plgpu.barrier_arrive(store_done_barrier.at[acc_slot])


def ragged_dot_kernel(a, b, group_sizes, config: TuningConfig, lowering_semantics=None):
    dtype = a.dtype
    if a.dtype != b.dtype:
        raise ValueError(f"Matmul LHS and RHS have incompatible dtypes {a.dtype} vs {b.dtype}")
    m, k = a.shape
    num_groups, k2, n = b.shape
    if num_groups != group_sizes.shape[0]:
        raise ValueError("RHS and group_sizes have incompatible shapes.")
    if k != k2:
        raise ValueError(f"Matmul LHS and RHS have incompatible shapes {a.shape} vs {b.shape[1:]}")
    collective = config.collective
    tile_m, tile_n, tile_k = (config.tile_m, config.tile_n, config.tile_k)
    block_tile_m = tile_m
    block_tile_n = tile_n
    if collective:
        tile_m *= 2
        tile_n *= 2
    n_iters = n // tile_n

    max_concurrent_steps = config.max_concurrent_steps
    epilogue_tile_n = config.epilogue_tile_n
    if tile_n % epilogue_tile_n != 0:
        raise ValueError(f"{tile_n=} must be divisible by {epilogue_tile_n=}")

    if m % tile_m != 0:
        raise ValueError(f"{m=} must be divisible by {tile_m=}")
    if n % tile_n != 0:
        raise ValueError(f"{n=} must be divisible by {tile_n=}")
    if k % tile_k != 0:
        raise ValueError(f"{k=} must be divisible by {tile_k=}")
    swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
    swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
    transforms = (
        plgpu.TilingTransform((8, swizzle_elems)),
        plgpu.SwizzleTransform(swizzle),
    )

    def kernel(a_gmem, b_gmem, group_sizes_gmem, out_gmem):
        group_sizes_regs = [group_sizes_gmem[i] for i in range(num_groups)]
        num_rows = sum(group_sizes_regs, start=jnp.int32(0))
        active_m_iters = lax.div(num_rows + tile_m - 1, tile_m) + num_groups - 1
        linear_grid = active_m_iters * n_iters
        # Upstream reads this unconditionally, but the launch only declares the
        # axis under `collective`, so the non-collective path fails to trace.
        cluster_idx = lax.axis_index("x") if collective else jnp.int32(0)

        @functools.partial(
            pl.run_scoped,
            a_smem=plgpu.SMEM((max_concurrent_steps, block_tile_m, tile_k), dtype, transforms=transforms),
            b_smem=plgpu.SMEM((max_concurrent_steps, tile_k, block_tile_n), dtype, transforms=transforms),
            # Temporary SMEM used for storing accumulator output to GMEM.
            acc_smem=plgpu.SMEM((block_tile_m, epilogue_tile_n), dtype),
            # a/b_tma_barrier
            a_tma_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=max_concurrent_steps),
            b_tma_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=max_concurrent_steps),
            # store_done_barrier
            store_done_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=1, orders_tensor_core=True),
            # mma_done_barrier
            mma_done_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=1, orders_tensor_core=True),
            # consumed_barrier
            consumed_barrier=plgpu.Barrier(
                num_arrivals=1,
                num_barriers=max_concurrent_steps,
                orders_tensor_core=True,
            ),
            # Accumulator TMEM
            acc_tmem=plgpu.TMEM((block_tile_m, tile_n), jnp.float32, collective=collective),
            collective_axes=("wg",),
        )
        def _scoped(**ref_kwargs):
            @plgpu.nd_loop(grid=(linear_grid,), collective_axes="sm")
            def mn_loop(loop_info: plgpu.NDLoopInfo):
                (linear_idx,) = loop_info.index
                local_index = loop_info.local_index
                m_index, n_index = plgpu.planar_snake(
                    linear_idx,
                    (active_m_iters, n_iters),
                    config.grid_minor_dim,
                    config.grid_tile_width,
                )
                with jax.named_scope("create_group_info"):
                    group_info = GroupInfo.create(group_sizes_regs, tile_m, m_index)
                do_matmul(
                    a_gmem,
                    b_gmem.at[group_info.group_id],
                    out_gmem,
                    # Upstream always passes three indices while `do_matmul` unpacks two
                    # unless collective, so the non-collective path cannot run.
                    grid_indices=(
                        (group_info.block, n_index, cluster_idx) if collective else (group_info.block, n_index)
                    ),
                    wg_axis="wg",
                    collective_axes=("x",) if collective else (),
                    local_index=local_index,
                    previous_total_k_iters=lax.select(local_index > 0, max_concurrent_steps, 0),
                    config=config,
                    group_info=group_info,
                    **ref_kwargs,
                )

    num_sms = jax.local_devices()[0].core_count
    # A fused megakernel must use one setting for transport and compute, and peer
    # transport only lowers under Lane.
    compiler_params = plgpu.CompilerParams(lowering_semantics=lowering_semantics or plgpu.LoweringSemantics.Warpgroup)
    f = plgpu.kernel(
        kernel,
        compiler_params=compiler_params,
        kernel_name=f"ragged_dot_kernel_{config!s}",
        out_type=jax.ShapeDtypeStruct((m, n), dtype),
        grid=(num_sms // 2,) if collective else (num_sms,),
        grid_names=("sm",),
        num_threads=2,
        thread_name="wg",
        cluster_names=("x",) if collective else (),
        cluster=(2,) if collective else (),
    )
    return f(a, b, group_sizes)


def ragged_dot_reference(a, b, g):
    return lax.ragged_dot(a, b, g, preferred_element_type=jnp.float16)


def sample_group_sizes(
    key: jax.Array,
    num_groups: int,
    num_elements: int,
    alpha: float = 10.0,
):
    """Sample group sizes.

    Args:
      key: PRNG key.
      num_groups: Number of groups to sample.
      num_elements: Total number of elements to sample.
      alpha: Shape parameter. The lower the alpha, the more imbalanced the
        group sizes will be. As alpha approaches infinity, the group sizes
        approach a uniform distribution.

    Returns:
      A jax.Array of shape (num_groups,) that sums to num_elements.
    """
    probs_key, sample_key = jax.random.split(key)
    probs = jax.random.dirichlet(probs_key, jnp.ones((num_groups,)) * alpha)
    return jax.random.multinomial(sample_key, num_elements, probs).astype(jnp.int32)
