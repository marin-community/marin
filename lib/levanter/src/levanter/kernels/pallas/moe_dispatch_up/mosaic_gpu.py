# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Mosaic GPU implementation of the MoE dispatch-up subkernel."""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental.pallas.ops.gpu.ragged_dot_mgpu import GroupInfo

from levanter.kernels.pallas.moe_dispatch_up.errors import MosaicGpuUnsupportedError
from levanter.kernels.pallas.moe_dispatch_up.reference import (
    MoeDispatchUpLayout,
    MoeDispatchUpSourceExpertPrepackedSend,
    MoeDispatchUpPrepackedSend,
    dispatch_prepacked_moe_dispatch_up_reference,
)


def _require_mgpu_runtime() -> None:
    gpu_devices = [device for device in jax.local_devices() if device.platform == "gpu"]
    if len(gpu_devices) < 2:
        raise MosaicGpuUnsupportedError(
            "MoE dispatch-up Mosaic GPU dispatch requires at least two local GPU devices; "
            f"found {len(gpu_devices)} GPU device(s)."
        )


def dispatch_prepacked_moe_dispatch_up_mosaic_gpu(
    prepacked: MoeDispatchUpPrepackedSend,
    *,
    recv_capacity: int | None = None,
) -> MoeDispatchUpLayout:
    """Validation-slice dispatch entrypoint for the Pallas MGPU backend.

    The API is intentionally separate from the reference implementation so the
    remote-ref kernel can replace this body without changing callers or tests.
    Until the CoreWeave validation kernel lands, explicit `mosaic_gpu` requests
    fail fast on non-MGPU hosts and use the reference layout only as a local
    bring-up path.
    """

    _require_mgpu_runtime()
    return dispatch_prepacked_moe_dispatch_up_reference(prepacked, recv_capacity=recv_capacity)


def dispatch_prepacked_moe_dispatch_up_mosaic_gpu_local(
    send_x_by_dst: jax.Array,
    send_row_by_dst: jax.Array,
    send_local_expert_by_dst: jax.Array,
    send_src_token_idx_by_dst: jax.Array,
    send_topk_slot_by_dst: jax.Array,
    send_router_weight_by_dst: jax.Array,
    send_count_by_dst: jax.Array,
    rows_per_expert: jax.Array,
    expert_base: jax.Array,
    *,
    axis_name: str,
    recv_capacity: int | None = None,
    copy_mode: str = "row_vector",
) -> MoeDispatchUpLayout:
    """Shard-local MGPU remote-dispatch validation primitive.

    The default `scratch` mode writes routed rows into destination-owned
    symmetric scratch, then locally compacts them into the expert-major receive
    layout. `row_vector` and `scalar` remain as direct remote-write diagnostic
    modes for comparison.
    """

    _require_mgpu_runtime()
    if send_x_by_dst.ndim != 3:
        raise ValueError(f"send_x_by_dst must have shape [EP, S, D], got {send_x_by_dst.shape}")
    ep_size, send_capacity, hidden = send_x_by_dst.shape
    if recv_capacity is None:
        recv_capacity = send_capacity
    if copy_mode not in ("scalar", "row_vector", "scratch"):
        raise ValueError(f"copy_mode must be 'scalar', 'row_vector', or 'scratch', got {copy_mode!r}")
    if copy_mode == "scratch":
        return _dispatch_prepacked_moe_dispatch_up_mosaic_gpu_scratch_local(
            send_x_by_dst,
            send_row_by_dst,
            send_local_expert_by_dst,
            send_src_token_idx_by_dst,
            send_topk_slot_by_dst,
            send_router_weight_by_dst,
            send_count_by_dst,
            rows_per_expert,
            expert_base,
            axis_name=axis_name,
            recv_capacity=recv_capacity,
        )

    def kernel_body(
        send_x_ref,
        send_row_ref,
        send_local_expert_ref,
        send_src_token_idx_ref,
        send_topk_slot_ref,
        send_router_weight_ref,
        send_count_ref,
        recv_x_ref,
        recv_valid_count_ref,
        recv_local_expert_ref,
        recv_src_rank_ref,
        recv_src_token_idx_ref,
        recv_topk_slot_ref,
        recv_router_weight_ref,
    ):
        recv_sem = pl.get_global(plgpu.SemaphoreType.REGULAR)
        src_rank = lax.axis_index(axis_name)

        def _zero_metadata():
            for recv_row in range(recv_capacity):
                recv_valid_count_ref[recv_row] = jnp.int32(0)
                recv_local_expert_ref[recv_row] = jnp.int32(0)
                recv_src_rank_ref[recv_row] = jnp.int32(0)
                recv_src_token_idx_ref[recv_row] = jnp.int32(0)
                recv_topk_slot_ref[recv_row] = jnp.int32(0)
                recv_router_weight_ref[recv_row] = jnp.zeros((), dtype=recv_router_weight_ref.dtype)

        def _zero_recv_x_scalar():
            for recv_row in range(recv_capacity):
                for hidden_idx in range(hidden):
                    recv_x_ref[recv_row, hidden_idx] = jnp.zeros((), dtype=recv_x_ref.dtype)

        def _zero_recv_x_vector():
            for recv_row in range(recv_capacity):
                recv_x_ref.at[recv_row, pl.ds(0, hidden)][...] = jnp.zeros((hidden,), dtype=recv_x_ref.dtype)

        def _copy_rows_scalar():
            for dst_rank in range(ep_size):
                remote_recv_x = plgpu.remote_ref(recv_x_ref, jnp.int32(dst_rank))
                remote_recv_valid_count = plgpu.remote_ref(recv_valid_count_ref, jnp.int32(dst_rank))
                remote_recv_local_expert = plgpu.remote_ref(recv_local_expert_ref, jnp.int32(dst_rank))
                remote_recv_src_rank = plgpu.remote_ref(recv_src_rank_ref, jnp.int32(dst_rank))
                remote_recv_src_token_idx = plgpu.remote_ref(recv_src_token_idx_ref, jnp.int32(dst_rank))
                remote_recv_topk_slot = plgpu.remote_ref(recv_topk_slot_ref, jnp.int32(dst_rank))
                remote_recv_router_weight = plgpu.remote_ref(recv_router_weight_ref, jnp.int32(dst_rank))

                for send_row in range(send_capacity):
                    dst_row = send_row_ref[dst_rank, send_row]
                    valid = (send_row < send_count_ref[dst_rank]) & (dst_row < recv_capacity)

                    @pl.when(valid)
                    def _copy_row():
                        for hidden_idx in range(hidden):
                            remote_recv_x[dst_row, hidden_idx] = send_x_ref[dst_rank, send_row, hidden_idx]
                        remote_recv_valid_count[dst_row] = jnp.int32(1)
                        remote_recv_local_expert[dst_row] = send_local_expert_ref[dst_rank, send_row]
                        remote_recv_src_rank[dst_row] = src_rank
                        remote_recv_src_token_idx[dst_row] = send_src_token_idx_ref[dst_rank, send_row]
                        remote_recv_topk_slot[dst_row] = send_topk_slot_ref[dst_rank, send_row]
                        remote_recv_router_weight[dst_row] = send_router_weight_ref[dst_rank, send_row]

                pl.semaphore_signal(recv_sem, device_id=jnp.int32(dst_rank))

        def _copy_rows_vector():
            for dst_rank in range(ep_size):
                remote_recv_x = plgpu.remote_ref(recv_x_ref, jnp.int32(dst_rank))
                remote_recv_valid_count = plgpu.remote_ref(recv_valid_count_ref, jnp.int32(dst_rank))
                remote_recv_local_expert = plgpu.remote_ref(recv_local_expert_ref, jnp.int32(dst_rank))
                remote_recv_src_rank = plgpu.remote_ref(recv_src_rank_ref, jnp.int32(dst_rank))
                remote_recv_src_token_idx = plgpu.remote_ref(recv_src_token_idx_ref, jnp.int32(dst_rank))
                remote_recv_topk_slot = plgpu.remote_ref(recv_topk_slot_ref, jnp.int32(dst_rank))
                remote_recv_router_weight = plgpu.remote_ref(recv_router_weight_ref, jnp.int32(dst_rank))

                for send_row in range(send_capacity):
                    dst_row = send_row_ref[dst_rank, send_row]
                    valid = (send_row < send_count_ref[dst_rank]) & (dst_row < recv_capacity)

                    @pl.when(valid)
                    def _copy_row():
                        remote_recv_x.at[dst_row, pl.ds(0, hidden)][...] = send_x_ref.at[
                            dst_rank, send_row, pl.ds(0, hidden)
                        ][...]
                        remote_recv_valid_count[dst_row] = jnp.int32(1)
                        remote_recv_local_expert[dst_row] = send_local_expert_ref[dst_rank, send_row]
                        remote_recv_src_rank[dst_row] = src_rank
                        remote_recv_src_token_idx[dst_row] = send_src_token_idx_ref[dst_rank, send_row]
                        remote_recv_topk_slot[dst_row] = send_topk_slot_ref[dst_rank, send_row]
                        remote_recv_router_weight[dst_row] = send_router_weight_ref[dst_rank, send_row]

                pl.semaphore_signal(recv_sem, device_id=jnp.int32(dst_rank))

        _zero_metadata()
        if copy_mode == "scalar":
            _zero_recv_x_scalar()
        else:
            _zero_recv_x_vector()

        for peer_rank in range(ep_size):
            pl.semaphore_signal(recv_sem, device_id=jnp.int32(peer_rank))
        pl.semaphore_wait(recv_sem, value=ep_size, decrement=False)

        if copy_mode == "scalar":
            _copy_rows_scalar()
        else:
            _copy_rows_vector()

        pl.semaphore_wait(recv_sem, value=2 * ep_size, decrement=False)

    (
        recv_x,
        recv_valid_count,
        recv_local_expert,
        recv_src_rank,
        recv_src_token_idx,
        recv_topk_slot,
        recv_router_weight,
    ) = plgpu.kernel(
        kernel_body,
        out_shape=[
            jax.ShapeDtypeStruct((recv_capacity, hidden), send_x_by_dst.dtype),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), send_router_weight_by_dst.dtype),
        ],
        grid=(1,),
        grid_names=("program",),
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
        ),
    )(
        send_x_by_dst,
        send_row_by_dst,
        send_local_expert_by_dst,
        send_src_token_idx_by_dst,
        send_topk_slot_by_dst,
        send_router_weight_by_dst,
        send_count_by_dst,
    )
    overflow_count = jnp.maximum(jnp.sum(rows_per_expert, dtype=jnp.int32) - recv_capacity, 0)
    return MoeDispatchUpLayout(
        recv_x,
        recv_valid_count > 0,
        rows_per_expert,
        expert_base,
        recv_local_expert,
        recv_src_rank,
        recv_src_token_idx,
        recv_topk_slot,
        recv_router_weight,
        overflow_count,
    )


def _dispatch_prepacked_moe_dispatch_up_mosaic_gpu_scratch_local(
    send_x_by_dst: jax.Array,
    send_row_by_dst: jax.Array,
    send_local_expert_by_dst: jax.Array,
    send_src_token_idx_by_dst: jax.Array,
    send_topk_slot_by_dst: jax.Array,
    send_router_weight_by_dst: jax.Array,
    send_count_by_dst: jax.Array,
    rows_per_expert: jax.Array,
    expert_base: jax.Array,
    *,
    axis_name: str,
    recv_capacity: int,
) -> MoeDispatchUpLayout:
    """Dispatch through symmetric scratch with one program per peer row."""

    ep_size, send_capacity, hidden = send_x_by_dst.shape

    def kernel_body(
        send_x_ref,
        send_row_ref,
        send_local_expert_ref,
        send_src_token_idx_ref,
        send_topk_slot_ref,
        send_router_weight_ref,
        send_count_ref,
        recv_x_ref,
        recv_valid_count_ref,
        recv_local_expert_ref,
        recv_src_rank_ref,
        recv_src_token_idx_ref,
        recv_topk_slot_ref,
        recv_router_weight_ref,
        scratch_x_ref,
        scratch_count_ref,
        scratch_dst_row_ref,
        scratch_local_expert_ref,
        scratch_src_token_idx_ref,
        scratch_topk_slot_ref,
        scratch_router_weight_ref,
    ):
        recv_sem = pl.get_global(plgpu.SemaphoreType.REGULAR)
        src_rank = lax.axis_index(axis_name)
        peer_rank = pl.program_id(0)
        send_row = pl.program_id(1)
        linear_row = peer_rank * send_capacity + send_row

        @pl.when(linear_row < recv_capacity)
        def _zero_recv_row():
            recv_x_ref.at[linear_row, pl.ds(0, hidden)][...] = jnp.zeros((hidden,), dtype=recv_x_ref.dtype)
            recv_valid_count_ref[linear_row] = jnp.int32(0)
            recv_local_expert_ref[linear_row] = jnp.int32(0)
            recv_src_rank_ref[linear_row] = jnp.int32(0)
            recv_src_token_idx_ref[linear_row] = jnp.int32(0)
            recv_topk_slot_ref[linear_row] = jnp.int32(0)
            recv_router_weight_ref[linear_row] = jnp.zeros((), dtype=recv_router_weight_ref.dtype)

        remote_scratch_x = plgpu.remote_ref(scratch_x_ref, jnp.int32(peer_rank))
        remote_scratch_count = plgpu.remote_ref(scratch_count_ref, jnp.int32(peer_rank))
        remote_scratch_dst_row = plgpu.remote_ref(scratch_dst_row_ref, jnp.int32(peer_rank))
        remote_scratch_local_expert = plgpu.remote_ref(scratch_local_expert_ref, jnp.int32(peer_rank))
        remote_scratch_src_token_idx = plgpu.remote_ref(scratch_src_token_idx_ref, jnp.int32(peer_rank))
        remote_scratch_topk_slot = plgpu.remote_ref(scratch_topk_slot_ref, jnp.int32(peer_rank))
        remote_scratch_router_weight = plgpu.remote_ref(scratch_router_weight_ref, jnp.int32(peer_rank))

        @pl.when(send_row == 0)
        def _write_count():
            remote_scratch_count[src_rank] = send_count_ref[peer_rank]

        dst_row = send_row_ref[peer_rank, send_row]
        send_valid = (send_row < send_count_ref[peer_rank]) & (dst_row < recv_capacity)

        @pl.when(send_valid)
        def _copy_row_to_scratch():
            remote_scratch_x.at[src_rank, send_row, pl.ds(0, hidden)][...] = send_x_ref.at[
                peer_rank, send_row, pl.ds(0, hidden)
            ][...]
            remote_scratch_dst_row[src_rank, send_row] = dst_row
            remote_scratch_local_expert[src_rank, send_row] = send_local_expert_ref[peer_rank, send_row]
            remote_scratch_src_token_idx[src_rank, send_row] = send_src_token_idx_ref[peer_rank, send_row]
            remote_scratch_topk_slot[src_rank, send_row] = send_topk_slot_ref[peer_rank, send_row]
            remote_scratch_router_weight[src_rank, send_row] = send_router_weight_ref[peer_rank, send_row]

        pl.semaphore_signal(recv_sem, device_id=jnp.int32(peer_rank))
        pl.semaphore_wait(recv_sem, value=ep_size * send_capacity, decrement=False)

        dst_row = scratch_dst_row_ref[peer_rank, send_row]
        recv_valid = (send_row < scratch_count_ref[peer_rank]) & (dst_row < recv_capacity)

        @pl.when(recv_valid)
        def _compact_row():
            scratch_row = scratch_x_ref.at[peer_rank, send_row, pl.ds(0, hidden)]
            recv_x_ref.at[dst_row, pl.ds(0, hidden)][...] = scratch_row[...]
            recv_valid_count_ref[dst_row] = jnp.int32(1)
            recv_local_expert_ref[dst_row] = scratch_local_expert_ref[peer_rank, send_row]
            recv_src_rank_ref[dst_row] = jnp.int32(peer_rank)
            recv_src_token_idx_ref[dst_row] = scratch_src_token_idx_ref[peer_rank, send_row]
            recv_topk_slot_ref[dst_row] = scratch_topk_slot_ref[peer_rank, send_row]
            recv_router_weight_ref[dst_row] = scratch_router_weight_ref[peer_rank, send_row]

    (
        recv_x,
        recv_valid_count,
        recv_local_expert,
        recv_src_rank,
        recv_src_token_idx,
        recv_topk_slot,
        recv_router_weight,
        _scratch_x,
        _scratch_count,
        _scratch_dst_row,
        _scratch_local_expert,
        _scratch_src_token_idx,
        _scratch_topk_slot,
        _scratch_router_weight,
    ) = plgpu.kernel(
        kernel_body,
        out_shape=[
            jax.ShapeDtypeStruct((recv_capacity, hidden), send_x_by_dst.dtype),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), send_router_weight_by_dst.dtype),
            jax.ShapeDtypeStruct((ep_size, send_capacity, hidden), send_x_by_dst.dtype),
            jax.ShapeDtypeStruct((ep_size,), jnp.int32),
            jax.ShapeDtypeStruct((ep_size, send_capacity), jnp.int32),
            jax.ShapeDtypeStruct((ep_size, send_capacity), jnp.int32),
            jax.ShapeDtypeStruct((ep_size, send_capacity), jnp.int32),
            jax.ShapeDtypeStruct((ep_size, send_capacity), jnp.int32),
            jax.ShapeDtypeStruct((ep_size, send_capacity), send_router_weight_by_dst.dtype),
        ],
        grid=(ep_size, send_capacity),
        grid_names=("peer", "row"),
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
        ),
    )(
        send_x_by_dst,
        send_row_by_dst,
        send_local_expert_by_dst,
        send_src_token_idx_by_dst,
        send_topk_slot_by_dst,
        send_router_weight_by_dst,
        send_count_by_dst,
    )
    overflow_count = jnp.maximum(jnp.sum(rows_per_expert, dtype=jnp.int32) - recv_capacity, 0)
    return MoeDispatchUpLayout(
        recv_x,
        recv_valid_count > 0,
        rows_per_expert,
        expert_base,
        recv_local_expert,
        recv_src_rank,
        recv_src_token_idx,
        recv_topk_slot,
        recv_router_weight,
        overflow_count,
    )


def _dispatch_prepacked_moe_dispatch_up_mosaic_gpu_scratch_ready_local(
    send_x_by_dst: jax.Array,
    send_row_by_dst: jax.Array,
    send_local_expert_by_dst: jax.Array,
    send_src_token_idx_by_dst: jax.Array,
    send_topk_slot_by_dst: jax.Array,
    send_router_weight_by_dst: jax.Array,
    send_count_by_dst: jax.Array,
    rows_per_expert: jax.Array,
    expert_base: jax.Array,
    send_expert_base_by_dst: jax.Array,
    send_expert_count_by_dst: jax.Array,
    recv_source_expert_base: jax.Array,
    recv_source_expert_count: jax.Array,
    *,
    axis_name: str,
    recv_capacity: int,
    ready_block_m: int = 64,
    rows_per_program: int = 1,
) -> tuple[MoeDispatchUpLayout, jax.Array, jax.Array]:
    """Scratch dispatch that exposes clipped ready rows per source/expert."""

    if rows_per_program < 1:
        raise ValueError(f"rows_per_program must be positive, got {rows_per_program}")
    ep_size, send_capacity, hidden = send_x_by_dst.shape
    local_experts = rows_per_expert.shape[0]
    recv_blocks = math.ceil(recv_capacity / ready_block_m)
    send_programs = math.ceil(send_capacity / rows_per_program)
    grid_rows = max(send_programs, local_experts, recv_blocks)
    ready_rows = jnp.minimum(jnp.sum(rows_per_expert, dtype=jnp.int32), jnp.int32(recv_capacity))

    def kernel_body(
        send_x_ref,
        send_row_ref,
        send_local_expert_ref,
        send_src_token_idx_ref,
        send_topk_slot_ref,
        send_router_weight_ref,
        send_count_ref,
        send_expert_base_ref,
        send_expert_count_ref,
        recv_source_expert_base_ref,
        recv_source_expert_count_ref,
        ready_rows_ref,
        recv_x_ref,
        recv_valid_count_ref,
        recv_local_expert_ref,
        recv_src_rank_ref,
        recv_src_token_idx_ref,
        recv_topk_slot_ref,
        recv_router_weight_ref,
        ready_count_ref,
        ready_block_count_ref,
        scratch_x_ref,
        scratch_count_ref,
        scratch_dst_row_ref,
        scratch_local_expert_ref,
        scratch_src_token_idx_ref,
        scratch_topk_slot_ref,
        scratch_router_weight_ref,
    ):
        recv_sem = pl.get_global(plgpu.SemaphoreType.REGULAR)
        src_rank = lax.axis_index(axis_name)
        peer_rank = pl.program_id(0)
        row_or_expert = pl.program_id(1)
        send_program_start = row_or_expert * rows_per_program

        @pl.when(row_or_expert < send_programs)
        def _zero_recv_rows():
            for row_offset in range(rows_per_program):
                send_row = send_program_start + row_offset
                linear_row = peer_rank * send_capacity + send_row

                @pl.when((send_row < send_capacity) & (linear_row < recv_capacity))
                def _zero_recv_row():
                    recv_x_ref.at[linear_row, pl.ds(0, hidden)][...] = jnp.zeros((hidden,), dtype=recv_x_ref.dtype)
                    recv_valid_count_ref[linear_row] = jnp.int32(0)
                    recv_local_expert_ref[linear_row] = jnp.int32(0)
                    recv_src_rank_ref[linear_row] = jnp.int32(0)
                    recv_src_token_idx_ref[linear_row] = jnp.int32(0)
                    recv_topk_slot_ref[linear_row] = jnp.int32(0)
                    recv_router_weight_ref[linear_row] = jnp.zeros((), dtype=recv_router_weight_ref.dtype)

        @pl.when(row_or_expert < local_experts)
        def _zero_ready_count():
            ready_count_ref[peer_rank, row_or_expert] = jnp.int32(0)

        @pl.when((peer_rank == 0) & (row_or_expert < recv_blocks))
        def _zero_ready_block_count():
            ready_block_count_ref[row_or_expert] = jnp.int32(0)

        remote_scratch_x = plgpu.remote_ref(scratch_x_ref, jnp.int32(peer_rank))
        remote_scratch_count = plgpu.remote_ref(scratch_count_ref, jnp.int32(peer_rank))
        remote_scratch_dst_row = plgpu.remote_ref(scratch_dst_row_ref, jnp.int32(peer_rank))
        remote_scratch_local_expert = plgpu.remote_ref(scratch_local_expert_ref, jnp.int32(peer_rank))
        remote_scratch_src_token_idx = plgpu.remote_ref(scratch_src_token_idx_ref, jnp.int32(peer_rank))
        remote_scratch_topk_slot = plgpu.remote_ref(scratch_topk_slot_ref, jnp.int32(peer_rank))
        remote_scratch_router_weight = plgpu.remote_ref(scratch_router_weight_ref, jnp.int32(peer_rank))

        @pl.when(row_or_expert == 0)
        def _write_count():
            remote_scratch_count[src_rank] = send_count_ref[peer_rank]

        @pl.when(row_or_expert < local_experts)
        def _touch_source_expert_ranges():
            # These loads keep the producer-side range contract in the compiled
            # path; a later coordinator uses the same base/count pair to signal
            # per-source/expert completion instead of whole-buffer completion.
            _ = send_expert_base_ref[peer_rank, row_or_expert] + send_expert_count_ref[peer_rank, row_or_expert]

        @pl.when(row_or_expert < send_programs)
        def _copy_source_rows():
            for row_offset in range(rows_per_program):
                send_row = send_program_start + row_offset
                safe_send_row = jnp.minimum(send_row, jnp.int32(send_capacity - 1))
                dst_row = send_row_ref[peer_rank, safe_send_row]
                send_valid = (send_row < send_count_ref[peer_rank]) & (dst_row < recv_capacity)

                @pl.when(send_valid)
                def _copy_row_to_scratch():
                    remote_scratch_x.at[src_rank, safe_send_row, pl.ds(0, hidden)][...] = send_x_ref.at[
                        peer_rank, safe_send_row, pl.ds(0, hidden)
                    ][...]
                    remote_scratch_dst_row[src_rank, safe_send_row] = dst_row
                    remote_scratch_local_expert[src_rank, safe_send_row] = send_local_expert_ref[
                        peer_rank, safe_send_row
                    ]
                    remote_scratch_src_token_idx[src_rank, safe_send_row] = send_src_token_idx_ref[
                        peer_rank, safe_send_row
                    ]
                    remote_scratch_topk_slot[src_rank, safe_send_row] = send_topk_slot_ref[peer_rank, safe_send_row]
                    remote_scratch_router_weight[src_rank, safe_send_row] = send_router_weight_ref[
                        peer_rank, safe_send_row
                    ]

        pl.semaphore_signal(recv_sem, device_id=jnp.int32(peer_rank))
        pl.semaphore_wait(recv_sem, value=ep_size * grid_rows, decrement=False)

        @pl.when(row_or_expert < send_programs)
        def _compact_source_rows():
            for row_offset in range(rows_per_program):
                send_row = send_program_start + row_offset
                safe_send_row = jnp.minimum(send_row, jnp.int32(send_capacity - 1))
                dst_row = scratch_dst_row_ref[peer_rank, safe_send_row]
                recv_valid = (send_row < scratch_count_ref[peer_rank]) & (dst_row < recv_capacity)

                @pl.when(recv_valid)
                def _compact_row():
                    scratch_row = scratch_x_ref.at[peer_rank, safe_send_row, pl.ds(0, hidden)]
                    recv_x_ref.at[dst_row, pl.ds(0, hidden)][...] = scratch_row[...]
                    recv_valid_count_ref[dst_row] = jnp.int32(1)
                    recv_local_expert_ref[dst_row] = scratch_local_expert_ref[peer_rank, safe_send_row]
                    recv_src_rank_ref[dst_row] = jnp.int32(peer_rank)
                    recv_src_token_idx_ref[dst_row] = scratch_src_token_idx_ref[peer_rank, safe_send_row]
                    recv_topk_slot_ref[dst_row] = scratch_topk_slot_ref[peer_rank, safe_send_row]
                    recv_router_weight_ref[dst_row] = scratch_router_weight_ref[peer_rank, safe_send_row]

        pl.semaphore_signal(recv_sem, device_id=src_rank)
        pl.semaphore_wait(recv_sem, value=2 * ep_size * grid_rows, decrement=False)

        @pl.when(row_or_expert < local_experts)
        def _mark_ready_count():
            base = recv_source_expert_base_ref[peer_rank, row_or_expert]
            count = recv_source_expert_count_ref[peer_rank, row_or_expert]
            remaining_capacity = jnp.maximum(jnp.int32(recv_capacity) - base, jnp.int32(0))
            ready_count_ref[peer_rank, row_or_expert] = jnp.minimum(count, remaining_capacity)

        @pl.when((peer_rank == 0) & (row_or_expert < recv_blocks))
        def _mark_ready_block_count():
            block_start = row_or_expert * ready_block_m
            remaining_rows = jnp.maximum(ready_rows_ref[()] - block_start, jnp.int32(0))
            ready_block_count_ref[row_or_expert] = jnp.minimum(remaining_rows, jnp.int32(ready_block_m))

    (
        recv_x,
        recv_valid_count,
        recv_local_expert,
        recv_src_rank,
        recv_src_token_idx,
        recv_topk_slot,
        recv_router_weight,
        ready_count,
        ready_block_count,
        _scratch_x,
        _scratch_count,
        _scratch_dst_row,
        _scratch_local_expert,
        _scratch_src_token_idx,
        _scratch_topk_slot,
        _scratch_router_weight,
    ) = plgpu.kernel(
        kernel_body,
        out_shape=[
            jax.ShapeDtypeStruct((recv_capacity, hidden), send_x_by_dst.dtype),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), send_router_weight_by_dst.dtype),
            jax.ShapeDtypeStruct((ep_size, local_experts), jnp.int32),
            jax.ShapeDtypeStruct((recv_blocks,), jnp.int32),
            jax.ShapeDtypeStruct((ep_size, send_capacity, hidden), send_x_by_dst.dtype),
            jax.ShapeDtypeStruct((ep_size,), jnp.int32),
            jax.ShapeDtypeStruct((ep_size, send_capacity), jnp.int32),
            jax.ShapeDtypeStruct((ep_size, send_capacity), jnp.int32),
            jax.ShapeDtypeStruct((ep_size, send_capacity), jnp.int32),
            jax.ShapeDtypeStruct((ep_size, send_capacity), jnp.int32),
            jax.ShapeDtypeStruct((ep_size, send_capacity), send_router_weight_by_dst.dtype),
        ],
        grid=(ep_size, grid_rows),
        grid_names=("peer", "row_or_expert"),
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
        ),
    )(
        send_x_by_dst,
        send_row_by_dst,
        send_local_expert_by_dst,
        send_src_token_idx_by_dst,
        send_topk_slot_by_dst,
        send_router_weight_by_dst,
        send_count_by_dst,
        send_expert_base_by_dst,
        send_expert_count_by_dst,
        recv_source_expert_base,
        recv_source_expert_count,
        ready_rows,
    )
    overflow_count = jnp.maximum(jnp.sum(rows_per_expert, dtype=jnp.int32) - recv_capacity, 0)
    layout = MoeDispatchUpLayout(
        recv_x,
        recv_valid_count > 0,
        rows_per_expert,
        expert_base,
        recv_local_expert,
        recv_src_rank,
        recv_src_token_idx,
        recv_topk_slot,
        recv_router_weight,
        overflow_count,
    )
    return layout, ready_count, ready_block_count


def dispatch_prepacked_moe_dispatch_up_mosaic_gpu_ready_local(
    send_x_by_dst: jax.Array,
    send_row_by_dst: jax.Array,
    send_local_expert_by_dst: jax.Array,
    send_src_token_idx_by_dst: jax.Array,
    send_topk_slot_by_dst: jax.Array,
    send_router_weight_by_dst: jax.Array,
    send_count_by_dst: jax.Array,
    rows_per_expert: jax.Array,
    expert_base: jax.Array,
    send_expert_base_by_dst: jax.Array,
    send_expert_count_by_dst: jax.Array,
    recv_source_expert_base: jax.Array,
    recv_source_expert_count: jax.Array,
    *,
    axis_name: str,
    recv_capacity: int,
    ready_block_m: int = 64,
    rows_per_program: int = 1,
) -> tuple[MoeDispatchUpLayout, jax.Array, jax.Array]:
    """Scratch dispatch plus source/expert ready counts for overlap bring-up."""

    _require_mgpu_runtime()
    if send_x_by_dst.ndim != 3:
        raise ValueError(f"send_x_by_dst must have shape [EP, S, D], got {send_x_by_dst.shape}")
    ep_size, send_capacity, hidden = send_x_by_dst.shape
    local_experts = rows_per_expert.shape[0]
    if send_expert_base_by_dst.shape != (ep_size, local_experts):
        raise ValueError(
            "send_expert_base_by_dst must have shape "
            f"{(ep_size, local_experts)}, got {send_expert_base_by_dst.shape}"
        )
    if send_expert_count_by_dst.shape != (ep_size, local_experts):
        raise ValueError(
            "send_expert_count_by_dst must have shape "
            f"{(ep_size, local_experts)}, got {send_expert_count_by_dst.shape}"
        )
    if recv_source_expert_base.shape != (ep_size, local_experts):
        raise ValueError(
            "recv_source_expert_base must have shape "
            f"{(ep_size, local_experts)}, got {recv_source_expert_base.shape}"
        )
    if recv_source_expert_count.shape != (ep_size, local_experts):
        raise ValueError(
            "recv_source_expert_count must have shape "
            f"{(ep_size, local_experts)}, got {recv_source_expert_count.shape}"
        )

    return _dispatch_prepacked_moe_dispatch_up_mosaic_gpu_scratch_ready_local(
        send_x_by_dst,
        send_row_by_dst,
        send_local_expert_by_dst,
        send_src_token_idx_by_dst,
        send_topk_slot_by_dst,
        send_router_weight_by_dst,
        send_count_by_dst,
        rows_per_expert,
        expert_base,
        send_expert_base_by_dst,
        send_expert_count_by_dst,
        recv_source_expert_base,
        recv_source_expert_count,
        axis_name=axis_name,
        recv_capacity=recv_capacity,
        ready_block_m=ready_block_m,
        rows_per_program=rows_per_program,
    )


def _dispatch_prepacked_moe_dispatch_up_mosaic_gpu_direct_ready_local(
    send_x_by_dst: jax.Array,
    send_row_by_dst: jax.Array,
    send_local_expert_by_dst: jax.Array,
    send_src_token_idx_by_dst: jax.Array,
    send_topk_slot_by_dst: jax.Array,
    send_router_weight_by_dst: jax.Array,
    send_count_by_dst: jax.Array,
    rows_per_expert: jax.Array,
    expert_base: jax.Array,
    recv_source_expert_base: jax.Array,
    recv_source_expert_count: jax.Array,
    *,
    axis_name: str,
    recv_capacity: int,
    ready_block_m: int = 64,
    rows_per_program: int = 1,
) -> tuple[MoeDispatchUpLayout, jax.Array, jax.Array]:
    """Direct remote dispatch that exposes clipped readiness metadata."""

    if rows_per_program < 1:
        raise ValueError(f"rows_per_program must be positive, got {rows_per_program}")
    ep_size, send_capacity, hidden = send_x_by_dst.shape
    local_experts = rows_per_expert.shape[0]
    recv_blocks = math.ceil(recv_capacity / ready_block_m)
    send_programs = math.ceil(send_capacity / rows_per_program)
    zero_programs = math.ceil(recv_capacity / rows_per_program)
    grid_rows = max(send_programs, zero_programs, local_experts, recv_blocks)
    ready_rows = jnp.minimum(jnp.sum(rows_per_expert, dtype=jnp.int32), jnp.int32(recv_capacity))

    def kernel_body(
        send_x_ref,
        send_row_ref,
        send_local_expert_ref,
        send_src_token_idx_ref,
        send_topk_slot_ref,
        send_router_weight_ref,
        send_count_ref,
        recv_source_expert_base_ref,
        recv_source_expert_count_ref,
        ready_rows_ref,
        recv_x_ref,
        recv_valid_count_ref,
        recv_local_expert_ref,
        recv_src_rank_ref,
        recv_src_token_idx_ref,
        recv_topk_slot_ref,
        recv_router_weight_ref,
        ready_count_ref,
        ready_block_count_ref,
    ):
        recv_sem = pl.get_global(plgpu.SemaphoreType.REGULAR)
        src_rank = lax.axis_index(axis_name)
        peer_rank = pl.program_id(0)
        row_or_expert = pl.program_id(1)
        row_start = row_or_expert * rows_per_program

        @pl.when(row_or_expert < zero_programs)
        def _zero_recv_rows():
            for row_offset in range(rows_per_program):
                recv_row = row_start + row_offset

                @pl.when(recv_row < recv_capacity)
                def _zero_recv_row():
                    recv_x_ref.at[recv_row, pl.ds(0, hidden)][...] = jnp.zeros((hidden,), dtype=recv_x_ref.dtype)
                    recv_valid_count_ref[recv_row] = jnp.int32(0)
                    recv_local_expert_ref[recv_row] = jnp.int32(0)
                    recv_src_rank_ref[recv_row] = jnp.int32(0)
                    recv_src_token_idx_ref[recv_row] = jnp.int32(0)
                    recv_topk_slot_ref[recv_row] = jnp.int32(0)
                    recv_router_weight_ref[recv_row] = jnp.zeros((), dtype=recv_router_weight_ref.dtype)

        @pl.when(row_or_expert < local_experts)
        def _zero_ready_count():
            ready_count_ref[peer_rank, row_or_expert] = jnp.int32(0)

        @pl.when((peer_rank == 0) & (row_or_expert < recv_blocks))
        def _zero_ready_block_count():
            ready_block_count_ref[row_or_expert] = jnp.int32(0)

        remote_recv_x = plgpu.remote_ref(recv_x_ref, jnp.int32(peer_rank))
        remote_recv_valid_count = plgpu.remote_ref(recv_valid_count_ref, jnp.int32(peer_rank))
        remote_recv_local_expert = plgpu.remote_ref(recv_local_expert_ref, jnp.int32(peer_rank))
        remote_recv_src_rank = plgpu.remote_ref(recv_src_rank_ref, jnp.int32(peer_rank))
        remote_recv_src_token_idx = plgpu.remote_ref(recv_src_token_idx_ref, jnp.int32(peer_rank))
        remote_recv_topk_slot = plgpu.remote_ref(recv_topk_slot_ref, jnp.int32(peer_rank))
        remote_recv_router_weight = plgpu.remote_ref(recv_router_weight_ref, jnp.int32(peer_rank))

        pl.semaphore_signal(recv_sem, device_id=jnp.int32(peer_rank))
        pl.semaphore_wait(recv_sem, value=ep_size * grid_rows, decrement=False)

        @pl.when(row_or_expert < send_programs)
        def _copy_source_rows():
            for row_offset in range(rows_per_program):
                send_row = row_start + row_offset
                safe_send_row = jnp.minimum(send_row, jnp.int32(send_capacity - 1))
                dst_row = send_row_ref[peer_rank, safe_send_row]
                send_valid = (send_row < send_count_ref[peer_rank]) & (dst_row < recv_capacity)

                @pl.when(send_valid)
                def _copy_row_to_recv():
                    remote_recv_x.at[dst_row, pl.ds(0, hidden)][...] = send_x_ref.at[
                        peer_rank, safe_send_row, pl.ds(0, hidden)
                    ][...]
                    remote_recv_valid_count[dst_row] = jnp.int32(1)
                    remote_recv_local_expert[dst_row] = send_local_expert_ref[peer_rank, safe_send_row]
                    remote_recv_src_rank[dst_row] = src_rank
                    remote_recv_src_token_idx[dst_row] = send_src_token_idx_ref[peer_rank, safe_send_row]
                    remote_recv_topk_slot[dst_row] = send_topk_slot_ref[peer_rank, safe_send_row]
                    remote_recv_router_weight[dst_row] = send_router_weight_ref[peer_rank, safe_send_row]

        pl.semaphore_signal(recv_sem, device_id=jnp.int32(peer_rank))
        pl.semaphore_wait(recv_sem, value=2 * ep_size * grid_rows, decrement=False)

        @pl.when(row_or_expert < local_experts)
        def _mark_ready_count():
            base = recv_source_expert_base_ref[peer_rank, row_or_expert]
            count = recv_source_expert_count_ref[peer_rank, row_or_expert]
            remaining_capacity = jnp.maximum(jnp.int32(recv_capacity) - base, jnp.int32(0))
            ready_count_ref[peer_rank, row_or_expert] = jnp.minimum(count, remaining_capacity)

        @pl.when((peer_rank == 0) & (row_or_expert < recv_blocks))
        def _mark_ready_block_count():
            block_start = row_or_expert * ready_block_m
            remaining_rows = jnp.maximum(ready_rows_ref[()] - block_start, jnp.int32(0))
            ready_block_count_ref[row_or_expert] = jnp.minimum(remaining_rows, jnp.int32(ready_block_m))

    (
        recv_x,
        recv_valid_count,
        recv_local_expert,
        recv_src_rank,
        recv_src_token_idx,
        recv_topk_slot,
        recv_router_weight,
        ready_count,
        ready_block_count,
    ) = plgpu.kernel(
        kernel_body,
        out_shape=[
            jax.ShapeDtypeStruct((recv_capacity, hidden), send_x_by_dst.dtype),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), send_router_weight_by_dst.dtype),
            jax.ShapeDtypeStruct((ep_size, local_experts), jnp.int32),
            jax.ShapeDtypeStruct((recv_blocks,), jnp.int32),
        ],
        grid=(ep_size, grid_rows),
        grid_names=("peer", "row_or_expert"),
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
        ),
    )(
        send_x_by_dst,
        send_row_by_dst,
        send_local_expert_by_dst,
        send_src_token_idx_by_dst,
        send_topk_slot_by_dst,
        send_router_weight_by_dst,
        send_count_by_dst,
        recv_source_expert_base,
        recv_source_expert_count,
        ready_rows,
    )
    overflow_count = jnp.maximum(jnp.sum(rows_per_expert, dtype=jnp.int32) - recv_capacity, 0)
    layout = MoeDispatchUpLayout(
        recv_x,
        recv_valid_count > 0,
        rows_per_expert,
        expert_base,
        recv_local_expert,
        recv_src_rank,
        recv_src_token_idx,
        recv_topk_slot,
        recv_router_weight,
        overflow_count,
    )
    return layout, ready_count, ready_block_count


def dispatch_prepacked_moe_dispatch_up_mosaic_gpu_direct_ready_local(
    send_x_by_dst: jax.Array,
    send_row_by_dst: jax.Array,
    send_local_expert_by_dst: jax.Array,
    send_src_token_idx_by_dst: jax.Array,
    send_topk_slot_by_dst: jax.Array,
    send_router_weight_by_dst: jax.Array,
    send_count_by_dst: jax.Array,
    rows_per_expert: jax.Array,
    expert_base: jax.Array,
    recv_source_expert_base: jax.Array,
    recv_source_expert_count: jax.Array,
    *,
    axis_name: str,
    recv_capacity: int,
    ready_block_m: int = 64,
    rows_per_program: int = 1,
) -> tuple[MoeDispatchUpLayout, jax.Array, jax.Array]:
    """Direct remote dispatch plus source/expert and block ready counts."""

    _require_mgpu_runtime()
    if send_x_by_dst.ndim != 3:
        raise ValueError(f"send_x_by_dst must have shape [EP, S, D], got {send_x_by_dst.shape}")
    ep_size, _send_capacity, _hidden = send_x_by_dst.shape
    local_experts = rows_per_expert.shape[0]
    if recv_source_expert_base.shape != (ep_size, local_experts):
        raise ValueError(
            "recv_source_expert_base must have shape "
            f"{(ep_size, local_experts)}, got {recv_source_expert_base.shape}"
        )
    if recv_source_expert_count.shape != (ep_size, local_experts):
        raise ValueError(
            "recv_source_expert_count must have shape "
            f"{(ep_size, local_experts)}, got {recv_source_expert_count.shape}"
        )

    return _dispatch_prepacked_moe_dispatch_up_mosaic_gpu_direct_ready_local(
        send_x_by_dst,
        send_row_by_dst,
        send_local_expert_by_dst,
        send_src_token_idx_by_dst,
        send_topk_slot_by_dst,
        send_router_weight_by_dst,
        send_count_by_dst,
        rows_per_expert,
        expert_base,
        recv_source_expert_base,
        recv_source_expert_count,
        axis_name=axis_name,
        recv_capacity=recv_capacity,
        ready_block_m=ready_block_m,
        rows_per_program=rows_per_program,
    )


def _dispatch_prepacked_moe_dispatch_up_mosaic_gpu_source_expert_local(
    send_x_by_dst_expert: jax.Array,
    send_src_token_idx_by_dst_expert: jax.Array,
    send_topk_slot_by_dst_expert: jax.Array,
    send_router_weight_by_dst_expert: jax.Array,
    send_row_base_by_dst_expert: jax.Array,
    send_count_by_dst_expert: jax.Array,
    rows_per_expert: jax.Array,
    expert_base: jax.Array,
    recv_source_expert_base: jax.Array,
    recv_source_expert_count: jax.Array,
    *,
    axis_name: str,
    recv_capacity: int,
    ready_block_m: int = 64,
    rows_per_program: int = 1,
    zero_recv: bool = True,
) -> tuple[MoeDispatchUpLayout, jax.Array, jax.Array]:
    """Compact source/expert remote dispatch with readiness metadata."""

    if rows_per_program < 1:
        raise ValueError(f"rows_per_program must be positive, got {rows_per_program}")
    ep_size, local_experts, source_expert_capacity, hidden = send_x_by_dst_expert.shape
    copy_cols = 256 if hidden >= 256 and hidden % 256 == 0 else hidden
    recv_blocks = math.ceil(recv_capacity / ready_block_m)
    copy_chunks = math.ceil(source_expert_capacity / rows_per_program)
    zero_programs = math.ceil(recv_capacity / rows_per_program)
    copy_programs = local_experts * copy_chunks
    grid_rows = max(zero_programs if zero_recv else 0, copy_programs, local_experts, recv_blocks)
    ready_rows = jnp.minimum(jnp.sum(rows_per_expert, dtype=jnp.int32), jnp.int32(recv_capacity))

    def kernel_body(
        send_x_ref,
        send_src_token_idx_ref,
        send_topk_slot_ref,
        send_router_weight_ref,
        send_row_base_ref,
        send_count_ref,
        recv_source_expert_base_ref,
        recv_source_expert_count_ref,
        ready_rows_ref,
        recv_x_ref,
        recv_valid_count_ref,
        recv_local_expert_ref,
        recv_src_rank_ref,
        recv_src_token_idx_ref,
        recv_topk_slot_ref,
        recv_router_weight_ref,
        ready_count_ref,
        ready_block_count_ref,
    ):
        recv_sem = pl.get_global(plgpu.SemaphoreType.REGULAR)
        src_rank = lax.axis_index(axis_name)
        dst_rank = pl.program_id(0)
        row_or_group = pl.program_id(1)

        if zero_recv:

            @pl.when((dst_rank == 0) & (row_or_group < zero_programs))
            def _zero_recv_rows():
                row_start = row_or_group * rows_per_program
                for row_offset in range(rows_per_program):
                    recv_row = row_start + row_offset

                    @pl.when(recv_row < recv_capacity)
                    def _zero_recv_row():
                        for col_start in range(0, hidden, copy_cols):
                            recv_x_ref.at[recv_row, pl.ds(col_start, copy_cols)][...] = jnp.zeros(
                                (copy_cols,), dtype=recv_x_ref.dtype
                            )
                        recv_valid_count_ref[recv_row] = jnp.int32(0)
                        recv_local_expert_ref[recv_row] = jnp.int32(0)
                        recv_src_rank_ref[recv_row] = jnp.int32(0)
                        recv_src_token_idx_ref[recv_row] = jnp.int32(0)
                        recv_topk_slot_ref[recv_row] = jnp.int32(0)
                        recv_router_weight_ref[recv_row] = jnp.zeros((), dtype=recv_router_weight_ref.dtype)

        @pl.when(row_or_group < local_experts)
        def _zero_ready_count():
            ready_count_ref[dst_rank, row_or_group] = jnp.int32(0)

        @pl.when((dst_rank == 0) & (row_or_group < recv_blocks))
        def _zero_ready_block_count():
            ready_block_count_ref[row_or_group] = jnp.int32(0)

        if zero_recv:
            pl.semaphore_signal(recv_sem, device_id=dst_rank)
            pl.semaphore_wait(recv_sem, value=ep_size * grid_rows, decrement=False)

        remote_recv_x = plgpu.remote_ref(recv_x_ref, jnp.int32(dst_rank))
        remote_recv_valid_count = plgpu.remote_ref(recv_valid_count_ref, jnp.int32(dst_rank))
        remote_recv_local_expert = plgpu.remote_ref(recv_local_expert_ref, jnp.int32(dst_rank))
        remote_recv_src_rank = plgpu.remote_ref(recv_src_rank_ref, jnp.int32(dst_rank))
        remote_recv_src_token_idx = plgpu.remote_ref(recv_src_token_idx_ref, jnp.int32(dst_rank))
        remote_recv_topk_slot = plgpu.remote_ref(recv_topk_slot_ref, jnp.int32(dst_rank))
        remote_recv_router_weight = plgpu.remote_ref(recv_router_weight_ref, jnp.int32(dst_rank))

        @pl.when(row_or_group < copy_programs)
        def _copy_source_expert_rows():
            local_expert = row_or_group // copy_chunks
            chunk = row_or_group - local_expert * copy_chunks
            row_start = chunk * rows_per_program
            count = send_count_ref[dst_rank, local_expert]
            dst_base = send_row_base_ref[dst_rank, local_expert]

            for row_offset in range(rows_per_program):
                source_expert_row = row_start + row_offset
                safe_row = jnp.minimum(source_expert_row, jnp.int32(source_expert_capacity - 1))
                dst_row = dst_base + source_expert_row
                send_valid = (source_expert_row < count) & (dst_row < recv_capacity)

                @pl.when(send_valid)
                def _copy_row_to_recv():
                    for col_start in range(0, hidden, copy_cols):
                        remote_recv_x.at[dst_row, pl.ds(col_start, copy_cols)][...] = send_x_ref.at[
                            dst_rank, local_expert, safe_row, pl.ds(col_start, copy_cols)
                        ][...]
                    remote_recv_valid_count[dst_row] = jnp.int32(1)
                    remote_recv_local_expert[dst_row] = local_expert
                    remote_recv_src_rank[dst_row] = src_rank
                    remote_recv_src_token_idx[dst_row] = send_src_token_idx_ref[dst_rank, local_expert, safe_row]
                    remote_recv_topk_slot[dst_row] = send_topk_slot_ref[dst_rank, local_expert, safe_row]
                    remote_recv_router_weight[dst_row] = send_router_weight_ref[dst_rank, local_expert, safe_row]

        pl.semaphore_signal(recv_sem, device_id=dst_rank)
        pl.semaphore_wait(
            recv_sem,
            value=(2 if zero_recv else 1) * ep_size * grid_rows,
            decrement=False,
        )

        @pl.when(row_or_group < local_experts)
        def _mark_ready_count():
            base = recv_source_expert_base_ref[dst_rank, row_or_group]
            count = recv_source_expert_count_ref[dst_rank, row_or_group]
            remaining_capacity = jnp.maximum(jnp.int32(recv_capacity) - base, jnp.int32(0))
            ready_count_ref[dst_rank, row_or_group] = jnp.minimum(count, remaining_capacity)

        @pl.when((dst_rank == 0) & (row_or_group < recv_blocks))
        def _mark_ready_block_count():
            block_start = row_or_group * ready_block_m
            remaining_rows = jnp.maximum(ready_rows_ref[()] - block_start, jnp.int32(0))
            ready_block_count_ref[row_or_group] = jnp.minimum(remaining_rows, jnp.int32(ready_block_m))

    (
        recv_x,
        recv_valid_count,
        recv_local_expert,
        recv_src_rank,
        recv_src_token_idx,
        recv_topk_slot,
        recv_router_weight,
        ready_count,
        ready_block_count,
    ) = plgpu.kernel(
        kernel_body,
        out_shape=[
            jax.ShapeDtypeStruct((recv_capacity, hidden), send_x_by_dst_expert.dtype),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), send_router_weight_by_dst_expert.dtype),
            jax.ShapeDtypeStruct((ep_size, local_experts), jnp.int32),
            jax.ShapeDtypeStruct((recv_blocks,), jnp.int32),
        ],
        grid=(ep_size, grid_rows),
        grid_names=("dst", "row_or_group"),
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
        ),
    )(
        send_x_by_dst_expert,
        send_src_token_idx_by_dst_expert,
        send_topk_slot_by_dst_expert,
        send_router_weight_by_dst_expert,
        send_row_base_by_dst_expert,
        send_count_by_dst_expert,
        recv_source_expert_base,
        recv_source_expert_count,
        ready_rows,
    )
    overflow_count = jnp.maximum(jnp.sum(rows_per_expert, dtype=jnp.int32) - recv_capacity, 0)
    recv_valid = recv_valid_count > 0 if zero_recv else jnp.arange(recv_capacity, dtype=jnp.int32) < ready_rows
    layout = MoeDispatchUpLayout(
        recv_x,
        recv_valid,
        rows_per_expert,
        expert_base,
        recv_local_expert,
        recv_src_rank,
        recv_src_token_idx,
        recv_topk_slot,
        recv_router_weight,
        overflow_count,
    )
    return layout, ready_count, ready_block_count


def dispatch_prepacked_moe_dispatch_up_mosaic_gpu_source_expert_local(
    prepacked: MoeDispatchUpSourceExpertPrepackedSend,
    *,
    axis_name: str,
    recv_capacity: int,
    ready_block_m: int = 64,
    rows_per_program: int = 1,
    zero_recv: bool = True,
) -> tuple[MoeDispatchUpLayout, jax.Array, jax.Array]:
    """Compact source/expert dispatch plus source/expert and block ready counts."""

    _require_mgpu_runtime()
    if prepacked.send_x_by_dst_expert.ndim != 4:
        raise ValueError(
            "local compact send_x_by_dst_expert must have shape [EP, EL, C, D], "
            f"got {prepacked.send_x_by_dst_expert.shape}"
        )
    ep_size, local_experts, _capacity, _hidden = prepacked.send_x_by_dst_expert.shape
    if prepacked.send_count_by_dst_expert.shape != (ep_size, local_experts):
        raise ValueError(
            "local send_count_by_dst_expert must have shape "
            f"{(ep_size, local_experts)}, got {prepacked.send_count_by_dst_expert.shape}"
        )
    if prepacked.send_row_base_by_dst_expert.shape != (ep_size, local_experts):
        raise ValueError(
            "local send_row_base_by_dst_expert must have shape "
            f"{(ep_size, local_experts)}, got {prepacked.send_row_base_by_dst_expert.shape}"
        )
    if prepacked.recv_source_expert_base.shape != (ep_size, local_experts):
        raise ValueError(
            "local recv_source_expert_base must have shape "
            f"{(ep_size, local_experts)}, got {prepacked.recv_source_expert_base.shape}"
        )
    if prepacked.recv_source_expert_count.shape != (ep_size, local_experts):
        raise ValueError(
            "local recv_source_expert_count must have shape "
            f"{(ep_size, local_experts)}, got {prepacked.recv_source_expert_count.shape}"
        )

    return _dispatch_prepacked_moe_dispatch_up_mosaic_gpu_source_expert_local(
        prepacked.send_x_by_dst_expert,
        prepacked.send_src_token_idx_by_dst_expert,
        prepacked.send_topk_slot_by_dst_expert,
        prepacked.send_router_weight_by_dst_expert,
        prepacked.send_row_base_by_dst_expert,
        prepacked.send_count_by_dst_expert,
        prepacked.rows_per_expert,
        prepacked.expert_base,
        prepacked.recv_source_expert_base,
        prepacked.recv_source_expert_count,
        axis_name=axis_name,
        recv_capacity=recv_capacity,
        ready_block_m=ready_block_m,
        rows_per_program=rows_per_program,
        zero_recv=zero_recv,
    )


def _dispatch_prepacked_moe_dispatch_up_mosaic_gpu_source_expert_tiled_local(
    send_x_by_dst_expert: jax.Array,
    send_src_token_idx_by_dst_expert: jax.Array,
    send_topk_slot_by_dst_expert: jax.Array,
    send_router_weight_by_dst_expert: jax.Array,
    send_row_base_by_dst_expert: jax.Array,
    send_count_by_dst_expert: jax.Array,
    rows_per_expert: jax.Array,
    expert_base: jax.Array,
    recv_source_expert_base: jax.Array,
    recv_source_expert_count: jax.Array,
    *,
    axis_name: str,
    recv_capacity: int,
    ready_block_m: int = 64,
    copy_cols: int = 256,
    copy_rows: int = 1,
) -> tuple[MoeDispatchUpLayout, jax.Array, jax.Array]:
    """Compact source/expert remote dispatch tiled over row and hidden columns."""

    ep_size, local_experts, source_expert_capacity, hidden = send_x_by_dst_expert.shape
    if copy_cols < 1:
        raise ValueError(f"copy_cols must be positive, got {copy_cols}")
    if copy_cols > 256:
        raise ValueError(f"copy_cols must be <= 256 for Mosaic async copies, got {copy_cols}")
    if hidden % copy_cols != 0:
        raise ValueError(f"hidden={hidden} must be divisible by copy_cols={copy_cols}")
    if copy_rows < 1:
        raise ValueError(f"copy_rows must be positive, got {copy_rows}")
    if copy_rows > 256:
        raise ValueError(f"copy_rows must be <= 256 for Mosaic async copies, got {copy_rows}")
    recv_blocks = math.ceil(recv_capacity / ready_block_m)
    row_tiles = math.ceil(source_expert_capacity / copy_rows)
    row_or_block_count = max(row_tiles, recv_blocks)
    col_tiles = hidden // copy_cols
    ready_rows = jnp.minimum(jnp.sum(rows_per_expert, dtype=jnp.int32), jnp.int32(recv_capacity))

    def kernel_body(
        send_x_ref,
        send_src_token_idx_ref,
        send_topk_slot_ref,
        send_router_weight_ref,
        send_row_base_ref,
        send_count_ref,
        recv_source_expert_base_ref,
        recv_source_expert_count_ref,
        ready_rows_ref,
        recv_x_ref,
        recv_local_expert_ref,
        recv_src_rank_ref,
        recv_src_token_idx_ref,
        recv_topk_slot_ref,
        recv_router_weight_ref,
        ready_count_ref,
        ready_block_count_ref,
    ):
        src_rank = lax.axis_index(axis_name)
        dst_rank = pl.program_id(0)
        local_expert = pl.program_id(1)
        row_or_block = pl.program_id(2)
        col_tile = pl.program_id(3)
        source_row_start = row_or_block * copy_rows
        col_start = col_tile * copy_cols

        remote_recv_x = plgpu.remote_ref(recv_x_ref, jnp.int32(dst_rank))
        remote_recv_local_expert = plgpu.remote_ref(recv_local_expert_ref, jnp.int32(dst_rank))
        remote_recv_src_rank = plgpu.remote_ref(recv_src_rank_ref, jnp.int32(dst_rank))
        remote_recv_src_token_idx = plgpu.remote_ref(recv_src_token_idx_ref, jnp.int32(dst_rank))
        remote_recv_topk_slot = plgpu.remote_ref(recv_topk_slot_ref, jnp.int32(dst_rank))
        remote_recv_router_weight = plgpu.remote_ref(recv_router_weight_ref, jnp.int32(dst_rank))

        def _write_row_metadata(source_row, dst_row):
            remote_recv_local_expert[dst_row] = local_expert
            remote_recv_src_rank[dst_row] = src_rank
            remote_recv_src_token_idx[dst_row] = send_src_token_idx_ref[dst_rank, local_expert, source_row]
            remote_recv_topk_slot[dst_row] = send_topk_slot_ref[dst_rank, local_expert, source_row]
            remote_recv_router_weight[dst_row] = send_router_weight_ref[dst_rank, local_expert, source_row]

        def _copy_one_row(source_row, dst_row):
            @functools.partial(
                pl.run_scoped,
                tile_smem=plgpu.SMEM((copy_cols,), dtype=send_x_ref.dtype),
                barrier=plgpu.Barrier(),
            )
            def _copy_payload_tile(tile_smem, barrier):
                src_slice = send_x_ref.at[dst_rank, local_expert, source_row, pl.ds(col_start, copy_cols)]
                plgpu.copy_gmem_to_smem(src_slice, tile_smem, barrier)
                plgpu.barrier_wait(barrier)
                dst_slice = remote_recv_x.at[dst_row, pl.ds(col_start, copy_cols)]
                plgpu.copy_smem_to_gmem(tile_smem, dst_slice)
                plgpu.wait_smem_to_gmem(0, wait_read_only=False)

        @pl.when(row_or_block < row_tiles)
        def _copy_source_expert_tile():
            count = send_count_ref[dst_rank, local_expert]
            dst_base = send_row_base_ref[dst_rank, local_expert]
            dst_row_start = dst_base + source_row_start
            for row_offset in range(copy_rows):
                source_row = source_row_start + row_offset
                dst_row = dst_row_start + row_offset
                send_valid = (source_row < count) & (dst_row < recv_capacity)

                @pl.when(send_valid)
                def _copy_row_to_recv():
                    _copy_one_row(source_row, dst_row)

                    @pl.when(col_tile == 0)
                    def _write_partial_metadata():
                        _write_row_metadata(source_row, dst_row)

        @pl.when((col_tile == 0) & (row_or_block == 0))
        def _mark_ready_count():
            base = recv_source_expert_base_ref[dst_rank, local_expert]
            count = recv_source_expert_count_ref[dst_rank, local_expert]
            remaining_capacity = jnp.maximum(jnp.int32(recv_capacity) - base, jnp.int32(0))
            ready_count_ref[dst_rank, local_expert] = jnp.minimum(count, remaining_capacity)

        @pl.when((dst_rank == 0) & (local_expert == 0) & (col_tile == 0) & (row_or_block < recv_blocks))
        def _mark_ready_block_count():
            block_start = row_or_block * ready_block_m
            remaining_rows = jnp.maximum(ready_rows_ref[()] - block_start, jnp.int32(0))
            ready_block_count_ref[row_or_block] = jnp.minimum(remaining_rows, jnp.int32(ready_block_m))

    (
        recv_x,
        recv_local_expert,
        recv_src_rank,
        recv_src_token_idx,
        recv_topk_slot,
        recv_router_weight,
        ready_count,
        ready_block_count,
    ) = plgpu.kernel(
        kernel_body,
        out_shape=[
            jax.ShapeDtypeStruct((recv_capacity, hidden), send_x_by_dst_expert.dtype),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity,), send_router_weight_by_dst_expert.dtype),
            jax.ShapeDtypeStruct((ep_size, local_experts), jnp.int32),
            jax.ShapeDtypeStruct((recv_blocks,), jnp.int32),
        ],
        grid=(ep_size, local_experts, row_or_block_count, col_tiles),
        grid_names=("dst", "local_expert", "row_or_block", "col"),
    )(
        send_x_by_dst_expert,
        send_src_token_idx_by_dst_expert,
        send_topk_slot_by_dst_expert,
        send_router_weight_by_dst_expert,
        send_row_base_by_dst_expert,
        send_count_by_dst_expert,
        recv_source_expert_base,
        recv_source_expert_count,
        ready_rows,
    )
    overflow_count = jnp.maximum(jnp.sum(rows_per_expert, dtype=jnp.int32) - recv_capacity, 0)
    recv_valid = jnp.arange(recv_capacity, dtype=jnp.int32) < ready_rows
    layout = MoeDispatchUpLayout(
        recv_x,
        recv_valid,
        rows_per_expert,
        expert_base,
        recv_local_expert,
        recv_src_rank,
        recv_src_token_idx,
        recv_topk_slot,
        recv_router_weight,
        overflow_count,
    )
    return layout, ready_count, ready_block_count


def dispatch_prepacked_moe_dispatch_up_mosaic_gpu_source_expert_tiled_local(
    prepacked: MoeDispatchUpSourceExpertPrepackedSend,
    *,
    axis_name: str,
    recv_capacity: int,
    ready_block_m: int = 64,
    copy_cols: int = 256,
    copy_rows: int = 1,
) -> tuple[MoeDispatchUpLayout, jax.Array, jax.Array]:
    """Compact source/expert dispatch tiled over rows and hidden columns."""

    _require_mgpu_runtime()
    if prepacked.send_x_by_dst_expert.ndim != 4:
        raise ValueError(
            "local compact send_x_by_dst_expert must have shape [EP, EL, C, D], "
            f"got {prepacked.send_x_by_dst_expert.shape}"
        )
    ep_size, local_experts, _capacity, _hidden = prepacked.send_x_by_dst_expert.shape
    if prepacked.send_count_by_dst_expert.shape != (ep_size, local_experts):
        raise ValueError(
            "local send_count_by_dst_expert must have shape "
            f"{(ep_size, local_experts)}, got {prepacked.send_count_by_dst_expert.shape}"
        )

    return _dispatch_prepacked_moe_dispatch_up_mosaic_gpu_source_expert_tiled_local(
        prepacked.send_x_by_dst_expert,
        prepacked.send_src_token_idx_by_dst_expert,
        prepacked.send_topk_slot_by_dst_expert,
        prepacked.send_router_weight_by_dst_expert,
        prepacked.send_row_base_by_dst_expert,
        prepacked.send_count_by_dst_expert,
        prepacked.rows_per_expert,
        prepacked.expert_base,
        prepacked.recv_source_expert_base,
        prepacked.recv_source_expert_count,
        axis_name=axis_name,
        recv_capacity=recv_capacity,
        ready_block_m=ready_block_m,
        copy_cols=copy_cols,
        copy_rows=copy_rows,
    )


def compact_source_expert_ring_transport_mosaic_gpu_local(
    send_x_by_dst_expert: jax.Array,
    send_count_by_dst_expert: jax.Array,
    *,
    axis_name: str,
) -> tuple[jax.Array, jax.Array]:
    """Ordered ring-step transport for compact source/expert MoE payloads."""

    _require_mgpu_runtime()
    if send_x_by_dst_expert.ndim != 4:
        raise ValueError("send_x_by_dst_expert must have shape [EP, EL, C, D], " f"got {send_x_by_dst_expert.shape}")
    ep_size, local_experts, source_expert_capacity, hidden = send_x_by_dst_expert.shape
    if send_count_by_dst_expert.shape != (ep_size, local_experts):
        raise ValueError(
            "send_count_by_dst_expert must have shape "
            f"{(ep_size, local_experts)}, got {send_count_by_dst_expert.shape}"
        )
    num_sms = jax.local_devices()[0].core_count
    rows_per_destination = local_experts * source_expert_capacity

    def kernel_body(send_x_ref, send_count_ref, recv_x_ref, recv_count_ref):
        received_sem = pl.get_global(plgpu.SemaphoreType.REGULAR)
        src_rank = lax.axis_index(axis_name)
        sm = pl.program_id(0)

        for step in range(ep_size):
            dst_rank = lax.rem(src_rank + step, jnp.int32(ep_size))
            remote_recv_x = plgpu.remote_ref(recv_x_ref, dst_rank)
            remote_recv_count = plgpu.remote_ref(recv_count_ref, dst_rank)

            @pl.when(sm == 0)
            def _copy_counts():
                for local_expert in range(local_experts):
                    remote_recv_count[src_rank, local_expert] = send_count_ref[dst_rank, local_expert]

            for row_start in range(0, rows_per_destination, num_sms):
                linear_row = row_start + sm

                @pl.when(linear_row < rows_per_destination)
                def _copy_row():
                    local_expert = linear_row // source_expert_capacity
                    source_expert_row = linear_row - local_expert * source_expert_capacity
                    remote_recv_x.at[src_rank, local_expert, source_expert_row, pl.ds(0, hidden)][...] = send_x_ref.at[
                        dst_rank, local_expert, source_expert_row, pl.ds(0, hidden)
                    ][...]

            pl.semaphore_signal(received_sem, device_id=dst_rank)
            pl.semaphore_wait(received_sem, value=(step + 1) * num_sms, decrement=False)

    return plgpu.kernel(
        kernel_body,
        out_shape=[
            jax.ShapeDtypeStruct(
                (ep_size, local_experts, source_expert_capacity, hidden),
                send_x_by_dst_expert.dtype,
            ),
            jax.ShapeDtypeStruct((ep_size, local_experts), jnp.int32),
        ],
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
        ),
    )(send_x_by_dst_expert, send_count_by_dst_expert)


def compute_moe_up_mosaic_gpu_local(
    recv_x: jax.Array,
    rows_per_expert: jax.Array,
    w_gate_up_local: jax.Array,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    max_concurrent_steps: int = 2,
    grid_block_n: int = 1,
) -> jax.Array:
    """Local expert-major W13/SiLU kernel with a fused activation epilogue.

    This consumes milestone 1's destination-owned `recv_x` layout. It computes
    `gate = X @ W_gate`, `up = X @ W_up`, and stores only
    `silu(gate) * up`, avoiding a global `[tokens, 2 * d_ff]` temporary.
    """

    _require_mgpu_runtime()
    if recv_x.ndim != 2:
        raise ValueError(f"recv_x must have shape [R, D], got {recv_x.shape}")
    if w_gate_up_local.ndim != 3:
        raise ValueError(f"w_gate_up_local must have shape [EL, D, 2I], got {w_gate_up_local.shape}")
    recv_capacity, hidden = recv_x.shape
    local_experts, hidden_2, gate_up = w_gate_up_local.shape
    if hidden != hidden_2:
        raise ValueError(f"recv_x hidden={hidden} must match w_gate_up_local hidden={hidden_2}")
    if rows_per_expert.shape != (local_experts,):
        raise ValueError(f"rows_per_expert must have shape ({local_experts},), got {rows_per_expert.shape}")
    if gate_up % 2 != 0:
        raise ValueError(f"w_gate_up_local last dimension must be even, got {gate_up}")
    if hidden % block_k != 0:
        raise ValueError(f"hidden={hidden} must be divisible by block_k={block_k}")
    intermediate = gate_up // 2

    def kernel_body(rows_per_expert_gmem, recv_x_gmem, w_gate_up_gmem, out_gmem):
        grid_m = pl.cdiv(recv_capacity, block_m) + local_experts - 1
        grid_n = pl.cdiv(intermediate, block_n)
        grid = (grid_m * grid_n,)

        @plgpu.nd_loop(grid, collective_axes="sm")
        def mn_loop(loop_info: plgpu.NDLoopInfo):
            mi, ni = plgpu.planar_snake(loop_info.index[0], (grid_m, grid_n), 1, grid_block_n)
            group_info = GroupInfo.create(rows_per_expert_gmem, block_m, mi)

            def acc_scope(gate_acc_ref, up_acc_ref):
                def pipeline_body(_, x_smem, gate_w_smem, up_w_smem):
                    plgpu.wgmma(gate_acc_ref, x_smem, gate_w_smem)
                    plgpu.wgmma(up_acc_ref, x_smem, up_w_smem)

                gate_w_ref = w_gate_up_gmem.at[group_info.group_id, :, pl.ds(0, intermediate)]
                up_w_ref = w_gate_up_gmem.at[group_info.group_id, :, pl.ds(intermediate, intermediate)]
                plgpu.emit_pipeline(
                    pipeline_body,
                    grid=(hidden // block_k,),
                    in_specs=[
                        plgpu.BlockSpec(
                            (block_m, block_k),
                            lambda k: (group_info.block, k),
                            delay_release=1,
                        ),
                        plgpu.BlockSpec(
                            (block_k, block_n),
                            lambda k: (k, ni),
                            delay_release=1,
                        ),
                        plgpu.BlockSpec(
                            (block_k, block_n),
                            lambda k: (k, ni),
                            delay_release=1,
                        ),
                    ],
                    max_concurrent_steps=max_concurrent_steps,
                )(recv_x_gmem, gate_w_ref, up_w_ref)
                return gate_acc_ref[...], up_acc_ref[...]

            gate_acc, up_acc = pl.run_scoped(
                acc_scope,
                plgpu.ACC((block_m, block_n), jnp.float32),
                plgpu.ACC((block_m, block_n), jnp.float32),
            )

            @functools.partial(pl.run_scoped, out_smem=plgpu.SMEM((block_m, block_n), dtype=out_gmem.dtype))
            def store_scope(out_smem):
                activated = jax.nn.silu(gate_acc) * up_acc
                out_smem[...] = activated.astype(out_smem.dtype)
                plgpu.commit_smem()

                smem_start = group_info.start_within_block
                remaining_rows = min(block_m, recv_capacity)
                while remaining_rows > 0:
                    const_rows_len = 1 << int(math.log2(remaining_rows))
                    remaining_rows //= 2

                    @pl.when(group_info.actual_size & const_rows_len != 0)
                    def _():
                        out_smem_slice = out_smem.at[pl.ds(smem_start, const_rows_len)]
                        out_gmem_slice = out_gmem.at[
                            pl.ds(group_info.block_start + smem_start, const_rows_len),
                            pl.ds(ni * block_n, block_n),
                        ]
                        plgpu.copy_smem_to_gmem(out_smem_slice, out_gmem_slice)

                    smem_start += group_info.actual_size & const_rows_len
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    num_sms = jax.local_devices()[0].core_count
    kernel = plgpu.kernel(
        kernel_body,
        out_shape=jax.ShapeDtypeStruct((recv_capacity, intermediate), recv_x.dtype),
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
        ),
    )
    out = kernel(rows_per_expert, recv_x, w_gate_up_local)
    valid_rows = jnp.arange(recv_capacity, dtype=jnp.int32) < jnp.minimum(
        jnp.sum(rows_per_expert, dtype=jnp.int32), recv_capacity
    )
    return jnp.where(valid_rows[:, None], out, jnp.zeros((), dtype=out.dtype))


def compute_moe_up_mosaic_gpu_source_expert_padded_local(
    compact_x: jax.Array,
    rows_per_source_expert: jax.Array,
    w_gate_up_local: jax.Array,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    max_concurrent_steps: int = 2,
    grid_block_n: int = 1,
) -> jax.Array:
    """Local W13/SiLU over padded source/expert groups without duplicating weights."""

    _require_mgpu_runtime()
    if compact_x.ndim != 2:
        raise ValueError(f"compact_x must have shape [R, D], got {compact_x.shape}")
    if rows_per_source_expert.ndim != 1:
        raise ValueError(
            "rows_per_source_expert must have shape [source_expert_groups], " f"got {rows_per_source_expert.shape}"
        )
    if w_gate_up_local.ndim != 3:
        raise ValueError(f"w_gate_up_local must have shape [EL, D, 2I], got {w_gate_up_local.shape}")
    recv_capacity, hidden = compact_x.shape
    local_experts, hidden_2, gate_up = w_gate_up_local.shape
    source_expert_groups = rows_per_source_expert.shape[0]
    if source_expert_groups % local_experts != 0:
        raise ValueError(
            "source/expert group count must be divisible by local expert count; "
            f"got {source_expert_groups} groups and {local_experts} local experts"
        )
    if hidden != hidden_2:
        raise ValueError(f"compact_x hidden={hidden} must match w_gate_up_local hidden={hidden_2}")
    if gate_up % 2 != 0:
        raise ValueError(f"w_gate_up_local last dimension must be even, got {gate_up}")
    if hidden % block_k != 0:
        raise ValueError(f"hidden={hidden} must be divisible by block_k={block_k}")
    intermediate = gate_up // 2

    def kernel_body(rows_per_source_expert_gmem, compact_x_gmem, w_gate_up_gmem, out_gmem):
        grid_m = pl.cdiv(recv_capacity, block_m) + source_expert_groups - 1
        grid_n = pl.cdiv(intermediate, block_n)
        grid = (grid_m * grid_n,)

        @plgpu.nd_loop(grid, collective_axes="sm")
        def mn_loop(loop_info: plgpu.NDLoopInfo):
            mi, ni = plgpu.planar_snake(loop_info.index[0], (grid_m, grid_n), 1, grid_block_n)
            group_info = GroupInfo.create(rows_per_source_expert_gmem, block_m, mi)
            local_expert = group_info.group_id % local_experts

            def acc_scope(gate_acc_ref, up_acc_ref):
                def pipeline_body(_, x_smem, gate_w_smem, up_w_smem):
                    plgpu.wgmma(gate_acc_ref, x_smem, gate_w_smem)
                    plgpu.wgmma(up_acc_ref, x_smem, up_w_smem)

                gate_w_ref = w_gate_up_gmem.at[local_expert, :, pl.ds(0, intermediate)]
                up_w_ref = w_gate_up_gmem.at[local_expert, :, pl.ds(intermediate, intermediate)]
                plgpu.emit_pipeline(
                    pipeline_body,
                    grid=(hidden // block_k,),
                    in_specs=[
                        plgpu.BlockSpec(
                            (block_m, block_k),
                            lambda k: (group_info.block, k),
                            delay_release=1,
                        ),
                        plgpu.BlockSpec(
                            (block_k, block_n),
                            lambda k: (k, ni),
                            delay_release=1,
                        ),
                        plgpu.BlockSpec(
                            (block_k, block_n),
                            lambda k: (k, ni),
                            delay_release=1,
                        ),
                    ],
                    max_concurrent_steps=max_concurrent_steps,
                )(compact_x_gmem, gate_w_ref, up_w_ref)
                return gate_acc_ref[...], up_acc_ref[...]

            gate_acc, up_acc = pl.run_scoped(
                acc_scope,
                plgpu.ACC((block_m, block_n), jnp.float32),
                plgpu.ACC((block_m, block_n), jnp.float32),
            )

            @functools.partial(pl.run_scoped, out_smem=plgpu.SMEM((block_m, block_n), dtype=out_gmem.dtype))
            def store_scope(out_smem):
                activated = jax.nn.silu(gate_acc) * up_acc
                out_smem[...] = activated.astype(out_smem.dtype)
                plgpu.commit_smem()

                smem_start = group_info.start_within_block
                remaining_rows = min(block_m, recv_capacity)
                while remaining_rows > 0:
                    const_rows_len = 1 << int(math.log2(remaining_rows))
                    remaining_rows //= 2

                    @pl.when(group_info.actual_size & const_rows_len != 0)
                    def _():
                        out_smem_slice = out_smem.at[pl.ds(smem_start, const_rows_len)]
                        out_gmem_slice = out_gmem.at[
                            pl.ds(group_info.block_start + smem_start, const_rows_len),
                            pl.ds(ni * block_n, block_n),
                        ]
                        plgpu.copy_smem_to_gmem(out_smem_slice, out_gmem_slice)

                    smem_start += group_info.actual_size & const_rows_len
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    num_sms = jax.local_devices()[0].core_count
    kernel = plgpu.kernel(
        kernel_body,
        out_shape=jax.ShapeDtypeStruct((recv_capacity, intermediate), compact_x.dtype),
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
        ),
    )
    out = kernel(rows_per_source_expert, compact_x, w_gate_up_local)
    valid_rows = jnp.arange(recv_capacity, dtype=jnp.int32) < jnp.minimum(
        jnp.sum(rows_per_source_expert, dtype=jnp.int32), recv_capacity
    )
    return jnp.where(valid_rows[:, None], out, jnp.zeros((), dtype=out.dtype))


def compute_moe_up_mosaic_gpu_ready_local(
    recv_x: jax.Array,
    ready_count: jax.Array,
    w_gate_up_local: jax.Array,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    max_concurrent_steps: int = 2,
    grid_block_n: int = 1,
) -> jax.Array:
    """Local W13/SiLU scheduled over source/expert ready ranges."""

    _require_mgpu_runtime()
    if recv_x.ndim != 2:
        raise ValueError(f"recv_x must have shape [R, D], got {recv_x.shape}")
    if ready_count.ndim != 2:
        raise ValueError(f"ready_count must have shape [EP, EL], got {ready_count.shape}")
    if w_gate_up_local.ndim != 3:
        raise ValueError(f"w_gate_up_local must have shape [EL, D, 2I], got {w_gate_up_local.shape}")
    recv_capacity, hidden = recv_x.shape
    ep_size, local_experts = ready_count.shape
    if w_gate_up_local.shape[0] != local_experts:
        raise ValueError(
            "ready_count local expert dimension must match w_gate_up_local; "
            f"got {ready_count.shape} vs {w_gate_up_local.shape}"
        )
    if w_gate_up_local.shape[1] != hidden:
        raise ValueError(f"recv_x hidden={hidden} must match w_gate_up_local hidden={w_gate_up_local.shape[1]}")
    if w_gate_up_local.shape[2] % 2 != 0:
        raise ValueError(f"w_gate_up_local last dimension must be even, got {w_gate_up_local.shape[2]}")
    if hidden % block_k != 0:
        raise ValueError(f"hidden={hidden} must be divisible by block_k={block_k}")
    intermediate = w_gate_up_local.shape[2] // 2
    group_count = ep_size * local_experts
    ready_count_expert_major = jnp.swapaxes(ready_count, 0, 1).reshape((group_count,))

    def kernel_body(ready_count_gmem, recv_x_gmem, w_gate_up_gmem, out_gmem):
        grid_m = pl.cdiv(recv_capacity, block_m) + group_count - 1
        grid_n = pl.cdiv(intermediate, block_n)
        grid = (grid_m * grid_n,)

        @plgpu.nd_loop(grid, collective_axes="sm")
        def mn_loop(loop_info: plgpu.NDLoopInfo):
            mi, ni = plgpu.planar_snake(loop_info.index[0], (grid_m, grid_n), 1, grid_block_n)
            group_info = GroupInfo.create(ready_count_gmem, block_m, mi)
            local_expert = group_info.group_id // ep_size

            def acc_scope(gate_acc_ref, up_acc_ref):
                def pipeline_body(_, x_smem, gate_w_smem, up_w_smem):
                    plgpu.wgmma(gate_acc_ref, x_smem, gate_w_smem)
                    plgpu.wgmma(up_acc_ref, x_smem, up_w_smem)

                gate_w_ref = w_gate_up_gmem.at[local_expert, :, pl.ds(0, intermediate)]
                up_w_ref = w_gate_up_gmem.at[local_expert, :, pl.ds(intermediate, intermediate)]
                plgpu.emit_pipeline(
                    pipeline_body,
                    grid=(hidden // block_k,),
                    in_specs=[
                        plgpu.BlockSpec(
                            (block_m, block_k),
                            lambda k: (group_info.block, k),
                            delay_release=1,
                        ),
                        plgpu.BlockSpec(
                            (block_k, block_n),
                            lambda k: (k, ni),
                            delay_release=1,
                        ),
                        plgpu.BlockSpec(
                            (block_k, block_n),
                            lambda k: (k, ni),
                            delay_release=1,
                        ),
                    ],
                    max_concurrent_steps=max_concurrent_steps,
                )(recv_x_gmem, gate_w_ref, up_w_ref)
                return gate_acc_ref[...], up_acc_ref[...]

            gate_acc, up_acc = pl.run_scoped(
                acc_scope,
                plgpu.ACC((block_m, block_n), jnp.float32),
                plgpu.ACC((block_m, block_n), jnp.float32),
            )

            @functools.partial(pl.run_scoped, out_smem=plgpu.SMEM((block_m, block_n), dtype=out_gmem.dtype))
            def store_scope(out_smem):
                activated = jax.nn.silu(gate_acc) * up_acc
                out_smem[...] = activated.astype(out_smem.dtype)
                plgpu.commit_smem()

                smem_start = group_info.start_within_block
                remaining_rows = min(block_m, recv_capacity)
                while remaining_rows > 0:
                    const_rows_len = 1 << int(math.log2(remaining_rows))
                    remaining_rows //= 2

                    @pl.when(group_info.actual_size & const_rows_len != 0)
                    def _():
                        out_smem_slice = out_smem.at[pl.ds(smem_start, const_rows_len)]
                        out_gmem_slice = out_gmem.at[
                            pl.ds(group_info.block_start + smem_start, const_rows_len),
                            pl.ds(ni * block_n, block_n),
                        ]
                        plgpu.copy_smem_to_gmem(out_smem_slice, out_gmem_slice)

                    smem_start += group_info.actual_size & const_rows_len
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    num_sms = jax.local_devices()[0].core_count
    kernel = plgpu.kernel(
        kernel_body,
        out_shape=jax.ShapeDtypeStruct((recv_capacity, intermediate), recv_x.dtype),
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
        ),
    )
    out = kernel(ready_count_expert_major, recv_x, w_gate_up_local)
    valid_rows = jnp.arange(recv_capacity, dtype=jnp.int32) < jnp.minimum(
        jnp.sum(ready_count, dtype=jnp.int32), recv_capacity
    )
    return jnp.where(valid_rows[:, None], out, jnp.zeros((), dtype=out.dtype))


def compute_moe_up_mosaic_gpu_block_ready_local(
    recv_x: jax.Array,
    rows_per_expert: jax.Array,
    ready_block_count: jax.Array,
    w_gate_up_local: jax.Array,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    max_concurrent_steps: int = 2,
    grid_block_n: int = 1,
) -> jax.Array:
    """Local W13/SiLU scheduled by expert groups with block-ready row masks."""

    _require_mgpu_runtime()
    if recv_x.ndim != 2:
        raise ValueError(f"recv_x must have shape [R, D], got {recv_x.shape}")
    if rows_per_expert.ndim != 1:
        raise ValueError(f"rows_per_expert must have shape [EL], got {rows_per_expert.shape}")
    if ready_block_count.ndim != 1:
        raise ValueError(f"ready_block_count must have shape [B], got {ready_block_count.shape}")
    if w_gate_up_local.ndim != 3:
        raise ValueError(f"w_gate_up_local must have shape [EL, D, 2I], got {w_gate_up_local.shape}")
    recv_capacity, hidden = recv_x.shape
    local_experts, hidden_2, gate_up = w_gate_up_local.shape
    if hidden != hidden_2:
        raise ValueError(f"recv_x hidden={hidden} must match w_gate_up_local hidden={hidden_2}")
    if rows_per_expert.shape != (local_experts,):
        raise ValueError(f"rows_per_expert must have shape ({local_experts},), got {rows_per_expert.shape}")
    expected_blocks = math.ceil(recv_capacity / block_m)
    if ready_block_count.shape != (expected_blocks,):
        raise ValueError(f"ready_block_count must have shape ({expected_blocks},), got {ready_block_count.shape}")
    if gate_up % 2 != 0:
        raise ValueError(f"w_gate_up_local last dimension must be even, got {gate_up}")
    if hidden % block_k != 0:
        raise ValueError(f"hidden={hidden} must be divisible by block_k={block_k}")
    intermediate = gate_up // 2

    def kernel_body(rows_per_expert_gmem, ready_block_count_gmem, recv_x_gmem, w_gate_up_gmem, out_gmem):
        grid_m = pl.cdiv(recv_capacity, block_m) + local_experts - 1
        grid_n = pl.cdiv(intermediate, block_n)
        grid = (grid_m * grid_n,)

        @plgpu.nd_loop(grid, collective_axes="sm")
        def mn_loop(loop_info: plgpu.NDLoopInfo):
            mi, ni = plgpu.planar_snake(loop_info.index[0], (grid_m, grid_n), 1, grid_block_n)
            group_info = GroupInfo.create(rows_per_expert_gmem, block_m, mi)
            ready_rows_in_block = ready_block_count_gmem[group_info.block]
            ready_after_start = jnp.maximum(ready_rows_in_block - group_info.start_within_block, jnp.int32(0))
            ready_actual_size = jnp.minimum(group_info.actual_size, ready_after_start)

            def acc_scope(gate_acc_ref, up_acc_ref):
                def pipeline_body(_, x_smem, gate_w_smem, up_w_smem):
                    plgpu.wgmma(gate_acc_ref, x_smem, gate_w_smem)
                    plgpu.wgmma(up_acc_ref, x_smem, up_w_smem)

                gate_w_ref = w_gate_up_gmem.at[group_info.group_id, :, pl.ds(0, intermediate)]
                up_w_ref = w_gate_up_gmem.at[group_info.group_id, :, pl.ds(intermediate, intermediate)]
                plgpu.emit_pipeline(
                    pipeline_body,
                    grid=(hidden // block_k,),
                    in_specs=[
                        plgpu.BlockSpec(
                            (block_m, block_k),
                            lambda k: (group_info.block, k),
                            delay_release=1,
                        ),
                        plgpu.BlockSpec(
                            (block_k, block_n),
                            lambda k: (k, ni),
                            delay_release=1,
                        ),
                        plgpu.BlockSpec(
                            (block_k, block_n),
                            lambda k: (k, ni),
                            delay_release=1,
                        ),
                    ],
                    max_concurrent_steps=max_concurrent_steps,
                )(recv_x_gmem, gate_w_ref, up_w_ref)
                return gate_acc_ref[...], up_acc_ref[...]

            gate_acc, up_acc = pl.run_scoped(
                acc_scope,
                plgpu.ACC((block_m, block_n), jnp.float32),
                plgpu.ACC((block_m, block_n), jnp.float32),
            )

            @functools.partial(pl.run_scoped, out_smem=plgpu.SMEM((block_m, block_n), dtype=out_gmem.dtype))
            def store_scope(out_smem):
                activated = jax.nn.silu(gate_acc) * up_acc
                out_smem[...] = activated.astype(out_smem.dtype)
                plgpu.commit_smem()

                smem_start = group_info.start_within_block
                remaining_rows = min(block_m, recv_capacity)
                while remaining_rows > 0:
                    const_rows_len = 1 << int(math.log2(remaining_rows))
                    remaining_rows //= 2

                    @pl.when(ready_actual_size & const_rows_len != 0)
                    def _():
                        out_smem_slice = out_smem.at[pl.ds(smem_start, const_rows_len)]
                        out_gmem_slice = out_gmem.at[
                            pl.ds(group_info.block_start + smem_start, const_rows_len),
                            pl.ds(ni * block_n, block_n),
                        ]
                        plgpu.copy_smem_to_gmem(out_smem_slice, out_gmem_slice)

                    smem_start += ready_actual_size & const_rows_len
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    num_sms = jax.local_devices()[0].core_count
    kernel = plgpu.kernel(
        kernel_body,
        out_shape=jax.ShapeDtypeStruct((recv_capacity, intermediate), recv_x.dtype),
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
        ),
    )
    out = kernel(rows_per_expert, ready_block_count, recv_x, w_gate_up_local)
    valid_rows = jnp.arange(recv_capacity, dtype=jnp.int32) < jnp.minimum(
        jnp.sum(ready_block_count, dtype=jnp.int32), recv_capacity
    )
    return jnp.where(valid_rows[:, None], out, jnp.zeros((), dtype=out.dtype))
