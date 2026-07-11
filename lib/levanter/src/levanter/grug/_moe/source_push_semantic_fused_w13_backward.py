# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent semantic W13 backward with source-owned dX return/combine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax, shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as mgpu
from jax.sharding import AbstractMesh, Mesh, NamedSharding, PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.source_push_plan import (
    SOURCE_PUSH_MESH_AXIS,
    SourcePushSemanticPlan,
    source_push_semantic_queue_metadata_jax,
)
from levanter.grug._moe.source_push_semantic_fused_w13 import (
    SourcePushSemanticFusedW13Config,
    SourcePushSemanticFusedW13Metadata,
    source_push_semantic_fused_w13_metadata_jax,
)


WGMMA_SWIZZLE_BYTES = 128
WGMMA_TILE_M = 8
PERSISTENT_COMBINE_PROGRAMS = 32
RETURN_HIDDEN_TILES_PER_JOB = 2


@dataclass(frozen=True, slots=True)
class SourcePushSemanticFusedW13BackwardConfig:
    """Fixed Hopper tile and persistent-CTA dimensions."""

    compute_m: int = 64
    send_m: int = 256
    block_hidden: int = 128
    block_output: int = 128
    inbox_slots: int = 12
    combine_token_block: int = 64
    chunk_owner_programs_per_peer: int = 2
    helper_programs_per_peer: int = 14
    consumer_programs_per_peer: int = 32

    def validate(self) -> None:
        expected = {
            "compute_m": 64,
            "send_m": 256,
            "block_hidden": 128,
            "block_output": 128,
            "inbox_slots": 12,
            "combine_token_block": 64,
            "chunk_owner_programs_per_peer": 2,
            "helper_programs_per_peer": 14,
            "consumer_programs_per_peer": 32,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"the initial Hopper lowering requires {name}={value}, got {getattr(self, name)}")

    @property
    def compute_blocks_per_send(self) -> int:
        return self.send_m // self.compute_m

    @property
    def worker_programs_per_peer(self) -> int:
        return self.chunk_owner_programs_per_peer + self.helper_programs_per_peer + self.consumer_programs_per_peer


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SourcePushSemanticFusedW13BackwardMetadata:
    """Forward chunk metadata plus source-owned inverse return locations."""

    forward: SourcePushSemanticFusedW13Metadata
    route_dst_ordinal: Int[Array, "S T K"]
    route_chunk: Int[Array, "S T K"]
    route_block: Int[Array, "S T K"]
    route_row: Int[Array, "S T K"]
    route_valid: Bool[Array, "S T K"]
    combine_ready_generation: Int[Array, "S TokenBlock DstOrd Slot"]


class SourcePushSemanticFusedW13BackwardResult(NamedTuple):
    """Source-owned input gradient and destination-owned W13 gradient."""

    dx: Float[Array, "S T H"]
    dw13: Float[Array, "Dst E H O"]
    queue_overflow_routes: Int[Array, ""]
    layout_overflow_rows: Int[Array, ""]


@dataclass(frozen=True, slots=True)
class SourcePushSemanticFusedW13BackwardSchedule:
    """Persistent CTA counts and fixed slot fan-in for one rank."""

    hidden_tiles: int
    helper_tiles_per_block: int
    helper_tiles: int
    hidden_tile_jobs: int
    compute_jobs_per_chunk: int
    token_blocks: int
    rounds: int
    lifecycle_programs: int
    helper_programs: int
    consumer_programs: int
    peer_programs: int
    combine_programs: int
    active_combine_programs: int
    readiness_signals: int
    block_readiness_signals: int
    readiness_waits: int

    @property
    def total_programs(self) -> int:
        return self.peer_programs + self.combine_programs


def source_push_semantic_fused_w13_backward_schedule(
    *,
    ep_size: int,
    hidden_dim: int,
    tokens_per_source: int,
    send_chunks_per_dst: int,
    config: SourcePushSemanticFusedW13BackwardConfig = SourcePushSemanticFusedW13BackwardConfig(),
) -> SourcePushSemanticFusedW13BackwardSchedule:
    """Return the rolling-slot schedule and fixed compute fan-in."""

    config.validate()
    if ep_size <= 0 or hidden_dim <= 0 or tokens_per_source <= 0 or send_chunks_per_dst <= 0:
        raise ValueError(
            "ep_size, hidden_dim, tokens_per_source, and send_chunks_per_dst must be positive, "
            f"got {ep_size}, {hidden_dim}, {tokens_per_source}, and {send_chunks_per_dst}"
        )
    if hidden_dim % config.block_hidden:
        raise ValueError(f"hidden dim {hidden_dim} must be divisible by block_hidden={config.block_hidden}")

    hidden_tiles = hidden_dim // config.block_hidden
    hidden_tile_jobs = (hidden_tiles + RETURN_HIDDEN_TILES_PER_JOB - 1) // RETURN_HIDDEN_TILES_PER_JOB
    compute_jobs_per_chunk = config.compute_blocks_per_send * hidden_tile_jobs
    token_blocks = (tokens_per_source + config.combine_token_block - 1) // config.combine_token_block
    combine_programs = max(PERSISTENT_COMBINE_PROGRAMS, hidden_tiles)
    active_combine_programs = sum(
        min(token_blocks, (combine_programs + hidden_tiles - 1 - hidden_tile) // hidden_tiles)
        for hidden_tile in range(hidden_tiles)
    )
    return SourcePushSemanticFusedW13BackwardSchedule(
        hidden_tiles=hidden_tiles,
        helper_tiles_per_block=hidden_tiles,
        helper_tiles=config.compute_blocks_per_send * hidden_tiles,
        hidden_tile_jobs=hidden_tile_jobs,
        compute_jobs_per_chunk=compute_jobs_per_chunk,
        token_blocks=token_blocks,
        rounds=(send_chunks_per_dst + config.inbox_slots - 1) // config.inbox_slots,
        lifecycle_programs=ep_size * config.chunk_owner_programs_per_peer,
        helper_programs=ep_size * config.helper_programs_per_peer,
        consumer_programs=ep_size * config.consumer_programs_per_peer,
        peer_programs=ep_size * config.worker_programs_per_peer,
        combine_programs=combine_programs,
        active_combine_programs=active_combine_programs,
        readiness_signals=ep_size * send_chunks_per_dst,
        block_readiness_signals=ep_size * send_chunks_per_dst * config.compute_blocks_per_send,
        readiness_waits=token_blocks * hidden_tiles * ep_size * config.inbox_slots,
    )


def source_push_semantic_fused_w13_backward_metadata_jax(
    x: Float[Array, "S T H"],
    plan: SourcePushSemanticPlan,
    *,
    send_chunks_per_dst: int,
    rows_per_expert_capacity: int,
    config: SourcePushSemanticFusedW13BackwardConfig = SourcePushSemanticFusedW13BackwardConfig(),
) -> SourcePushSemanticFusedW13BackwardMetadata:
    """Lower semantic routes into outbound chunks and inverse dX return rows."""

    config.validate()
    forward_config = _forward_config(config)
    forward = source_push_semantic_fused_w13_metadata_jax(
        x,
        plan,
        send_chunks_per_dst=send_chunks_per_dst,
        rows_per_expert_capacity=rows_per_expert_capacity,
        config=forward_config,
    )
    entries_per_dst = send_chunks_per_dst * config.compute_blocks_per_send
    queue = source_push_semantic_queue_metadata_jax(
        plan,
        return_row_block=config.compute_m,
        entries_per_dst=entries_per_dst,
    )
    route_entry = queue.route_entry
    route_valid = queue.route_valid & (route_entry >= 0) & (route_entry < entries_per_dst)
    route_chunk = jnp.where(route_valid, route_entry // config.compute_blocks_per_send, 0).astype(jnp.int32)
    route_block = jnp.where(route_valid, route_entry % config.compute_blocks_per_send, 0).astype(jnp.int32)
    route_row = jnp.where(route_valid, queue.route_queue_row, 0).astype(jnp.int32)
    route_dst_ordinal = jnp.where(route_valid, queue.route_dst_ordinal, 0).astype(jnp.int32)
    source = jnp.arange(x.shape[0], dtype=jnp.int32)[:, None, None]
    sent_rows = forward.send_valid_rows.at[source, route_dst_ordinal, route_chunk, route_block].get()
    route_valid &= route_row < sent_rows
    route_chunk = jnp.where(route_valid, route_chunk, 0)
    route_block = jnp.where(route_valid, route_block, 0)
    route_row = jnp.where(route_valid, route_row, 0)
    route_dst_ordinal = jnp.where(route_valid, route_dst_ordinal, 0)

    source_count, tokens_per_source, topk = route_valid.shape
    token_blocks = (tokens_per_source + config.combine_token_block - 1) // config.combine_token_block
    route_generation = jnp.where(route_valid, route_chunk // config.inbox_slots + 1, 0)
    combine_ready_generation = (
        jnp.zeros(
            (source_count, token_blocks, source_count, config.inbox_slots),
            dtype=jnp.int32,
        )
        .at[
            jnp.arange(source_count, dtype=jnp.int32)[:, None, None],
            jnp.arange(tokens_per_source, dtype=jnp.int32)[None, :, None] // config.combine_token_block,
            route_dst_ordinal,
            route_chunk % config.inbox_slots,
        ]
        .max(route_generation, mode="drop")
    )
    assert topk == plan.topk
    return SourcePushSemanticFusedW13BackwardMetadata(
        forward=forward,
        route_dst_ordinal=route_dst_ordinal,
        route_chunk=route_chunk,
        route_block=route_block,
        route_row=route_row,
        route_valid=route_valid,
        combine_ready_generation=combine_ready_generation,
    )


def source_push_semantic_fused_w13_backward_reference_jax(
    x: Float[Array, "S T H"],
    dz13: Float[Array, "Dst E C O"],
    w13: Float[Array, "Dst E H O"],
    metadata: SourcePushSemanticFusedW13BackwardMetadata,
) -> tuple[Float[Array, "S T H"], Float[Array, "Dst E H O"]]:
    """Readable rematerialize, W13 backward, return, and source-combine reference."""

    forward = metadata.forward
    source_count, destination_count, _chunks, _blocks, compute_m = forward.token_ids.shape
    source = jnp.arange(source_count, dtype=jnp.int32)[:, None, None, None, None]
    dst_ordinal = jnp.arange(destination_count, dtype=jnp.int32)[None, :, None, None]
    destination = (jnp.arange(source_count, dtype=jnp.int32)[:, None, None, None] + dst_ordinal) % destination_count
    safe_expert = jnp.maximum(forward.send_expert, 0)
    row = jnp.arange(compute_m, dtype=jnp.int32)[None, None, None, None, :]
    row_valid = row < forward.send_valid_rows[..., None]

    gathered_x = x.at[source, forward.token_ids].get().astype(jnp.float32)
    gathered_x = jnp.where(row_valid[..., None], gathered_x, 0.0)
    expert_row = forward.send_row_start[..., None] + row
    gathered_dz = (
        dz13.at[
            destination[..., None],
            safe_expert[..., None],
            expert_row,
        ]
        .get()
        .astype(jnp.float32)
    )
    gathered_dz = jnp.where(row_valid[..., None], gathered_dz, 0.0)
    gathered_w_sharding = None
    if jax.sharding.get_abstract_mesh().are_all_axes_explicit:
        gathered_w_sharding = P(SOURCE_PUSH_MESH_AXIS, None, None, None, None, None)
    gathered_w = w13.at[destination, safe_expert].get(out_sharding=gathered_w_sharding).astype(jnp.float32)

    dx_routes = jnp.einsum("sdcbmo,sdcbho->sdcbmh", gathered_dz, gathered_w)
    dx = jnp.zeros((source_count, x.shape[1], x.shape[2]), dtype=jnp.float32)
    dx = dx.at[source, forward.token_ids].add(jnp.where(row_valid[..., None], dx_routes, 0.0))

    dw_partials = jnp.einsum("sdcbmh,sdcbmo->sdcbho", gathered_x, gathered_dz)
    live_block = jnp.any(row_valid, axis=-1)
    dw13 = jnp.zeros(w13.shape, dtype=jnp.float32)
    dw13 = dw13.at[destination, safe_expert].add(jnp.where(live_block[..., None, None], dw_partials, 0.0))
    return dx, dw13


def source_push_semantic_fused_w13_backward(
    x: Float[Array, "S T H"],
    dz13: Float[Array, "Dst E C O"],
    w13: Float[Array, "Dst E H O"],
    plan: SourcePushSemanticPlan,
    *,
    send_chunks_per_dst: int,
    rows_per_expert_capacity: int,
    config: SourcePushSemanticFusedW13BackwardConfig = SourcePushSemanticFusedW13BackwardConfig(),
    mesh: Mesh | AbstractMesh | None = None,
    interpret: bool = False,
) -> SourcePushSemanticFusedW13BackwardResult:
    """Run fused W13 backward and source dX return/combine."""

    _validate_request(x, dz13, w13, plan, rows_per_expert_capacity, config)
    metadata = source_push_semantic_fused_w13_backward_metadata_jax(
        x,
        plan,
        send_chunks_per_dst=send_chunks_per_dst,
        rows_per_expert_capacity=rows_per_expert_capacity,
        config=config,
    )
    if interpret:
        dx, dw13 = source_push_semantic_fused_w13_backward_reference_jax(x, dz13, w13, metadata)
    else:
        if jax.default_backend() != "gpu":
            raise NotImplementedError("persistent semantic fused W13 backward requires a GPU backend")
        if mesh is None:
            mesh = jax.sharding.get_abstract_mesh()
            if mesh.empty:
                raise ValueError("mesh is required for persistent semantic fused W13 backward")
        dx, dw13 = _source_push_semantic_fused_w13_backward_sharded(
            x,
            dz13,
            w13,
            metadata,
            config=config,
            mesh=mesh,
        )
    return SourcePushSemanticFusedW13BackwardResult(
        dx=dx,
        dw13=dw13,
        queue_overflow_routes=metadata.forward.queue_overflow_routes,
        layout_overflow_rows=metadata.forward.layout_overflow_rows,
    )


def _source_push_semantic_fused_w13_backward_sharded(
    x: Array,
    dz13: Array,
    w13: Array,
    metadata: SourcePushSemanticFusedW13BackwardMetadata,
    *,
    config: SourcePushSemanticFusedW13BackwardConfig,
    mesh: Mesh | AbstractMesh,
) -> tuple[Array, Array]:
    ep_size = x.shape[0]
    if mesh.shape[SOURCE_PUSH_MESH_AXIS] != ep_size:
        raise ValueError(f"mesh size must match EP size {ep_size}, got {mesh.shape[SOURCE_PUSH_MESH_AXIS]}")
    kernel = _make_source_push_semantic_fused_w13_backward_kernel(
        ep_size=ep_size,
        tokens_per_source=x.shape[1],
        hidden_dim=x.shape[2],
        output_dim=dz13.shape[-1],
        experts_per_rank=w13.shape[1],
        rows_per_expert_capacity=dz13.shape[2],
        send_chunks_per_dst=metadata.forward.send_chunks_per_dst,
        dtype=x.dtype,
        config=config,
    )

    def local_fn(
        x_local,
        token_ids_local,
        send_valid_local,
        recv_expert_local,
        recv_row_local,
        recv_valid_local,
        route_dst_local,
        route_chunk_local,
        route_block_local,
        route_row_local,
        route_valid_local,
        combine_ready_local,
        dz_local,
        w_local,
    ):
        _x_inbox, _dx_return, dx_local, dw_local = kernel(
            x_local[0],
            token_ids_local[0],
            send_valid_local[0],
            recv_expert_local[0],
            recv_row_local[0],
            recv_valid_local[0],
            route_dst_local[0],
            route_chunk_local[0],
            route_block_local[0],
            route_row_local[0],
            route_valid_local[0],
            combine_ready_local[0],
            dz_local[0],
            w_local[0],
        )
        return dx_local[None, ...], dw_local[None, ...]

    source_3d = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None))
    source_4d = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None))
    destination_3d = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None))
    destination_4d = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None))
    source_token_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None, None))
    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        ),
        out_specs=(P(SOURCE_PUSH_MESH_AXIS, None, None), P(SOURCE_PUSH_MESH_AXIS, None, None, None)),
        check_vma=False,
    )(
        jax.sharding.reshard(x, source_3d),
        jax.sharding.reshard(metadata.forward.token_ids, source_token_sharding),
        jax.sharding.reshard(metadata.forward.send_valid_rows, source_4d),
        jax.sharding.reshard(metadata.forward.recv_expert, destination_4d),
        jax.sharding.reshard(metadata.forward.recv_row_start, destination_4d),
        jax.sharding.reshard(metadata.forward.recv_valid_rows, destination_4d),
        jax.sharding.reshard(metadata.route_dst_ordinal, destination_3d),
        jax.sharding.reshard(metadata.route_chunk, destination_3d),
        jax.sharding.reshard(metadata.route_block, destination_3d),
        jax.sharding.reshard(metadata.route_row, destination_3d),
        jax.sharding.reshard(metadata.route_valid, destination_3d),
        jax.sharding.reshard(metadata.combine_ready_generation, source_4d),
        jax.sharding.reshard(dz13, destination_4d),
        jax.sharding.reshard(w13, destination_4d),
    )


def _make_source_push_semantic_fused_w13_backward_kernel(
    *,
    ep_size: int,
    tokens_per_source: int,
    hidden_dim: int,
    output_dim: int,
    experts_per_rank: int,
    rows_per_expert_capacity: int,
    send_chunks_per_dst: int,
    dtype: jnp.dtype,
    config: SourcePushSemanticFusedW13BackwardConfig,
):
    config.validate()
    _validate_kernel_dimensions(hidden_dim, output_dim, config)
    schedule = source_push_semantic_fused_w13_backward_schedule(
        ep_size=ep_size,
        hidden_dim=hidden_dim,
        tokens_per_source=tokens_per_source,
        send_chunks_per_dst=send_chunks_per_dst,
        config=config,
    )
    hidden_tiles = schedule.hidden_tiles
    output_tiles = output_dim // config.block_output
    helper_start = config.chunk_owner_programs_per_peer
    consumer_start = helper_start + config.helper_programs_per_peer
    dw_tiles = experts_per_rank * hidden_tiles * output_tiles

    def body(
        x_ref,
        token_ids_ref,
        send_valid_ref,
        recv_expert_ref,
        recv_row_ref,
        recv_valid_ref,
        route_dst_ref,
        route_chunk_ref,
        route_block_ref,
        route_row_ref,
        route_valid_ref,
        combine_ready_ref,
        dz_ref,
        w_ref,
        x_inbox_ref,
        dx_return_ref,
        dx_ref,
        dw_ref,
    ) -> None:
        empty_sem = pl.get_global(mgpu.SemaphoreType.REGULAR((ep_size, config.inbox_slots)))
        prepare_sem = pl.get_global(mgpu.SemaphoreType.REGULAR((ep_size, config.inbox_slots)))
        helper_done_sem = pl.get_global(
            mgpu.SemaphoreType.REGULAR((ep_size, config.inbox_slots, config.compute_blocks_per_send))
        )
        block_ready_sem = pl.get_global(
            mgpu.SemaphoreType.REGULAR((ep_size, config.inbox_slots, config.compute_blocks_per_send))
        )
        done_sem = pl.get_global(mgpu.SemaphoreType.REGULAR((ep_size, config.inbox_slots)))
        ready_sem = pl.get_global(mgpu.SemaphoreType.REGULAR((ep_size, config.inbox_slots)))
        dw_init_done_sem = pl.get_global(mgpu.SemaphoreType.REGULAR((dw_tiles,)))
        rank = lax.axis_index(SOURCE_PUSH_MESH_AXIS)
        worker = pl.program_id(0)

        def _signal_remote(sem, peer, index) -> None:
            pl.semaphore_signal(
                sem.at[index],
                device_id=peer,
                device_id_type=pl.DeviceIdType.LOGICAL,
            )

        @pl.when(worker < schedule.peer_programs)
        def _producer() -> None:
            peer_ordinal = worker // config.worker_programs_per_peer
            peer_worker = worker % config.worker_programs_per_peer

            @pl.loop(0, math.ceil(dw_tiles / schedule.peer_programs))
            def _initialize_owned_dw(iteration) -> None:
                tile = iteration * schedule.peer_programs + worker

                @pl.when(tile < dw_tiles)
                def _zero_tile() -> None:
                    expert = tile // (hidden_tiles * output_tiles)
                    remainder = tile % (hidden_tiles * output_tiles)
                    hidden_tile = remainder // output_tiles
                    output_tile = remainder % output_tiles
                    dw_ref[
                        expert,
                        pl.ds(hidden_tile * config.block_hidden, config.block_hidden),
                        pl.ds(output_tile * config.block_output, config.block_output),
                    ] = jnp.zeros((config.block_hidden, config.block_output), dtype=jnp.float32)
                    pl.semaphore_signal(dw_init_done_sem.at[tile])

            def _run_peer(static_destination_ordinal: int) -> None:
                static_source_ordinal = (-static_destination_ordinal) % ep_size
                destination = (rank + static_destination_ordinal) % ep_size
                source = (rank + static_source_ordinal) % ep_size
                destination_inbox = x_inbox_ref
                source_return = dx_return_ref
                if static_destination_ordinal != 0:
                    destination_inbox = mgpu.remote_ref(
                        x_inbox_ref,
                        destination,
                        device_id_type=pl.DeviceIdType.LOGICAL,
                    )
                if static_source_ordinal != 0:
                    source_return = mgpu.remote_ref(
                        dx_return_ref,
                        source,
                        device_id_type=pl.DeviceIdType.LOGICAL,
                    )

                @pl.when(peer_worker < config.chunk_owner_programs_per_peer)
                def _chunk_owner() -> None:
                    owner = peer_worker

                    @pl.loop(0, config.inbox_slots)
                    def _initialize_owned_slot(slot) -> None:
                        send_task = static_destination_ordinal * config.inbox_slots + slot

                        @pl.when((send_task % config.chunk_owner_programs_per_peer) == owner)
                        def _initialize_slot() -> None:
                            pl.semaphore_signal(empty_sem.at[static_destination_ordinal, slot])

                    @pl.loop(0, schedule.rounds)
                    def _round_loop(round_index) -> None:
                        @pl.loop(0, config.inbox_slots)
                        def _publish_slot_loop(slot) -> None:
                            send_task = static_destination_ordinal * config.inbox_slots + slot
                            chunk = round_index * config.inbox_slots + slot

                            @pl.when(
                                (chunk < send_chunks_per_dst)
                                & ((send_task % config.chunk_owner_programs_per_peer) == owner)
                            )
                            def _publish_owned_chunk() -> None:
                                pl.semaphore_wait(
                                    empty_sem.at[static_destination_ordinal, slot],
                                    value=round_index + 1,
                                    decrement=False,
                                )
                                pl.semaphore_signal(prepare_sem.at[static_destination_ordinal, slot])

                        @pl.loop(0, config.inbox_slots)
                        def _ready_slot_loop(slot) -> None:
                            send_task = static_destination_ordinal * config.inbox_slots + slot
                            chunk = round_index * config.inbox_slots + slot

                            @pl.when(
                                (chunk < send_chunks_per_dst)
                                & ((send_task % config.chunk_owner_programs_per_peer) == owner)
                            )
                            def _publish_owned_blocks() -> None:
                                for block in range(config.compute_blocks_per_send):
                                    pl.semaphore_wait(
                                        helper_done_sem.at[static_destination_ordinal, slot, block],
                                        value=(round_index + 1) * schedule.helper_tiles_per_block,
                                        decrement=False,
                                    )
                                    if static_destination_ordinal == 0:
                                        pl.semaphore_signal(block_ready_sem.at[0, slot, block])
                                    else:
                                        _signal_remote(
                                            block_ready_sem,
                                            destination,
                                            (static_source_ordinal, slot, block),
                                        )

                        @pl.loop(0, config.inbox_slots)
                        def _complete_slot_loop(slot) -> None:
                            send_task = static_destination_ordinal * config.inbox_slots + slot
                            chunk = round_index * config.inbox_slots + slot

                            @pl.when(
                                (chunk < send_chunks_per_dst)
                                & ((send_task % config.chunk_owner_programs_per_peer) == owner)
                            )
                            def _complete_owned_chunk() -> None:
                                pl.semaphore_wait(
                                    done_sem.at[static_source_ordinal, slot],
                                    value=(round_index + 1) * schedule.compute_jobs_per_chunk,
                                    decrement=False,
                                )
                                if static_source_ordinal == 0:
                                    pl.semaphore_signal(ready_sem.at[static_destination_ordinal, slot])
                                    pl.semaphore_signal(empty_sem.at[static_destination_ordinal, slot])
                                else:
                                    _signal_remote(
                                        ready_sem,
                                        source,
                                        (static_destination_ordinal, slot),
                                    )
                                    _signal_remote(
                                        empty_sem,
                                        source,
                                        (static_destination_ordinal, slot),
                                    )

                @pl.when(peer_worker >= helper_start)
                def _data_worker() -> None:
                    helper = peer_worker - helper_start
                    consumer = peer_worker - consumer_start

                    def _prepare_x_tile(chunk, slot, tile) -> None:
                        block = tile // hidden_tiles
                        hidden_tile = tile % hidden_tiles
                        hidden_start = hidden_tile * config.block_hidden
                        valid_rows = send_valid_ref[static_destination_ordinal, chunk, block]

                        @pl.when(valid_rows > 0)
                        def _copy_live_tile() -> None:
                            def _copy_scope(tile_smem) -> None:
                                @pl.loop(0, config.compute_m)
                                def _row_loop(row) -> None:
                                    token = token_ids_ref[static_destination_ordinal, chunk, block, row]
                                    tile_smem[row, :] = jnp.where(
                                        row < valid_rows,
                                        x_ref[token, pl.ds(hidden_start, config.block_hidden)],
                                        jnp.zeros((config.block_hidden,), dtype=dtype),
                                    )

                                mgpu.commit_smem()
                                mgpu.copy_smem_to_gmem(
                                    tile_smem,
                                    destination_inbox.at[
                                        rank,
                                        slot,
                                        pl.ds(block * config.compute_m, config.compute_m),
                                        pl.ds(hidden_start, config.block_hidden),
                                    ],
                                )
                                mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                            pl.run_scoped(
                                _copy_scope,
                                tile_smem=mgpu.SMEM((config.compute_m, config.block_hidden), dtype=dtype),
                            )

                    def _compute_hidden_tile(chunk, slot, block, hidden_tile) -> None:
                        valid_rows = recv_valid_ref[static_source_ordinal, chunk, block]
                        hidden_start = hidden_tile * config.block_hidden
                        source_return[
                            static_destination_ordinal,
                            chunk,
                            pl.ds(block * config.compute_m, config.compute_m),
                            pl.ds(hidden_start, config.block_hidden),
                        ] = jnp.zeros((config.compute_m, config.block_hidden), dtype=dtype)

                        @pl.when(valid_rows > 0)
                        def _live_tile() -> None:
                            expert = recv_expert_ref[static_source_ordinal, chunk, block]
                            row_start = recv_row_ref[static_source_ordinal, chunk, block]

                            def _dx_acc_scope(acc_ref) -> None:
                                def _dx_smem_scope(dz_smem, w_smem, barrier) -> None:
                                    row = mgpu.layout_cast(
                                        lax.broadcasted_iota(
                                            jnp.int32,
                                            (config.compute_m, config.block_output),
                                            0,
                                        ),
                                        mgpu.Layout.WGMMA,
                                    )
                                    row_valid = (row < valid_rows).astype(jnp.float32)

                                    @pl.loop(0, output_tiles)
                                    def _output_loop(output_tile) -> None:
                                        output_start = output_tile * config.block_output
                                        mgpu.copy_gmem_to_smem(
                                            dz_ref.at[
                                                expert,
                                                pl.ds(row_start, config.compute_m),
                                                pl.ds(output_start, config.block_output),
                                            ],
                                            dz_smem,
                                            barrier,
                                        )
                                        mgpu.copy_gmem_to_smem(
                                            w_ref.at[
                                                expert,
                                                pl.ds(hidden_start, config.block_hidden),
                                                pl.ds(output_start, config.block_output),
                                            ],
                                            w_smem,
                                            barrier,
                                        )
                                        mgpu.barrier_wait(barrier)
                                        dz_smem[:, :] = (dz_smem[...].astype(jnp.float32) * row_valid).astype(dtype)
                                        mgpu.commit_smem()
                                        mgpu.wgmma(
                                            acc_ref,
                                            dz_smem,
                                            mgpu.transpose_ref(w_smem, (1, 0)),
                                        )
                                        mgpu.wgmma_wait(0)

                                pl.run_scoped(
                                    _dx_smem_scope,
                                    dz_smem=_wgmma_smem((config.compute_m, config.block_output), dtype),
                                    w_smem=_wgmma_smem((config.block_hidden, config.block_output), dtype),
                                    barrier=mgpu.Barrier(num_arrivals=2),
                                )
                                # Mosaic must lower this accumulator fragment directly to GMEM.
                                source_return[
                                    static_destination_ordinal,
                                    chunk,
                                    pl.ds(block * config.compute_m, config.compute_m),
                                    pl.ds(hidden_start, config.block_hidden),
                                ] = acc_ref[...].astype(dtype)

                            pl.run_scoped(
                                _dx_acc_scope,
                                acc_ref=mgpu.ACC((config.compute_m, config.block_hidden)),
                            )

                            @pl.loop(0, output_tiles)
                            def _dw_output_loop(output_tile) -> None:
                                def _dw_acc_scope(acc_ref) -> None:
                                    def _dw_smem_scope(x_smem, dz_smem, barrier) -> None:
                                        mgpu.copy_gmem_to_smem(
                                            x_inbox_ref.at[
                                                source,
                                                slot,
                                                pl.ds(block * config.compute_m, config.compute_m),
                                                pl.ds(hidden_start, config.block_hidden),
                                            ],
                                            x_smem,
                                            barrier,
                                        )
                                        mgpu.copy_gmem_to_smem(
                                            dz_ref.at[
                                                expert,
                                                pl.ds(row_start, config.compute_m),
                                                pl.ds(
                                                    output_tile * config.block_output,
                                                    config.block_output,
                                                ),
                                            ],
                                            dz_smem,
                                            barrier,
                                        )
                                        mgpu.barrier_wait(barrier)
                                        row = mgpu.layout_cast(
                                            lax.broadcasted_iota(
                                                jnp.int32,
                                                (config.compute_m, config.block_output),
                                                0,
                                            ),
                                            mgpu.Layout.WGMMA,
                                        )
                                        row_valid = (row < valid_rows).astype(jnp.float32)
                                        dz_smem[:, :] = (dz_smem[...].astype(jnp.float32) * row_valid).astype(dtype)
                                        mgpu.commit_smem()
                                        mgpu.wgmma(
                                            acc_ref,
                                            mgpu.transpose_ref(x_smem, (1, 0)),
                                            dz_smem,
                                        )
                                        mgpu.wgmma_wait(0)

                                    pl.run_scoped(
                                        _dw_smem_scope,
                                        x_smem=_wgmma_smem((config.compute_m, config.block_hidden), dtype),
                                        dz_smem=_wgmma_smem((config.compute_m, config.block_output), dtype),
                                        barrier=mgpu.Barrier(num_arrivals=2),
                                    )
                                    dw_tile = (
                                        expert * hidden_tiles * output_tiles + hidden_tile * output_tiles + output_tile
                                    )
                                    pl.semaphore_wait(
                                        dw_init_done_sem.at[dw_tile],
                                        value=1,
                                        decrement=False,
                                    )
                                    mgpu.atomic_add(
                                        dw_ref.at[
                                            expert,
                                            pl.ds(hidden_start, config.block_hidden),
                                            pl.ds(
                                                output_tile * config.block_output,
                                                config.block_output,
                                            ),
                                        ],
                                        acc_ref[...],
                                    )

                                pl.run_scoped(
                                    _dw_acc_scope,
                                    acc_ref=mgpu.ACC((config.block_hidden, config.block_output)),
                                )

                    @pl.loop(0, send_chunks_per_dst)
                    def _chunk_loop(chunk) -> None:
                        slot = chunk % config.inbox_slots
                        generation = chunk // config.inbox_slots + 1

                        @pl.when(peer_worker < consumer_start)
                        def _prepare_chunk() -> None:
                            pl.semaphore_wait(
                                prepare_sem.at[static_destination_ordinal, slot],
                                value=generation,
                                decrement=False,
                            )

                            @pl.loop(0, math.ceil(schedule.helper_tiles / config.helper_programs_per_peer))
                            def _helper_loop(iteration) -> None:
                                tile = iteration * config.helper_programs_per_peer + helper

                                @pl.when(tile < schedule.helper_tiles)
                                def _prepare_owned_tile() -> None:
                                    _prepare_x_tile(chunk, slot, tile)
                                    block = tile // schedule.helper_tiles_per_block
                                    pl.semaphore_signal(helper_done_sem.at[static_destination_ordinal, slot, block])

                        @pl.when(peer_worker >= consumer_start)
                        def _consume_chunk() -> None:
                            @pl.loop(0, math.ceil(schedule.compute_jobs_per_chunk / config.consumer_programs_per_peer))
                            def _job_loop(iteration) -> None:
                                job = iteration * config.consumer_programs_per_peer + consumer

                                @pl.when(job < schedule.compute_jobs_per_chunk)
                                def _owned_job() -> None:
                                    block = job // schedule.hidden_tile_jobs
                                    hidden_job = job % schedule.hidden_tile_jobs
                                    pl.semaphore_wait(
                                        block_ready_sem.at[static_source_ordinal, slot, block],
                                        value=generation,
                                        decrement=False,
                                    )

                                    @pl.loop(0, RETURN_HIDDEN_TILES_PER_JOB)
                                    def _hidden_loop(hidden_offset) -> None:
                                        hidden_tile = hidden_job * RETURN_HIDDEN_TILES_PER_JOB + hidden_offset

                                        @pl.when(hidden_tile < hidden_tiles)
                                        def _owned_hidden_tile() -> None:
                                            _compute_hidden_tile(chunk, slot, block, hidden_tile)

                                    pl.semaphore_signal(done_sem.at[static_source_ordinal, slot])

            branches = tuple((lambda ordinal: lambda _: _run_peer(ordinal))(i) for i in range(ep_size))
            lax.switch(peer_ordinal, branches, None)

        @pl.when(worker >= schedule.peer_programs)
        def _combine() -> None:
            consumer = worker - schedule.peer_programs
            hidden_tile = consumer % hidden_tiles
            consumer_lane = consumer // hidden_tiles
            consumers_for_tile = (schedule.combine_programs + hidden_tiles - 1 - hidden_tile) // hidden_tiles
            hidden_start = hidden_tile * config.block_hidden

            @pl.when(consumer_lane < schedule.token_blocks)
            def _active_consumer() -> None:
                @pl.loop(0, schedule.token_blocks)
                def _token_block_loop(iteration) -> None:
                    token_block = consumer_lane + iteration * consumers_for_tile

                    @pl.when(token_block < schedule.token_blocks)
                    def _combine_token_block() -> None:
                        for destination_ordinal in range(ep_size):
                            for slot in range(config.inbox_slots):
                                required_generation = combine_ready_ref[token_block, destination_ordinal, slot]

                                @pl.when(required_generation > 0)
                                def _wait_for_required_round() -> None:
                                    pl.semaphore_wait(
                                        ready_sem.at[destination_ordinal, slot],
                                        value=required_generation,
                                        decrement=False,
                                    )

                        token_start = token_block * config.combine_token_block

                        @pl.loop(0, config.combine_token_block)
                        def _token_loop(token_offset) -> None:
                            token = token_start + token_offset

                            @pl.when(token < tokens_per_source)
                            def _combine_token() -> None:
                                acc = jnp.zeros((config.block_hidden,), dtype=jnp.float32)
                                for route_slot in range(route_valid_ref.shape[-1]):
                                    destination_ordinal = route_dst_ref[token, route_slot]
                                    chunk = route_chunk_ref[token, route_slot]
                                    block = route_block_ref[token, route_slot]
                                    row = route_row_ref[token, route_slot]
                                    route_value = dx_return_ref[
                                        destination_ordinal,
                                        chunk,
                                        block * config.compute_m + row,
                                        pl.ds(hidden_start, config.block_hidden),
                                    ].astype(jnp.float32)
                                    valid = route_valid_ref[token, route_slot] != 0
                                    acc += jnp.where(
                                        valid,
                                        route_value,
                                        jnp.zeros((), dtype=jnp.float32),
                                    )

                                dx_ref[token, pl.ds(hidden_start, config.block_hidden)] = acc

    inbox_shape = (ep_size, config.inbox_slots, config.send_m, hidden_dim)
    return_shape = (ep_size, send_chunks_per_dst, config.send_m, hidden_dim)
    return mgpu.kernel(
        body,
        out_shape=(
            jax.ShapeDtypeStruct(inbox_shape, dtype),
            jax.ShapeDtypeStruct(return_shape, dtype),
            jax.ShapeDtypeStruct((tokens_per_source, hidden_dim), jnp.float32),
            jax.ShapeDtypeStruct((experts_per_rank, hidden_dim, output_dim), jnp.float32),
        ),
        grid=(schedule.total_programs,),
        grid_names=("worker_program",),
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
    )


def _forward_config(config: SourcePushSemanticFusedW13BackwardConfig) -> SourcePushSemanticFusedW13Config:
    return SourcePushSemanticFusedW13Config(
        compute_m=config.compute_m,
        send_m=config.send_m,
        block_n=config.block_output,
        block_k=config.block_hidden,
        send_k=config.send_m,
        inbox_slots=config.inbox_slots,
    )


def _wgmma_smem(shape: tuple[int, int], dtype: jnp.dtype):
    swizzle_elements = WGMMA_SWIZZLE_BYTES // jnp.dtype(dtype).itemsize
    if shape[-2] % WGMMA_TILE_M or shape[-1] % swizzle_elements:
        raise ValueError(f"WGMMA SMEM shape {shape} is incompatible with 128-byte swizzling")
    return mgpu.SMEM(
        shape,
        dtype=dtype,
        transforms=(
            mgpu.TilingTransform((WGMMA_TILE_M, swizzle_elements)),
            mgpu.SwizzleTransform(WGMMA_SWIZZLE_BYTES),
        ),
    )


def _validate_kernel_dimensions(
    hidden_dim: int,
    output_dim: int,
    config: SourcePushSemanticFusedW13BackwardConfig,
) -> None:
    if hidden_dim % config.send_m or hidden_dim % config.block_hidden:
        raise ValueError(
            f"hidden_dim must be divisible by send_m and block_hidden, got {hidden_dim=} "
            f"{config.send_m=} {config.block_hidden=}"
        )
    if output_dim % config.block_output:
        raise ValueError(f"output_dim must be divisible by block_output, got {output_dim=} {config.block_output=}")


def _validate_request(
    x: Array,
    dz13: Array,
    w13: Array,
    plan: SourcePushSemanticPlan,
    rows_per_expert_capacity: int,
    config: SourcePushSemanticFusedW13BackwardConfig,
) -> None:
    config.validate()
    if x.ndim != 3:
        raise ValueError(f"x must have shape [source, token, hidden], got {x.shape}")
    if dz13.ndim != 4 or w13.ndim != 4:
        raise ValueError(f"dz13 and w13 must be rank four, got {dz13.shape=} {w13.shape=}")
    source_count, destination_count, experts_per_rank = plan.xcounts.shape
    if source_count != destination_count or x.shape[:2] != (source_count, plan.tokens_per_source):
        raise ValueError(f"x shape {x.shape} is incompatible with semantic plan shape {plan.xcounts.shape}")
    if dz13.shape[:3] != (destination_count, experts_per_rank, rows_per_expert_capacity):
        raise ValueError(
            f"dz13 leading shape {dz13.shape[:3]} must be "
            f"{(destination_count, experts_per_rank, rows_per_expert_capacity)}"
        )
    if w13.shape != (destination_count, experts_per_rank, x.shape[-1], dz13.shape[-1]):
        raise ValueError(
            f"w13 shape {w13.shape} must be " f"{(destination_count, experts_per_rank, x.shape[-1], dz13.shape[-1])}"
        )
    _validate_kernel_dimensions(x.shape[-1], dz13.shape[-1], config)
