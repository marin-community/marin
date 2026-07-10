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


@dataclass(frozen=True, slots=True)
class SourcePushSemanticFusedW13BackwardConfig:
    """Fixed physical configuration for the initial Hopper stage."""

    compute_m: int = 64
    send_m: int = 256
    send_k: int = 256
    block_hidden: int = 128
    block_output: int = 128
    inbox_slots: int = 12
    producer_programs_per_peer: int = 16
    compute_programs_per_peer: int = 8
    combine_programs_per_peer: int = 8

    def validate(self) -> None:
        expected = {
            "compute_m": 64,
            "send_m": 256,
            "send_k": 256,
            "block_hidden": 128,
            "block_output": 128,
            "inbox_slots": 12,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"the initial Hopper lowering requires {name}={value}, got {getattr(self, name)}")
        for name in ("producer_programs_per_peer", "compute_programs_per_peer", "combine_programs_per_peer"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")

    @property
    def compute_blocks_per_send(self) -> int:
        return self.send_m // self.compute_m


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
    send_return_consumed_target: Int[Array, "S DstOrd C"]
    recv_return_consumed_target: Int[Array, "Dst SrcOrd C"]


class SourcePushSemanticFusedW13BackwardResult(NamedTuple):
    """Source-owned input gradient and destination-owned W13 gradient."""

    dx: Float[Array, "S T H"]
    dw13: Float[Array, "Dst E H O"]
    queue_overflow_routes: Int[Array, ""]
    layout_overflow_rows: Int[Array, ""]


@dataclass(frozen=True, slots=True)
class SourcePushSemanticFusedW13BackwardGenerationAccounting:
    """Cumulative semaphore targets for one source/destination generation."""

    slot: int
    generation: int
    empty_generation: int
    send_done_generation: int
    full_generation: int
    dx_ready_generation: int
    compute_done_generation: int
    returned_route_tiles: int
    released_generation: int


def source_push_semantic_fused_w13_backward_generation_accounting(
    chunk: int,
    *,
    ep_size: int,
    hidden_dim: int,
    valid_rows: int,
    config: SourcePushSemanticFusedW13BackwardConfig = SourcePushSemanticFusedW13BackwardConfig(),
) -> SourcePushSemanticFusedW13BackwardGenerationAccounting:
    """Return fixed and data-dependent semaphore increments for a chunk."""

    config.validate()
    if chunk < 0:
        raise ValueError(f"chunk must be nonnegative, got {chunk}")
    if ep_size <= 0:
        raise ValueError(f"ep_size must be positive, got {ep_size}")
    if hidden_dim % config.send_k or hidden_dim % config.block_hidden:
        raise ValueError(f"hidden_dim {hidden_dim} must be divisible by send_k and block_hidden")
    if not 0 <= valid_rows <= config.send_m:
        raise ValueError(f"valid_rows must be in [0, {config.send_m}], got {valid_rows}")
    generation = chunk // config.inbox_slots + 1
    producer_tiles = config.compute_blocks_per_send * (hidden_dim // config.send_k)
    return SourcePushSemanticFusedW13BackwardGenerationAccounting(
        slot=chunk % config.inbox_slots,
        generation=generation,
        empty_generation=generation,
        send_done_generation=generation * producer_tiles,
        full_generation=generation,
        dx_ready_generation=generation,
        compute_done_generation=generation * config.compute_programs_per_peer,
        returned_route_tiles=valid_rows * (hidden_dim // config.block_hidden),
        released_generation=generation + 1,
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

    hidden_tiles = x.shape[-1] // config.block_hidden
    rows_per_chunk = jnp.sum(forward.send_valid_rows, axis=-1, dtype=jnp.int32)
    increments = rows_per_chunk * hidden_tiles
    send_target = jnp.zeros_like(increments)
    for slot in range(config.inbox_slots):
        slot_increments = increments[:, :, slot :: config.inbox_slots]
        send_target = send_target.at[:, :, slot :: config.inbox_slots].set(jnp.cumsum(slot_increments, axis=2))

    source_count = x.shape[0]
    destination = jnp.arange(source_count, dtype=jnp.int32)[:, None]
    source_ordinal = jnp.arange(source_count, dtype=jnp.int32)[None, :]
    recv_source = (destination + source_ordinal) % source_count
    recv_dst_ordinal = (-source_ordinal) % source_count
    recv_target = send_target.at[recv_source, recv_dst_ordinal].get()
    return SourcePushSemanticFusedW13BackwardMetadata(
        forward=forward,
        route_dst_ordinal=route_dst_ordinal,
        route_chunk=route_chunk,
        route_block=route_block,
        route_row=route_row,
        route_valid=route_valid,
        send_return_consumed_target=send_target,
        recv_return_consumed_target=recv_target,
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
    dz13 = jnp.where(metadata.forward.valid[..., None], dz13, jnp.zeros((), dtype=dz13.dtype))
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
        recv_return_target_local,
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
            recv_return_target_local[0],
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
        jax.sharding.reshard(metadata.recv_return_consumed_target, destination_3d),
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
    blocks = config.compute_blocks_per_send
    hidden_tiles = hidden_dim // config.block_hidden
    output_tiles = output_dim // config.block_output
    producer_tiles = blocks * (hidden_dim // config.send_k)
    producer_tiles_per_program = math.ceil(producer_tiles / config.producer_programs_per_peer)
    total_compute_programs = ep_size * config.compute_programs_per_peer
    dw_tiles = experts_per_rank * hidden_tiles * output_tiles
    worker_programs = (
        config.producer_programs_per_peer + config.compute_programs_per_peer + config.combine_programs_per_peer
    )

    def body(
        x_ref,
        token_ids_ref,
        send_valid_ref,
        recv_expert_ref,
        recv_row_ref,
        recv_valid_ref,
        recv_return_target_ref,
        dz_ref,
        w_ref,
        x_inbox_ref,
        dx_return_ref,
        dx_ref,
        dw_ref,
    ) -> None:
        empty_sem = pl.get_global(mgpu.SemaphoreType.REGULAR((ep_size, config.inbox_slots)))
        send_done_sem = pl.get_global(mgpu.SemaphoreType.REGULAR((ep_size, config.inbox_slots)))
        full_sem = pl.get_global(mgpu.SemaphoreType.REGULAR((ep_size, config.inbox_slots)))
        compute_done_sem = pl.get_global(mgpu.SemaphoreType.REGULAR((ep_size, config.inbox_slots)))
        dx_full_sem = pl.get_global(mgpu.SemaphoreType.REGULAR((ep_size, config.inbox_slots, blocks, hidden_tiles)))
        return_consumed_sem = pl.get_global(mgpu.SemaphoreType.REGULAR((ep_size, config.inbox_slots)))
        dx_init_done_sem = pl.get_global(mgpu.SemaphoreType.REGULAR(()))
        dx_init_ready_sem = pl.get_global(mgpu.SemaphoreType.REGULAR(()))
        dw_init_done_sem = pl.get_global(mgpu.SemaphoreType.REGULAR((dw_tiles,)))
        rank = lax.axis_index(SOURCE_PUSH_MESH_AXIS)
        peer_ordinal = pl.program_id(0)
        worker = pl.program_id(1)

        def _signal_remote(sem, peer, index) -> None:
            pl.semaphore_signal(
                sem.at[index],
                device_id=peer,
                device_id_type=pl.DeviceIdType.LOGICAL,
            )

        @pl.when(worker < config.producer_programs_per_peer)
        def _producer() -> None:
            producer = worker

            def _send_peer(static_peer_ordinal: int) -> None:
                dst = (rank + static_peer_ordinal) % ep_size
                remote_inbox = None
                if static_peer_ordinal != 0:
                    remote_inbox = mgpu.remote_ref(x_inbox_ref, dst, device_id_type=pl.DeviceIdType.LOGICAL)

                @pl.loop(0, send_chunks_per_dst)
                def _chunk_loop(chunk) -> None:
                    slot = chunk % config.inbox_slots
                    generation = chunk // config.inbox_slots + 1

                    @pl.loop(0, producer_tiles_per_program)
                    def _tile_loop(tile_iteration) -> None:
                        tile = producer + tile_iteration * config.producer_programs_per_peer

                        @pl.when(tile < producer_tiles)
                        def _send_tile() -> None:
                            block = tile // (hidden_dim // config.send_k)
                            k_tile = tile % (hidden_dim // config.send_k)
                            k_start = k_tile * config.send_k
                            pl.semaphore_wait(empty_sem.at[dst, slot], value=generation, decrement=False)

                            def _copy_scope(tile_smem) -> None:
                                valid_rows = send_valid_ref[static_peer_ordinal, chunk, block]

                                @pl.loop(0, config.compute_m)
                                def _row_loop(row) -> None:
                                    token = token_ids_ref[static_peer_ordinal, chunk, block, row]
                                    tile_smem[row, :] = jnp.where(
                                        row < valid_rows,
                                        x_ref[token, pl.ds(k_start, config.send_k)],
                                        jnp.zeros((config.send_k,), dtype=dtype),
                                    )

                                mgpu.commit_smem()
                                destination_ref = x_inbox_ref if static_peer_ordinal == 0 else remote_inbox
                                mgpu.copy_smem_to_gmem(
                                    tile_smem,
                                    destination_ref.at[
                                        rank,
                                        slot,
                                        pl.ds(block * config.compute_m, config.compute_m),
                                        pl.ds(k_start, config.send_k),
                                    ],
                                )
                                mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                            pl.run_scoped(
                                _copy_scope,
                                tile_smem=mgpu.SMEM((config.compute_m, config.send_k), dtype=dtype),
                            )
                            pl.semaphore_signal(send_done_sem.at[dst, slot])

                    @pl.when(producer == 0)
                    def _publish() -> None:
                        pl.semaphore_wait(
                            send_done_sem.at[dst, slot],
                            value=generation * producer_tiles,
                            decrement=False,
                        )
                        if static_peer_ordinal == 0:
                            pl.semaphore_signal(full_sem.at[rank, slot])
                        else:
                            _signal_remote(full_sem, dst, (rank, slot))

            branches = tuple((lambda ordinal: lambda _: _send_peer(ordinal))(i) for i in range(ep_size))
            lax.switch(peer_ordinal, branches, None)

        @pl.when(
            (worker >= config.producer_programs_per_peer)
            & (worker < config.producer_programs_per_peer + config.compute_programs_per_peer)
        )
        def _compute() -> None:
            local_compute = worker - config.producer_programs_per_peer
            global_compute_id = peer_ordinal * config.compute_programs_per_peer + local_compute

            # Every dW tile has one initializer; peer-local compute groups atomically add source partials.
            @pl.loop(0, math.ceil(dw_tiles / total_compute_programs))
            def _initialize_owned_dw(iteration) -> None:
                tile = iteration * total_compute_programs + global_compute_id

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

            @pl.when(local_compute == 0)
            def _initialize_empty_credit() -> None:
                def _init_source(static_src_ordinal: int) -> None:
                    src = (rank + static_src_ordinal) % ep_size

                    @pl.loop(0, config.inbox_slots)
                    def _slot_loop(slot) -> None:
                        if static_src_ordinal == 0:
                            pl.semaphore_signal(empty_sem.at[rank, slot])
                        else:
                            _signal_remote(empty_sem, src, (rank, slot))

                branches = tuple((lambda ordinal: lambda _: _init_source(ordinal))(i) for i in range(ep_size))
                lax.switch(peer_ordinal, branches, None)

            def _dx_job(src: int, chunk, slot, generation, block, hidden_tile) -> None:
                valid_rows = recv_valid_ref[src, chunk, block]

                @pl.when(valid_rows > 0)
                def _live_dx() -> None:
                    expert = recv_expert_ref[src, chunk, block]
                    row_start = recv_row_ref[src, chunk, block]

                    def _acc_scope(acc_ref) -> None:
                        def _smem_scope(dz_smem, w_smem, barrier) -> None:
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
                                        pl.ds(hidden_tile * config.block_hidden, config.block_hidden),
                                        pl.ds(output_start, config.block_output),
                                    ],
                                    w_smem,
                                    barrier,
                                )
                                mgpu.barrier_wait(barrier)
                                mgpu.commit_smem()
                                mgpu.wgmma(acc_ref, dz_smem, mgpu.transpose_ref(w_smem, (1, 0)))
                                mgpu.wgmma_wait(0)

                        pl.run_scoped(
                            _smem_scope,
                            dz_smem=_wgmma_smem((config.compute_m, config.block_output), dtype),
                            w_smem=_wgmma_smem((config.block_hidden, config.block_output), dtype),
                            barrier=mgpu.Barrier(num_arrivals=2),
                        )

                        source_rank = (rank + src) % ep_size
                        destination_ref = dx_return_ref
                        if src != 0:
                            destination_ref = mgpu.remote_ref(
                                dx_return_ref,
                                source_rank,
                                device_id_type=pl.DeviceIdType.LOGICAL,
                            )
                        destination_ref[
                            rank,
                            slot,
                            pl.ds(block * config.compute_m, config.compute_m),
                            pl.ds(hidden_tile * config.block_hidden, config.block_hidden),
                        ] = acc_ref[...].astype(dtype)

                    pl.run_scoped(
                        _acc_scope,
                        acc_ref=mgpu.ACC((config.compute_m, config.block_hidden)),
                    )

                # Signal every generation, including masked blocks, so slot reuse has a fixed target.
                source_rank = (rank + src) % ep_size
                if src == 0:
                    pl.semaphore_signal(dx_full_sem.at[rank, slot, block, hidden_tile])
                else:
                    _signal_remote(
                        dx_full_sem,
                        source_rank,
                        (rank, slot, block, hidden_tile),
                    )

            def _dw_job(src: int, chunk, slot, block, dw_tile) -> None:
                valid_rows = recv_valid_ref[src, chunk, block]
                expert = dw_tile // (hidden_tiles * output_tiles)
                remainder = dw_tile % (hidden_tiles * output_tiles)
                hidden_tile = remainder // output_tiles
                output_tile = remainder % output_tiles
                block_expert = recv_expert_ref[src, chunk, block]

                @pl.when((valid_rows > 0) & (expert == block_expert))
                def _live_dw() -> None:
                    row_start = recv_row_ref[src, chunk, block]

                    def _acc_scope(acc_ref) -> None:
                        def _smem_scope(x_smem, dz_smem, barrier) -> None:
                            mgpu.copy_gmem_to_smem(
                                x_inbox_ref.at[
                                    (rank + src) % ep_size,
                                    slot,
                                    pl.ds(block * config.compute_m, config.compute_m),
                                    pl.ds(hidden_tile * config.block_hidden, config.block_hidden),
                                ],
                                x_smem,
                                barrier,
                            )
                            mgpu.copy_gmem_to_smem(
                                dz_ref.at[
                                    expert,
                                    pl.ds(row_start, config.compute_m),
                                    pl.ds(output_tile * config.block_output, config.block_output),
                                ],
                                dz_smem,
                                barrier,
                            )
                            mgpu.barrier_wait(barrier)
                            mgpu.commit_smem()
                            mgpu.wgmma(acc_ref, mgpu.transpose_ref(x_smem, (1, 0)), dz_smem)
                            mgpu.wgmma_wait(0)

                        pl.run_scoped(
                            _smem_scope,
                            x_smem=_wgmma_smem((config.compute_m, config.block_hidden), dtype),
                            dz_smem=_wgmma_smem((config.compute_m, config.block_output), dtype),
                            barrier=mgpu.Barrier(num_arrivals=2),
                        )
                        dw_index = (
                            expert,
                            pl.ds(hidden_tile * config.block_hidden, config.block_hidden),
                            pl.ds(output_tile * config.block_output, config.block_output),
                        )
                        pl.semaphore_wait(dw_init_done_sem.at[dw_tile], value=1, decrement=False)
                        mgpu.atomic_add(dw_ref.at[dw_index], acc_ref[...])

                    pl.run_scoped(
                        _acc_scope,
                        acc_ref=mgpu.ACC((config.block_hidden, config.block_output)),
                    )

            def _compute_source(static_src: int) -> None:

                @pl.loop(0, send_chunks_per_dst)
                def _chunk_loop(chunk) -> None:
                    slot = chunk % config.inbox_slots
                    generation = chunk // config.inbox_slots + 1
                    pl.semaphore_wait(
                        full_sem.at[(rank + static_src) % ep_size, slot], value=generation, decrement=False
                    )

                    @pl.loop(0, math.ceil((blocks * hidden_tiles) / config.compute_programs_per_peer))
                    def _dx_loop(iteration) -> None:
                        job = iteration * config.compute_programs_per_peer + local_compute

                        @pl.when(job < blocks * hidden_tiles)
                        def _owned_dx() -> None:
                            block = job // hidden_tiles
                            hidden_tile = job % hidden_tiles
                            _dx_job(static_src, chunk, slot, generation, block, hidden_tile)

                    @pl.loop(0, math.ceil(dw_tiles / config.compute_programs_per_peer))
                    def _dw_loop(iteration) -> None:
                        dw_tile = iteration * config.compute_programs_per_peer + local_compute

                        @pl.when(dw_tile < dw_tiles)
                        def _owned_dw() -> None:
                            for block in range(blocks):
                                _dw_job(static_src, chunk, slot, block, dw_tile)

                    pl.semaphore_signal(compute_done_sem.at[(rank + static_src) % ep_size, slot])

                    @pl.when(local_compute == 0)
                    def _release() -> None:
                        src_rank = (rank + static_src) % ep_size
                        pl.semaphore_wait(
                            compute_done_sem.at[src_rank, slot],
                            value=generation * config.compute_programs_per_peer,
                            decrement=False,
                        )
                        return_target = recv_return_target_ref[static_src, chunk]
                        pl.semaphore_wait(
                            return_consumed_sem.at[src_rank, slot],
                            value=return_target,
                            decrement=False,
                        )
                        if static_src == 0:
                            pl.semaphore_signal(empty_sem.at[rank, slot])
                        else:
                            _signal_remote(empty_sem, src_rank, (rank, slot))

            branches = tuple(
                (lambda source_ordinal: lambda _: _compute_source(source_ordinal))((-phase) % ep_size)
                for phase in range(ep_size)
            )
            lax.switch(peer_ordinal, branches, None)

        @pl.when(worker >= config.producer_programs_per_peer + config.compute_programs_per_peer)
        def _combine() -> None:
            local_combine = worker - config.producer_programs_per_peer - config.compute_programs_per_peer
            initialize_jobs = tokens_per_source * hidden_tiles

            # One peer-local resident group initializes dX before any route
            # accumulation. A rank-wide CTA barrier would exceed Hopper
            # residency at the target EP size.
            @pl.when(peer_ordinal == 0)
            def _initialize_dx() -> None:
                @pl.loop(0, math.ceil(initialize_jobs / config.combine_programs_per_peer))
                def _initialize_dx_loop(iteration) -> None:
                    job = iteration * config.combine_programs_per_peer + local_combine

                    @pl.when(job < initialize_jobs)
                    def _zero_owned_dx() -> None:
                        token = job // hidden_tiles
                        hidden_tile = job % hidden_tiles
                        hidden_start = hidden_tile * config.block_hidden
                        dx_ref[token, pl.ds(hidden_start, config.block_hidden)] = jnp.zeros(
                            (config.block_hidden,), dtype=jnp.float32
                        )

                pl.semaphore_signal(dx_init_done_sem)

                @pl.when(local_combine == 0)
                def _publish_dx_initialized() -> None:
                    pl.semaphore_wait(
                        dx_init_done_sem,
                        value=config.combine_programs_per_peer,
                        decrement=False,
                    )
                    pl.semaphore_signal(dx_init_ready_sem)

            pl.semaphore_wait(dx_init_ready_sem, value=1, decrement=False)

            def _consume_destination(static_dst_ordinal: int) -> None:
                dst = (rank + static_dst_ordinal) % ep_size

                @pl.loop(0, send_chunks_per_dst)
                def _chunk_loop(chunk) -> None:
                    slot = chunk % config.inbox_slots
                    generation = chunk // config.inbox_slots + 1

                    @pl.loop(0, math.ceil((blocks * hidden_tiles) / config.combine_programs_per_peer))
                    def _physical_tile_loop(iteration) -> None:
                        tile = iteration * config.combine_programs_per_peer + local_combine

                        @pl.when(tile < blocks * hidden_tiles)
                        def _consume_owned_tile() -> None:
                            block = tile // hidden_tiles
                            hidden_tile = tile % hidden_tiles
                            hidden_start = hidden_tile * config.block_hidden
                            pl.semaphore_wait(
                                dx_full_sem.at[dst, slot, block, hidden_tile],
                                value=generation,
                                decrement=False,
                            )
                            valid_rows = send_valid_ref[static_dst_ordinal, chunk, block]

                            @pl.loop(0, config.compute_m)
                            def _row_loop(row) -> None:
                                @pl.when(row < valid_rows)
                                def _scatter_live_row() -> None:
                                    token = token_ids_ref[static_dst_ordinal, chunk, block, row]
                                    # Keep the row axis so lane lowering distributes the atomic value as a
                                    # matrix tile. A rank-1 GMEM value is replicated across lanes here.
                                    dx_tile = dx_return_ref[
                                        dst,
                                        slot,
                                        pl.ds(block * config.compute_m + row, 1),
                                        pl.ds(hidden_start, config.block_hidden),
                                    ].astype(jnp.float32)
                                    mgpu.atomic_add(
                                        dx_ref.at[
                                            pl.ds(token, 1),
                                            pl.ds(hidden_start, config.block_hidden),
                                        ],
                                        dx_tile,
                                    )

                                    if static_dst_ordinal == 0:
                                        pl.semaphore_signal(return_consumed_sem.at[rank, slot])
                                    else:
                                        _signal_remote(return_consumed_sem, dst, (rank, slot))

            branches = tuple((lambda ordinal: lambda _: _consume_destination(ordinal))(i) for i in range(ep_size))
            lax.switch(peer_ordinal, branches, None)

    scratch_shape = (ep_size, config.inbox_slots, config.send_m, hidden_dim)
    return mgpu.kernel(
        body,
        out_shape=(
            jax.ShapeDtypeStruct(scratch_shape, dtype),
            jax.ShapeDtypeStruct(scratch_shape, dtype),
            jax.ShapeDtypeStruct((tokens_per_source, hidden_dim), jnp.float32),
            jax.ShapeDtypeStruct((experts_per_rank, hidden_dim, output_dim), jnp.float32),
        ),
        grid=(ep_size, worker_programs),
        grid_names=("peer_phase", "worker_program"),
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
    )


def _forward_config(config: SourcePushSemanticFusedW13BackwardConfig) -> SourcePushSemanticFusedW13Config:
    return SourcePushSemanticFusedW13Config(
        compute_m=config.compute_m,
        send_m=config.send_m,
        block_n=config.block_output,
        block_k=config.block_hidden,
        send_k=config.send_k,
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
    if hidden_dim % config.send_k or hidden_dim % config.block_hidden:
        raise ValueError(
            f"hidden_dim must be divisible by send_k and block_hidden, got {hidden_dim=} "
            f"{config.send_k=} {config.block_hidden=}"
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
