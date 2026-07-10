# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent expert W2 fused with direct source return and source combine."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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
from levanter.grug._moe.source_push_semantic_inbox_pallas import source_push_semantic_inbox_layout_jax


WGMMA_SWIZZLE_BYTES = 128
WGMMA_TILE_M = 8


@dataclass(frozen=True, slots=True)
class SourcePushSemanticFusedW2ReturnConfig:
    """Fixed Hopper tile and persistent-CTA dimensions."""

    compute_m: int = 64
    block_k: int = 64
    block_n: int = 128
    producer_programs_per_peer: int = 16
    combine_programs: int = 32
    combine_token_block: int = 64

    def validate(self) -> None:
        if self.compute_m != 64:
            raise ValueError(f"the initial Hopper lowering requires compute_m=64, got {self.compute_m}")
        if self.block_k != 64:
            raise ValueError(f"the initial Hopper lowering requires block_k=64, got {self.block_k}")
        if self.block_n != 128:
            raise ValueError(f"the initial Hopper lowering requires block_n=128, got {self.block_n}")
        if self.producer_programs_per_peer != 16:
            raise ValueError(
                "the initial Hopper lowering requires 16 W2 producer programs per peer, "
                f"got {self.producer_programs_per_peer}"
            )
        if self.combine_programs != 32:
            raise ValueError(f"the initial Hopper lowering requires 32 combine programs, got {self.combine_programs}")
        if self.combine_token_block != 64:
            raise ValueError(
                f"the initial Hopper lowering requires combine_token_block=64, got {self.combine_token_block}"
            )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SourcePushSemanticFusedW2ReturnMetadata:
    """Destination W2 rows and source-owned inverse route queue."""

    queue_local_expert: Int[Array, "S DstOrd Q"]
    queue_local_row_start: Int[Array, "S DstOrd Q"]
    queue_valid_rows: Int[Array, "S DstOrd Q"]
    recv_local_expert: Int[Array, "Dst SrcOrd Q"]
    recv_expert_row_start: Int[Array, "Dst SrcOrd Q"]
    recv_valid_rows: Int[Array, "Dst SrcOrd Q"]
    queue_dst_ordinal: Int[Array, "S T K"]
    queue_entry: Int[Array, "S T K"]
    queue_row: Int[Array, "S T K"]
    route_weight: Float[Array, "S T K"]
    route_valid: Bool[Array, "S T K"]
    expert_valid: Bool[Array, "Dst E C"]
    queue_overflow_routes: Int[Array, ""]
    layout_overflow_rows: Int[Array, ""]
    rows_per_expert_capacity: int = field(metadata={"static": True})
    entries_per_dst: int = field(metadata={"static": True})


class SourcePushSemanticFusedW2ReturnResult(NamedTuple):
    """Source-sharded output and the bf16 route values saved for backward."""

    y: Float[Array, "S T H"]
    return_y: Float[Array, "S DstOrd Q M H"]
    expert_valid: Bool[Array, "Dst E C"]
    queue_overflow_routes: Int[Array, ""]
    layout_overflow_rows: Int[Array, ""]


@dataclass(frozen=True, slots=True)
class SourcePushSemanticFusedW2ReturnSemaphoreAccounting:
    """One-shot readiness generations for a final source-owned route tile."""

    destination_ordinal: int
    entry: int
    hidden_tile: int
    producer_signal_generation: int
    combine_wait_generation: int


def source_push_semantic_fused_w2_return_semaphore_accounting(
    destination_ordinal: int,
    entry: int,
    hidden_tile: int,
) -> SourcePushSemanticFusedW2ReturnSemaphoreAccounting:
    """Describe the publish/wait generation for one final return tile.

    Return storage is the final backward residual, not a recyclable staging
    slot. Each tile is therefore published exactly once in a kernel invocation.
    """

    if destination_ordinal < 0 or entry < 0 or hidden_tile < 0:
        raise ValueError("destination_ordinal, entry, and hidden_tile must be nonnegative")
    return SourcePushSemanticFusedW2ReturnSemaphoreAccounting(
        destination_ordinal=destination_ordinal,
        entry=entry,
        hidden_tile=hidden_tile,
        producer_signal_generation=1,
        combine_wait_generation=1,
    )


def source_push_semantic_fused_w2_return_metadata_jax(
    plan: SourcePushSemanticPlan,
    *,
    rows_per_expert_capacity: int,
    entries_per_dst: int,
    config: SourcePushSemanticFusedW2ReturnConfig = SourcePushSemanticFusedW2ReturnConfig(),
) -> SourcePushSemanticFusedW2ReturnMetadata:
    """Lower semantic routes to source-padded W2 and direct-return metadata."""

    config.validate()
    if rows_per_expert_capacity <= 0:
        raise ValueError(f"rows_per_expert_capacity must be positive, got {rows_per_expert_capacity}")
    if entries_per_dst <= 0:
        raise ValueError(f"entries_per_dst must be positive, got {entries_per_dst}")
    source_count, destination_count, _ = plan.xcounts.shape
    if source_count != destination_count:
        raise ValueError(f"source and destination counts must match, got {plan.xcounts.shape}")

    queue = source_push_semantic_queue_metadata_jax(
        plan,
        return_row_block=config.compute_m,
        entries_per_dst=entries_per_dst,
    )
    layout = source_push_semantic_inbox_layout_jax(
        plan,
        queue,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )

    source_index = jnp.arange(source_count, dtype=jnp.int32)[:, None, None]
    destination_ordinal_index = jnp.arange(destination_count, dtype=jnp.int32)[None, :, None]
    actual_destination = (source_index + destination_ordinal_index) % destination_count
    safe_queue_expert = jnp.maximum(queue.local_expert, 0)
    queue_expert_row_start = (
        layout.src_base_by_expert.at[actual_destination, source_index, safe_queue_expert].get() + queue.local_row_start
    )
    queue_valid_rows = jnp.clip(
        rows_per_expert_capacity - queue_expert_row_start,
        0,
        queue.valid_rows,
    ).astype(jnp.int32)

    destination = jnp.arange(destination_count, dtype=jnp.int32)[:, None]
    source_ordinal = jnp.arange(source_count, dtype=jnp.int32)[None, :]
    source = (destination + source_ordinal) % source_count
    destination_ordinal = jnp.broadcast_to((-source_ordinal) % destination_count, source.shape)
    recv_local_expert = queue.local_expert.at[source, destination_ordinal].get()
    recv_local_row_start = queue.local_row_start.at[source, destination_ordinal].get()
    recv_valid_rows = queue_valid_rows.at[source, destination_ordinal].get()
    safe_expert = jnp.maximum(recv_local_expert, 0)
    recv_source_base = layout.src_base_by_expert.at[
        destination[..., None],
        source[..., None],
        safe_expert,
    ].get()
    recv_expert_row_start = jnp.where(
        recv_valid_rows > 0,
        recv_source_base + recv_local_row_start,
        0,
    ).astype(jnp.int32)

    source_pair = jnp.arange(source_count, dtype=jnp.int32)[:, None, None]
    safe_token = jnp.where(plan.valid_mask, jnp.maximum(plan.token_ids, 0), plan.tokens_per_source)
    safe_route_slot = jnp.where(plan.valid_mask, jnp.maximum(plan.route_slots, 0), plan.topk)
    route_weight = jnp.zeros(
        (source_count, plan.tokens_per_source, plan.topk),
        dtype=plan.route_weights.dtype,
    )
    route_weight = route_weight.at[source_pair, safe_token, safe_route_slot].set(plan.route_weights, mode="drop")
    route_queue_valid_rows = queue_valid_rows.at[
        jnp.arange(source_count, dtype=jnp.int32)[:, None, None],
        queue.route_dst_ordinal,
        queue.route_entry,
    ].get(mode="clip")
    route_valid = queue.route_valid & (queue.route_queue_row < route_queue_valid_rows)
    route_weight = jnp.where(route_valid, route_weight, jnp.zeros((), dtype=route_weight.dtype))

    return SourcePushSemanticFusedW2ReturnMetadata(
        queue_local_expert=queue.local_expert,
        queue_local_row_start=queue.local_row_start,
        queue_valid_rows=queue_valid_rows,
        recv_local_expert=recv_local_expert,
        recv_expert_row_start=recv_expert_row_start,
        recv_valid_rows=recv_valid_rows,
        queue_dst_ordinal=queue.route_dst_ordinal,
        queue_entry=queue.route_entry,
        queue_row=queue.route_queue_row,
        route_weight=route_weight,
        route_valid=route_valid,
        expert_valid=layout.valid,
        queue_overflow_routes=queue.overflow_routes,
        layout_overflow_rows=layout.overflow_rows,
        rows_per_expert_capacity=rows_per_expert_capacity,
        entries_per_dst=entries_per_dst,
    )


def source_push_semantic_fused_w2_return_reference_jax(
    z_expert: Float[Array, "Dst E C twoI"],
    w_down: Float[Array, "Dst E I H"],
    metadata: SourcePushSemanticFusedW2ReturnMetadata,
) -> tuple[Float[Array, "S T H"], Float[Array, "S DstOrd Q M H"]]:
    """Obvious SwiGLU, W2, direct-return, and fp32 source-combine reference."""

    _validate_reference_request(z_expert, w_down, metadata)
    intermediate_dim = w_down.shape[-2]
    gate = z_expert[..., :intermediate_dim].astype(jnp.float32)
    up = z_expert[..., intermediate_dim:].astype(jnp.float32)
    h_expert = jax.nn.silu(gate) * up
    route_y_expert = jnp.einsum(
        "deci,deih->dech",
        h_expert,
        w_down.astype(jnp.float32),
        preferred_element_type=jnp.float32,
    )
    route_y_expert = jnp.where(
        metadata.expert_valid[..., None],
        route_y_expert,
        jnp.zeros((), dtype=route_y_expert.dtype),
    )

    source_count = metadata.queue_local_expert.shape[0]
    source = jnp.arange(source_count, dtype=jnp.int32)[:, None, None]
    destination_ordinal = jnp.arange(source_count, dtype=jnp.int32)[None, :, None]
    destination = (source + destination_ordinal) % source_count
    safe_expert = jnp.maximum(metadata.queue_local_expert, 0)

    source_ordinal = (-destination_ordinal) % source_count
    queue_entry = jnp.arange(metadata.entries_per_dst, dtype=jnp.int32)[None, None, :]
    expert_row_start = metadata.recv_expert_row_start.at[destination, source_ordinal, queue_entry].get()
    queue_row = jnp.arange(SourcePushSemanticFusedW2ReturnConfig().compute_m, dtype=jnp.int32)
    queue_row = queue_row[None, None, None, :]
    expert_row = expert_row_start[..., None] + queue_row
    safe_expert_row = jnp.minimum(expert_row, metadata.rows_per_expert_capacity - 1)
    return_y = route_y_expert.at[
        destination[..., None],
        safe_expert[..., None],
        safe_expert_row,
    ].get()
    return_valid = queue_row < metadata.queue_valid_rows[..., None]
    return_y = jnp.where(return_valid[..., None], return_y, jnp.zeros((), dtype=return_y.dtype)).astype(jnp.bfloat16)

    route_value = return_y.at[
        jnp.arange(source_count, dtype=jnp.int32)[:, None, None],
        metadata.queue_dst_ordinal,
        metadata.queue_entry,
        metadata.queue_row,
    ].get(mode="clip")
    weighted = route_value.astype(jnp.float32) * metadata.route_weight.astype(jnp.float32)[..., None]
    weighted = jnp.where(metadata.route_valid[..., None], weighted, jnp.zeros((), dtype=weighted.dtype))
    y = jnp.sum(weighted, axis=2, dtype=jnp.float32).astype(jnp.bfloat16)
    return y, return_y


def source_push_semantic_fused_w2_return(
    z_expert: Float[Array, "Dst E C twoI"],
    w_down: Float[Array, "Dst E I H"],
    plan: SourcePushSemanticPlan,
    *,
    entries_per_dst: int,
    config: SourcePushSemanticFusedW2ReturnConfig = SourcePushSemanticFusedW2ReturnConfig(),
    mesh: Mesh | AbstractMesh | None = None,
    interpret: bool = False,
) -> SourcePushSemanticFusedW2ReturnResult:
    """Run persistent W2/direct-return/source-combine or its JAX reference."""

    _validate_request(z_expert, w_down, plan, entries_per_dst, config)
    metadata = source_push_semantic_fused_w2_return_metadata_jax(
        plan,
        rows_per_expert_capacity=z_expert.shape[2],
        entries_per_dst=entries_per_dst,
        config=config,
    )
    if interpret:
        y, return_y = source_push_semantic_fused_w2_return_reference_jax(z_expert, w_down, metadata)
    else:
        if jax.default_backend() != "gpu":
            raise NotImplementedError("persistent semantic fused W2 return requires a GPU backend")
        if mesh is None:
            mesh = jax.sharding.get_abstract_mesh()
            if mesh.empty:
                raise ValueError("mesh is required for persistent semantic fused W2 return")
        y, return_y = _source_push_semantic_fused_w2_return_sharded(
            z_expert,
            w_down,
            metadata,
            config=config,
            mesh=mesh,
        )
    return SourcePushSemanticFusedW2ReturnResult(
        y=y,
        return_y=return_y,
        expert_valid=metadata.expert_valid,
        queue_overflow_routes=metadata.queue_overflow_routes,
        layout_overflow_rows=metadata.layout_overflow_rows,
    )


def _source_push_semantic_fused_w2_return_sharded(
    z_expert: Array,
    w_down: Array,
    metadata: SourcePushSemanticFusedW2ReturnMetadata,
    *,
    config: SourcePushSemanticFusedW2ReturnConfig,
    mesh: Mesh | AbstractMesh,
) -> tuple[Array, Array]:
    source_count = metadata.queue_local_expert.shape[0]
    if mesh.shape[SOURCE_PUSH_MESH_AXIS] != source_count:
        raise ValueError(
            f"mesh {SOURCE_PUSH_MESH_AXIS!r} size must match source count {source_count}, "
            f"got {mesh.shape[SOURCE_PUSH_MESH_AXIS]}"
        )
    kernel = _make_source_push_semantic_fused_w2_return_kernel(
        ep_size=source_count,
        experts_per_rank=z_expert.shape[1],
        rows_per_expert_capacity=z_expert.shape[2],
        intermediate_dim=w_down.shape[-2],
        hidden_dim=w_down.shape[-1],
        tokens_per_source=metadata.queue_dst_ordinal.shape[1],
        topk=metadata.queue_dst_ordinal.shape[2],
        entries_per_dst=metadata.entries_per_dst,
        dtype=z_expert.dtype,
        config=config,
    )

    def local_fn(
        z_local,
        w_local,
        recv_expert_local,
        recv_row_local,
        recv_valid_local,
        queue_dst_local,
        queue_entry_local,
        queue_row_local,
        route_weight_local,
        route_valid_local,
    ):
        return_y_local, y_local = kernel(
            z_local[0],
            w_local[0],
            recv_expert_local[0],
            recv_row_local[0],
            recv_valid_local[0],
            queue_dst_local[0],
            queue_entry_local[0],
            queue_row_local[0],
            route_weight_local[0],
            route_valid_local[0].astype(jnp.int32),
        )
        return y_local[None, ...], return_y_local[None, ...]

    destination_4d = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None))
    destination_3d = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None))
    source_3d = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None))
    y, return_y = shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        out_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
        ),
        check_vma=False,
    )(
        jax.sharding.reshard(z_expert, destination_4d),
        jax.sharding.reshard(w_down, destination_4d),
        jax.sharding.reshard(metadata.recv_local_expert, destination_3d),
        jax.sharding.reshard(metadata.recv_expert_row_start, destination_3d),
        jax.sharding.reshard(metadata.recv_valid_rows, destination_3d),
        jax.sharding.reshard(metadata.queue_dst_ordinal, source_3d),
        jax.sharding.reshard(metadata.queue_entry, source_3d),
        jax.sharding.reshard(metadata.queue_row, source_3d),
        jax.sharding.reshard(metadata.route_weight, source_3d),
        jax.sharding.reshard(metadata.route_valid, source_3d),
    )
    return y, return_y


def _make_source_push_semantic_fused_w2_return_kernel(
    *,
    ep_size: int,
    experts_per_rank: int,
    rows_per_expert_capacity: int,
    intermediate_dim: int,
    hidden_dim: int,
    tokens_per_source: int,
    topk: int,
    entries_per_dst: int,
    dtype: jnp.dtype,
    config: SourcePushSemanticFusedW2ReturnConfig,
):
    del experts_per_rank, rows_per_expert_capacity
    config.validate()
    hidden_tiles = hidden_dim // config.block_n
    intermediate_tiles = intermediate_dim // config.block_k
    producer_jobs = entries_per_dst * hidden_tiles
    producer_jobs_per_program = math.ceil(producer_jobs / config.producer_programs_per_peer)
    token_blocks = math.ceil(tokens_per_source / config.combine_token_block)
    combine_jobs = token_blocks * hidden_tiles
    combine_jobs_per_program = math.ceil(combine_jobs / config.combine_programs)
    worker_programs = config.producer_programs_per_peer + config.combine_programs

    def body(
        z_ref,
        w_ref,
        recv_expert_ref,
        recv_row_ref,
        recv_valid_ref,
        queue_dst_ref,
        queue_entry_ref,
        queue_row_ref,
        route_weight_ref,
        route_valid_ref,
        return_y_ref,
        y_ref,
    ) -> None:
        ready_sem = pl.get_global(mgpu.SemaphoreType.REGULAR((ep_size, entries_per_dst, hidden_tiles)))
        rank = lax.axis_index(SOURCE_PUSH_MESH_AXIS)
        peer_ordinal = pl.program_id(0)
        worker = pl.program_id(1)

        def _signal_ready(peer, destination_ordinal, entry, hidden_tile) -> None:
            pl.semaphore_signal(
                ready_sem.at[destination_ordinal, entry, hidden_tile],
                device_id=peer,
                device_id_type=pl.DeviceIdType.LOGICAL,
            )

        @pl.when(worker < config.producer_programs_per_peer)
        def _producer() -> None:
            producer = worker

            def _produce_for_source(static_source_ordinal: int) -> None:
                source = (rank + static_source_ordinal) % ep_size
                destination_ordinal = (-static_source_ordinal) % ep_size
                if static_source_ordinal == 0:
                    destination_ref = return_y_ref
                else:
                    destination_ref = mgpu.remote_ref(
                        return_y_ref,
                        source,
                        device_id_type=pl.DeviceIdType.LOGICAL,
                    )

                @pl.loop(0, producer_jobs_per_program)
                def _producer_job_loop(job_iteration) -> None:
                    job = producer + job_iteration * config.producer_programs_per_peer

                    @pl.when(job < producer_jobs)
                    def _produce_job() -> None:
                        entry = job // hidden_tiles
                        hidden_tile = job % hidden_tiles
                        hidden_start = hidden_tile * config.block_n
                        valid_rows = recv_valid_ref[static_source_ordinal, entry]
                        expert = jnp.maximum(recv_expert_ref[static_source_ordinal, entry], 0)
                        expert_row_start = recv_row_ref[static_source_ordinal, entry]

                        def _acc_scope(acc_ref) -> Array:
                            def _smem_scope(h_smem, w_smem, weight_barrier) -> None:
                                @pl.loop(0, intermediate_tiles)
                                def _intermediate_loop(intermediate_tile) -> None:
                                    intermediate_start = intermediate_tile * config.block_k
                                    mgpu.copy_gmem_to_smem(
                                        w_ref.at[
                                            expert,
                                            pl.ds(intermediate_start, config.block_k),
                                            pl.ds(hidden_start, config.block_n),
                                        ],
                                        w_smem,
                                        weight_barrier,
                                    )
                                    gate = z_ref[
                                        expert,
                                        pl.ds(expert_row_start, config.compute_m),
                                        pl.ds(intermediate_start, config.block_k),
                                    ].astype(jnp.float32)
                                    up = z_ref[
                                        expert,
                                        pl.ds(expert_row_start, config.compute_m),
                                        pl.ds(intermediate_dim + intermediate_start, config.block_k),
                                    ].astype(jnp.float32)
                                    row = mgpu.layout_cast(
                                        lax.broadcasted_iota(
                                            jnp.int32,
                                            (config.compute_m, config.block_k),
                                            0,
                                        ),
                                        mgpu.Layout.WGMMA,
                                    )
                                    row_valid = (row < valid_rows).astype(jnp.float32)
                                    h_smem[:, :] = (jax.nn.silu(gate) * up * row_valid).astype(dtype)
                                    mgpu.barrier_wait(weight_barrier)
                                    mgpu.commit_smem()
                                    mgpu.wgmma(acc_ref, h_smem, w_smem)
                                    mgpu.wgmma_wait(0)

                            pl.run_scoped(
                                _smem_scope,
                                h_smem=_wgmma_smem((config.compute_m, config.block_k), dtype),
                                w_smem=_wgmma_smem((config.block_k, config.block_n), dtype),
                                weight_barrier=mgpu.Barrier(num_arrivals=1),
                            )
                            return acc_ref[...].astype(jnp.bfloat16)

                        output = pl.run_scoped(
                            _acc_scope,
                            acc_ref=mgpu.ACC((config.compute_m, config.block_n)),
                        )

                        def _store_scope(output_smem) -> None:
                            output_smem[:, :] = output
                            mgpu.commit_smem()
                            mgpu.copy_smem_to_gmem(
                                output_smem,
                                destination_ref.at[
                                    destination_ordinal,
                                    entry,
                                    pl.ds(0, config.compute_m),
                                    pl.ds(hidden_start, config.block_n),
                                ],
                            )
                            mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                        pl.run_scoped(
                            _store_scope,
                            output_smem=mgpu.SMEM((config.compute_m, config.block_n), dtype=jnp.bfloat16),
                        )
                        if static_source_ordinal == 0:
                            pl.semaphore_signal(ready_sem.at[destination_ordinal, entry, hidden_tile])
                        else:
                            _signal_ready(source, destination_ordinal, entry, hidden_tile)

                # Every queue entry is published, including masked entries. This
                # lets invalid routes safely use queue index zero without a
                # conditional semaphore wait in the combine hot path.

            branches = tuple((lambda ordinal: lambda _: _produce_for_source(ordinal))(i) for i in range(ep_size))
            lax.switch(peer_ordinal, branches, None)

        @pl.when((peer_ordinal == 0) & (worker >= config.producer_programs_per_peer))
        def _combine() -> None:
            consumer = worker - config.producer_programs_per_peer

            @pl.loop(0, combine_jobs_per_program)
            def _combine_job_loop(job_iteration) -> None:
                job = consumer + job_iteration * config.combine_programs

                @pl.when(job < combine_jobs)
                def _combine_job() -> None:
                    token_block = job // hidden_tiles
                    hidden_tile = job % hidden_tiles
                    hidden_start = hidden_tile * config.block_n
                    token_start = token_block * config.combine_token_block

                    @pl.loop(0, config.combine_token_block)
                    def _token_loop(token_offset) -> None:
                        token = token_start + token_offset

                        @pl.when(token < tokens_per_source)
                        def _combine_token() -> None:
                            acc = jnp.zeros((config.block_n,), dtype=jnp.float32)
                            for route_slot in range(topk):
                                destination_ordinal = queue_dst_ref[token, route_slot]
                                entry = queue_entry_ref[token, route_slot]
                                queue_row = queue_row_ref[token, route_slot]
                                pl.semaphore_wait(
                                    ready_sem.at[destination_ordinal, entry, hidden_tile],
                                    value=1,
                                    decrement=False,
                                )
                                route_value = return_y_ref[
                                    destination_ordinal,
                                    entry,
                                    queue_row,
                                    pl.ds(hidden_start, config.block_n),
                                ].astype(jnp.float32)
                                weight = route_weight_ref[token, route_slot].astype(jnp.float32)
                                valid = route_valid_ref[token, route_slot] != 0
                                acc += jnp.where(valid, route_value * weight, jnp.zeros((), dtype=jnp.float32))

                            y_ref[token, pl.ds(hidden_start, config.block_n)] = acc.astype(jnp.bfloat16)

    return_shape = (ep_size, entries_per_dst, config.compute_m, hidden_dim)
    y_shape = (tokens_per_source, hidden_dim)
    return mgpu.kernel(
        body,
        out_shape=(
            jax.ShapeDtypeStruct(return_shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(y_shape, jnp.bfloat16),
        ),
        grid=(ep_size, worker_programs),
        grid_names=("peer_phase", "worker_program"),
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
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


def _validate_reference_request(
    z_expert: Array,
    w_down: Array,
    metadata: SourcePushSemanticFusedW2ReturnMetadata,
) -> None:
    if z_expert.ndim != 4 or w_down.ndim != 4:
        raise ValueError(f"z_expert and w_down must be rank four, got {z_expert.shape} and {w_down.shape}")
    if z_expert.shape[:2] != w_down.shape[:2]:
        raise ValueError(f"destination/expert dimensions must match, got {z_expert.shape} and {w_down.shape}")
    if z_expert.shape[2] != metadata.rows_per_expert_capacity:
        raise ValueError(f"z row capacity {z_expert.shape[2]} must match metadata {metadata.rows_per_expert_capacity}")
    if z_expert.shape[-1] != 2 * w_down.shape[-2]:
        raise ValueError(f"z output dim must equal 2 * W2 intermediate dim, got {z_expert.shape} and {w_down.shape}")


def _validate_request(
    z_expert: Array,
    w_down: Array,
    plan: SourcePushSemanticPlan,
    entries_per_dst: int,
    config: SourcePushSemanticFusedW2ReturnConfig,
) -> None:
    config.validate()
    if entries_per_dst <= 0:
        raise ValueError(f"entries_per_dst must be positive, got {entries_per_dst}")
    if z_expert.ndim != 4 or w_down.ndim != 4:
        raise ValueError(f"z_expert and w_down must be rank four, got {z_expert.shape} and {w_down.shape}")
    destination_count, experts_per_rank = plan.xcounts.shape[1:]
    if z_expert.shape[:2] != (destination_count, experts_per_rank):
        raise ValueError(
            f"z destination/expert shape {z_expert.shape[:2]} must be {(destination_count, experts_per_rank)}"
        )
    if w_down.shape[:2] != (destination_count, experts_per_rank):
        raise ValueError(
            f"w_down destination/expert shape {w_down.shape[:2]} must be {(destination_count, experts_per_rank)}"
        )
    if z_expert.shape[-1] != 2 * w_down.shape[-2]:
        raise ValueError(f"z output dim must equal 2 * W2 intermediate dim, got {z_expert.shape} and {w_down.shape}")
    if z_expert.dtype != jnp.bfloat16 or w_down.dtype != jnp.bfloat16:
        raise ValueError(f"fused W2 return requires bfloat16 inputs, got {z_expert.dtype} and {w_down.dtype}")
    if z_expert.shape[2] % config.compute_m:
        raise ValueError(f"expert row capacity {z_expert.shape[2]} must be divisible by compute_m={config.compute_m}")
    if w_down.shape[-2] % config.block_k:
        raise ValueError(f"intermediate dim {w_down.shape[-2]} must be divisible by block_k={config.block_k}")
    if w_down.shape[-1] % config.block_n:
        raise ValueError(f"hidden dim {w_down.shape[-1]} must be divisible by block_n={config.block_n}")
