# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent expert W2 fused with direct source return and source combine."""

from __future__ import annotations

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
# Leave residency headroom below the 132-SM H100 SXM target. Every program
# participates in the completion barrier, so the grid must co-reside.
PERSISTENT_W2_PROGRAMS = 128
PERSISTENT_COMBINE_PROGRAMS = 32
N_GROUPS_PER_JOB = 2
W2_PIPELINE_STAGES = 2
W2_PIPELINE_MAX_PENDING_WGMMA_GROUPS = W2_PIPELINE_STAGES * N_GROUPS_PER_JOB


@dataclass(frozen=True, slots=True)
class SourcePushSemanticFusedW2ReturnConfig:
    """Fixed Hopper tile and persistent-CTA dimensions."""

    compute_m: int = 64
    block_k: int = 64
    block_n: int = 128
    combine_token_block: int = 64

    def validate(self) -> None:
        if self.compute_m != 64:
            raise ValueError(f"the initial Hopper lowering requires compute_m=64, got {self.compute_m}")
        if self.block_k != 64:
            raise ValueError(f"the initial Hopper lowering requires block_k=64, got {self.block_k}")
        if self.block_n != 128:
            raise ValueError(f"the initial Hopper lowering requires block_n=128, got {self.block_n}")
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
class SourcePushSemanticFusedW2ReturnSchedule:
    """Resident CTA work distribution and completion-barrier counts for one rank."""

    hidden_tiles: int
    token_blocks: int
    w2_jobs: int
    min_w2_jobs_per_program: int
    max_w2_jobs_per_program: int
    producer_program_start: int
    producer_programs: int
    combine_program_start: int
    combine_programs: int
    active_combine_programs: int
    readiness_signals: int
    readiness_waits: int

    @property
    def total_programs(self) -> int:
        return max(self.producer_programs, self.combine_program_start + self.combine_programs)


@dataclass(frozen=True, slots=True)
class _W2PipelineStructure:
    intermediate_tiles: int
    tile_pairs: int
    stages: int
    smem_bytes: int
    max_pending_wgmma_groups: int


def _w2_pipeline_structure(
    *,
    intermediate_dim: int,
    dtype: jnp.dtype,
    config: SourcePushSemanticFusedW2ReturnConfig,
) -> _W2PipelineStructure:
    intermediate_tiles = intermediate_dim // config.block_k
    elements_per_stage = 3 * config.compute_m * config.block_k + N_GROUPS_PER_JOB * config.block_k * config.block_n
    return _W2PipelineStructure(
        intermediate_tiles=intermediate_tiles,
        tile_pairs=(intermediate_tiles + W2_PIPELINE_STAGES - 1) // W2_PIPELINE_STAGES,
        stages=W2_PIPELINE_STAGES,
        smem_bytes=W2_PIPELINE_STAGES * elements_per_stage * jnp.dtype(dtype).itemsize,
        max_pending_wgmma_groups=W2_PIPELINE_MAX_PENDING_WGMMA_GROUPS,
    )


def source_push_semantic_fused_w2_return_schedule(
    *,
    ep_size: int,
    hidden_dim: int,
    tokens_per_source: int,
    entries_per_dst: int,
    config: SourcePushSemanticFusedW2ReturnConfig = SourcePushSemanticFusedW2ReturnConfig(),
) -> SourcePushSemanticFusedW2ReturnSchedule:
    """Return the fixed persistent schedule and its readiness traffic."""

    config.validate()
    if ep_size <= 0 or hidden_dim <= 0 or tokens_per_source <= 0 or entries_per_dst <= 0:
        raise ValueError(
            "ep_size, hidden_dim, tokens_per_source, and entries_per_dst must be positive, "
            f"got {ep_size}, {hidden_dim}, {tokens_per_source}, and {entries_per_dst}"
        )
    if hidden_dim % config.block_n:
        raise ValueError(f"hidden dim {hidden_dim} must be divisible by block_n={config.block_n}")

    hidden_tiles = hidden_dim // config.block_n
    if hidden_tiles > PERSISTENT_W2_PROGRAMS:
        raise ValueError(f"hidden tiles {hidden_tiles} exceed the resident grid size {PERSISTENT_W2_PROGRAMS}")
    token_blocks = (tokens_per_source + config.combine_token_block - 1) // config.combine_token_block
    combine_programs = max(PERSISTENT_COMBINE_PROGRAMS, hidden_tiles)
    active_combine_programs = sum(
        min(token_blocks, (combine_programs + hidden_tiles - 1 - hidden_tile) // hidden_tiles)
        for hidden_tile in range(hidden_tiles)
    )
    hidden_tile_jobs = (hidden_tiles + N_GROUPS_PER_JOB - 1) // N_GROUPS_PER_JOB
    w2_jobs = ep_size * entries_per_dst * hidden_tile_jobs
    return SourcePushSemanticFusedW2ReturnSchedule(
        hidden_tiles=hidden_tiles,
        token_blocks=token_blocks,
        w2_jobs=w2_jobs,
        min_w2_jobs_per_program=w2_jobs // PERSISTENT_W2_PROGRAMS,
        max_w2_jobs_per_program=(w2_jobs + PERSISTENT_W2_PROGRAMS - 1) // PERSISTENT_W2_PROGRAMS,
        producer_program_start=0,
        producer_programs=PERSISTENT_W2_PROGRAMS,
        combine_program_start=0,
        combine_programs=combine_programs,
        active_combine_programs=active_combine_programs,
        readiness_signals=PERSISTENT_W2_PROGRAMS * ep_size,
        readiness_waits=active_combine_programs * ep_size,
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
    schedule = source_push_semantic_fused_w2_return_schedule(
        ep_size=ep_size,
        hidden_dim=hidden_dim,
        tokens_per_source=tokens_per_source,
        entries_per_dst=entries_per_dst,
        config=config,
    )
    hidden_tiles = schedule.hidden_tiles
    hidden_tile_jobs = (hidden_tiles + N_GROUPS_PER_JOB - 1) // N_GROUPS_PER_JOB
    pipeline = _w2_pipeline_structure(intermediate_dim=intermediate_dim, dtype=dtype, config=config)
    token_blocks = schedule.token_blocks
    combine_programs = schedule.combine_programs
    worker_programs = schedule.total_programs

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
        completion_sem = pl.get_global(mgpu.SemaphoreType.REGULAR((ep_size,)))
        rank = lax.axis_index(SOURCE_PUSH_MESH_AXIS)
        worker = pl.program_id(0)

        def _signal_completion(peer) -> None:
            pl.semaphore_signal(
                completion_sem.at[rank],
                device_id=peer,
                device_id_type=pl.DeviceIdType.LOGICAL,
            )

        def _produce_for_source(static_source_ordinal: int, job_args) -> None:
            entry, hidden_tile_job = job_args
            hidden_tile = hidden_tile_job * N_GROUPS_PER_JOB
            hidden_start = hidden_tile * config.block_n
            has_second_hidden_tile = hidden_tile + 1 < hidden_tiles
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

            valid_rows = recv_valid_ref[static_source_ordinal, entry]

            @pl.when(valid_rows > 0)
            def _produce_entry() -> None:
                expert = recv_expert_ref[static_source_ordinal, entry]
                expert_row_start = recv_row_ref[static_source_ordinal, entry]

                def _acc_scope(acc_0_ref, acc_1_ref) -> None:
                    def _smem_scope(
                        gate_0_smem,
                        up_0_smem,
                        h_0_smem,
                        w_00_smem,
                        w_01_smem,
                        gate_1_smem,
                        up_1_smem,
                        h_1_smem,
                        w_10_smem,
                        w_11_smem,
                        activation_0_barrier,
                        w_01_barrier,
                        activation_1_barrier,
                        w_11_barrier,
                    ) -> None:
                        row = mgpu.layout_cast(
                            lax.broadcasted_iota(jnp.int32, (config.compute_m, config.block_k), 0),
                            mgpu.Layout.WGMMA,
                        )
                        row_valid = (row < valid_rows).astype(jnp.float32)

                        def _load_stage(
                            intermediate_tile,
                            gate_smem,
                            up_smem,
                            w_0_smem,
                            w_1_smem,
                            activation_barrier,
                            w_1_barrier,
                        ) -> None:
                            intermediate_start = intermediate_tile * config.block_k
                            mgpu.copy_gmem_to_smem(
                                w_ref.at[
                                    expert,
                                    pl.ds(intermediate_start, config.block_k),
                                    pl.ds(hidden_start, config.block_n),
                                ],
                                w_0_smem,
                                activation_barrier,
                            )
                            mgpu.copy_gmem_to_smem(
                                z_ref.at[
                                    expert,
                                    pl.ds(expert_row_start, config.compute_m),
                                    pl.ds(intermediate_start, config.block_k),
                                ],
                                gate_smem,
                                activation_barrier,
                            )
                            mgpu.copy_gmem_to_smem(
                                z_ref.at[
                                    expert,
                                    pl.ds(expert_row_start, config.compute_m),
                                    pl.ds(intermediate_dim + intermediate_start, config.block_k),
                                ],
                                up_smem,
                                activation_barrier,
                            )

                            @pl.when(has_second_hidden_tile)
                            def _load_second_weight_tile() -> None:
                                mgpu.copy_gmem_to_smem(
                                    w_ref.at[
                                        expert,
                                        pl.ds(intermediate_start, config.block_k),
                                        pl.ds(hidden_start + config.block_n, config.block_n),
                                    ],
                                    w_1_smem,
                                    w_1_barrier,
                                )

                        def _wait_stage(activation_barrier, w_1_barrier) -> None:
                            mgpu.barrier_wait(activation_barrier)

                            @pl.when(has_second_hidden_tile)
                            def _wait_for_second_weight_tile() -> None:
                                mgpu.barrier_wait(w_1_barrier)

                        def _issue_stage(gate_smem, up_smem, h_smem, w_0_smem, w_1_smem) -> None:
                            gate = gate_smem[...].astype(jnp.float32)
                            up = up_smem[...].astype(jnp.float32)
                            h_smem[:, :] = (jax.nn.silu(gate) * up * row_valid).astype(dtype)
                            mgpu.commit_smem()
                            mgpu.wgmma(acc_0_ref, h_smem, w_0_smem)

                            @pl.when(has_second_hidden_tile)
                            def _accumulate_second_hidden_tile() -> None:
                                mgpu.wgmma(acc_1_ref, h_smem, w_1_smem)

                        def _wait_for_reusable_stage() -> None:
                            @pl.when(has_second_hidden_tile)
                            def _wait_for_two_groups() -> None:
                                mgpu.wgmma_wait(2)

                            @pl.when(jnp.logical_not(has_second_hidden_tile))
                            def _wait_for_one_group() -> None:
                                mgpu.wgmma_wait(1)

                        _load_stage(
                            0,
                            gate_0_smem,
                            up_0_smem,
                            w_00_smem,
                            w_01_smem,
                            activation_0_barrier,
                            w_01_barrier,
                        )

                        @pl.loop(0, pipeline.tile_pairs)
                        def _intermediate_pair_loop(pair) -> None:
                            # A reuse wait retires the older stage but leaves the current tile's groups in flight.
                            even_tile = pair * W2_PIPELINE_STAGES
                            _wait_stage(activation_0_barrier, w_01_barrier)
                            _issue_stage(gate_0_smem, up_0_smem, h_0_smem, w_00_smem, w_01_smem)

                            odd_tile = even_tile + 1

                            @pl.when(odd_tile < pipeline.intermediate_tiles)
                            def _issue_odd_tile() -> None:
                                _wait_for_reusable_stage()
                                _load_stage(
                                    odd_tile,
                                    gate_1_smem,
                                    up_1_smem,
                                    w_10_smem,
                                    w_11_smem,
                                    activation_1_barrier,
                                    w_11_barrier,
                                )
                                _wait_stage(activation_1_barrier, w_11_barrier)
                                _issue_stage(gate_1_smem, up_1_smem, h_1_smem, w_10_smem, w_11_smem)

                                next_even_tile = even_tile + W2_PIPELINE_STAGES

                                @pl.when(next_even_tile < pipeline.intermediate_tiles)
                                def _prefetch_next_even_tile() -> None:
                                    _wait_for_reusable_stage()
                                    _load_stage(
                                        next_even_tile,
                                        gate_0_smem,
                                        up_0_smem,
                                        w_00_smem,
                                        w_01_smem,
                                        activation_0_barrier,
                                        w_01_barrier,
                                    )

                        mgpu.wgmma_wait(0)

                    pl.run_scoped(
                        _smem_scope,
                        gate_0_smem=_wgmma_smem((config.compute_m, config.block_k), dtype),
                        up_0_smem=_wgmma_smem((config.compute_m, config.block_k), dtype),
                        h_0_smem=_wgmma_smem((config.compute_m, config.block_k), dtype),
                        w_00_smem=_wgmma_smem((config.block_k, config.block_n), dtype),
                        w_01_smem=_wgmma_smem((config.block_k, config.block_n), dtype),
                        gate_1_smem=_wgmma_smem((config.compute_m, config.block_k), dtype),
                        up_1_smem=_wgmma_smem((config.compute_m, config.block_k), dtype),
                        h_1_smem=_wgmma_smem((config.compute_m, config.block_k), dtype),
                        w_10_smem=_wgmma_smem((config.block_k, config.block_n), dtype),
                        w_11_smem=_wgmma_smem((config.block_k, config.block_n), dtype),
                        activation_0_barrier=mgpu.Barrier(num_arrivals=3),
                        w_01_barrier=mgpu.Barrier(num_arrivals=1),
                        activation_1_barrier=mgpu.Barrier(num_arrivals=3),
                        w_11_barrier=mgpu.Barrier(num_arrivals=1),
                    )
                    acc_0 = mgpu.wgmma_accumulator_load(acc_0_ref, wait_n=None)
                    destination_ref[
                        destination_ordinal,
                        entry,
                        pl.ds(0, config.compute_m),
                        pl.ds(hidden_start, config.block_n),
                    ] = acc_0.astype(jnp.bfloat16)

                    @pl.when(has_second_hidden_tile)
                    def _store_second_hidden_tile() -> None:
                        acc_1 = mgpu.wgmma_accumulator_load(acc_1_ref, wait_n=None)
                        destination_ref[
                            destination_ordinal,
                            entry,
                            pl.ds(0, config.compute_m),
                            pl.ds(hidden_start + config.block_n, config.block_n),
                        ] = acc_1.astype(jnp.bfloat16)

                pl.run_scoped(
                    _acc_scope,
                    acc_0_ref=mgpu.ACC((config.compute_m, config.block_n)),
                    acc_1_ref=mgpu.ACC((config.compute_m, config.block_n)),
                )

        branches = tuple((lambda ordinal: lambda args: _produce_for_source(ordinal, args))(i) for i in range(ep_size))

        @pl.loop(0, schedule.max_w2_jobs_per_program)
        def _w2_job_loop(iteration) -> None:
            job = worker + iteration * schedule.producer_programs

            @pl.when(job < schedule.w2_jobs)
            def _w2_job() -> None:
                source_ordinal = job // (entries_per_dst * hidden_tile_jobs)
                source_job = job % (entries_per_dst * hidden_tile_jobs)
                entry = source_job // hidden_tile_jobs
                hidden_tile_job = source_job % hidden_tile_jobs
                lax.switch(source_ordinal, branches, (entry, hidden_tile_job))

        for peer_ordinal in range(ep_size):
            if peer_ordinal == 0:
                pl.semaphore_signal(completion_sem.at[rank])
            else:
                _signal_completion((rank + peer_ordinal) % ep_size)

        @pl.when(worker < schedule.combine_programs)
        def _combine() -> None:
            consumer = worker
            hidden_tile = consumer % hidden_tiles
            consumer_lane = consumer // hidden_tiles
            consumers_for_tile = (combine_programs + hidden_tiles - 1 - hidden_tile) // hidden_tiles
            hidden_start = hidden_tile * config.block_n

            @pl.when(consumer_lane < token_blocks)
            def _active_consumer() -> None:
                for destination_ordinal in range(ep_size):
                    pl.semaphore_wait(
                        completion_sem.at[destination_ordinal],
                        value=schedule.producer_programs,
                        decrement=False,
                    )

                @pl.loop(0, token_blocks)
                def _token_block_loop(iteration) -> None:
                    token_block = consumer_lane + iteration * consumers_for_tile

                    @pl.when(token_block < token_blocks)
                    def _combine_token_block() -> None:
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
                                    route_value = return_y_ref[
                                        destination_ordinal,
                                        entry,
                                        queue_row,
                                        pl.ds(hidden_start, config.block_n),
                                    ].astype(jnp.float32)
                                    weight = route_weight_ref[token, route_slot].astype(jnp.float32)
                                    valid = route_valid_ref[token, route_slot] != 0
                                    acc += jnp.where(
                                        valid,
                                        route_value * weight,
                                        jnp.zeros((), dtype=jnp.float32),
                                    )

                                y_ref[token, pl.ds(hidden_start, config.block_n)] = acc.astype(jnp.bfloat16)

    return_shape = (ep_size, entries_per_dst, config.compute_m, hidden_dim)
    y_shape = (tokens_per_source, hidden_dim)
    return mgpu.kernel(
        body,
        out_shape=(
            jax.ShapeDtypeStruct(return_shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(y_shape, jnp.bfloat16),
        ),
        grid=(worker_programs,),
        grid_names=("worker_program",),
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
