# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent source dcombine/dy routing fused with expert W2 backward."""

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
from levanter.grug._moe.source_push_semantic_inbox_pallas import source_push_semantic_inbox_metadata_jax


_WGMMA_SWIZZLE_BYTES = 128
_WGMMA_TILE_M = 8


@dataclass(frozen=True, slots=True)
class SourcePushSemanticFusedW2BackwardConfig:
    """Fixed physical shape for the first Hopper persistent lowering."""

    compute_m: int = 64
    send_m: int = 256
    intermediate_block: int = 128
    hidden_block: int = 128
    send_hidden_block: int = 256
    inbox_slots: int = 12
    producer_programs_per_peer: int = 16
    consumer_programs: int = 32

    def validate(self) -> None:
        expected = {
            "compute_m": 64,
            "send_m": 256,
            "intermediate_block": 128,
            "hidden_block": 128,
            "send_hidden_block": 256,
            "inbox_slots": 12,
            "producer_programs_per_peer": 16,
            "consumer_programs": 32,
        }
        for name, value in expected.items():
            actual = getattr(self, name)
            if actual != value:
                raise ValueError(f"the initial Hopper lowering requires {name}={value}, got {actual}")

    @property
    def compute_blocks_per_send(self) -> int:
        return self.send_m // self.compute_m

    def consumer_programs_per_source(self, source_count: int) -> int:
        """Partition the fixed consumer pool across source-owned work queues."""

        if source_count <= 0 or self.consumer_programs % source_count:
            raise ValueError(
                f"consumer_programs={self.consumer_programs} must be divisible by source_count={source_count}"
            )
        return self.consumer_programs // source_count


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SourcePushSemanticFusedW2BackwardMetadata:
    """Source queue rows and destination expert-major receive descriptors."""

    token_ids: Int[Array, "S DstOrd C B M"]
    route_slots: Int[Array, "S DstOrd C B M"]
    route_weights: Float[Array, "S DstOrd C B M"]
    row_valid: Bool[Array, "S DstOrd C B M"]
    send_expert: Int[Array, "S DstOrd C B"]
    send_row_start: Int[Array, "S DstOrd C B"]
    send_valid_rows: Int[Array, "S DstOrd C B"]
    recv_expert: Int[Array, "Dst SrcOrd C B"]
    recv_row_start: Int[Array, "Dst SrcOrd C B"]
    recv_valid_rows: Int[Array, "Dst SrcOrd C B"]
    valid: Bool[Array, "Dst E C"]
    queue_overflow_routes: Int[Array, ""]
    layout_overflow_rows: Int[Array, ""]
    live_send_blocks: Int[Array, ""]
    masked_send_blocks: Int[Array, ""]
    rows_per_expert_capacity: int = field(metadata={"static": True})
    send_chunks_per_dst: int = field(metadata={"static": True})
    topk: int = field(metadata={"static": True})


class SourcePushSemanticFusedW2BackwardResult(NamedTuple):
    """W2 gradients without a replicated or expert-major route-dy tensor."""

    d_h: Float[Array, "Dst E C I"]
    d_w2: Float[Array, "Dst E I H"]
    d_route_weight: Float[Array, "S T K"]
    valid: Bool[Array, "Dst E C"]
    queue_overflow_routes: Int[Array, ""]
    layout_overflow_rows: Int[Array, ""]
    live_send_blocks: Int[Array, ""]
    masked_send_blocks: Int[Array, ""]


@dataclass(frozen=True, slots=True)
class SourcePushSemanticFusedW2BackwardGenerationAccounting:
    """Cumulative semaphore targets for one source send chunk."""

    slot: int
    generation: int
    empty_generation: int
    send_done_generation: int
    full_generation: int
    consumer_done_generation: int
    released_generation: int


def source_push_semantic_fused_w2_backward_generation_accounting(
    chunk: int,
    *,
    hidden_dim: int,
    intermediate_dim: int,
    experts_per_rank: int,
    ep_size: int,
    send_chunks_per_dst: int,
    config: SourcePushSemanticFusedW2BackwardConfig = SourcePushSemanticFusedW2BackwardConfig(),
) -> SourcePushSemanticFusedW2BackwardGenerationAccounting:
    """Return the slot semaphore generations for one source-owned chunk."""

    config.validate()
    _validate_kernel_dimensions(hidden_dim, intermediate_dim, config)
    if chunk < 0 or chunk >= send_chunks_per_dst:
        raise ValueError(f"invalid chunk coordinate {chunk=} for {send_chunks_per_dst=}")
    generation = chunk // config.inbox_slots + 1
    producer_tiles = config.compute_blocks_per_send * (hidden_dim // config.send_hidden_block)
    return SourcePushSemanticFusedW2BackwardGenerationAccounting(
        slot=chunk % config.inbox_slots,
        generation=generation,
        empty_generation=generation,
        send_done_generation=generation * producer_tiles,
        full_generation=generation,
        consumer_done_generation=generation * config.consumer_programs_per_source(ep_size),
        released_generation=generation + 1,
    )


def source_push_semantic_fused_w2_backward_metadata_jax(
    dy: Float[Array, "S T H"],
    plan: SourcePushSemanticPlan,
    *,
    send_chunks_per_dst: int,
    rows_per_expert_capacity: int,
    config: SourcePushSemanticFusedW2BackwardConfig = SourcePushSemanticFusedW2BackwardConfig(),
) -> SourcePushSemanticFusedW2BackwardMetadata:
    """Lower semantic route rows into B256 sends made of four B64 compute blocks."""

    config.validate()
    if send_chunks_per_dst <= 0:
        raise ValueError(f"send_chunks_per_dst must be positive, got {send_chunks_per_dst}")
    if rows_per_expert_capacity % config.compute_m:
        raise ValueError(
            f"rows_per_expert_capacity must be divisible by compute_m={config.compute_m}, "
            f"got {rows_per_expert_capacity}"
        )
    entries_per_dst = send_chunks_per_dst * config.compute_blocks_per_send
    queue = source_push_semantic_queue_metadata_jax(
        plan,
        return_row_block=config.compute_m,
        entries_per_dst=entries_per_dst,
    )
    inbox = source_push_semantic_inbox_metadata_jax(
        dy,
        plan,
        queue,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )
    source_count, destination_count, _ = queue.local_expert.shape
    block_shape = (source_count, destination_count, send_chunks_per_dst, config.compute_blocks_per_send)
    send_meta = inbox.send_meta.reshape(*block_shape, -1)
    token_ids = inbox.token_ids.reshape(*block_shape, config.compute_m)
    row_valid = inbox.valid_mask.reshape(*block_shape, config.compute_m)

    source = jnp.arange(source_count, dtype=jnp.int32)[:, None, None, None]
    dst_ordinal = jnp.arange(destination_count, dtype=jnp.int32)[None, :, None, None]
    destination = (source + dst_ordinal) % destination_count
    safe_expert = jnp.maximum(send_meta[..., 1], 0)
    expert_base = inbox.layout.expert_base.at[destination, safe_expert].get()
    send_row_start = jnp.where(send_meta[..., 3] > 0, send_meta[..., 2] - expert_base, 0)

    local_row = jnp.arange(config.compute_m, dtype=jnp.int32)[None, None, None, None, :]
    pair_row = (
        plan.pair_expert_base.at[source, destination, safe_expert].get()[..., None]
        + queue.local_row_start.reshape(*block_shape)[..., None]
        + local_row
    )
    safe_pair_row = jnp.minimum(pair_row, plan.assignment_ids.shape[-1] - 1)
    route_slots = plan.route_slots.at[source[..., None], destination[..., None], safe_pair_row].get()
    route_weights = plan.route_weights.at[source[..., None], destination[..., None], safe_pair_row].get()
    route_slots = jnp.where(row_valid, route_slots, 0).astype(jnp.int32)
    route_weights = jnp.where(row_valid, route_weights, jnp.zeros((), dtype=route_weights.dtype))

    destination_index = jnp.arange(destination_count, dtype=jnp.int32)[:, None]
    source_ordinal = jnp.arange(source_count, dtype=jnp.int32)[None, :]
    recv_source = (destination_index + source_ordinal) % source_count
    recv_dst_ordinal = (-source_ordinal) % destination_count

    def _to_recv(value: Array) -> Array:
        return value.at[recv_source, recv_dst_ordinal].get()

    send_expert = send_meta[..., 1].astype(jnp.int32)
    send_valid_rows = send_meta[..., 3].astype(jnp.int32)
    live_send_blocks = jnp.sum(send_valid_rows > 0, dtype=jnp.int32)
    return SourcePushSemanticFusedW2BackwardMetadata(
        token_ids=token_ids,
        route_slots=route_slots,
        route_weights=route_weights,
        row_valid=row_valid,
        send_expert=send_expert,
        send_row_start=send_row_start.astype(jnp.int32),
        send_valid_rows=send_valid_rows,
        recv_expert=_to_recv(send_expert),
        recv_row_start=_to_recv(send_row_start).astype(jnp.int32),
        recv_valid_rows=_to_recv(send_valid_rows),
        valid=inbox.layout.valid,
        queue_overflow_routes=queue.overflow_routes,
        layout_overflow_rows=inbox.layout.overflow_rows,
        live_send_blocks=live_send_blocks,
        masked_send_blocks=jnp.asarray(send_valid_rows.size, dtype=jnp.int32) - live_send_blocks,
        rows_per_expert_capacity=rows_per_expert_capacity,
        send_chunks_per_dst=send_chunks_per_dst,
        topk=plan.topk,
    )


def source_push_semantic_fused_w2_backward_reference_jax(
    dy: Float[Array, "S T H"],
    return_y: Float[Array, "S DstOrd Q M H"],
    h_expert: Float[Array, "Dst E C I"],
    w_down: Float[Array, "Dst E I H"],
    metadata: SourcePushSemanticFusedW2BackwardMetadata,
) -> tuple[Float[Array, "Dst E C I"], Float[Array, "Dst E I H"], Float[Array, "S T K"]]:
    """Obvious gather/matmul/scatter reference for the fused stage."""

    source_count, destination_count, _chunks, _blocks, compute_m = metadata.token_ids.shape
    source = jnp.arange(source_count, dtype=jnp.int32)[:, None, None, None, None]
    gathered_sharding = None
    if jax.sharding.get_abstract_mesh().are_all_axes_explicit:
        gathered_sharding = P(SOURCE_PUSH_MESH_AXIS, None, None, None, None, None)
    dy_rows = dy.at[source, metadata.token_ids].get(out_sharding=gathered_sharding).astype(jnp.float32)
    dy_rows = jnp.where(metadata.row_valid[..., None], dy_rows, jnp.zeros((), dtype=jnp.float32))
    dy_route = dy_rows * metadata.route_weights[..., None].astype(jnp.float32)

    dst_ordinal = jnp.arange(destination_count, dtype=jnp.int32)[None, :, None, None]
    destination = jnp.arange(source_count, dtype=jnp.int32)[:, None, None, None] + dst_ordinal
    destination %= destination_count
    safe_expert = jnp.maximum(metadata.send_expert, 0)
    weights = w_down.at[destination, safe_expert].get(out_sharding=gathered_sharding).astype(jnp.float32)
    queue_dh = jnp.einsum("sdcbmh,sdcbih->sdcbmi", dy_route, weights, preferred_element_type=jnp.float32)

    row = jnp.arange(compute_m, dtype=jnp.int32)[None, None, None, None, :]
    row_valid = metadata.row_valid
    scatter_destination = jnp.broadcast_to(destination[..., None], row_valid.shape)
    scatter_expert = jnp.broadcast_to(safe_expert[..., None], row_valid.shape)
    scatter_row = metadata.send_row_start[..., None] + row
    scatter_row = jnp.where(row_valid, scatter_row, metadata.rows_per_expert_capacity)
    dy_route_for_expert = dy_route
    if gathered_sharding is not None:
        replicated_rows = P(None, None, None, None, None)
        replicated_values = P(None, None, None, None, None, None)
        queue_dh = jax.sharding.reshard(queue_dh, replicated_values)
        row_valid = jax.sharding.reshard(row_valid, replicated_rows)
        scatter_destination = jax.sharding.reshard(scatter_destination, replicated_rows)
        scatter_expert = jax.sharding.reshard(scatter_expert, replicated_rows)
        scatter_row = jax.sharding.reshard(scatter_row, replicated_rows)
        dy_route_for_expert = jax.sharding.reshard(dy_route, replicated_values)
    d_h = jnp.zeros(
        (*h_expert.shape[:3], h_expert.shape[-1]),
        dtype=jnp.float32,
    )
    d_h = jnp.pad(d_h, ((0, 0), (0, 0), (0, 1), (0, 0)))
    d_h = d_h.at[scatter_destination, scatter_expert, scatter_row].set(
        jnp.where(row_valid[..., None], queue_dh, jnp.zeros((), dtype=jnp.float32))
    )
    d_h = d_h[..., : metadata.rows_per_expert_capacity, :]

    safe_row = jnp.minimum(metadata.send_row_start[..., None] + row, h_expert.shape[2] - 1)
    if gathered_sharding is not None:
        safe_row = jax.sharding.reshard(safe_row, P(None, None, None, None, None))
    h_rows_sharding = None if gathered_sharding is None else P(None, None, None, None, None, None)
    h_rows = h_expert.at[scatter_destination, scatter_expert, safe_row].get(out_sharding=h_rows_sharding)
    h_rows = h_rows.astype(jnp.float32)
    h_rows = jnp.where(row_valid[..., None], h_rows, jnp.zeros((), dtype=jnp.float32))
    dw2 = jnp.zeros(w_down.shape, dtype=jnp.float32)
    for dst in range(w_down.shape[0]):
        for expert in range(w_down.shape[1]):
            mask = row_valid & (scatter_destination == dst) & (scatter_expert == expert)
            h_masked = jnp.where(mask[..., None], h_rows, jnp.zeros((), dtype=jnp.float32))
            dy_masked = jnp.where(mask[..., None], dy_route_for_expert, jnp.zeros((), dtype=jnp.float32))
            part = jnp.einsum("sdcbmi,sdcbmh->ih", h_masked, dy_masked, preferred_element_type=jnp.float32)
            dw2 = dw2.at[dst, expert].set(part)

    entry = (
        jnp.arange(_chunks, dtype=jnp.int32)[None, None, :, None, None] * metadata.token_ids.shape[3]
        + jnp.arange(_blocks, dtype=jnp.int32)[None, None, None, :, None]
    )
    queue_row = jnp.arange(compute_m, dtype=jnp.int32)[None, None, None, None, :]
    route_y = (
        return_y.at[source, dst_ordinal[..., None], entry, queue_row]
        .get(out_sharding=gathered_sharding)
        .astype(jnp.float32)
    )
    queue_d_route = jnp.sum(dy_rows * route_y, axis=-1)
    queue_d_route = jnp.where(row_valid, queue_d_route, jnp.zeros((), dtype=jnp.float32))

    def _scatter_source_route_gradient(token_ids, route_slots, values):
        d_route_source = jnp.zeros((dy.shape[1], metadata.topk), dtype=jnp.float32)
        return d_route_source.at[token_ids, route_slots].add(values)

    d_route = jax.vmap(_scatter_source_route_gradient)(
        metadata.token_ids,
        metadata.route_slots,
        queue_d_route,
    )
    return (
        jnp.where(metadata.valid[..., None], d_h, jnp.zeros((), dtype=d_h.dtype)),
        dw2,
        d_route,
    )


def source_push_semantic_fused_w2_backward(
    dy: Float[Array, "S T H"],
    return_y: Float[Array, "S DstOrd Q M H"],
    h_expert: Float[Array, "Dst E C I"],
    w_down: Float[Array, "Dst E I H"],
    plan: SourcePushSemanticPlan,
    *,
    send_chunks_per_dst: int,
    rows_per_expert_capacity: int,
    config: SourcePushSemanticFusedW2BackwardConfig = SourcePushSemanticFusedW2BackwardConfig(),
    mesh: Mesh | AbstractMesh | None = None,
    interpret: bool = False,
) -> SourcePushSemanticFusedW2BackwardResult:
    """Fuse source-owned dcombine/dy routing with destination W2 backward."""

    _validate_request(
        dy,
        return_y,
        h_expert,
        w_down,
        plan,
        send_chunks_per_dst,
        rows_per_expert_capacity,
        config,
    )
    metadata = source_push_semantic_fused_w2_backward_metadata_jax(
        dy,
        plan,
        send_chunks_per_dst=send_chunks_per_dst,
        rows_per_expert_capacity=rows_per_expert_capacity,
        config=config,
    )
    if interpret:
        d_h, d_w2, d_route = source_push_semantic_fused_w2_backward_reference_jax(
            dy, return_y, h_expert, w_down, metadata
        )
    else:
        if jax.default_backend() != "gpu":
            raise NotImplementedError("persistent semantic fused W2 backward requires a GPU backend")
        if mesh is None:
            mesh = jax.sharding.get_abstract_mesh()
            if mesh.empty:
                raise ValueError("mesh is required for persistent semantic fused W2 backward")
        d_h, d_w2, d_route = _source_push_semantic_fused_w2_backward_sharded(
            dy,
            return_y,
            h_expert,
            w_down,
            metadata,
            config=config,
            mesh=mesh,
        )
    d_h = jnp.where(metadata.valid[..., None], d_h, jnp.zeros((), dtype=d_h.dtype))
    return SourcePushSemanticFusedW2BackwardResult(
        d_h=d_h,
        d_w2=d_w2,
        d_route_weight=d_route,
        valid=metadata.valid,
        queue_overflow_routes=metadata.queue_overflow_routes,
        layout_overflow_rows=metadata.layout_overflow_rows,
        live_send_blocks=metadata.live_send_blocks,
        masked_send_blocks=metadata.masked_send_blocks,
    )


def _source_push_semantic_fused_w2_backward_sharded(
    dy: Array,
    return_y: Array,
    h_expert: Array,
    w_down: Array,
    metadata: SourcePushSemanticFusedW2BackwardMetadata,
    *,
    config: SourcePushSemanticFusedW2BackwardConfig,
    mesh: Mesh | AbstractMesh,
) -> tuple[Array, Array, Array]:
    if mesh.shape[SOURCE_PUSH_MESH_AXIS] != dy.shape[0]:
        raise ValueError(
            f"mesh {SOURCE_PUSH_MESH_AXIS!r} size must match source count {dy.shape[0]}, "
            f"got {mesh.shape[SOURCE_PUSH_MESH_AXIS]}"
        )
    kernel = _make_source_push_semantic_fused_w2_backward_kernel(
        ep_size=dy.shape[0],
        hidden_dim=dy.shape[-1],
        intermediate_dim=h_expert.shape[-1],
        experts_per_rank=h_expert.shape[1],
        rows_per_expert_capacity=metadata.rows_per_expert_capacity,
        send_chunks_per_dst=metadata.send_chunks_per_dst,
        dtype=dy.dtype,
        config=config,
    )

    def local_fn(
        dy_local,
        route_y_local,
        token_local,
        slot_local,
        weight_local,
        send_valid_local,
        recv_expert_local,
        recv_row_local,
        recv_valid_local,
        h_local,
        w_local,
    ):
        _inbox, d_h_local, d_w2_local, queue_d_route_local = kernel(
            dy_local[0],
            route_y_local[0],
            token_local[0],
            weight_local[0],
            send_valid_local[0],
            recv_expert_local[0],
            recv_row_local[0],
            recv_valid_local[0],
            h_local[0],
            w_local[0],
        )
        compute_m = token_local.shape[-1]
        row = jnp.arange(compute_m, dtype=jnp.int32)
        row_valid = row < send_valid_local[0, ..., None]
        d_route_local = jnp.zeros((dy_local.shape[1], metadata.topk), dtype=jnp.float32)
        d_route_local = d_route_local.at[token_local[0], slot_local[0]].add(
            jnp.where(row_valid, queue_d_route_local, jnp.zeros((), dtype=queue_d_route_local.dtype))
        )
        return d_h_local[None], d_w2_local[None], d_route_local[None]

    source_3d = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None))
    source_4d = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None))
    source_5d = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None, None))
    source_route_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None, None))
    destination_4d = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None))
    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        ),
        out_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        check_vma=False,
    )(
        jax.sharding.reshard(dy, source_3d),
        jax.sharding.reshard(return_y, source_5d),
        jax.sharding.reshard(metadata.token_ids, source_route_sharding),
        jax.sharding.reshard(metadata.route_slots, source_route_sharding),
        jax.sharding.reshard(metadata.route_weights, source_route_sharding),
        jax.sharding.reshard(metadata.send_valid_rows, source_4d),
        jax.sharding.reshard(metadata.recv_expert, destination_4d),
        jax.sharding.reshard(metadata.recv_row_start, destination_4d),
        jax.sharding.reshard(metadata.recv_valid_rows, destination_4d),
        jax.sharding.reshard(h_expert, destination_4d),
        jax.sharding.reshard(w_down, destination_4d),
    )


def _make_source_push_semantic_fused_w2_backward_kernel(
    *,
    ep_size: int,
    hidden_dim: int,
    intermediate_dim: int,
    experts_per_rank: int,
    rows_per_expert_capacity: int,
    send_chunks_per_dst: int,
    dtype: jnp.dtype,
    config: SourcePushSemanticFusedW2BackwardConfig,
):
    """Build one Lane-lowered kernel containing peer producers and WGMMA consumers."""

    config.validate()
    _validate_kernel_dimensions(hidden_dim, intermediate_dim, config)
    blocks = config.compute_blocks_per_send
    send_hidden_tiles = hidden_dim // config.send_hidden_block
    producer_tiles = blocks * send_hidden_tiles
    producer_tiles_per_program = (
        producer_tiles + config.producer_programs_per_peer - 1
    ) // config.producer_programs_per_peer
    intermediate_tiles = intermediate_dim // config.intermediate_block
    hidden_tiles = hidden_dim // config.hidden_block
    dh_jobs = config.inbox_slots * blocks * intermediate_tiles
    dw2_output_tiles = experts_per_rank * intermediate_tiles * hidden_tiles
    dw2_jobs = dw2_output_tiles
    consumer_jobs = dh_jobs + dw2_jobs
    consumer_programs_per_source = config.consumer_programs_per_source(ep_size)
    consumer_jobs_per_program = (consumer_jobs + consumer_programs_per_source - 1) // consumer_programs_per_source
    zero_jobs_per_program = (dw2_output_tiles + consumer_programs_per_source - 1) // consumer_programs_per_source
    worker_programs = config.producer_programs_per_peer + consumer_programs_per_source

    def body(
        dy_ref,
        route_y_ref,
        token_ids_ref,
        route_weights_ref,
        send_valid_ref,
        recv_expert_ref,
        recv_row_ref,
        recv_valid_ref,
        h_ref,
        w_ref,
        inbox_ref,
        d_h_ref,
        d_w2_ref,
        queue_d_route_ref,
    ) -> None:
        empty_sem = pl.get_global(mgpu.SemaphoreType.REGULAR((ep_size, config.inbox_slots)))
        send_done_sem = pl.get_global(mgpu.SemaphoreType.REGULAR((ep_size, config.inbox_slots)))
        full_sem = pl.get_global(mgpu.SemaphoreType.REGULAR((ep_size, config.inbox_slots)))
        consumer_done_sem = pl.get_global(mgpu.SemaphoreType.REGULAR((ep_size, config.inbox_slots)))
        init_done_sem = pl.get_global(mgpu.SemaphoreType.REGULAR(()))
        init_ready_sem = pl.get_global(mgpu.SemaphoreType.REGULAR(()))
        rank = lax.axis_index(SOURCE_PUSH_MESH_AXIS)
        peer_ordinal = pl.program_id(0)
        worker = pl.program_id(1)

        def _signal_remote(sem, peer, local_index, slot) -> None:
            pl.semaphore_signal(sem.at[local_index, slot], device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL)

        @pl.when((peer_ordinal == 0) & (worker >= config.producer_programs_per_peer))
        def _initialize() -> None:
            consumer = worker - config.producer_programs_per_peer

            @pl.loop(0, zero_jobs_per_program)
            def _zero_dw2(job_iteration) -> None:
                job = consumer + job_iteration * consumer_programs_per_source

                @pl.when(job < dw2_output_tiles)
                def _zero_owned_tile() -> None:
                    expert = job // (intermediate_tiles * hidden_tiles)
                    rem = job % (intermediate_tiles * hidden_tiles)
                    i_tile = rem // hidden_tiles
                    h_tile = rem % hidden_tiles
                    d_w2_ref[
                        expert,
                        pl.ds(i_tile * config.intermediate_block, config.intermediate_block),
                        pl.ds(h_tile * config.hidden_block, config.hidden_block),
                    ] = jnp.zeros((config.intermediate_block, config.hidden_block), dtype=jnp.float32)

            pl.semaphore_signal(init_done_sem)

            @pl.when(consumer == 0)
            def _publish_initial_state() -> None:
                pl.semaphore_wait(init_done_sem, value=consumer_programs_per_source, decrement=False)
                for static_source_ordinal in range(ep_size):
                    src = (rank + static_source_ordinal) % ep_size

                    @pl.loop(0, config.inbox_slots)
                    def _slot_loop(slot) -> None:
                        if static_source_ordinal == 0:
                            pl.semaphore_signal(empty_sem.at[rank, slot])
                        else:
                            _signal_remote(empty_sem, src, rank, slot)

                pl.semaphore_signal(init_ready_sem)

        @pl.when(worker < config.producer_programs_per_peer)
        def _producer() -> None:
            producer = worker
            pl.semaphore_wait(init_ready_sem, value=1, decrement=False)

            def _send_peer(static_peer_ordinal: int) -> None:
                dst = (rank + static_peer_ordinal) % ep_size
                remote_inbox = None
                if static_peer_ordinal != 0:
                    remote_inbox = mgpu.remote_ref(inbox_ref, dst, device_id_type=pl.DeviceIdType.LOGICAL)

                @pl.loop(0, send_chunks_per_dst)
                def _chunk_loop(chunk) -> None:
                    slot = chunk % config.inbox_slots
                    generation = chunk // config.inbox_slots + 1

                    @pl.loop(0, producer_tiles_per_program)
                    def _tile_loop(tile_iteration) -> None:
                        tile = producer + tile_iteration * config.producer_programs_per_peer

                        @pl.when(tile < producer_tiles)
                        def _send_tile() -> None:
                            block = tile // send_hidden_tiles
                            hidden_tile = tile % send_hidden_tiles
                            hidden_start = hidden_tile * config.send_hidden_block
                            pl.semaphore_wait(empty_sem.at[dst, slot], value=generation, decrement=False)

                            def _copy_scope(tile_smem) -> None:
                                valid_rows = send_valid_ref[static_peer_ordinal, chunk, block]

                                @pl.loop(0, config.compute_m)
                                def _row_loop(row) -> None:
                                    token = token_ids_ref[static_peer_ordinal, chunk, block, row]
                                    weight = route_weights_ref[static_peer_ordinal, chunk, block, row]
                                    live = row < valid_rows
                                    source_dy = dy_ref[token, pl.ds(hidden_start, config.send_hidden_block)]
                                    tile_smem[row, :] = jnp.where(
                                        live,
                                        source_dy * weight.astype(source_dy.dtype),
                                        jnp.zeros((config.send_hidden_block,), dtype=dtype),
                                    )

                                    @pl.when(hidden_tile == 0)
                                    def _compute_route_weight_gradient() -> None:
                                        acc = jnp.asarray(0.0, dtype=jnp.float32)
                                        entry = chunk * blocks + block
                                        for route_hidden_start in range(0, hidden_dim, config.send_hidden_block):
                                            dy_part = dy_ref[
                                                token, pl.ds(route_hidden_start, config.send_hidden_block)
                                            ].astype(jnp.float32)
                                            route_part = route_y_ref[
                                                static_peer_ordinal,
                                                entry,
                                                row,
                                                pl.ds(route_hidden_start, config.send_hidden_block),
                                            ].astype(jnp.float32)
                                            acc += jnp.sum(dy_part * route_part)
                                        queue_d_route_ref[static_peer_ordinal, chunk, block, row] = jnp.where(
                                            live, acc, jnp.asarray(0.0, dtype=jnp.float32)
                                        )

                                mgpu.commit_smem()
                                destination_ref = inbox_ref if static_peer_ordinal == 0 else remote_inbox
                                mgpu.copy_smem_to_gmem(
                                    tile_smem,
                                    destination_ref.at[
                                        rank,
                                        slot,
                                        pl.ds(block * config.compute_m, config.compute_m),
                                        pl.ds(hidden_start, config.send_hidden_block),
                                    ],
                                )
                                mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                            pl.run_scoped(
                                _copy_scope,
                                tile_smem=mgpu.SMEM((config.compute_m, config.send_hidden_block), dtype=dtype),
                            )
                            pl.semaphore_signal(send_done_sem.at[dst, slot])

                    @pl.when(producer == 0)
                    def _publish_full() -> None:
                        pl.semaphore_wait(
                            send_done_sem.at[dst, slot], value=generation * producer_tiles, decrement=False
                        )
                        if static_peer_ordinal == 0:
                            pl.semaphore_signal(full_sem.at[rank, slot])
                        else:
                            _signal_remote(full_sem, dst, rank, slot)

            branches = tuple((lambda ordinal: lambda _: _send_peer(ordinal))(i) for i in range(ep_size))
            lax.switch(peer_ordinal, branches, None)

        @pl.when(worker >= config.producer_programs_per_peer)
        def _consumer() -> None:
            consumer = worker - config.producer_programs_per_peer
            pl.semaphore_wait(init_ready_sem, value=1, decrement=False)

            def _consume_source(static_source_ordinal: int) -> None:
                src = (rank + static_source_ordinal) % ep_size

                chunk_windows = (send_chunks_per_dst + config.inbox_slots - 1) // config.inbox_slots

                @pl.loop(0, chunk_windows)
                def _window_loop(window) -> None:
                    window_start = window * config.inbox_slots
                    generation = window + 1

                    @pl.loop(0, consumer_jobs_per_program)
                    def _job_loop(job_iteration) -> None:
                        job = consumer + job_iteration * consumer_programs_per_source

                        @pl.when(job < consumer_jobs)
                        def _consume_job() -> None:
                            @pl.when(job < dh_jobs)
                            def _compute_dh() -> None:
                                chunk_offset = job // (blocks * intermediate_tiles)
                                rem = job % (blocks * intermediate_tiles)
                                block = rem // intermediate_tiles
                                i_tile = rem % intermediate_tiles
                                chunk = window_start + chunk_offset

                                @pl.when(chunk < send_chunks_per_dst)
                                def _compute_resident_dh() -> None:
                                    pl.semaphore_wait(
                                        full_sem.at[src, chunk_offset], value=generation, decrement=False
                                    )
                                    valid_rows = recv_valid_ref[static_source_ordinal, chunk, block]

                                    @pl.when(valid_rows > 0)
                                    def _compute_live() -> None:
                                        expert = recv_expert_ref[static_source_ordinal, chunk, block]

                                        def _acc_scope(acc_ref) -> None:
                                            def _smem_scope(dy_smem, w_smem, barrier) -> None:
                                                @pl.loop(0, hidden_tiles)
                                                def _hidden_loop(h_tile) -> None:
                                                    h_start = h_tile * config.hidden_block
                                                    mgpu.copy_gmem_to_smem(
                                                        inbox_ref.at[
                                                            src,
                                                            chunk_offset,
                                                            pl.ds(block * config.compute_m, config.compute_m),
                                                            pl.ds(h_start, config.hidden_block),
                                                        ],
                                                        dy_smem,
                                                        barrier,
                                                    )
                                                    mgpu.copy_gmem_to_smem(
                                                        w_ref.at[
                                                            expert,
                                                            pl.ds(
                                                                i_tile * config.intermediate_block,
                                                                config.intermediate_block,
                                                            ),
                                                            pl.ds(h_start, config.hidden_block),
                                                        ],
                                                        w_smem,
                                                        barrier,
                                                    )
                                                    mgpu.barrier_wait(barrier)
                                                    mgpu.commit_smem()
                                                    mgpu.wgmma(acc_ref, dy_smem, mgpu.transpose_ref(w_smem, (1, 0)))
                                                    mgpu.wgmma_wait(0)

                                            pl.run_scoped(
                                                _smem_scope,
                                                dy_smem=_wgmma_smem((config.compute_m, config.hidden_block), dtype),
                                                w_smem=_wgmma_smem(
                                                    (config.intermediate_block, config.hidden_block), dtype
                                                ),
                                                barrier=mgpu.Barrier(num_arrivals=2),
                                            )
                                            row_start = recv_row_ref[static_source_ordinal, chunk, block]
                                            d_h_ref[
                                                expert,
                                                pl.ds(row_start, config.compute_m),
                                                pl.ds(
                                                    i_tile * config.intermediate_block,
                                                    config.intermediate_block,
                                                ),
                                            ] = acc_ref[...]

                                        pl.run_scoped(
                                            _acc_scope,
                                            acc_ref=mgpu.ACC((config.compute_m, config.intermediate_block)),
                                        )

                            @pl.when(job >= dh_jobs)
                            def _compute_dw2() -> None:
                                dw_job = job - dh_jobs
                                expert = dw_job // (intermediate_tiles * hidden_tiles)
                                rem = dw_job % (intermediate_tiles * hidden_tiles)
                                i_tile = rem // hidden_tiles
                                h_tile = rem % hidden_tiles

                                owns_rows = jnp.asarray(False)
                                for static_chunk_offset in range(config.inbox_slots):
                                    chunk = window_start + static_chunk_offset
                                    safe_chunk = jnp.minimum(chunk, send_chunks_per_dst - 1)
                                    for static_block in range(blocks):
                                        valid_rows = recv_valid_ref[static_source_ordinal, safe_chunk, static_block]
                                        block_expert = recv_expert_ref[static_source_ordinal, safe_chunk, static_block]
                                        owns_rows |= (
                                            (chunk < send_chunks_per_dst) & (valid_rows > 0) & (block_expert == expert)
                                        )

                                @pl.when(owns_rows)
                                def _compute_window_dw2() -> None:

                                    def _acc_scope(acc_ref) -> None:
                                        def _smem_scope(h_smem, dy_smem, barrier) -> None:
                                            for static_chunk_offset in range(config.inbox_slots):
                                                chunk = window_start + static_chunk_offset
                                                safe_chunk = jnp.minimum(chunk, send_chunks_per_dst - 1)
                                                chunk_owns_rows = jnp.asarray(False)
                                                for static_block in range(blocks):
                                                    valid_rows = recv_valid_ref[
                                                        static_source_ordinal, safe_chunk, static_block
                                                    ]
                                                    block_expert = recv_expert_ref[
                                                        static_source_ordinal, safe_chunk, static_block
                                                    ]
                                                    chunk_owns_rows |= (valid_rows > 0) & (block_expert == expert)

                                                @pl.when((chunk < send_chunks_per_dst) & chunk_owns_rows)
                                                def _accumulate_chunk() -> None:
                                                    pl.semaphore_wait(
                                                        full_sem.at[src, static_chunk_offset],
                                                        value=generation,
                                                        decrement=False,
                                                    )
                                                    for static_block in range(blocks):
                                                        valid_rows = recv_valid_ref[
                                                            static_source_ordinal, safe_chunk, static_block
                                                        ]
                                                        block_expert = recv_expert_ref[
                                                            static_source_ordinal, safe_chunk, static_block
                                                        ]

                                                        @pl.when((valid_rows > 0) & (block_expert == expert))
                                                        def _accumulate_block() -> None:
                                                            row_start = recv_row_ref[
                                                                static_source_ordinal, safe_chunk, static_block
                                                            ]
                                                            mgpu.copy_gmem_to_smem(
                                                                h_ref.at[
                                                                    expert,
                                                                    pl.ds(row_start, config.compute_m),
                                                                    pl.ds(
                                                                        i_tile * config.intermediate_block,
                                                                        config.intermediate_block,
                                                                    ),
                                                                ],
                                                                h_smem,
                                                                barrier,
                                                            )
                                                            mgpu.copy_gmem_to_smem(
                                                                inbox_ref.at[
                                                                    src,
                                                                    static_chunk_offset,
                                                                    pl.ds(
                                                                        static_block * config.compute_m,
                                                                        config.compute_m,
                                                                    ),
                                                                    pl.ds(
                                                                        h_tile * config.hidden_block,
                                                                        config.hidden_block,
                                                                    ),
                                                                ],
                                                                dy_smem,
                                                                barrier,
                                                            )
                                                            mgpu.barrier_wait(barrier)
                                                            mgpu.commit_smem()
                                                            mgpu.wgmma(
                                                                acc_ref,
                                                                mgpu.transpose_ref(h_smem, (1, 0)),
                                                                dy_smem,
                                                            )
                                                            mgpu.wgmma_wait(0)

                                        pl.run_scoped(
                                            _smem_scope,
                                            h_smem=_wgmma_smem((config.compute_m, config.intermediate_block), dtype),
                                            dy_smem=_wgmma_smem((config.compute_m, config.hidden_block), dtype),
                                            barrier=mgpu.Barrier(num_arrivals=2),
                                        )
                                        mgpu.atomic_add(
                                            d_w2_ref.at[
                                                expert,
                                                pl.ds(
                                                    i_tile * config.intermediate_block,
                                                    config.intermediate_block,
                                                ),
                                                pl.ds(h_tile * config.hidden_block, config.hidden_block),
                                            ],
                                            acc_ref[...],
                                        )

                                    pl.run_scoped(
                                        _acc_scope,
                                        acc_ref=mgpu.ACC((config.intermediate_block, config.hidden_block)),
                                    )

                    for static_chunk_offset in range(config.inbox_slots):

                        @pl.when(window_start + static_chunk_offset < send_chunks_per_dst)
                        def _mark_consumer_done() -> None:
                            pl.semaphore_signal(consumer_done_sem.at[src, static_chunk_offset])

                    @pl.when((window % consumer_programs_per_source) == consumer)
                    def _release_window() -> None:
                        for static_chunk_offset in range(config.inbox_slots):

                            @pl.when(window_start + static_chunk_offset < send_chunks_per_dst)
                            def _release_slot() -> None:
                                pl.semaphore_wait(
                                    consumer_done_sem.at[src, static_chunk_offset],
                                    value=generation * consumer_programs_per_source,
                                    decrement=False,
                                )
                                if static_source_ordinal == 0:
                                    pl.semaphore_signal(empty_sem.at[rank, static_chunk_offset])
                                else:
                                    _signal_remote(empty_sem, src, rank, static_chunk_offset)

            branches = tuple(
                (lambda ordinal: lambda _: _consume_source((-ordinal) % ep_size))(i) for i in range(ep_size)
            )
            lax.switch(peer_ordinal, branches, None)

    return mgpu.kernel(
        body,
        out_shape=(
            jax.ShapeDtypeStruct((ep_size, config.inbox_slots, config.send_m, hidden_dim), dtype),
            jax.ShapeDtypeStruct((experts_per_rank, rows_per_expert_capacity, intermediate_dim), jnp.float32),
            jax.ShapeDtypeStruct((experts_per_rank, intermediate_dim, hidden_dim), jnp.float32),
            jax.ShapeDtypeStruct(
                (
                    ep_size,
                    send_chunks_per_dst,
                    config.compute_blocks_per_send,
                    config.compute_m,
                ),
                jnp.float32,
            ),
        ),
        grid=(ep_size, worker_programs),
        grid_names=("peer_phase", "worker_program"),
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
    )


def _wgmma_smem(shape: tuple[int, int], dtype: jnp.dtype):
    swizzle_elements = _WGMMA_SWIZZLE_BYTES // jnp.dtype(dtype).itemsize
    if shape[-2] % _WGMMA_TILE_M or shape[-1] % swizzle_elements:
        raise ValueError(f"WGMMA SMEM shape {shape} is incompatible with 128-byte swizzling")
    return mgpu.SMEM(
        shape,
        dtype=dtype,
        transforms=(
            mgpu.TilingTransform((_WGMMA_TILE_M, swizzle_elements)),
            mgpu.SwizzleTransform(_WGMMA_SWIZZLE_BYTES),
        ),
    )


def _validate_kernel_dimensions(
    hidden_dim: int,
    intermediate_dim: int,
    config: SourcePushSemanticFusedW2BackwardConfig,
) -> None:
    if hidden_dim % config.hidden_block or hidden_dim % config.send_hidden_block:
        raise ValueError(f"hidden_dim must be divisible by hidden_block and send_hidden_block, got {hidden_dim=}")
    if intermediate_dim % config.intermediate_block:
        raise ValueError(f"intermediate_dim must be divisible by intermediate_block, got {intermediate_dim=}")


def _validate_request(
    dy: Array,
    return_y: Array,
    h_expert: Array,
    w_down: Array,
    plan: SourcePushSemanticPlan,
    send_chunks_per_dst: int,
    rows_per_expert_capacity: int,
    config: SourcePushSemanticFusedW2BackwardConfig,
) -> None:
    config.validate()
    if dy.ndim != 3:
        raise ValueError(f"dy must have shape [source, token, hidden], got {dy.shape}")
    source_count, destination_count, experts_per_rank = plan.xcounts.shape
    expected_return_shape = (
        source_count,
        destination_count,
        send_chunks_per_dst * config.compute_blocks_per_send,
        config.compute_m,
        dy.shape[-1],
    )
    if return_y.shape != expected_return_shape:
        raise ValueError(f"return_y shape {return_y.shape} must be {expected_return_shape}")
    if h_expert.ndim != 4 or w_down.ndim != 4:
        raise ValueError(f"h_expert and w_down must be rank four, got {h_expert.shape=} {w_down.shape=}")
    if source_count != destination_count or dy.shape[:2] != (source_count, plan.tokens_per_source):
        raise ValueError(f"dy shape {dy.shape} is incompatible with semantic plan {plan.xcounts.shape}")
    if h_expert.shape[:3] != (destination_count, experts_per_rank, rows_per_expert_capacity):
        raise ValueError(
            f"h_expert leading shape {h_expert.shape[:3]} must be "
            f"{(destination_count, experts_per_rank, rows_per_expert_capacity)}"
        )
    if w_down.shape != (destination_count, experts_per_rank, h_expert.shape[-1], dy.shape[-1]):
        raise ValueError(
            f"w_down shape {w_down.shape} must be "
            f"{(destination_count, experts_per_rank, h_expert.shape[-1], dy.shape[-1])}"
        )
    if dy.dtype != jnp.bfloat16 or h_expert.dtype != jnp.bfloat16 or w_down.dtype != jnp.bfloat16:
        raise ValueError(
            f"WGMMA inputs dy, h_expert, and w_down must use bfloat16, got "
            f"{dy.dtype}, {h_expert.dtype}, {w_down.dtype}"
        )
    if rows_per_expert_capacity % config.compute_m:
        raise ValueError(
            f"rows_per_expert_capacity must be divisible by compute_m={config.compute_m}, "
            f"got {rows_per_expert_capacity}"
        )
    _validate_kernel_dimensions(dy.shape[-1], h_expert.shape[-1], config)
