# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent semantic W13 backward with source-owned dX return/combine."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import NamedTuple, cast

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
WGMMA_PIPELINE_STAGES = 2
# A separate persistent dW owner grid would wait for stage-ready signals while
# the 288-CTA route grid could leave the required producers nonresident. Keep
# the complete producer -> {dX, dW} partial order in one co-resident grid below
# the 132-SM H100 SXM target instead.
PERSISTENT_BACKWARD_PROGRAMS = 128
STAGING_PROGRAMS = 48
DX_PROGRAMS = 40
DW_PROGRAMS = 40
assert STAGING_PROGRAMS + DX_PROGRAMS + DW_PROGRAMS == PERSISTENT_BACKWARD_PROGRAMS


class _SourcePushSemanticFusedW13BackwardDiagnostic(StrEnum):
    FULL = "full"
    STAGING_ONLY = "staging_only"
    STAGING_DX = "staging_dx"
    STAGING_DW = "staging_dw"
    STAGING_DX_DW_NO_COMBINE = "staging_dx_dw_no_combine"
    FULL_SERIAL_COMBINE = "full_serial_combine"

    @property
    def includes_dx(self) -> bool:
        return self in (self.FULL, self.STAGING_DX, self.STAGING_DX_DW_NO_COMBINE, self.FULL_SERIAL_COMBINE)

    @property
    def includes_dw(self) -> bool:
        return self in (self.FULL, self.STAGING_DW, self.STAGING_DX_DW_NO_COMBINE, self.FULL_SERIAL_COMBINE)

    @property
    def includes_combine(self) -> bool:
        return self in (self.FULL, self.STAGING_DX, self.FULL_SERIAL_COMBINE)

    @property
    def serializes_combine(self) -> bool:
        return self in (self.FULL, self.FULL_SERIAL_COMBINE)

    @property
    def combines_on_dx_workers(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class SourcePushSemanticFusedW13BackwardConfig:
    """Fixed Hopper tile and persistent-CTA dimensions."""

    compute_m: int = 64
    send_m: int = 256
    send_hidden_block: int = 256
    block_hidden: int = 128
    block_output: int = 128
    inbox_slots: int = 12
    combine_token_block: int = 64

    def validate(self) -> None:
        expected = {
            "compute_m": 64,
            "send_hidden_block": 256,
            "block_hidden": 128,
            "block_output": 128,
            "inbox_slots": 12,
            "combine_token_block": 64,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"the initial Hopper lowering requires {name}={value}, got {getattr(self, name)}")
        if self.send_m <= 0 or self.send_m % self.compute_m:
            raise ValueError(
                f"send_m must be a positive multiple of compute_m, got {self.send_m=} and {self.compute_m=}"
            )

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
    combine_ready_generation: Int[Array, "S TokenBlock DstOrd Slot"]
    compact_valid_rows: Int[Array, "Dst E MBlock"]


class SourcePushSemanticFusedW13BackwardResult(NamedTuple):
    """Source-owned input gradient and destination-owned W13 gradient."""

    dx: Float[Array, "S T H"]
    dw13: Float[Array, "Dst E H O"]
    queue_overflow_routes: Int[Array, ""]
    layout_overflow_rows: Int[Array, ""]


class _SourcePushSemanticFusedW13BackwardDiagnosticResult(NamedTuple):
    x_expert: Float[Array, "Dst E C H"]
    dx: Float[Array, "S T H"]
    dw13: Float[Array, "Dst E H O"]
    queue_overflow_routes: Int[Array, ""]
    layout_overflow_rows: Int[Array, ""]


class _SourcePushSemanticFusedW13BackwardRoleLayout(NamedTuple):
    dx_program_start: int
    dw_program_start: int
    total_programs: int


def _source_push_semantic_fused_w13_backward_role_layout(
    diagnostic: _SourcePushSemanticFusedW13BackwardDiagnostic,
) -> _SourcePushSemanticFusedW13BackwardRoleLayout:
    dx_program_start = STAGING_PROGRAMS
    dw_program_start = dx_program_start + (DX_PROGRAMS if diagnostic.includes_dx else 0)
    total_programs = dw_program_start + (DW_PROGRAMS if diagnostic.includes_dw else 0)
    return _SourcePushSemanticFusedW13BackwardRoleLayout(dx_program_start, dw_program_start, total_programs)


@dataclass(frozen=True, slots=True)
class SourcePushSemanticFusedW13BackwardSchedule:
    """Resident CTA partition and fixed semantic work counts for one rank."""

    hidden_tiles: int
    compact_m_blocks: int
    staging_jobs: int
    dx_jobs: int
    token_blocks: int
    rounds: int
    staging_programs: int
    dx_program_start: int
    dx_programs: int
    dw_program_start: int
    dw_programs: int
    combine_programs: int
    active_combine_programs: int
    max_stage_readiness_signals: int
    compact_readiness_slots: int
    dx_readiness_signals: int
    readiness_waits: int

    @property
    def total_programs(self) -> int:
        return PERSISTENT_BACKWARD_PROGRAMS


def source_push_semantic_fused_w13_backward_schedule(
    *,
    ep_size: int,
    experts_per_rank: int,
    rows_per_expert_capacity: int,
    hidden_dim: int,
    tokens_per_source: int,
    send_chunks_per_dst: int,
    config: SourcePushSemanticFusedW13BackwardConfig = SourcePushSemanticFusedW13BackwardConfig(),
) -> SourcePushSemanticFusedW13BackwardSchedule:
    """Return the co-resident semantic staging and output-owner schedule."""

    config.validate()
    if (
        ep_size <= 0
        or experts_per_rank <= 0
        or rows_per_expert_capacity <= 0
        or hidden_dim <= 0
        or tokens_per_source <= 0
        or send_chunks_per_dst <= 0
    ):
        raise ValueError(
            "ep_size, experts_per_rank, rows_per_expert_capacity, hidden_dim, tokens_per_source, and "
            "send_chunks_per_dst must be positive, "
            f"got {ep_size}, {experts_per_rank}, {rows_per_expert_capacity}, {hidden_dim}, "
            f"{tokens_per_source}, and {send_chunks_per_dst}"
        )
    if hidden_dim % config.block_hidden:
        raise ValueError(f"hidden dim {hidden_dim} must be divisible by block_hidden={config.block_hidden}")
    if rows_per_expert_capacity % config.compute_m:
        raise ValueError(
            f"rows_per_expert_capacity must be divisible by compute_m={config.compute_m}, "
            f"got {rows_per_expert_capacity}"
        )

    hidden_tiles = hidden_dim // config.block_hidden
    compact_m_blocks = rows_per_expert_capacity // config.compute_m
    if hidden_tiles > STAGING_PROGRAMS:
        raise ValueError(f"hidden tiles {hidden_tiles} exceed the resident combine group {STAGING_PROGRAMS}")
    block_jobs = ep_size * send_chunks_per_dst * config.compute_blocks_per_send
    token_blocks = (tokens_per_source + config.combine_token_block - 1) // config.combine_token_block
    combine_programs = STAGING_PROGRAMS
    active_combine_programs = sum(
        min(token_blocks, (combine_programs + hidden_tiles - 1 - hidden_tile) // hidden_tiles)
        for hidden_tile in range(hidden_tiles)
    )
    return SourcePushSemanticFusedW13BackwardSchedule(
        hidden_tiles=hidden_tiles,
        compact_m_blocks=compact_m_blocks,
        staging_jobs=block_jobs * hidden_tiles,
        dx_jobs=block_jobs * hidden_tiles,
        token_blocks=token_blocks,
        rounds=(send_chunks_per_dst + config.inbox_slots - 1) // config.inbox_slots,
        staging_programs=STAGING_PROGRAMS,
        dx_program_start=STAGING_PROGRAMS,
        dx_programs=DX_PROGRAMS,
        dw_program_start=STAGING_PROGRAMS + DX_PROGRAMS,
        dw_programs=DW_PROGRAMS,
        combine_programs=combine_programs,
        active_combine_programs=active_combine_programs,
        max_stage_readiness_signals=block_jobs * hidden_tiles,
        compact_readiness_slots=experts_per_rank * compact_m_blocks * hidden_tiles,
        dx_readiness_signals=ep_size * send_chunks_per_dst,
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
    if rows_per_expert_capacity % config.compute_m:
        raise ValueError(
            f"rows_per_expert_capacity must be divisible by compute_m={config.compute_m}, "
            f"got {rows_per_expert_capacity}"
        )
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
    destination_count, _source_count, _chunks, _blocks = forward.recv_valid_rows.shape
    experts_per_rank = plan.xcounts.shape[-1]
    compact_m_blocks = rows_per_expert_capacity // config.compute_m
    destination = jnp.arange(destination_count, dtype=jnp.int32)[:, None, None, None]
    safe_expert = jnp.maximum(forward.recv_expert, 0)
    compact_block = jnp.where(
        forward.recv_valid_rows > 0,
        forward.recv_row_start // config.compute_m,
        0,
    )
    compact_valid_rows = (
        jnp.zeros((destination_count, experts_per_rank, compact_m_blocks), dtype=jnp.int32)
        .at[destination, safe_expert, compact_block]
        .max(forward.recv_valid_rows, mode="drop")
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
        compact_valid_rows=compact_valid_rows,
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


def _source_push_semantic_fused_w13_staging_reference_jax(
    x: Float[Array, "S T H"],
    metadata: SourcePushSemanticFusedW13BackwardMetadata,
    *,
    experts_per_rank: int,
    rows_per_expert_capacity: int,
) -> Float[Array, "Dst E C H"]:
    forward = metadata.forward
    source_count, destination_count, _chunks, _blocks, compute_m = forward.token_ids.shape
    source = jnp.arange(source_count, dtype=jnp.int32)[:, None, None, None, None]
    dst_ordinal = jnp.arange(destination_count, dtype=jnp.int32)[None, :, None, None]
    destination = (jnp.arange(source_count, dtype=jnp.int32)[:, None, None, None] + dst_ordinal) % destination_count
    row = jnp.arange(compute_m, dtype=jnp.int32)[None, None, None, None, :]
    row_valid = row < forward.send_valid_rows[..., None]
    expert = jnp.maximum(forward.send_expert, 0)[..., None]
    expert_row = forward.send_row_start[..., None] + row
    gathered_x = x.at[source, forward.token_ids].get()
    gathered_x = jnp.where(row_valid[..., None], gathered_x, 0.0)
    x_expert = jnp.zeros(
        (destination_count, experts_per_rank, rows_per_expert_capacity, x.shape[-1]),
        dtype=x.dtype,
    )
    return x_expert.at[destination[..., None], expert, expert_row].add(gathered_x)


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


def _source_push_semantic_fused_w13_backward_diagnostic(
    x: Float[Array, "S T H"],
    dz13: Float[Array, "Dst E C O"],
    w13: Float[Array, "Dst E H O"],
    plan: SourcePushSemanticPlan,
    *,
    send_chunks_per_dst: int,
    rows_per_expert_capacity: int,
    diagnostic: _SourcePushSemanticFusedW13BackwardDiagnostic,
    config: SourcePushSemanticFusedW13BackwardConfig = SourcePushSemanticFusedW13BackwardConfig(),
    mesh: Mesh | AbstractMesh | None = None,
    interpret: bool = False,
) -> _SourcePushSemanticFusedW13BackwardDiagnosticResult:
    """Run a benchmark-only compile-time role isolation variant."""

    if diagnostic is _SourcePushSemanticFusedW13BackwardDiagnostic.FULL:
        raise ValueError("the full path must use source_push_semantic_fused_w13_backward")
    _validate_request(x, dz13, w13, plan, rows_per_expert_capacity, config)
    metadata = source_push_semantic_fused_w13_backward_metadata_jax(
        x,
        plan,
        send_chunks_per_dst=send_chunks_per_dst,
        rows_per_expert_capacity=rows_per_expert_capacity,
        config=config,
    )
    if interpret:
        x_expert = _source_push_semantic_fused_w13_staging_reference_jax(
            x,
            metadata,
            experts_per_rank=w13.shape[1],
            rows_per_expert_capacity=rows_per_expert_capacity,
        )
        reference_dx, reference_dw13 = source_push_semantic_fused_w13_backward_reference_jax(x, dz13, w13, metadata)
        dx = reference_dx if diagnostic.includes_combine else jnp.zeros_like(x, dtype=jnp.float32)
        dw13 = reference_dw13 if diagnostic.includes_dw else jnp.zeros_like(w13, dtype=jnp.float32)
    else:
        if jax.default_backend() != "gpu":
            raise NotImplementedError("persistent semantic fused W13 backward diagnostics require a GPU backend")
        if mesh is None:
            mesh = jax.sharding.get_abstract_mesh()
            if mesh.empty:
                raise ValueError("mesh is required for persistent semantic fused W13 backward diagnostics")
        x_expert, dx, dw13 = _source_push_semantic_fused_w13_backward_sharded(
            x,
            dz13,
            w13,
            metadata,
            config=config,
            mesh=mesh,
            diagnostic=diagnostic,
        )
    return _SourcePushSemanticFusedW13BackwardDiagnosticResult(
        x_expert=x_expert,
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
    diagnostic: _SourcePushSemanticFusedW13BackwardDiagnostic = _SourcePushSemanticFusedW13BackwardDiagnostic.FULL,
) -> tuple[Array, ...]:
    ep_size = x.shape[0]
    if mesh.shape[SOURCE_PUSH_MESH_AXIS] != ep_size:
        raise ValueError(f"mesh size must match EP size {ep_size}, got {mesh.shape[SOURCE_PUSH_MESH_AXIS]}")
    kernel_diagnostic = diagnostic
    if diagnostic is _SourcePushSemanticFusedW13BackwardDiagnostic.FULL:
        kernel_diagnostic = _SourcePushSemanticFusedW13BackwardDiagnostic.STAGING_DX_DW_NO_COMBINE
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
        diagnostic=kernel_diagnostic,
    )

    def local_fn(
        x_local,
        token_ids_local,
        send_valid_local,
        send_expert_local,
        send_row_local,
        recv_expert_local,
        recv_row_local,
        recv_valid_local,
        compact_valid_rows_local,
        route_dst_local,
        route_chunk_local,
        route_block_local,
        route_row_local,
        route_valid_local,
        combine_ready_local,
        dz_local,
        w_local,
    ):
        x_expert_local, dx_return_local, dx_local, dw_local = kernel(
            x_local[0],
            token_ids_local[0],
            send_valid_local[0],
            send_expert_local[0],
            send_row_local[0],
            recv_expert_local[0],
            recv_row_local[0],
            recv_valid_local[0],
            compact_valid_rows_local[0],
            route_dst_local[0],
            route_chunk_local[0],
            route_block_local[0],
            route_row_local[0],
            route_valid_local[0],
            combine_ready_local[0],
            dz_local[0],
            w_local[0],
        )
        if diagnostic is _SourcePushSemanticFusedW13BackwardDiagnostic.FULL:
            # x_expert is remote-written scratch used inside the custom call.
            x_guard = lax.optimization_barrier(x_expert_local[0, 0, 0].astype(jnp.float32))
            dw_local = dw_local.at[0, 0, 0].add(x_guard - x_guard)
            return dx_return_local[None, ...], dw_local[None, ...]
        if diagnostic is _SourcePushSemanticFusedW13BackwardDiagnostic.STAGING_ONLY:
            return (x_expert_local[None, ...],)
        if diagnostic is _SourcePushSemanticFusedW13BackwardDiagnostic.FULL_SERIAL_COMBINE:
            return x_expert_local[None, ...], dx_local[None, ...], dw_local[None, ...]
        if diagnostic.includes_combine:
            return x_expert_local[None, ...], dx_local[None, ...]
        return x_expert_local[None, ...], dw_local[None, ...]

    source_3d = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None))
    source_4d = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None))
    destination_3d = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None))
    destination_4d = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None))
    source_token_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None, None))
    if diagnostic is _SourcePushSemanticFusedW13BackwardDiagnostic.FULL:
        out_specs = (
            P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        )
    elif diagnostic is _SourcePushSemanticFusedW13BackwardDiagnostic.STAGING_ONLY:
        out_specs = (P(SOURCE_PUSH_MESH_AXIS, None, None, None),)
    elif diagnostic is _SourcePushSemanticFusedW13BackwardDiagnostic.FULL_SERIAL_COMBINE:
        out_specs = (
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        )
    elif diagnostic.includes_combine:
        out_specs = (
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        )
    else:
        out_specs = (
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        )
    outputs = shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        ),
        out_specs=out_specs,
        check_vma=False,
    )(
        jax.sharding.reshard(x, source_3d),
        jax.sharding.reshard(metadata.forward.token_ids, source_token_sharding),
        jax.sharding.reshard(metadata.forward.send_valid_rows, source_4d),
        jax.sharding.reshard(metadata.forward.send_expert, source_4d),
        jax.sharding.reshard(metadata.forward.send_row_start, source_4d),
        jax.sharding.reshard(metadata.forward.recv_expert, destination_4d),
        jax.sharding.reshard(metadata.forward.recv_row_start, destination_4d),
        jax.sharding.reshard(metadata.forward.recv_valid_rows, destination_4d),
        jax.sharding.reshard(metadata.compact_valid_rows, destination_3d),
        jax.sharding.reshard(metadata.route_dst_ordinal, destination_3d),
        jax.sharding.reshard(metadata.route_chunk, destination_3d),
        jax.sharding.reshard(metadata.route_block, destination_3d),
        jax.sharding.reshard(metadata.route_row, destination_3d),
        jax.sharding.reshard(metadata.route_valid, destination_3d),
        jax.sharding.reshard(metadata.combine_ready_generation, source_4d),
        jax.sharding.reshard(dz13, destination_4d),
        jax.sharding.reshard(w13, destination_4d),
    )
    if diagnostic is _SourcePushSemanticFusedW13BackwardDiagnostic.FULL:
        dx_return, dw13 = cast(tuple[Array, Array], outputs)
        dx = _source_push_semantic_dx_combine_sharded(
            dx_return,
            metadata.route_dst_ordinal,
            metadata.route_chunk,
            metadata.route_block,
            metadata.route_row,
            metadata.route_valid,
            config=config,
            mesh=mesh,
        )
        return dx, dw13
    if diagnostic is _SourcePushSemanticFusedW13BackwardDiagnostic.STAGING_ONLY:
        (x_expert,) = cast(tuple[Array], outputs)
        return x_expert, jnp.zeros_like(x, dtype=jnp.float32), jnp.zeros_like(w13, dtype=jnp.float32)
    if diagnostic is _SourcePushSemanticFusedW13BackwardDiagnostic.FULL_SERIAL_COMBINE:
        return cast(tuple[Array, Array, Array], outputs)
    if diagnostic.includes_combine:
        x_expert, dx = cast(tuple[Array, Array], outputs)
        return x_expert, dx, jnp.zeros_like(w13, dtype=jnp.float32)
    x_expert, dw13 = cast(tuple[Array, Array], outputs)
    return x_expert, jnp.zeros_like(x, dtype=jnp.float32), dw13


def _source_push_semantic_dx_combine_sharded(
    dx_return: Array,
    route_dst_ordinal: Array,
    route_chunk: Array,
    route_block: Array,
    route_row: Array,
    route_valid: Array,
    *,
    config: SourcePushSemanticFusedW13BackwardConfig,
    mesh: Mesh | AbstractMesh,
) -> Array:
    tokens_per_source = route_valid.shape[1]
    hidden_dim = dx_return.shape[-1]
    kernel = _make_source_push_semantic_dx_combine_kernel(
        tokens_per_source=tokens_per_source,
        hidden_dim=hidden_dim,
        topk=route_valid.shape[-1],
        dtype=dx_return.dtype,
        config=config,
    )

    def local_fn(
        dx_return_local, route_dst_local, route_chunk_local, route_block_local, route_row_local, route_valid_local
    ):
        dx_local = kernel(
            dx_return_local[0],
            route_dst_local[0],
            route_chunk_local[0],
            route_block_local[0],
            route_row_local[0],
            route_valid_local[0],
        )
        return dx_local[None, ...]

    source_3d = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None))
    source_5d = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None, None))
    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None),
        check_vma=False,
    )(
        jax.sharding.reshard(dx_return, source_5d),
        jax.sharding.reshard(route_dst_ordinal, source_3d),
        jax.sharding.reshard(route_chunk, source_3d),
        jax.sharding.reshard(route_block, source_3d),
        jax.sharding.reshard(route_row, source_3d),
        jax.sharding.reshard(route_valid.astype(jnp.int32), source_3d),
    )


def _make_source_push_semantic_dx_combine_kernel(
    *,
    tokens_per_source: int,
    hidden_dim: int,
    topk: int,
    dtype: jnp.dtype,
    config: SourcePushSemanticFusedW13BackwardConfig,
):
    if hidden_dim % config.block_hidden:
        raise ValueError(f"hidden dim {hidden_dim} must be divisible by block_hidden={config.block_hidden}")
    token_blocks = math.ceil(tokens_per_source / config.combine_token_block)
    hidden_tiles = hidden_dim // config.block_hidden

    def body(
        dx_return_ref,
        route_dst_ref,
        route_chunk_ref,
        route_block_ref,
        route_row_ref,
        route_valid_ref,
        dx_ref,
    ) -> None:
        token_block = pl.program_id(0)
        hidden_tile = pl.program_id(1)
        token_start = token_block * config.combine_token_block
        hidden_start = hidden_tile * config.block_hidden

        @pl.loop(0, config.combine_token_block)
        def _token_loop(token_offset) -> None:
            token = token_start + token_offset

            @pl.when(token < tokens_per_source)
            def _combine_token() -> None:
                acc = jnp.zeros((config.block_hidden,), dtype=jnp.float32)
                for route_slot in range(topk):
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
                    acc += jnp.where(valid, route_value, jnp.zeros((), dtype=jnp.float32))

                dx_ref[token, pl.ds(hidden_start, config.block_hidden)] = acc

    return mgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((tokens_per_source, hidden_dim), jnp.float32),
        grid=(token_blocks, hidden_tiles),
        grid_names=("token_block", "hidden_tile"),
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
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
    diagnostic: _SourcePushSemanticFusedW13BackwardDiagnostic = _SourcePushSemanticFusedW13BackwardDiagnostic.FULL,
):
    config.validate()
    _validate_kernel_dimensions(hidden_dim, output_dim, config)
    schedule = source_push_semantic_fused_w13_backward_schedule(
        ep_size=ep_size,
        experts_per_rank=experts_per_rank,
        rows_per_expert_capacity=rows_per_expert_capacity,
        hidden_dim=hidden_dim,
        tokens_per_source=tokens_per_source,
        send_chunks_per_dst=send_chunks_per_dst,
        config=config,
    )
    hidden_tiles = schedule.hidden_tiles
    output_tiles = output_dim // config.block_output
    dw_tiles = experts_per_rank * hidden_tiles * output_tiles
    role_layout = _source_push_semantic_fused_w13_backward_role_layout(diagnostic)
    dx_program_start = role_layout.dx_program_start
    dw_program_start = role_layout.dw_program_start
    total_programs = role_layout.total_programs

    def body(
        x_ref,
        token_ids_ref,
        send_valid_ref,
        send_expert_ref,
        send_row_ref,
        recv_expert_ref,
        recv_row_ref,
        recv_valid_ref,
        compact_valid_rows_ref,
        route_dst_ref,
        route_chunk_ref,
        route_block_ref,
        route_row_ref,
        route_valid_ref,
        combine_ready_ref,
        dz_ref,
        w_ref,
        x_expert_ref,
        dx_return_ref,
        dx_ref,
        dw_ref,
    ) -> None:
        if diagnostic.includes_dw:
            compact_ready_sem = pl.get_global(
                mgpu.SemaphoreType.REGULAR((experts_per_rank, schedule.compact_m_blocks, hidden_tiles))
            )
        if diagnostic.includes_dx:
            dx_done_sem = pl.get_global(mgpu.SemaphoreType.REGULAR((ep_size, send_chunks_per_dst)))
            ready_sem = pl.get_global(mgpu.SemaphoreType.REGULAR((ep_size, config.inbox_slots)))
        if diagnostic.serializes_combine:
            dw_done_sem = pl.get_global(mgpu.SemaphoreType.REGULAR((1,)))
        rank = lax.axis_index(SOURCE_PUSH_MESH_AXIS)
        worker = pl.program_id(0)

        def _signal_remote(sem, peer, index) -> None:
            pl.semaphore_signal(
                sem.at[index],
                device_id=peer,
                device_id_type=pl.DeviceIdType.LOGICAL,
            )

        def _combine_routes(combine_worker, combine_programs: int) -> None:
            hidden_tile = combine_worker % hidden_tiles
            consumer_lane = combine_worker // hidden_tiles
            consumers_for_tile = (combine_programs + hidden_tiles - 1 - hidden_tile) // hidden_tiles
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
                                    acc += jnp.where(valid, route_value, jnp.zeros((), dtype=jnp.float32))

                                dx_ref[token, pl.ds(hidden_start, config.block_hidden)] = acc

        @pl.when(worker < schedule.staging_programs)
        def _stage_and_combine() -> None:
            producer = worker
            staging_jobs_per_peer = send_chunks_per_dst * config.compute_blocks_per_send * hidden_tiles

            def _stage_peer(static_destination_ordinal: int) -> None:
                destination = (rank + static_destination_ordinal) % ep_size
                destination_x = x_expert_ref
                if static_destination_ordinal != 0:
                    destination_x = mgpu.remote_ref(
                        x_expert_ref,
                        destination,
                        device_id_type=pl.DeviceIdType.LOGICAL,
                    )

                @pl.loop(0, math.ceil(staging_jobs_per_peer / schedule.staging_programs))
                def _stage_loop(iteration) -> None:
                    job = iteration * schedule.staging_programs + producer

                    @pl.when(job < staging_jobs_per_peer)
                    def _stage_owned_tile() -> None:
                        hidden_tile = job % hidden_tiles
                        block_job = job // hidden_tiles
                        block = block_job % config.compute_blocks_per_send
                        chunk = block_job // config.compute_blocks_per_send
                        valid_rows = send_valid_ref[static_destination_ordinal, chunk, block]

                        @pl.when(valid_rows > 0)
                        def _stage_live_tile() -> None:
                            expert = send_expert_ref[static_destination_ordinal, chunk, block]
                            row_start = send_row_ref[static_destination_ordinal, chunk, block]
                            hidden_start = hidden_tile * config.block_hidden

                            def _copy_scope(x_smem) -> None:
                                @pl.loop(0, config.compute_m)
                                def _row_loop(row) -> None:
                                    token = token_ids_ref[static_destination_ordinal, chunk, block, row]
                                    x_smem[row, :] = jnp.where(
                                        row < valid_rows,
                                        x_ref[token, pl.ds(hidden_start, config.block_hidden)],
                                        jnp.zeros((config.block_hidden,), dtype=dtype),
                                    )

                                mgpu.commit_smem()
                                mgpu.copy_smem_to_gmem(
                                    x_smem,
                                    destination_x.at[
                                        expert,
                                        pl.ds(row_start, config.compute_m),
                                        pl.ds(hidden_start, config.block_hidden),
                                    ],
                                )
                                mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                            pl.run_scoped(
                                _copy_scope,
                                x_smem=mgpu.SMEM((config.compute_m, config.block_hidden), dtype=dtype),
                            )
                            if diagnostic.includes_dw:
                                compact_block = row_start // config.compute_m
                                if static_destination_ordinal == 0:
                                    pl.semaphore_signal(compact_ready_sem.at[expert, compact_block, hidden_tile])
                                else:
                                    _signal_remote(
                                        compact_ready_sem,
                                        destination,
                                        (expert, compact_block, hidden_tile),
                                    )

            for destination_ordinal in range(ep_size):
                _stage_peer(destination_ordinal)

            if diagnostic.includes_combine and not diagnostic.combines_on_dx_workers:
                if diagnostic.serializes_combine:
                    pl.semaphore_wait(dw_done_sem.at[0], value=schedule.dw_programs, decrement=False)
                _combine_routes(producer, schedule.staging_programs)

        def _compute_and_publish_dx() -> None:
            consumer = worker - dx_program_start
            dx_jobs_per_source = send_chunks_per_dst * config.compute_blocks_per_send * hidden_tiles

            def _compute_source(static_source_ordinal: int) -> None:
                source = (rank + static_source_ordinal) % ep_size
                destination_ordinal = (-static_source_ordinal) % ep_size
                source_return = dx_return_ref
                if static_source_ordinal != 0:
                    source_return = mgpu.remote_ref(
                        dx_return_ref,
                        source,
                        device_id_type=pl.DeviceIdType.LOGICAL,
                    )

                @pl.loop(0, math.ceil(dx_jobs_per_source / schedule.dx_programs))
                def _dx_loop(iteration) -> None:
                    job = iteration * schedule.dx_programs + consumer

                    @pl.when(job < dx_jobs_per_source)
                    def _compute_owned_dx() -> None:
                        hidden_tile = job % hidden_tiles
                        block_job = job // hidden_tiles
                        block = block_job % config.compute_blocks_per_send
                        chunk = block_job // config.compute_blocks_per_send
                        valid_rows = recv_valid_ref[static_source_ordinal, chunk, block]
                        hidden_start = hidden_tile * config.block_hidden

                        source_return[
                            destination_ordinal,
                            chunk,
                            pl.ds(block * config.compute_m, config.compute_m),
                            pl.ds(hidden_start, config.block_hidden),
                        ] = jnp.zeros((config.compute_m, config.block_hidden), dtype=dtype)

                        @pl.when(valid_rows > 0)
                        def _compute_live_dx() -> None:
                            expert = recv_expert_ref[static_source_ordinal, chunk, block]
                            row_start = recv_row_ref[static_source_ordinal, chunk, block]

                            def _acc_scope(acc_ref) -> None:
                                def _smem_scope(dz_smem, w_smem, barrier) -> None:
                                    row = mgpu.layout_cast(
                                        lax.broadcasted_iota(jnp.int32, (config.compute_m, config.block_output), 0),
                                        mgpu.Layout.WGMMA,
                                    )
                                    row_valid = (row < valid_rows).astype(jnp.float32)

                                    def _copy_output_tile(output_tile, stage) -> None:
                                        output_start = output_tile * config.block_output
                                        mgpu.copy_gmem_to_smem(
                                            dz_ref.at[
                                                expert,
                                                pl.ds(row_start, config.compute_m),
                                                pl.ds(output_start, config.block_output),
                                            ],
                                            dz_smem.at[stage],
                                            barrier.at[stage],
                                        )
                                        mgpu.copy_gmem_to_smem(
                                            w_ref.at[
                                                expert,
                                                pl.ds(hidden_start, config.block_hidden),
                                                pl.ds(output_start, config.block_output),
                                            ],
                                            w_smem.at[stage],
                                            barrier.at[stage],
                                        )

                                    _copy_output_tile(jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))

                                    @pl.loop(0, output_tiles)
                                    def _output_loop(output_tile) -> None:
                                        stage = output_tile % WGMMA_PIPELINE_STAGES
                                        mgpu.barrier_wait(barrier.at[stage])
                                        dz_stage = dz_smem.at[stage]
                                        w_stage = w_smem.at[stage]
                                        dz_stage[:, :] = (dz_stage[...].astype(jnp.float32) * row_valid).astype(dtype)
                                        mgpu.commit_smem()
                                        mgpu.wgmma(acc_ref, dz_stage, mgpu.transpose_ref(w_stage, (1, 0)))

                                        @pl.when(output_tile + 1 < output_tiles)
                                        def _prefetch_next_output_tile() -> None:
                                            # The older group owns the stage about to be reused.
                                            mgpu.wgmma_wait(1)
                                            next_tile = output_tile + 1
                                            _copy_output_tile(next_tile, next_tile % WGMMA_PIPELINE_STAGES)

                                    mgpu.wgmma_wait(0)

                                pl.run_scoped(
                                    _smem_scope,
                                    dz_smem=_wgmma_smem(
                                        (WGMMA_PIPELINE_STAGES, config.compute_m, config.block_output), dtype
                                    ),
                                    w_smem=_wgmma_smem(
                                        (WGMMA_PIPELINE_STAGES, config.block_hidden, config.block_output), dtype
                                    ),
                                    barrier=mgpu.Barrier(
                                        num_arrivals=2,
                                        num_barriers=WGMMA_PIPELINE_STAGES,
                                    ),
                                )
                                source_return[
                                    destination_ordinal,
                                    chunk,
                                    pl.ds(block * config.compute_m, config.compute_m),
                                    pl.ds(hidden_start, config.block_hidden),
                                ] = acc_ref[...].astype(dtype)

                            pl.run_scoped(
                                _acc_scope,
                                acc_ref=mgpu.ACC((config.compute_m, config.block_hidden)),
                            )

                        pl.semaphore_signal(dx_done_sem.at[static_source_ordinal, chunk])

                # Preserve cumulative slot generations by publishing chunks in order.
                @pl.loop(0, math.ceil(config.inbox_slots / schedule.dx_programs))
                def _publish_dx_loop(iteration) -> None:
                    slot = iteration * schedule.dx_programs + consumer

                    @pl.when(slot < config.inbox_slots)
                    def _publish_owned_slot() -> None:
                        @pl.loop(0, schedule.rounds)
                        def _round_loop(round_index) -> None:
                            chunk = round_index * config.inbox_slots + slot

                            @pl.when(chunk < send_chunks_per_dst)
                            def _publish_chunk() -> None:
                                pl.semaphore_wait(
                                    dx_done_sem.at[static_source_ordinal, chunk],
                                    value=config.compute_blocks_per_send * hidden_tiles,
                                    decrement=False,
                                )
                                if static_source_ordinal == 0:
                                    pl.semaphore_signal(ready_sem.at[0, slot])
                                else:
                                    _signal_remote(ready_sem, source, (destination_ordinal, slot))

            for source_ordinal in range(ep_size):
                _compute_source(source_ordinal)

            if diagnostic.includes_combine and diagnostic.combines_on_dx_workers:
                _combine_routes(consumer, schedule.dx_programs)

        if diagnostic.includes_dx:
            pl.when((worker >= dx_program_start) & (worker < dw_program_start))(_compute_and_publish_dx)

        def _own_dw_tiles() -> None:
            owner = worker - dw_program_start

            @pl.loop(0, math.ceil(dw_tiles / schedule.dw_programs))
            def _tile_loop(iteration) -> None:
                tile = iteration * schedule.dw_programs + owner

                @pl.when(tile < dw_tiles)
                def _own_tile() -> None:
                    expert = tile // (hidden_tiles * output_tiles)
                    remainder = tile % (hidden_tiles * output_tiles)
                    hidden_tile = remainder // output_tiles
                    output_tile = remainder % output_tiles
                    hidden_start = hidden_tile * config.block_hidden
                    output_start = output_tile * config.block_output

                    def _acc_scope(acc_ref) -> None:
                        def _smem_scope(x_smem, dz_smem, barrier) -> None:
                            def _copy_compact_block(compact_block, stage) -> None:
                                valid_rows = compact_valid_rows_ref[expert, compact_block]

                                @pl.when(valid_rows > 0)
                                def _copy_live_block() -> None:
                                    row_start = compact_block * config.compute_m
                                    pl.semaphore_wait(
                                        compact_ready_sem.at[expert, compact_block, hidden_tile],
                                        value=1,
                                        decrement=False,
                                    )
                                    mgpu.copy_gmem_to_smem(
                                        x_expert_ref.at[
                                            expert,
                                            pl.ds(row_start, config.compute_m),
                                            pl.ds(hidden_start, config.block_hidden),
                                        ],
                                        x_smem.at[stage],
                                        barrier.at[stage],
                                    )
                                    mgpu.copy_gmem_to_smem(
                                        dz_ref.at[
                                            expert,
                                            pl.ds(row_start, config.compute_m),
                                            pl.ds(output_start, config.block_output),
                                        ],
                                        dz_smem.at[stage],
                                        barrier.at[stage],
                                    )

                            def _accumulate_compact_block(compact_block, stage) -> None:
                                valid_rows = compact_valid_rows_ref[expert, compact_block]

                                @pl.when(valid_rows > 0)
                                def _accumulate_live_block() -> None:
                                    mgpu.barrier_wait(barrier.at[stage])
                                    row = mgpu.layout_cast(
                                        lax.broadcasted_iota(
                                            jnp.int32,
                                            (config.compute_m, config.block_output),
                                            0,
                                        ),
                                        mgpu.Layout.WGMMA,
                                    )
                                    row_valid = (row < valid_rows).astype(jnp.float32)
                                    x_stage = x_smem.at[stage]
                                    dz_stage = dz_smem.at[stage]
                                    dz_stage[:, :] = (dz_stage[...].astype(jnp.float32) * row_valid).astype(dtype)
                                    mgpu.commit_smem()
                                    mgpu.wgmma(
                                        acc_ref,
                                        mgpu.transpose_ref(x_stage, (1, 0)),
                                        dz_stage,
                                    )

                            @pl.loop(0, math.ceil(schedule.compact_m_blocks / WGMMA_PIPELINE_STAGES))
                            def _compact_m_pair_loop(pair) -> None:
                                first_block = pair * WGMMA_PIPELINE_STAGES
                                _copy_compact_block(first_block, jnp.asarray(0, dtype=jnp.int32))
                                _accumulate_compact_block(first_block, jnp.asarray(0, dtype=jnp.int32))

                                second_block = first_block + 1

                                @pl.when(second_block < schedule.compact_m_blocks)
                                def _pipeline_second_block() -> None:
                                    _copy_compact_block(second_block, jnp.asarray(1, dtype=jnp.int32))
                                    _accumulate_compact_block(second_block, jnp.asarray(1, dtype=jnp.int32))

                                # Compact ownership can contain holes, so do not carry a
                                # conditional WGMMA across reuse of either pair stage.
                                mgpu.wgmma_wait(0)

                        pl.run_scoped(
                            _smem_scope,
                            x_smem=_wgmma_smem((WGMMA_PIPELINE_STAGES, config.compute_m, config.block_hidden), dtype),
                            dz_smem=_wgmma_smem((WGMMA_PIPELINE_STAGES, config.compute_m, config.block_output), dtype),
                            barrier=mgpu.Barrier(
                                num_arrivals=2,
                                num_barriers=WGMMA_PIPELINE_STAGES,
                            ),
                        )
                        dw_ref[
                            expert,
                            pl.ds(hidden_start, config.block_hidden),
                            pl.ds(output_start, config.block_output),
                        ] = acc_ref[...]

                    pl.run_scoped(
                        _acc_scope,
                        acc_ref=mgpu.ACC((config.block_hidden, config.block_output)),
                    )

            if diagnostic.serializes_combine:
                pl.semaphore_signal(dw_done_sem.at[0])

        if diagnostic.includes_dw:
            pl.when(worker >= dw_program_start)(_own_dw_tiles)

    x_expert_shape = (experts_per_rank, rows_per_expert_capacity, hidden_dim)
    return_shape = (ep_size, send_chunks_per_dst, config.send_m, hidden_dim)
    return mgpu.kernel(
        body,
        out_shape=(
            jax.ShapeDtypeStruct(x_expert_shape, dtype),
            jax.ShapeDtypeStruct(return_shape, dtype),
            jax.ShapeDtypeStruct((tokens_per_source, hidden_dim), jnp.float32),
            jax.ShapeDtypeStruct((experts_per_rank, hidden_dim, output_dim), jnp.float32),
        ),
        grid=(total_programs,),
        grid_names=("worker_program",),
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
    )


def _forward_config(config: SourcePushSemanticFusedW13BackwardConfig) -> SourcePushSemanticFusedW13Config:
    return replace(
        SourcePushSemanticFusedW13Config(),
        compute_m=config.compute_m,
        send_m=config.send_m,
        block_n=config.block_output,
        block_k=config.block_hidden,
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
    if hidden_dim % config.send_hidden_block or hidden_dim % config.block_hidden:
        raise ValueError(
            f"hidden_dim must be divisible by send_hidden_block and block_hidden, got {hidden_dim=} "
            f"{config.send_hidden_block=} {config.block_hidden=}"
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
    if rows_per_expert_capacity % config.compute_m:
        raise ValueError(
            f"rows_per_expert_capacity must be divisible by compute_m={config.compute_m}, "
            f"got {rows_per_expert_capacity}"
        )
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
