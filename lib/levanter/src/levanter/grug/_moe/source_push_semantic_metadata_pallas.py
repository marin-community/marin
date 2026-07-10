# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Pallas-oriented metadata builders for pair-flat source-push MoE routing.

This module is intentionally scoped to the slot-free semantic metadata contract.
The first implemented Pallas stage is the Sonic-style tiled histogram:
``selected_experts[S, T, K] -> tile_counts[S, tile, dst, local_expert]``.
The later local-rank/scatter stage can consume these counts plus tile prefixes
without changing the semantic plan API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as mgpu
from jaxtyping import Array, Float, Int

from levanter.grug._moe.ep_common import _clip_receiver_group_sizes
from levanter.grug._moe.source_push_plan import (
    INVALID_ASSIGNMENT_ID,
    SourcePushSemanticPlan,
    _exclusive_cumsum_jax,
    _source_push_semantic_reverse_route_from_metadata_jax,
)


DEFAULT_SOURCE_PUSH_SEMANTIC_METADATA_TILE_ASSIGNMENTS = 128


@dataclass(frozen=True, slots=True)
class SourcePushSemanticMetadataPallasBlockSizes:
    """Tile sizes for Pallas semantic metadata construction."""

    tile_assignments: int = DEFAULT_SOURCE_PUSH_SEMANTIC_METADATA_TILE_ASSIGNMENTS

    @classmethod
    def get_default(cls) -> "SourcePushSemanticMetadataPallasBlockSizes":
        return cls()


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SourcePushSemanticTileMetadata:
    """Tiled routing counts and prefix metadata for pair-flat semantic rows."""

    tile_counts: Int[Array, "S Tiles Dst E"]
    tile_pair_base: Int[Array, "S Tiles Dst E"]
    xcounts: Int[Array, "S Dst E"]
    pair_expert_base: Int[Array, "S Dst E"]
    rows_per_local_expert: Int[Array, "Dst E"]
    expert_base: Int[Array, "Dst E"]
    src_base_by_expert: Int[Array, "Dst S E"]
    routing_dropped_routes: Int[Array, ""]
    tokens_per_source: int = field(metadata={"static": True})
    topk: int = field(metadata={"static": True})
    tile_assignments: int = field(metadata={"static": True})


def source_push_semantic_tile_histogram_pallas_mgpu(
    selected_experts: Int[Array, "S T K"],
    *,
    ep_size: int,
    experts_per_rank: int,
    block_sizes: SourcePushSemanticMetadataPallasBlockSizes | None = None,
    interpret: bool = False,
) -> Int[Array, "S Tiles Dst E"]:
    """Count routes per source/tile/destination/local-expert using a Pallas kernel.

    This is the first stage of the planned metadata builder. It avoids a global
    sort by preserving assignment-tile order, making the remaining row-rank
    problem local to each tile.
    """

    block_sizes = SourcePushSemanticMetadataPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_tile_histogram_request(selected_experts, ep_size, experts_per_rank, block_sizes)
    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU source-push semantic metadata requires a GPU backend")

    source_count, tokens_per_source, topk = selected_experts.shape
    assignments_per_source = tokens_per_source * topk
    tile_count = math.ceil(assignments_per_source / block_sizes.tile_assignments)
    output_shape = jax.ShapeDtypeStruct((source_count, tile_count, ep_size, experts_per_rank), jnp.int32)
    kernel = _make_source_push_semantic_tile_histogram_kernel(
        tile_assignments=block_sizes.tile_assignments,
        topk=topk,
        assignments_per_source=assignments_per_source,
        experts_per_rank=experts_per_rank,
    )
    return pl.pallas_call(
        kernel,
        in_specs=(pl.BlockSpec(memory_space=mgpu.GMEM),),
        out_specs=pl.BlockSpec(memory_space=mgpu.GMEM),
        out_shape=output_shape,
        grid=(source_count, tile_count),
        interpret=interpret,
        name="source_push_semantic_tile_histogram_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_source_push_semantic_tile_histogram_cost_estimate(selected_experts, output_shape),
    )(selected_experts.astype(jnp.int32))


def source_push_semantic_tile_metadata_from_counts_jax(
    tile_counts: Int[Array, "S Tiles Dst E"],
    *,
    tokens_per_source: int,
    topk: int,
    capacity_factor: float = 1.25,
    tile_assignments: int = DEFAULT_SOURCE_PUSH_SEMANTIC_METADATA_TILE_ASSIGNMENTS,
) -> SourcePushSemanticTileMetadata:
    """Derive pair-flat offsets from Pallas tile histogram counts with JAX ops."""

    if tile_counts.ndim != 4:
        raise ValueError(f"tile_counts must have shape [S, tiles, D, E], got {tile_counts.shape}")
    if tokens_per_source <= 0:
        raise ValueError(f"tokens_per_source must be positive, got {tokens_per_source}")
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if capacity_factor <= 0:
        raise ValueError(f"capacity_factor must be positive, got {capacity_factor}")
    if tile_assignments <= 0:
        raise ValueError(f"tile_assignments must be positive, got {tile_assignments}")

    source_count, _tile_count, dst_count, experts_per_rank = tile_counts.shape
    if source_count != dst_count:
        raise ValueError(f"tile_counts source/destination dims must match, got {tile_counts.shape}")

    assignments_per_source = tokens_per_source * topk
    global_experts = dst_count * experts_per_rank
    group_sizes = jnp.sum(tile_counts, axis=1, dtype=jnp.int32).reshape(source_count, global_experts)
    receiver_capacity = max(experts_per_rank, int(math.ceil(capacity_factor * assignments_per_source)))
    clipped_group_sizes = _clip_receiver_group_sizes(
        group_sizes,
        local_expert_size=experts_per_rank,
        receiver_capacity=receiver_capacity,
    )
    xcounts = clipped_group_sizes.reshape(source_count, dst_count, experts_per_rank)
    pair_expert_base = _exclusive_cumsum_jax(xcounts, axis=2)
    rows_per_local_expert = jnp.sum(xcounts, axis=0, dtype=jnp.int32)
    expert_base = _exclusive_cumsum_jax(rows_per_local_expert, axis=1)
    src_base_by_expert = jnp.transpose(_exclusive_cumsum_jax(xcounts, axis=0), (1, 0, 2))

    tile_prefix = _exclusive_cumsum_jax(tile_counts, axis=1)
    tile_pair_base = jnp.minimum(tile_prefix, xcounts[:, None, :, :])
    routing_dropped_routes = assignments_per_source * source_count - jnp.sum(xcounts, dtype=jnp.int32)
    return SourcePushSemanticTileMetadata(
        tile_counts=tile_counts,
        tile_pair_base=tile_pair_base,
        xcounts=xcounts,
        pair_expert_base=pair_expert_base,
        rows_per_local_expert=rows_per_local_expert,
        expert_base=expert_base,
        src_base_by_expert=src_base_by_expert,
        routing_dropped_routes=routing_dropped_routes,
        tokens_per_source=tokens_per_source,
        topk=topk,
        tile_assignments=tile_assignments,
    )


def build_source_push_semantic_tile_metadata_pallas_mgpu(
    selected_experts: Int[Array, "S T K"],
    *,
    ep_size: int,
    experts_per_rank: int,
    capacity_factor: float = 1.25,
    block_sizes: SourcePushSemanticMetadataPallasBlockSizes | None = None,
    interpret: bool = False,
) -> SourcePushSemanticTileMetadata:
    """Build the implemented Pallas metadata stage plus JAX prefix offsets."""

    block_sizes = SourcePushSemanticMetadataPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    tile_counts = source_push_semantic_tile_histogram_pallas_mgpu(
        selected_experts,
        ep_size=ep_size,
        experts_per_rank=experts_per_rank,
        block_sizes=block_sizes,
        interpret=interpret,
    )
    return source_push_semantic_tile_metadata_from_counts_jax(
        tile_counts,
        tokens_per_source=selected_experts.shape[1],
        topk=selected_experts.shape[2],
        capacity_factor=capacity_factor,
        tile_assignments=block_sizes.tile_assignments,
    )


def build_source_push_semantic_plan_pallas_mgpu(
    selected_experts: Int[Array, "S T K"],
    route_weights: Float[Array, "S T K"],
    *,
    ep_size: int,
    experts_per_rank: int,
    rows_per_src_dst_capacity: int,
    rows_per_expert_capacity: int | None = None,
    capacity_factor: float = 1.25,
    block_sizes: SourcePushSemanticMetadataPallasBlockSizes | None = None,
    interpret: bool = False,
) -> SourcePushSemanticPlan:
    """Build pair-flat semantic metadata with Pallas histogram and scatter stages.

    After pair clipping, ``rows_per_expert_capacity`` optionally caps each
    destination expert across sources, with earlier sources taking precedence.
    """

    if rows_per_expert_capacity is not None and rows_per_expert_capacity <= 0:
        raise ValueError(f"rows_per_expert_capacity must be positive, got {rows_per_expert_capacity}")

    unclipped_tile_metadata = build_source_push_semantic_tile_metadata_pallas_mgpu(
        selected_experts,
        ep_size=ep_size,
        experts_per_rank=experts_per_rank,
        capacity_factor=capacity_factor,
        block_sizes=block_sizes,
        interpret=interpret,
    )
    clipped_pair_counts = jnp.sum(unclipped_tile_metadata.xcounts, axis=2, dtype=jnp.int32)
    stored_pair_counts = jnp.minimum(clipped_pair_counts, rows_per_src_dst_capacity)
    metadata_overflow_routes = jnp.sum(clipped_pair_counts - stored_pair_counts, dtype=jnp.int32)
    tile_metadata = source_push_semantic_tile_metadata_apply_pair_capacity_jax(
        unclipped_tile_metadata,
        rows_per_src_dst_capacity=rows_per_src_dst_capacity,
    )
    if rows_per_expert_capacity is not None:
        pair_clipped_counts = tile_metadata.xcounts
        source_expert_base = _exclusive_cumsum_jax(pair_clipped_counts, axis=0)
        expert_clipped_counts = jnp.clip(
            rows_per_expert_capacity - source_expert_base,
            0,
            pair_clipped_counts,
        )
        metadata_overflow_routes += jnp.sum(pair_clipped_counts - expert_clipped_counts, dtype=jnp.int32)
        tile_metadata = _source_push_semantic_tile_metadata_with_counts_jax(
            tile_metadata,
            expert_clipped_counts,
        )
    assignment_ids, pair_weights = source_push_semantic_row_scatter_pallas_mgpu(
        selected_experts,
        route_weights,
        tile_metadata,
        rows_per_src_dst_capacity=rows_per_src_dst_capacity,
        interpret=interpret,
    )
    valid_mask = assignment_ids != INVALID_ASSIGNMENT_ID
    safe_assignment_ids = jnp.maximum(assignment_ids, 0)
    token_ids = jnp.where(valid_mask, safe_assignment_ids // selected_experts.shape[2], INVALID_ASSIGNMENT_ID)
    route_slots = jnp.where(valid_mask, safe_assignment_ids % selected_experts.shape[2], INVALID_ASSIGNMENT_ID)
    reverse_route = _source_push_semantic_reverse_route_from_metadata_jax(
        assignment_ids=assignment_ids,
        token_ids=token_ids,
        route_slots=route_slots,
        valid_mask=valid_mask,
        xcounts=tile_metadata.xcounts,
        pair_expert_base=tile_metadata.pair_expert_base,
        src_base_by_expert=tile_metadata.src_base_by_expert,
        tokens_per_source=selected_experts.shape[1],
        topk=selected_experts.shape[2],
    )
    dropped_routes = tile_metadata.routing_dropped_routes + metadata_overflow_routes
    return SourcePushSemanticPlan(
        assignment_ids=assignment_ids,
        token_ids=token_ids,
        route_slots=route_slots,
        route_weights=pair_weights,
        valid_mask=valid_mask,
        xcounts=tile_metadata.xcounts,
        pair_expert_base=tile_metadata.pair_expert_base,
        rows_per_local_expert=tile_metadata.rows_per_local_expert,
        expert_base=tile_metadata.expert_base,
        src_base_by_expert=tile_metadata.src_base_by_expert,
        reverse_route=reverse_route,
        routing_dropped_routes=tile_metadata.routing_dropped_routes,
        metadata_overflow_routes=metadata_overflow_routes,
        dropped_routes=dropped_routes,
        tokens_per_source=selected_experts.shape[1],
        topk=selected_experts.shape[2],
    )


def source_push_semantic_tile_metadata_apply_pair_capacity_jax(
    tile_metadata: SourcePushSemanticTileMetadata,
    *,
    rows_per_src_dst_capacity: int,
) -> SourcePushSemanticTileMetadata:
    """Clip per-expert counts to a fixed pair-flat row capacity."""

    if rows_per_src_dst_capacity <= 0:
        raise ValueError(f"rows_per_src_dst_capacity must be positive, got {rows_per_src_dst_capacity}")

    unclipped_counts = tile_metadata.xcounts
    pair_expert_base_unclipped = _exclusive_cumsum_jax(unclipped_counts, axis=2)
    stored_counts = jnp.clip(rows_per_src_dst_capacity - pair_expert_base_unclipped, 0, unclipped_counts)
    return _source_push_semantic_tile_metadata_with_counts_jax(tile_metadata, stored_counts)


def _source_push_semantic_tile_metadata_with_counts_jax(
    tile_metadata: SourcePushSemanticTileMetadata,
    counts: Int[Array, "S Dst E"],
) -> SourcePushSemanticTileMetadata:
    pair_expert_base = _exclusive_cumsum_jax(counts, axis=2)
    rows_per_local_expert = jnp.sum(counts, axis=0, dtype=jnp.int32)
    expert_base = _exclusive_cumsum_jax(rows_per_local_expert, axis=1)
    src_base_by_expert = jnp.transpose(_exclusive_cumsum_jax(counts, axis=0), (1, 0, 2))
    tile_prefix = _exclusive_cumsum_jax(tile_metadata.tile_counts, axis=1)
    tile_pair_base = jnp.minimum(tile_prefix, counts[:, None, :, :])
    return replace(
        tile_metadata,
        tile_pair_base=tile_pair_base,
        xcounts=counts,
        pair_expert_base=pair_expert_base,
        rows_per_local_expert=rows_per_local_expert,
        expert_base=expert_base,
        src_base_by_expert=src_base_by_expert,
    )


def source_push_semantic_row_scatter_pallas_mgpu(
    selected_experts: Int[Array, "S T K"],
    route_weights: Float[Array, "S T K"],
    tile_metadata: SourcePushSemanticTileMetadata,
    *,
    rows_per_src_dst_capacity: int,
    interpret: bool = False,
) -> tuple[Int[Array, "S Dst R"], Float[Array, "S Dst R"]]:
    """Scatter source assignments into pair-flat rows using tile-local ranks."""

    _validate_row_scatter_request(selected_experts, route_weights, tile_metadata, rows_per_src_dst_capacity)
    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU source-push semantic row scatter requires a GPU backend")

    source_count, tokens_per_source, topk = selected_experts.shape
    assignments_per_source = tokens_per_source * topk
    dst_count = tile_metadata.xcounts.shape[1]
    output_shape = (
        jax.ShapeDtypeStruct((source_count, dst_count, rows_per_src_dst_capacity), jnp.int32),
        jax.ShapeDtypeStruct((source_count, dst_count, rows_per_src_dst_capacity), route_weights.dtype),
    )
    assignment_init = jnp.full(output_shape[0].shape, INVALID_ASSIGNMENT_ID, dtype=jnp.int32)
    weight_init = jnp.zeros(output_shape[1].shape, dtype=route_weights.dtype)
    kernel = _make_source_push_semantic_row_scatter_kernel(
        tile_assignments=tile_metadata.tile_assignments,
        topk=topk,
        assignments_per_source=assignments_per_source,
        experts_per_rank=tile_metadata.xcounts.shape[2],
    )
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        kernel,
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=(gmem_spec, gmem_spec),
        out_shape=output_shape,
        grid=(source_count, tile_metadata.tile_counts.shape[1]),
        input_output_aliases={5: 0, 6: 1},
        interpret=interpret,
        name="source_push_semantic_row_scatter_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_source_push_semantic_row_scatter_cost_estimate(
            selected_experts,
            route_weights,
            tile_metadata,
            output_shape,
        ),
    )(
        selected_experts.astype(jnp.int32),
        route_weights,
        tile_metadata.tile_pair_base,
        tile_metadata.xcounts,
        tile_metadata.pair_expert_base,
        assignment_init,
        weight_init,
    )


def _make_source_push_semantic_tile_histogram_kernel(
    *,
    tile_assignments: int,
    topk: int,
    assignments_per_source: int,
    experts_per_rank: int,
):
    def kernel(
        selected_experts_ref: Int[pl.Ref, "S T K"],
        tile_counts_ref: Int[pl.Ref, "S Tiles Dst E"],
    ) -> None:
        src = pl.program_id(0)
        tile = pl.program_id(1)
        tile_start = tile * tile_assignments

        for dst in range(tile_counts_ref.shape[2]):
            for local_expert in range(experts_per_rank):
                tile_counts_ref[src, tile, dst, local_expert] = jnp.int32(0)

        for row in range(tile_assignments):
            assignment = tile_start + row
            valid = assignment < assignments_per_source
            safe_assignment = jnp.minimum(assignment, assignments_per_source - 1)
            token = safe_assignment // topk
            route_slot = safe_assignment - token * topk
            global_expert = selected_experts_ref[src, token, route_slot]
            dst = global_expert // experts_per_rank
            local_expert = global_expert - dst * experts_per_rank
            current = tile_counts_ref[src, tile, dst, local_expert]
            tile_counts_ref[src, tile, dst, local_expert] = current + valid.astype(jnp.int32)

    return kernel


def _make_source_push_semantic_row_scatter_kernel(
    *,
    tile_assignments: int,
    topk: int,
    assignments_per_source: int,
    experts_per_rank: int,
):
    def kernel(
        selected_experts_ref: Int[pl.Ref, "S T K"],
        route_weights_ref: Float[pl.Ref, "S T K"],
        tile_pair_base_ref: Int[pl.Ref, "S Tiles Dst E"],
        xcounts_ref: Int[pl.Ref, "S Dst E"],
        pair_expert_base_ref: Int[pl.Ref, "S Dst E"],
        _assignment_ids_init_ref: Int[pl.Ref, "S Dst R"],
        _pair_weights_init_ref: Float[pl.Ref, "S Dst R"],
        assignment_ids_ref: Int[pl.Ref, "S Dst R"],
        pair_weights_ref: Float[pl.Ref, "S Dst R"],
    ) -> None:
        src = pl.program_id(0)
        tile = pl.program_id(1)
        tile_start = tile * tile_assignments

        for row in range(tile_assignments):
            assignment = tile_start + row
            valid_assignment = assignment < assignments_per_source
            safe_assignment = jnp.minimum(assignment, assignments_per_source - 1)
            token = safe_assignment // topk
            route_slot = safe_assignment - token * topk
            global_expert = selected_experts_ref[src, token, route_slot]
            dst = global_expert // experts_per_rank
            local_expert = global_expert - dst * experts_per_rank

            tile_local_rank = jnp.int32(0)
            for previous_row in range(tile_assignments):
                previous_assignment = tile_start + previous_row
                previous_valid = (previous_row < row) & (previous_assignment < assignments_per_source)
                safe_previous_assignment = jnp.minimum(previous_assignment, assignments_per_source - 1)
                previous_token = safe_previous_assignment // topk
                previous_route_slot = safe_previous_assignment - previous_token * topk
                previous_global_expert = selected_experts_ref[src, previous_token, previous_route_slot]
                tile_local_rank += (previous_valid & (previous_global_expert == global_expert)).astype(jnp.int32)

            local_row = tile_pair_base_ref[src, tile, dst, local_expert] + tile_local_rank
            keep = valid_assignment & (local_row < xcounts_ref[src, dst, local_expert])
            pair_row = pair_expert_base_ref[src, dst, local_expert] + local_row
            safe_pair_row = jnp.minimum(pair_row, assignment_ids_ref.shape[2] - 1)
            assignment_ids_ref[src, dst, safe_pair_row] = jnp.where(
                keep,
                assignment,
                assignment_ids_ref[src, dst, safe_pair_row],
            )
            pair_weights_ref[src, dst, safe_pair_row] = jnp.where(
                keep,
                route_weights_ref[src, token, route_slot],
                pair_weights_ref[src, dst, safe_pair_row],
            )

    return kernel


def _source_push_semantic_tile_histogram_cost_estimate(
    selected_experts: Array,
    output_shape: jax.ShapeDtypeStruct,
) -> pl.CostEstimate:
    input_bytes = math.prod(selected_experts.shape) * selected_experts.dtype.itemsize
    output_bytes = math.prod(output_shape.shape) * jnp.dtype(output_shape.dtype).itemsize
    return pl.CostEstimate(
        flops=0,
        transcendentals=0,
        bytes_accessed=input_bytes + output_bytes,
        remote_bytes_transferred=0,
    )


def _source_push_semantic_row_scatter_cost_estimate(
    selected_experts: Array,
    route_weights: Array,
    tile_metadata: SourcePushSemanticTileMetadata,
    output_shape: tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct],
) -> pl.CostEstimate:
    input_bytes = (
        math.prod(selected_experts.shape) * selected_experts.dtype.itemsize
        + math.prod(route_weights.shape) * route_weights.dtype.itemsize
        + math.prod(tile_metadata.tile_pair_base.shape) * tile_metadata.tile_pair_base.dtype.itemsize
        + math.prod(tile_metadata.xcounts.shape) * tile_metadata.xcounts.dtype.itemsize
        + math.prod(tile_metadata.pair_expert_base.shape) * tile_metadata.pair_expert_base.dtype.itemsize
    )
    output_bytes = sum(math.prod(shape.shape) * jnp.dtype(shape.dtype).itemsize for shape in output_shape)
    return pl.CostEstimate(
        flops=0,
        transcendentals=0,
        bytes_accessed=input_bytes + output_bytes,
        remote_bytes_transferred=0,
    )


def _validate_tile_histogram_request(
    selected_experts: Array,
    ep_size: int,
    experts_per_rank: int,
    block_sizes: SourcePushSemanticMetadataPallasBlockSizes,
) -> None:
    if selected_experts.ndim != 3:
        raise ValueError(f"selected_experts must have shape [S, T, K], got {selected_experts.shape}")
    if ep_size <= 0:
        raise ValueError(f"ep_size must be positive, got {ep_size}")
    if experts_per_rank <= 0:
        raise ValueError(f"experts_per_rank must be positive, got {experts_per_rank}")
    if selected_experts.shape[0] != ep_size:
        raise ValueError(f"selected_experts leading dim must match ep_size={ep_size}, got {selected_experts.shape[0]}")
    if block_sizes.tile_assignments <= 0:
        raise ValueError(f"tile_assignments must be positive, got {block_sizes.tile_assignments}")


def _validate_row_scatter_request(
    selected_experts: Array,
    route_weights: Array,
    tile_metadata: SourcePushSemanticTileMetadata,
    rows_per_src_dst_capacity: int,
) -> None:
    if selected_experts.ndim != 3:
        raise ValueError(f"selected_experts must have shape [S, T, K], got {selected_experts.shape}")
    if route_weights.shape != selected_experts.shape:
        raise ValueError(
            f"route_weights shape {route_weights.shape} must match selected_experts shape {selected_experts.shape}"
        )
    if rows_per_src_dst_capacity <= 0:
        raise ValueError(f"rows_per_src_dst_capacity must be positive, got {rows_per_src_dst_capacity}")
    source_count = selected_experts.shape[0]
    if tile_metadata.xcounts.ndim != 3:
        raise ValueError(f"xcounts must have shape [S, D, E], got {tile_metadata.xcounts.shape}")
    if tile_metadata.xcounts.shape[0] != source_count:
        raise ValueError(
            f"xcounts source dim {tile_metadata.xcounts.shape[0]} must match selected_experts source dim {source_count}"
        )
    if tile_metadata.tile_pair_base.shape != tile_metadata.tile_counts.shape:
        raise ValueError(
            f"tile_pair_base shape {tile_metadata.tile_pair_base.shape} must match "
            f"tile_counts shape {tile_metadata.tile_counts.shape}"
        )
    if tile_metadata.tile_counts.shape[0] != source_count:
        raise ValueError(
            f"tile_counts source dim {tile_metadata.tile_counts.shape[0]} must match "
            f"selected_experts source dim {source_count}"
        )
