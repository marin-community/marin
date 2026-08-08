# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Executable CPU index-plane plans for routed relations."""

from dataclasses import dataclass

import numpy as np

from tile_lifetime.expert_parallel_plan import ExpertParallelPlan
from tile_lifetime.ir import DType
from tile_lifetime.tile_program import (
    TileOp,
    TilePrimitive,
    TileProgram,
    TileProgramStage,
    optimize_tile_program,
)


class RelationPlanError(ValueError):
    """Structured rejection of a relation or its bounded capacity."""

    def __init__(self, reasons: tuple[str, ...]):
        self.reasons = reasons
        super().__init__("; ".join(reasons))


@dataclass(frozen=True)
class OrderedRelationFoldProgram:
    """Compiler-owned ordered reduction over returned relation partitions."""

    partition_count: int
    accumulation_dtype: DType
    output_dtype: DType
    order: str
    tile_program: TileProgram


def compile_ordered_relation_fold(
    *,
    partition_count: int,
    accumulation_dtype: DType,
    output_dtype: DType,
) -> OrderedRelationFoldProgram:
    """Compile base-plus-partials addition from generic tile primitives."""
    if partition_count <= 0:
        raise ValueError("relation fold requires at least one partition")
    operations: list[TileOp] = [
        TileOp(
            TilePrimitive.LOAD_STATE,
            TileProgramStage.FINALIZATION,
            ("base",),
            ("accumulator.0",),
            (("dtype", accumulation_dtype.value),),
        )
    ]
    accumulator = "accumulator.0"
    for partition in range(partition_count):
        partial = f"partial.{partition}"
        loaded = f"partial.{partition}.loaded"
        next_accumulator = f"accumulator.{partition + 1}"
        operations.extend(
            (
                TileOp(
                    TilePrimitive.LOAD_TILE,
                    TileProgramStage.FINALIZATION,
                    (partial,),
                    (loaded,),
                    (("partition", str(partition)),),
                ),
                TileOp(
                    TilePrimitive.ADD,
                    TileProgramStage.FINALIZATION,
                    (accumulator, loaded),
                    (next_accumulator,),
                    (("rounding", "rn"),),
                ),
            )
        )
        accumulator = next_accumulator
    operations.extend(
        (
            TileOp(
                TilePrimitive.CONVERT,
                TileProgramStage.FINALIZATION,
                (accumulator,),
                ("output.converted",),
                (("dtype", output_dtype.value), ("rounding", "rn")),
            ),
            TileOp(
                TilePrimitive.STORE,
                TileProgramStage.FINALIZATION,
                ("output.converted",),
                ("output",),
            ),
        )
    )
    program = optimize_tile_program(tuple(operations), required_outputs=("output",))
    return OrderedRelationFoldProgram(
        partition_count=partition_count,
        accumulation_dtype=accumulation_dtype,
        output_dtype=output_dtype,
        order="partition ascending, explicit round-to-nearest FP32 add",
        tile_program=program,
    )


@dataclass(frozen=True)
class RelationPlan:
    """Inspectable source-to-destination permutation with padded destination groups."""

    source_item_count: int
    route_slots: int
    destination_count: int
    destination_rank_count: int
    merge_order: str
    edge_valid: np.ndarray
    source_item: np.ndarray
    route_slot: np.ndarray
    destination_item: np.ndarray
    destination_rank: np.ndarray
    destination_local_item: np.ndarray
    weight: np.ndarray
    route_to_destination_row: np.ndarray
    destination_row_to_route: np.ndarray
    row_source_item: np.ndarray
    row_route_slot: np.ndarray
    row_destination_item: np.ndarray
    row_destination_rank: np.ndarray
    row_destination_local_item: np.ndarray
    row_weight: np.ndarray
    row_valid: np.ndarray
    row_padding: np.ndarray
    group_destination_item: np.ndarray
    group_destination_rank: np.ndarray
    group_destination_local_item: np.ndarray
    group_count: np.ndarray
    group_padded_count: np.ndarray
    group_offset: np.ndarray
    exchange_source_item: np.ndarray
    exchange_destination_rank: np.ndarray
    route_to_exchange_row: np.ndarray

    @property
    def route_count(self) -> int:
        """Number of unpadded source-route assignments."""
        return int(np.count_nonzero(self.edge_valid))

    @property
    def destination_row_count(self) -> int:
        """Number of padded rows across all destination groups."""
        return int(self.destination_row_to_route.shape[0])

    @property
    def destination_edge_offsets(self) -> np.ndarray:
        """Offsets into the compact destination-major edge arrays."""
        return np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(self.group_count, dtype=np.int32)))

    @property
    def grouped_route_indices(self) -> np.ndarray:
        """Valid rectangular route indices in stable destination-major order."""
        return self.destination_row_to_route[self.row_valid]

    @property
    def grouped_source_item(self) -> np.ndarray:
        """Source identity for every compact destination-major edge."""
        return self.source_item[self.grouped_route_indices]

    @property
    def grouped_route_slot(self) -> np.ndarray:
        """Source-local slot for every compact destination-major edge."""
        return self.route_slot[self.grouped_route_indices]

    def dispatch(self, payload: np.ndarray) -> np.ndarray:
        """Gather source payload into padded destination-group order."""
        _validate_payload(payload, self.source_item_count, "source payload")
        output = np.zeros((self.destination_row_count, *payload.shape[1:]), dtype=payload.dtype)
        output[self.row_valid] = payload[self.row_source_item[self.row_valid]]
        return output

    def inverse_dispatch(self, destination_payload: np.ndarray, *, fill_value: int | float = 0) -> np.ndarray:
        """Restore destination-row payload to source-item, route-slot order."""
        _validate_payload(destination_payload, self.destination_row_count, "destination payload")
        flat = np.full(
            (self.source_item_count * self.route_slots, *destination_payload.shape[1:]),
            fill_value,
            dtype=destination_payload.dtype,
        )
        valid_routes = np.flatnonzero(self.edge_valid.reshape(-1))
        flat[valid_routes] = destination_payload[self.route_to_destination_row[valid_routes]]
        return flat.reshape(self.source_item_count, self.route_slots, *destination_payload.shape[1:])

    def weighted_merge(self, destination_payload: np.ndarray) -> np.ndarray:
        """Restore source order, multiply by route weights, and reduce route slots."""
        restored = self.inverse_dispatch(destination_payload)
        accumulation_dtype = np.result_type(restored.dtype, self.weight.dtype, np.float32)
        output = np.zeros((self.source_item_count, *restored.shape[2:]), dtype=accumulation_dtype)
        for source_item in range(self.source_item_count):
            for route_slot in range(self.route_slots):
                if not self.edge_valid[source_item, route_slot]:
                    continue
                output[source_item] += (
                    restored[source_item, route_slot].astype(accumulation_dtype) * self.weight[source_item, route_slot]
                )
        return output

    def dispatch_coalesced(self, payload: np.ndarray) -> np.ndarray:
        """Gather one payload row per distinct (source item, destination rank)."""
        _validate_payload(payload, self.source_item_count, "source payload")
        return payload[self.exchange_source_item]

    def expand_coalesced(self, exchange_payload: np.ndarray) -> np.ndarray:
        """Expand coalesced transport rows into padded destination-group rows."""
        _validate_payload(exchange_payload, self.exchange_source_item.shape[0], "coalesced exchange payload")
        output = np.zeros((self.destination_row_count, *exchange_payload.shape[1:]), dtype=exchange_payload.dtype)
        valid_routes = self.destination_row_to_route[self.row_valid]
        output[self.row_valid] = exchange_payload[self.route_to_exchange_row[valid_routes]]
        return output

    def dump(self) -> str:
        """Render a compact stable description for diagnostics and snapshots."""
        lines = [
            (
                f"RelationPlan sources={self.source_item_count} slots={self.route_slots} "
                f"routes={self.route_count} destination_rows={self.destination_row_count} "
                f"exchange_rows={self.exchange_source_item.shape[0]} merge_order={self.merge_order}"
            )
        ]
        for group_index in range(self.group_count.shape[0]):
            lines.append(
                f"  rank={int(self.group_destination_rank[group_index])} "
                f"item={int(self.group_destination_local_item[group_index])} "
                f"count={int(self.group_count[group_index])} "
                f"padded={int(self.group_padded_count[group_index])} "
                f"offset={int(self.group_offset[group_index])}"
            )
        return "\n".join(lines)


def build_relation_plan(
    destination_indices: np.ndarray,
    weights: np.ndarray,
    *,
    edge_valid: np.ndarray | None = None,
    destination_rank_by_item: np.ndarray,
    destination_local_item_by_item: np.ndarray,
    padding_quantum: int,
    max_routes_per_rank: int | None = None,
    max_padded_rows_per_rank: int | None = None,
) -> RelationPlan:
    """Build a stable grouped relation from source indices and weights."""
    destination_indices = np.asarray(destination_indices)
    weights = np.asarray(weights, dtype=np.float32)
    if edge_valid is None:
        edge_valid = np.ones(destination_indices.shape, dtype=np.bool_)
    else:
        edge_valid = np.asarray(edge_valid, dtype=np.bool_)
    destination_rank_by_item = np.asarray(destination_rank_by_item)
    destination_local_item_by_item = np.asarray(destination_local_item_by_item)
    reasons = _relation_reasons(
        destination_indices,
        weights,
        edge_valid,
        destination_rank_by_item,
        destination_local_item_by_item,
        padding_quantum=padding_quantum,
        max_routes_per_rank=max_routes_per_rank,
        max_padded_rows_per_rank=max_padded_rows_per_rank,
    )
    if reasons:
        raise RelationPlanError(reasons)

    source_item_count, route_slots = destination_indices.shape
    source_item = np.repeat(np.arange(source_item_count, dtype=np.int32), route_slots)
    route_slot = np.tile(np.arange(route_slots, dtype=np.int32), source_item_count)
    destination_item = destination_indices.reshape(-1).astype(np.int32, copy=False)
    flat_edge_valid = edge_valid.reshape(-1)
    valid_route_indices = np.flatnonzero(flat_edge_valid)
    destination_rank = np.full(destination_item.shape, -1, dtype=np.int32)
    destination_local_item = np.full(destination_item.shape, -1, dtype=np.int32)
    destination_rank[valid_route_indices] = destination_rank_by_item[destination_item[valid_route_indices]]
    destination_local_item[valid_route_indices] = destination_local_item_by_item[destination_item[valid_route_indices]]
    flat_weight = weights.reshape(-1)

    group_order = np.lexsort((destination_local_item_by_item, destination_rank_by_item))
    group_destination_rank = destination_rank_by_item[group_order].astype(np.int32, copy=False)
    group_destination_local_item = destination_local_item_by_item[group_order].astype(np.int32, copy=False)
    group_count = np.zeros(group_order.shape[0], dtype=np.int32)
    group_index_by_destination = np.empty(group_order.shape[0], dtype=np.int32)
    group_index_by_destination[group_order] = np.arange(group_order.shape[0], dtype=np.int32)
    np.add.at(group_count, group_index_by_destination[destination_item[valid_route_indices]], 1)
    group_padded_count = _round_up(group_count, padding_quantum)
    group_offset = np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(group_padded_count[:-1], dtype=np.int32)))

    row_count = int(np.sum(group_padded_count, dtype=np.int64))
    destination_row_to_route = np.full(row_count, -1, dtype=np.int32)
    route_to_destination_row = np.full(source_item.shape[0], -1, dtype=np.int32)
    valid_route_groups = group_index_by_destination[destination_item[valid_route_indices]]
    stable_group_order = np.argsort(valid_route_groups, kind="stable")
    grouped_routes = valid_route_indices[stable_group_order]
    grouped_route_groups = valid_route_groups[stable_group_order]
    compact_group_offset = np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(group_count[:-1], dtype=np.int32)))
    within_group = np.arange(valid_route_indices.shape[0], dtype=np.int32) - np.repeat(
        compact_group_offset,
        group_count,
    )
    grouped_destination_rows = group_offset[grouped_route_groups] + within_group
    destination_row_to_route[grouped_destination_rows] = grouped_routes
    route_to_destination_row[grouped_routes] = grouped_destination_rows

    row_valid = destination_row_to_route >= 0
    row_source_item = np.full(row_count, -1, dtype=np.int32)
    row_route_slot = np.full(row_count, -1, dtype=np.int32)
    row_destination_item = np.full(row_count, -1, dtype=np.int32)
    row_destination_rank = np.repeat(group_destination_rank, group_padded_count).astype(np.int32, copy=False)
    row_destination_local_item = np.repeat(group_destination_local_item, group_padded_count).astype(np.int32, copy=False)
    row_weight = np.zeros(row_count, dtype=np.float32)
    valid_routes = destination_row_to_route[row_valid]
    row_source_item[row_valid] = source_item[valid_routes]
    row_route_slot[row_valid] = route_slot[valid_routes]
    row_destination_item[row_valid] = destination_item[valid_routes]
    row_weight[row_valid] = flat_weight[valid_routes]

    exchange_pairs = np.stack((destination_rank[valid_route_indices], source_item[valid_route_indices]), axis=1)
    if exchange_pairs.shape[0]:
        unique_pairs, exchange_rows = np.unique(exchange_pairs, axis=0, return_inverse=True)
        route_to_exchange_row = np.full(source_item.shape[0], -1, dtype=np.int32)
        route_to_exchange_row[valid_route_indices] = exchange_rows.astype(np.int32, copy=False)
    else:
        unique_pairs = np.empty((0, 2), dtype=np.int32)
        route_to_exchange_row = np.full(source_item.shape[0], -1, dtype=np.int32)

    _check_capacity(
        destination_rank[valid_route_indices],
        row_destination_rank,
        row_valid,
        rank_count=int(np.max(destination_rank_by_item)) + 1,
        max_routes_per_rank=max_routes_per_rank,
        max_padded_rows_per_rank=max_padded_rows_per_rank,
    )
    return RelationPlan(
        source_item_count=source_item_count,
        route_slots=route_slots,
        destination_count=destination_rank_by_item.shape[0],
        destination_rank_count=int(np.max(destination_rank_by_item)) + 1,
        merge_order="source_item ascending, then route_slot ascending, FP32 accumulation",
        edge_valid=edge_valid,
        source_item=source_item,
        route_slot=route_slot,
        destination_item=destination_item,
        destination_rank=destination_rank,
        destination_local_item=destination_local_item,
        weight=weights,
        route_to_destination_row=route_to_destination_row,
        destination_row_to_route=destination_row_to_route,
        row_source_item=row_source_item,
        row_route_slot=row_route_slot,
        row_destination_item=row_destination_item,
        row_destination_rank=row_destination_rank,
        row_destination_local_item=row_destination_local_item,
        row_weight=row_weight,
        row_valid=row_valid,
        row_padding=~row_valid,
        group_destination_item=group_order.astype(np.int32, copy=False),
        group_destination_rank=group_destination_rank,
        group_destination_local_item=group_destination_local_item,
        group_count=group_count,
        group_padded_count=group_padded_count,
        group_offset=group_offset,
        exchange_source_item=unique_pairs[:, 1].astype(np.int32, copy=False),
        exchange_destination_rank=unique_pairs[:, 0].astype(np.int32, copy=False),
        route_to_exchange_row=route_to_exchange_row,
    )


def build_expert_parallel_relation_plan(
    plan: ExpertParallelPlan,
    destination_indices: np.ndarray,
    weights: np.ndarray,
) -> RelationPlan:
    """Instantiate the generic index plane from an expert-parallel plan contract."""
    global_experts = plan.ownership.global_expert_count
    local_experts = plan.ownership.local_expert_count
    destination_items = np.arange(global_experts, dtype=np.int32)
    return build_relation_plan(
        destination_indices,
        weights,
        destination_rank_by_item=destination_items // local_experts,
        destination_local_item_by_item=destination_items % local_experts,
        padding_quantum=plan.segments.padding_quantum,
        max_routes_per_rank=plan.capacity.receiver_assignment_capacity,
        max_padded_rows_per_rank=plan.capacity.padded_local_capacity,
    )


def build_partitioned_merge_rows(
    returned_source_items: np.ndarray,
    partition_counts: np.ndarray,
    *,
    source_item_count: int,
) -> np.ndarray:
    """Map one unique partial per partition/source pair to its returned row.

    The resulting ``[partition, source_item]`` table is a physical input to an
    ordered compiler fold. Missing contributions are encoded as ``-1``. This
    function only plans indices; it does not perform a semantic reduction.
    """
    returned_source_items = np.asarray(returned_source_items)
    partition_counts = np.asarray(partition_counts)
    reasons: list[str] = []
    if returned_source_items.ndim != 1:
        reasons.append("returned source items must be one-dimensional")
    if partition_counts.ndim != 1:
        reasons.append("partition counts must be one-dimensional")
    if source_item_count <= 0:
        reasons.append("source item count must be positive")
    if reasons:
        raise RelationPlanError(tuple(reasons))
    if not np.issubdtype(returned_source_items.dtype, np.integer):
        reasons.append("returned source items must have integer dtype")
    if not np.issubdtype(partition_counts.dtype, np.integer):
        reasons.append("partition counts must have integer dtype")
    if np.any(partition_counts < 0):
        reasons.append("partition counts must be non-negative")
    if int(np.sum(partition_counts, dtype=np.int64)) != returned_source_items.shape[0]:
        reasons.append("partition counts do not cover the returned source-item rows")
    if np.any(returned_source_items < 0) or np.any(returned_source_items >= source_item_count):
        reasons.append("returned source item is outside the source domain")
    if reasons:
        raise RelationPlanError(tuple(reasons))

    rows = np.full((partition_counts.shape[0], source_item_count), -1, dtype=np.int64)
    offset = 0
    for partition, count_value in enumerate(partition_counts):
        count = int(count_value)
        source_items = returned_source_items[offset : offset + count]
        if np.unique(source_items).shape[0] != count:
            raise RelationPlanError((f"partition {partition} contains duplicate source items",))
        rows[partition, source_items] = np.arange(offset, offset + count, dtype=np.int64)
        offset += count
    return rows


def _relation_reasons(
    destination_indices: np.ndarray,
    weights: np.ndarray,
    edge_valid: np.ndarray,
    destination_rank_by_item: np.ndarray,
    destination_local_item_by_item: np.ndarray,
    *,
    padding_quantum: int,
    max_routes_per_rank: int | None,
    max_padded_rows_per_rank: int | None,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if destination_indices.ndim != 2:
        reasons.append("destination indices must have shape [source_item, route_slot]")
    if weights.shape != destination_indices.shape:
        reasons.append("weights must match destination-index shape")
    if edge_valid.shape != destination_indices.shape:
        reasons.append("edge validity must match destination-index shape")
    if destination_rank_by_item.ndim != 1 or destination_local_item_by_item.shape != destination_rank_by_item.shape:
        reasons.append("destination ownership arrays must be one-dimensional and identically shaped")
    if padding_quantum <= 0:
        reasons.append("padding quantum must be positive")
    if max_routes_per_rank is not None and max_routes_per_rank <= 0:
        reasons.append("maximum routes per rank must be positive")
    if max_padded_rows_per_rank is not None and max_padded_rows_per_rank <= 0:
        reasons.append("maximum padded rows per rank must be positive")
    if reasons:
        return tuple(reasons)
    if not np.issubdtype(destination_indices.dtype, np.integer):
        reasons.append("destination indices must have integer dtype")
    if destination_rank_by_item.shape[0] == 0:
        reasons.append("destination ownership mapping must not be empty")
    valid_destinations = destination_indices[edge_valid]
    if np.any(valid_destinations < 0) or np.any(valid_destinations >= destination_rank_by_item.shape[0]):
        reasons.append("destination index is outside the ownership mapping")
    if np.any(destination_rank_by_item < 0) or np.any(destination_local_item_by_item < 0):
        reasons.append("destination ownership coordinates must be non-negative")
    ownership_pairs = np.stack((destination_rank_by_item, destination_local_item_by_item), axis=1)
    if np.unique(ownership_pairs, axis=0).shape[0] != ownership_pairs.shape[0]:
        reasons.append("destination ownership pairs must be unique")
    return tuple(reasons)


def _check_capacity(
    route_rank: np.ndarray,
    row_rank: np.ndarray,
    row_valid: np.ndarray,
    *,
    rank_count: int,
    max_routes_per_rank: int | None,
    max_padded_rows_per_rank: int | None,
) -> None:
    route_counts = np.bincount(route_rank, minlength=rank_count)
    padded_counts = np.bincount(row_rank, minlength=rank_count)
    reasons: list[str] = []
    if max_routes_per_rank is not None:
        overflowing = np.flatnonzero(route_counts > max_routes_per_rank)
        for rank in overflowing:
            reasons.append(
                f"destination rank {rank} has {int(route_counts[rank])} routes, exceeding capacity {max_routes_per_rank}"
            )
    if max_padded_rows_per_rank is not None:
        overflowing = np.flatnonzero(padded_counts > max_padded_rows_per_rank)
        for rank in overflowing:
            reasons.append(
                f"destination rank {rank} has {int(padded_counts[rank])} padded rows, "
                f"exceeding capacity {max_padded_rows_per_rank}"
            )
    if reasons:
        raise RelationPlanError(tuple(reasons))
    assert int(np.count_nonzero(row_valid)) == route_rank.shape[0]


def _round_up(values: np.ndarray, quantum: int) -> np.ndarray:
    return ((values + quantum - 1) // quantum * quantum).astype(np.int32, copy=False)


def _validate_payload(payload: np.ndarray, leading_size: int, name: str) -> None:
    if not isinstance(payload, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if payload.ndim == 0 or payload.shape[0] != leading_size:
        raise ValueError(f"{name} leading dimension must be {leading_size}, got {payload.shape}")
