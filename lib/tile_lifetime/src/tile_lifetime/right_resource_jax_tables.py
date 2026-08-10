# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize generic right-resource work tables as JAX operands."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from tile_lifetime.relation import RelationPlan
from tile_lifetime.right_resource_event_schedule import RightResourceFoldEventSchedule

_PACKED_ITEM_BITS = 24
_PACKED_SLOT_MASK = 0xFF


@dataclass(frozen=True)
class RightResourceWorkTables:
    """Deterministic grouped work metadata consumed by a physical template."""

    right_to_left_offsets: np.ndarray
    right_to_left_sources: np.ndarray
    partial_slot_sources: np.ndarray
    split_counts: np.ndarray
    scheduler_metadata: np.ndarray
    work_count: np.ndarray
    left_offsets: np.ndarray
    right_payload_offsets: np.ndarray

    @property
    def work_capacity(self) -> int:
        """Return the statically allocated grouped-task capacity."""
        return int(self.scheduler_metadata.shape[0])


@dataclass(frozen=True)
class JaxRightResourceWorkTables:
    """JAX-owned runtime operands for grouped resource computation."""

    right_to_left_offsets: jax.Array
    right_to_left_sources: jax.Array
    partial_slot_sources: jax.Array
    split_counts: jax.Array
    scheduler_metadata: jax.Array
    work_count: jax.Array
    left_offsets: jax.Array
    right_payload_offsets: jax.Array
    work_capacity: int


def derive_right_resource_work_tables(
    relation: RelationPlan,
    schedule: RightResourceFoldEventSchedule,
) -> RightResourceWorkTables:
    """Derive a deterministic CSR, worklist, and Fold-slot ownership."""
    descriptor = schedule.descriptor
    partition_count = descriptor.edge_partition_count
    right_count = relation.destination_count
    left_count = relation.source_item_count
    if relation.destination_rank_count != 1 or np.any(relation.group_destination_rank != 0):
        raise ValueError("right-resource physical tables require one destination placement")
    local_item_by_right = np.empty(right_count, dtype=np.int32)
    local_item_by_right[relation.group_destination_item] = relation.group_destination_local_item
    if not np.array_equal(np.sort(local_item_by_right), np.arange(right_count, dtype=np.int32)):
        raise ValueError("right-resource physical tables require a dense local resource domain")
    slots_by_partition = _slots_by_partition(descriptor.edge_partition_by_slot, partition_count)
    selected_counts = {len(slots) for slots in slots_by_partition}
    if len(selected_counts) != 1:
        raise ValueError("right-resource physical tables require an equal edge-slot count per partition")
    selected_count = selected_counts.pop()
    edge_capacity = left_count * selected_count

    offsets = np.zeros((partition_count, right_count + 1), dtype=np.int32)
    sources = np.full((partition_count, edge_capacity), -1, dtype=np.int32)
    partial_sources = np.full_like(sources, -1)
    split_counts = np.zeros((left_count, partition_count), dtype=np.int32)
    grouped_edges: dict[tuple[int, int], list[int]] = {}
    within_partition_slot = {
        route_slot: selected_slot
        for partition_slots in slots_by_partition
        for selected_slot, route_slot in enumerate(partition_slots)
    }
    for edge in np.flatnonzero(relation.edge_valid.reshape(-1)):
        route_slot = int(relation.route_slot[edge])
        partition = descriptor.edge_partition_by_slot[route_slot]
        right_item = int(relation.destination_item[edge])
        grouped_edges.setdefault((partition, right_item), []).append(int(edge))
        split_counts[int(relation.source_item[edge]), partition] += 1

    metadata = []
    for partition in range(partition_count):
        cursor = 0
        for right_item in range(right_count):
            edges = grouped_edges.get((partition, right_item), ())
            offsets[partition, right_item] = cursor
            ordered_edges = sorted(
                edges,
                key=lambda edge: (
                    int(relation.source_item[edge]),
                    within_partition_slot[int(relation.route_slot[edge])],
                ),
            )
            for edge in ordered_edges:
                left_item = int(relation.source_item[edge])
                selected_slot = within_partition_slot[int(relation.route_slot[edge])]
                sources[partition, cursor] = left_item
                partial_sources[partition, cursor] = left_item | (
                    (selected_slot & _PACKED_SLOT_MASK) << _PACKED_ITEM_BITS
                )
                cursor += 1
            for local_begin in range(0, len(ordered_edges), descriptor.edge_capacity_per_task):
                count = min(descriptor.edge_capacity_per_task, len(ordered_edges) - local_begin)
                metadata.append((partition, right_item, local_begin, count, 0, int(local_item_by_right[right_item])))
        offsets[partition, right_count] = cursor

    scheduler_metadata = np.asarray(metadata, dtype=np.int32)
    if scheduler_metadata.shape != (schedule.grouping.task_count, 6):
        raise ValueError(
            "generic right-resource task decomposition disagrees with physical work tables: "
            f"expected {(schedule.grouping.task_count, 6)}, found {scheduler_metadata.shape}"
        )
    left_offsets = np.asarray([0, left_count], dtype=np.int32)
    right_payload_offsets = np.asarray([0, local_item_by_right.shape[0] * descriptor.right_item_extent], dtype=np.int32)
    return RightResourceWorkTables(
        right_to_left_offsets=offsets,
        right_to_left_sources=sources,
        partial_slot_sources=partial_sources,
        split_counts=split_counts,
        scheduler_metadata=scheduler_metadata,
        work_count=np.asarray([schedule.grouping.task_count], dtype=np.int32),
        left_offsets=left_offsets,
        right_payload_offsets=right_payload_offsets,
    )


def right_resource_work_tables_as_jax(tables: RightResourceWorkTables) -> JaxRightResourceWorkTables:
    """Transfer generic work tables into JAX-owned device operands."""
    return JaxRightResourceWorkTables(
        right_to_left_offsets=jnp.asarray(tables.right_to_left_offsets),
        right_to_left_sources=jnp.asarray(tables.right_to_left_sources),
        partial_slot_sources=jnp.asarray(tables.partial_slot_sources),
        split_counts=jnp.asarray(tables.split_counts),
        scheduler_metadata=jnp.asarray(tables.scheduler_metadata),
        work_count=jnp.asarray(tables.work_count),
        left_offsets=jnp.asarray(tables.left_offsets),
        right_payload_offsets=jnp.asarray(tables.right_payload_offsets),
        work_capacity=tables.work_capacity,
    )


def _slots_by_partition(
    partition_by_slot: tuple[int, ...],
    partition_count: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(slot for slot, partition in enumerate(partition_by_slot) if partition == target)
        for target in range(partition_count)
    )
