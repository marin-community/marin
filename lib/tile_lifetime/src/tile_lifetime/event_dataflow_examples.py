# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic task-dependence builders used by Event Tensor experiments."""

from __future__ import annotations

from tile_lifetime.event_dataflow import (
    EventDataflowProgram,
    EventMemoryScope,
    EventSchedulingMode,
    MemoryVisibility,
    TaskAxis,
    TaskDependence,
    TaskFamily,
    TaskRelation,
    derive_event_tensor_plan,
)
from tile_lifetime.relation import RelationPlan


def split_fold_dependence(*, row_count: int, partition_count: int) -> TaskDependence:
    """Build the task dependence induced by splitting a row Fold into partials."""
    if row_count <= 0:
        raise ValueError("split Fold row count must be positive")
    if partition_count <= 0:
        raise ValueError("split Fold partition count must be positive")
    partial = TaskFamily("partial_fold", (TaskAxis("row", row_count), TaskAxis("partition", partition_count)))
    finalize = TaskFamily("fold_finalize", (TaskAxis("row", row_count),))
    pairs = tuple(((row, partition), (row,)) for row in range(row_count) for partition in range(partition_count))
    return TaskDependence(
        TaskRelation.from_pairs(partial, finalize, pairs),
        MemoryVisibility(EventMemoryScope.DEVICE),
    )


def relation_segment_dependence(
    relation: RelationPlan,
    *,
    producer_name: str = "relation_edge_ready",
    consumer_name: str = "segment_consumer",
) -> TaskDependence:
    """Map each valid runtime relation edge to its destination segment task."""
    producer = TaskFamily(producer_name, (TaskAxis("edge", relation.route_count),))
    consumer = TaskFamily(consumer_name, (TaskAxis("destination", relation.destination_count),))
    grouped_routes = relation.grouped_route_indices
    pairs = tuple(((edge,), (int(relation.destination_item[route]),)) for edge, route in enumerate(grouped_routes))
    return TaskDependence(
        TaskRelation.from_pairs(producer, consumer, pairs),
        MemoryVisibility(EventMemoryScope.DEVICE),
    )


def tiled_collective_dependence(
    *,
    output_tile_count: int,
    destination_count: int,
    partials_per_destination: int,
) -> TaskDependence:
    """Build a tiled producer to destination-owned communication consumer graph."""
    if output_tile_count <= 0:
        raise ValueError("output tile count must be positive")
    if destination_count <= 0:
        raise ValueError("destination count must be positive")
    if partials_per_destination <= 0:
        raise ValueError("partials per destination must be positive")
    producer = TaskFamily(
        "contract_output_tile",
        (
            TaskAxis("output_tile", output_tile_count),
            TaskAxis("destination", destination_count),
            TaskAxis("partial", partials_per_destination),
        ),
        placement="matrix_workers",
    )
    consumer = TaskFamily(
        "placement_change_tile",
        (TaskAxis("output_tile", output_tile_count), TaskAxis("destination", destination_count)),
        placement="transport_workers",
    )
    pairs = tuple(
        ((output_tile, destination, partial), (output_tile, destination))
        for output_tile in range(output_tile_count)
        for destination in range(destination_count)
        for partial in range(partials_per_destination)
    )
    return TaskDependence(
        TaskRelation.from_pairs(producer, consumer, pairs),
        MemoryVisibility(EventMemoryScope.SYSTEM),
    )


def single_dependence_event_program(
    dependence: TaskDependence,
    *,
    name: str,
    scheduling_mode: EventSchedulingMode,
) -> EventDataflowProgram:
    """Derive a complete two-family reference program from one exact relation."""
    plan = derive_event_tensor_plan(dependence, name=name, scheduling_mode=scheduling_mode)
    source = dependence.relation.source
    target = dependence.relation.target
    assert isinstance(source, TaskFamily)
    assert isinstance(target, TaskFamily)
    return EventDataflowProgram((source, target), (dependence,), (plan,))
