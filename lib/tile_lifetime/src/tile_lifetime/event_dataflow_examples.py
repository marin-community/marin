# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic task-dependence builders used by Event Tensor experiments."""

from __future__ import annotations

from tile_lifetime.event_dataflow import (
    EventDataflowProgram,
    EventGenerationPolicy,
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


def split_fold_dependence(
    *,
    row_count: int,
    partition_count: int,
    visibility_scope: EventMemoryScope = EventMemoryScope.DEVICE,
) -> TaskDependence:
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
        MemoryVisibility(visibility_scope),
    )


def relation_segment_dependence(
    relation: RelationPlan,
    *,
    producer_name: str = "relation_edge_ready",
    consumer_name: str = "segment_consumer",
) -> TaskDependence:
    """Map each valid runtime relation edge to its physical segment task."""
    producer = TaskFamily(producer_name, (TaskAxis("edge", relation.route_count),))
    consumer = TaskFamily(consumer_name, (TaskAxis("segment", relation.destination_count),))
    edge_offsets = relation.destination_edge_offsets
    pairs = tuple(
        ((edge,), (segment,))
        for segment, count in enumerate(relation.group_count)
        for edge in range(int(edge_offsets[segment]), int(edge_offsets[segment]) + int(count))
    )
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


def pipelined_contract_fold_program(
    *,
    generation_count: int,
    pipeline_depth: int,
) -> EventDataflowProgram:
    """Derive a phased Contract-Fold-Contract graph with bounded slot reuse."""
    if generation_count <= 0:
        raise ValueError("pipeline generation count must be positive")
    if pipeline_depth <= 0:
        raise ValueError("pipeline depth must be positive")
    phase_and_slot = (TaskAxis("generation", generation_count), TaskAxis("pipeline_slot", pipeline_depth))
    first_contract = TaskFamily("first_contract", phase_and_slot, placement="matrix_workers")
    fold_update = TaskFamily("fold_update", phase_and_slot, placement="reduction_workers")
    second_contract = TaskFamily("second_contract", phase_and_slot, placement="matrix_workers")
    fold_finalize = TaskFamily("fold_finalize", (TaskAxis("generation", generation_count),))
    visibility = MemoryVisibility(EventMemoryScope.CTA)

    pointwise_pairs = tuple(
        ((generation, slot), (generation, slot))
        for generation in range(generation_count)
        for slot in range(pipeline_depth)
    )
    first_to_fold = TaskDependence(
        TaskRelation.from_pairs(first_contract, fold_update, pointwise_pairs),
        visibility,
    )
    fold_to_second = TaskDependence(
        TaskRelation.from_pairs(fold_update, second_contract, pointwise_pairs),
        visibility,
    )
    second_to_finalize = TaskDependence(
        TaskRelation.from_pairs(
            second_contract,
            fold_finalize,
            tuple(
                ((generation, slot), (generation,))
                for generation in range(generation_count)
                for slot in range(pipeline_depth)
            ),
        ),
        visibility,
    )
    reuse = TaskDependence(
        TaskRelation.from_pairs(
            fold_finalize,
            first_contract,
            tuple(
                ((generation - 1,), (generation, slot))
                for generation in range(1, generation_count)
                for slot in range(pipeline_depth)
            ),
        ),
        visibility,
    )
    dependences = (reuse, first_to_fold, fold_to_second, second_to_finalize)
    plans = tuple(
        derive_event_tensor_plan(
            dependence,
            name=f"{dependence.relation.source.name}_to_{dependence.relation.target.name}",
            generation_policy=EventGenerationPolicy.PHASED,
        )
        for dependence in dependences
    )
    return EventDataflowProgram(
        (first_contract, fold_update, second_contract, fold_finalize),
        dependences,
        plans,
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
