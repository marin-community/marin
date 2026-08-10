# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive generic task dataflow from scheduled tensor programs."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, prod

from tile_lifetime.collective_transport import CollectiveCompletionPlan
from tile_lifetime.event_buffering import BoundedBufferPlan, derive_bounded_buffer_plan
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
from tile_lifetime.ir import DType
from tile_lifetime.relation import RelationPlan
from tile_lifetime.right_resource_event_schedule import RightResourcePipelineDescriptor
from tile_lifetime.sm100_routed_lowering import SM100RoutedStreamingLowering
from tile_lifetime.streaming_attention import AttentionScoreAxis, StreamingAttentionProgram
from tile_lifetime.streaming_event_schedule import StreamingContractFoldDescriptor

_DTYPE_BYTES = {
    DType.BF16: 2,
    DType.FP32: 4,
    DType.FP64: 8,
}


def sm100_routed_right_resource_descriptor(
    lowering: SM100RoutedStreamingLowering,
) -> RightResourcePipelineDescriptor:
    """Bind one routed normalized-Fold lowering to generic schedule roles.

    This adapter is the semantic boundary: the returned descriptor contains no
    query, key, value, attention, or expert role names. The schedule-level
    derivation therefore remains reusable for any relation-grouped body that
    stages one right-side resource and emits partitioned Fold contributions.
    """
    element_bytes = 2
    operand_count = 2
    feature_dimension = 128
    return RightResourcePipelineDescriptor(
        grouped_body_name="grouped_contract_fold_body",
        fold_finalize_name="partial_state_fold_finalize",
        edge_partition_by_slot=tuple(
            route_slot // lowering.selected_count for route_slot in range(lowering.relation.route_slots)
        ),
        edge_partition_count=lowering.key_value_heads,
        edge_capacity_per_task=lowering.schedule.right_edges_per_task,
        right_item_extent=lowering.schedule.right_block_size,
        resource_buffer_depth=lowering.schedule.right_stages,
        resource_payload_bytes=(lowering.schedule.right_block_size * feature_dimension * operand_count * element_bytes),
    )


@dataclass(frozen=True)
class StreamingFoldTaskDataflow:
    """Task graph derived from one Contract/normalized-Fold/Contract program."""

    program: EventDataflowProgram
    key_value_stage: TaskFamily
    qk_contract: TaskFamily
    fold_partial: TaskFamily
    pv_contract: TaskFamily
    finalize: TaskFamily
    row_tile_count: int
    fold_partition_count: int
    pipeline_depth: int
    key_value_buffer: BoundedBufferPlan


@dataclass(frozen=True)
class SegmentedContractTaskDataflow:
    """Runtime relation readiness feeding generic segmented Contract tiles."""

    program: EventDataflowProgram
    edge_ready: TaskFamily
    contract: TaskFamily
    segment_count: int
    output_tile_count: int


@dataclass(frozen=True)
class CollectiveCompletionSchedule:
    """Bounded schedule choice for one placement-changing Fold."""

    tile_count: int
    scheduling_mode: EventSchedulingMode

    def __post_init__(self) -> None:
        if self.tile_count <= 0:
            raise ValueError("collective completion tile count must be positive")


@dataclass(frozen=True)
class CollectiveCompletionTaskDataflow:
    """System-visible transport completions feeding generic Fold tiles.

    The source tasks represent completion of a placement transition, not a
    selected NCCL, XLA, or device-side transport implementation. A target
    lowering may erase or replace them once it proves the corresponding
    transport completion and visibility contract.
    """

    completion: CollectiveCompletionPlan
    schedule: CollectiveCompletionSchedule
    program: EventDataflowProgram
    partial_tile: TaskFamily
    transport_completion: TaskFamily
    fold_tile: TaskFamily
    contribution_devices: tuple[int, ...]
    contribution_groups: tuple[int, ...]


def streaming_contract_fold_event_descriptor(
    program: StreamingAttentionProgram,
) -> StreamingContractFoldDescriptor:
    """Erase normalized-attention names into a generic physical descriptor."""
    query_axes = tuple(axis for axis in program.qk.output.axes if axis.label == AttentionScoreAxis.QUERY.value)
    fold_axes = tuple(axis for axis in program.qk.output.axes if axis.label == AttentionScoreAxis.KEY.value)
    if len(query_axes) != 1 or len(fold_axes) != 1:
        raise ValueError("streaming Contract/Fold adaptation requires one resident and one Fold axis")
    query, key = program.qk.inputs
    value = program.pv.inputs[1]
    dtypes = (query.dtype, key.dtype, value.dtype)
    if len(set(dtypes)) != 1:
        raise ValueError("streaming Contract/Fold operands must share one physical element type")
    try:
        element_bytes = _DTYPE_BYTES[query.dtype]
    except KeyError as error:
        raise ValueError(f"streaming Contract/Fold has unsupported physical dtype {query.dtype.value}") from error
    return StreamingContractFoldDescriptor(
        first_contract_name=program.qk.name,
        fold_update_name="streaming_fold_update",
        second_contract_name=program.pv.name,
        finalize_name=program.finalize.name,
        fold_extent=fold_axes[0].extent,
        resident_tile_size=program.schedule.query_tile_size,
        streamed_tile_size=program.schedule.key_value_tile_size,
        pipeline_depth=program.schedule.pipeline_depth,
        resident_reduction_dimension=query.shape[-1],
        streamed_reduction_dimension=key.shape[-1],
        output_dimension=value.shape[-1],
        element_bytes=element_bytes,
    )


def streaming_fold_task_dataflow(
    program: StreamingAttentionProgram,
    *,
    visibility_scope: EventMemoryScope = EventMemoryScope.CTA,
) -> StreamingFoldTaskDataflow:
    """Mechanically tile a streaming normalized weighted Fold into task families."""
    row_axes = program.state.row_max.axes
    query_axes = tuple(axis for axis in row_axes if axis.label == AttentionScoreAxis.QUERY.value)
    if len(query_axes) != 1:
        raise ValueError("streaming Fold task derivation requires one query axis")
    fold_axes = tuple(axis for axis in program.qk.output.axes if axis not in row_axes)
    if len(fold_axes) != 1:
        raise ValueError("streaming Fold task derivation requires one Fold axis")
    query_axis = query_axes[0]
    fold_axis = fold_axes[0]
    row_tile_count = prod(
        ceil(axis.extent / program.schedule.query_tile_size) if axis == query_axis else axis.extent for axis in row_axes
    )
    fold_partition_count = ceil(fold_axis.extent / program.schedule.key_value_tile_size)
    tiled_axes = (TaskAxis("row_tile", row_tile_count), TaskAxis("fold_partition", fold_partition_count))
    row_axis = (TaskAxis("row_tile", row_tile_count),)
    key_value_stage = TaskFamily("key_value_stage", tiled_axes, placement="transfer_workers")
    qk_contract = TaskFamily(program.qk.name, tiled_axes, placement="matrix_workers")
    fold_partial = TaskFamily("normalized_exp_fold_partial", tiled_axes, placement="reduction_workers")
    pv_contract = TaskFamily(program.pv.name, tiled_axes, placement="matrix_workers")
    finalize = TaskFamily(program.finalize.name, row_axis, placement="reduction_workers")
    visibility = MemoryVisibility(visibility_scope)
    pointwise = tuple(
        ((row_tile, partition), (row_tile, partition))
        for row_tile in range(row_tile_count)
        for partition in range(fold_partition_count)
    )
    stage_to_qk = TaskDependence(TaskRelation.from_pairs(key_value_stage, qk_contract, pointwise), visibility)
    stage_to_pv = TaskDependence(TaskRelation.from_pairs(key_value_stage, pv_contract, pointwise), visibility)
    qk_to_fold = TaskDependence(TaskRelation.from_pairs(qk_contract, fold_partial, pointwise), visibility)
    fold_to_pv = TaskDependence(TaskRelation.from_pairs(fold_partial, pv_contract, pointwise), visibility)
    pv_to_finalize = TaskDependence(
        TaskRelation.from_pairs(
            pv_contract,
            finalize,
            tuple(
                ((row_tile, partition), (row_tile,))
                for row_tile in range(row_tile_count)
                for partition in range(fold_partition_count)
            ),
        ),
        visibility,
    )
    initial_dependences = (stage_to_qk, stage_to_pv, qk_to_fold, fold_to_pv, pv_to_finalize)
    initial_plans = tuple(
        derive_event_tensor_plan(
            dependence,
            name=f"{dependence.relation.source.name}_to_{dependence.relation.target.name}",
        )
        for dependence in initial_dependences
    )
    initial_program = EventDataflowProgram(
        (key_value_stage, qk_contract, fold_partial, pv_contract, finalize),
        initial_dependences,
        initial_plans,
    )
    item_coordinates = key_value_stage.coordinates
    key_value_buffer = derive_bounded_buffer_plan(
        name="key_value_pipeline",
        program=initial_program,
        producer=key_value_stage,
        uses=(stage_to_qk.relation, stage_to_pv.relation),
        capacity=row_tile_count * program.schedule.pipeline_depth,
        slot_for={
            coordinate: coordinate[0] * program.schedule.pipeline_depth + coordinate[1] % program.schedule.pipeline_depth
            for coordinate in item_coordinates
        },
        generation_for={coordinate: coordinate[1] // program.schedule.pipeline_depth for coordinate in item_coordinates},
        visibility=visibility,
    )
    dependences = (*initial_dependences, *key_value_buffer.reuse_dependences)
    event_plans = tuple(
        derive_event_tensor_plan(
            dependence,
            name=f"{dependence.relation.source.name}_to_{dependence.relation.target.name}",
            generation_policy=(
                EventGenerationPolicy.PHASED
                if dependence in key_value_buffer.reuse_dependences
                else EventGenerationPolicy.PER_INVOCATION
            ),
        )
        for dependence in dependences
    )
    dataflow = EventDataflowProgram(
        (key_value_stage, qk_contract, fold_partial, pv_contract, finalize),
        dependences,
        event_plans,
    )
    return StreamingFoldTaskDataflow(
        program=dataflow,
        key_value_stage=key_value_stage,
        qk_contract=qk_contract,
        fold_partial=fold_partial,
        pv_contract=pv_contract,
        finalize=finalize,
        row_tile_count=row_tile_count,
        fold_partition_count=fold_partition_count,
        pipeline_depth=program.schedule.pipeline_depth,
        key_value_buffer=key_value_buffer,
    )


def relation_segmented_contract_task_dataflow(
    relation: RelationPlan,
    *,
    output_tile_count: int,
    visibility_scope: EventMemoryScope = EventMemoryScope.DEVICE,
) -> SegmentedContractTaskDataflow:
    """Derive relation-edge readiness for generic segmented Contract tiles."""
    if output_tile_count <= 0:
        raise ValueError("segmented Contract output tile count must be positive")
    edge_ready = TaskFamily("relation_edge_ready", (TaskAxis("edge", relation.route_count),), placement="transport")
    contract = TaskFamily(
        "segmented_contract",
        (TaskAxis("segment", relation.destination_count), TaskAxis("output_tile", output_tile_count)),
        placement="matrix_workers",
    )
    offsets = relation.destination_edge_offsets
    pairs = tuple(
        ((edge,), (segment, output_tile))
        for segment, count in enumerate(relation.group_count)
        for output_tile in range(output_tile_count)
        for edge in range(int(offsets[segment]), int(offsets[segment]) + int(count))
    )
    dependence = TaskDependence(
        TaskRelation.from_pairs(edge_ready, contract, pairs),
        MemoryVisibility(visibility_scope),
    )
    plan = derive_event_tensor_plan(dependence, name="relation_edges_to_segmented_contract")
    return SegmentedContractTaskDataflow(
        program=EventDataflowProgram((edge_ready, contract), (dependence,), (plan,)),
        edge_ready=edge_ready,
        contract=contract,
        segment_count=relation.destination_count,
        output_tile_count=output_tile_count,
    )


def collective_completion_task_dataflow(
    completion: CollectiveCompletionPlan,
    *,
    schedule: CollectiveCompletionSchedule,
) -> CollectiveCompletionTaskDataflow:
    """Derive readiness for a tiled partial-value completion.

    Replica-group membership determines the producer-to-Fold relation and the
    Event Tensor indegree. The Fold operator affects the consumer's semantic
    body but not the readiness construction.
    """
    replica_groups = completion.transport.replica_domain.groups
    contribution_devices = tuple(device for group in replica_groups for device in group)
    contribution_groups = tuple(group_index for group_index, group in enumerate(replica_groups) for _ in group)
    tile_axis = TaskAxis("tile", schedule.tile_count)
    partial_tile = TaskFamily(
        "partial_value_tile",
        (TaskAxis("contribution", len(contribution_devices)), tile_axis),
        placement=completion.transport.source_value,
    )
    transport_completion = TaskFamily(
        "placement_transition_completion",
        (TaskAxis("contribution", len(contribution_devices)), tile_axis),
        placement=completion.transport.destination_value,
    )
    fold_tile = TaskFamily(
        "collective_fold_tile",
        (TaskAxis("replica_group", len(replica_groups)), tile_axis),
        placement=completion.transport.destination_value,
    )
    pointwise_pairs = tuple(
        ((contribution, tile), (contribution, tile))
        for contribution in range(len(contribution_devices))
        for tile in range(schedule.tile_count)
    )
    fold_pairs = tuple(
        ((contribution, tile), (group, tile))
        for contribution, group in enumerate(contribution_groups)
        for tile in range(schedule.tile_count)
    )
    partial_to_transport = TaskDependence(
        TaskRelation.from_pairs(partial_tile, transport_completion, pointwise_pairs),
        MemoryVisibility(EventMemoryScope.DEVICE),
    )
    transport_to_fold = TaskDependence(
        TaskRelation.from_pairs(transport_completion, fold_tile, fold_pairs),
        MemoryVisibility(EventMemoryScope.SYSTEM),
    )
    partial_event_plan = derive_event_tensor_plan(
        partial_to_transport,
        name="partial_value_to_placement_transition",
        scheduling_mode=schedule.scheduling_mode,
    )
    completion_event_plan = derive_event_tensor_plan(
        transport_to_fold,
        name="placement_transition_to_collective_fold",
        scheduling_mode=schedule.scheduling_mode,
    )
    return CollectiveCompletionTaskDataflow(
        completion=completion,
        schedule=schedule,
        program=EventDataflowProgram(
            task_families=(partial_tile, transport_completion, fold_tile),
            dependences=(partial_to_transport, transport_to_fold),
            event_plans=(partial_event_plan, completion_event_plan),
        ),
        partial_tile=partial_tile,
        transport_completion=transport_completion,
        fold_tile=fold_tile,
        contribution_devices=contribution_devices,
        contribution_groups=contribution_groups,
    )
