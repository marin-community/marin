# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Event Tensor schedules for relation-grouped right-resource computation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from tile_lifetime.event_buffering import (
    BoundedBufferPlan,
    EventRealizationAudit,
    EventRealizationKind,
    derive_bounded_buffer_plan,
    erased_event_realization,
    physical_event_realization,
    verify_event_realizations,
)
from tile_lifetime.event_dataflow import (
    EventDataflowProgram,
    EventGenerationPolicy,
    EventMemoryScope,
    EventStorageBinding,
    EventTensorRuntimeInputs,
    MemoryVisibility,
    TaskAxis,
    TaskDependence,
    TaskFamily,
    TaskRelation,
    derive_event_tensor_plan,
    event_tensor_runtime_inputs,
    verify_event_dataflow_program,
)
from tile_lifetime.relation import RelationPlan


@dataclass(frozen=True)
class RightResourcePipelineDescriptor:
    """Physical facts for one relation-grouped body followed by a Fold."""

    grouped_body_name: str
    fold_finalize_name: str
    edge_partition_by_slot: tuple[int, ...]
    edge_partition_count: int
    edge_capacity_per_task: int
    resource_buffer_depth: int
    resource_payload_bytes: int

    def __post_init__(self) -> None:
        names = (self.grouped_body_name, self.fold_finalize_name)
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("right-resource task names must be non-empty and distinct")
        if self.edge_partition_count <= 0:
            raise ValueError("edge partition count must be positive")
        if len(self.edge_partition_by_slot) == 0:
            raise ValueError("edge partition map must not be empty")
        if any(partition < 0 or partition >= self.edge_partition_count for partition in self.edge_partition_by_slot):
            raise ValueError("edge partition map contains an out-of-range partition")
        if self.edge_capacity_per_task <= 0:
            raise ValueError("right-resource edge capacity must be positive")
        if self.resource_buffer_depth <= 0 or self.resource_payload_bytes <= 0:
            raise ValueError("right-resource buffer depth and payload size must be positive")


@dataclass(frozen=True)
class RightResourceGrouping:
    """Stable active right-resource groups derived from relation edges."""

    resource_partition: tuple[int, ...]
    resource_item: tuple[int, ...]
    resource_edge_offsets: tuple[int, ...]
    resource_edges: tuple[int, ...]

    @property
    def task_count(self) -> int:
        """Return the number of active grouped-body tasks."""
        return len(self.resource_item)

    @property
    def edge_count(self) -> int:
        """Return the number of relation edges covered by active tasks."""
        return len(self.resource_edges)


@dataclass(frozen=True)
class RightResourceFoldEventSchedule:
    """Verified outer readiness around a grouped body and partial-state Fold."""

    descriptor: RightResourcePipelineDescriptor
    grouping: RightResourceGrouping
    program: EventDataflowProgram
    resource_stage: TaskFamily
    grouped_body: TaskFamily
    fold_finalize: TaskFamily
    resource_buffer: BoundedBufferPlan
    realization: EventRealizationAudit
    resource_runtime_inputs: EventTensorRuntimeInputs
    fold_runtime_inputs: EventTensorRuntimeInputs
    reuse_runtime_inputs: tuple[EventTensorRuntimeInputs, ...]
    program_fingerprint: str
    runtime_fingerprint: str


def derive_right_resource_fold_event_schedule(
    relation: RelationPlan,
    descriptor: RightResourcePipelineDescriptor,
) -> RightResourceFoldEventSchedule:
    """Derive exact relation grouping, bounded staging, and Fold readiness."""
    if len(descriptor.edge_partition_by_slot) != relation.route_slots:
        raise ValueError("edge partition map must cover every relation route slot")
    grouping = _right_resource_grouping(relation, descriptor)
    if grouping.task_count == 0:
        raise ValueError("right-resource scheduling requires at least one valid relation edge")

    task_axis = TaskAxis("resource_task", grouping.task_count)
    resource_stage = TaskFamily("right_resource_stage", (task_axis,), placement="transfer_workers")
    grouped_body = TaskFamily(descriptor.grouped_body_name, (task_axis,), placement="matrix_workers")
    fold_finalize = TaskFamily(
        descriptor.fold_finalize_name,
        (
            TaskAxis("left_item", relation.source_item_count),
            TaskAxis("edge_partition", descriptor.edge_partition_count),
        ),
        placement="reduction_workers",
    )
    cluster_visibility = MemoryVisibility(EventMemoryScope.CLUSTER)
    device_visibility = MemoryVisibility(EventMemoryScope.DEVICE)
    pointwise = tuple(((task,), (task,)) for task in range(grouping.task_count))
    resource_to_body = TaskDependence(
        TaskRelation.from_pairs(resource_stage, grouped_body, pointwise),
        cluster_visibility,
    )
    body_to_fold = TaskDependence(
        TaskRelation.from_pairs(
            grouped_body,
            fold_finalize,
            _body_to_fold_pairs(relation, grouping, descriptor),
        ),
        device_visibility,
    )
    resource_plan = derive_event_tensor_plan(
        resource_to_body,
        name="right_resource_ready",
        generation_policy=EventGenerationPolicy.PHASED,
    )
    fold_plan = derive_event_tensor_plan(body_to_fold, name="partial_state_ready")
    base_program = EventDataflowProgram(
        (resource_stage, grouped_body, fold_finalize),
        (resource_to_body, body_to_fold),
        (resource_plan, fold_plan),
    )
    slots = {(task,): task % descriptor.resource_buffer_depth for task in range(grouping.task_count)}
    generations = {(task,): task // descriptor.resource_buffer_depth for task in range(grouping.task_count)}
    resource_buffer = derive_bounded_buffer_plan(
        name="right_resource_buffer",
        program=base_program,
        producer=resource_stage,
        uses=(resource_to_body.relation,),
        capacity=descriptor.resource_buffer_depth,
        slot_for=slots,
        generation_for=generations,
        visibility=cluster_visibility,
    )
    reuse_plans = tuple(
        derive_event_tensor_plan(
            dependence,
            name=f"right_resource_reuse_{index}",
            generation_policy=EventGenerationPolicy.PHASED,
        )
        for index, dependence in enumerate(resource_buffer.reuse_dependences)
    )
    program = EventDataflowProgram(
        base_program.task_families,
        (*base_program.dependences, *resource_buffer.reuse_dependences),
        (*base_program.event_plans, *reuse_plans),
    )
    verify_event_dataflow_program(program)
    realization = verify_event_realizations(
        program,
        (
            physical_event_realization(
                resource_plan,
                mechanism="backend-selected staged-resource completion",
                reason="the grouped body consumes a bounded right-side resource after cluster-visible staging",
            ),
            erased_event_realization(
                fold_plan,
                kind=EventRealizationKind.ERASED_STREAM_ORDER,
                mechanism="same device stream",
                reason="the generated Fold finalizer launches after the grouped-body kernel on the same stream",
            ),
            *(
                physical_event_realization(
                    plan,
                    mechanism="backend-selected staged-resource release",
                    reason="a bounded resource slot cannot be reused before its grouped body completes",
                )
                for plan in reuse_plans
            ),
        ),
    )
    storage = EventStorageBinding(
        resource_plan.domain,
        resource_buffer.slots,
        resource_buffer.generations,
    )
    resource_runtime = event_tensor_runtime_inputs(resource_plan, storage_binding=storage)
    fold_runtime = event_tensor_runtime_inputs(fold_plan)
    reuse_runtime = tuple(
        event_tensor_runtime_inputs(
            plan,
            storage_binding=EventStorageBinding(plan.domain, resource_buffer.slots, resource_buffer.generations),
        )
        for plan in reuse_plans
    )
    program_payload = {
        "body": descriptor.grouped_body_name,
        "fold": descriptor.fold_finalize_name,
        "left_items": relation.source_item_count,
        "edge_partitions": descriptor.edge_partition_count,
        "edge_capacity_per_task": descriptor.edge_capacity_per_task,
        "resource_tasks": grouping.task_count,
        "buffer_depth": descriptor.resource_buffer_depth,
        "resource_payload_bytes": descriptor.resource_payload_bytes,
        "realizations": tuple((entry.plan_name, entry.kind.value) for entry in realization.entries),
    }
    runtime_payload = {
        "resource_partition": grouping.resource_partition,
        "resource_item": grouping.resource_item,
        "resource_edge_offsets": grouping.resource_edge_offsets,
        "resource_edges": grouping.resource_edges,
        "fold_counts": fold_runtime.event_initial_counts,
        "fold_sources": fold_runtime.event_sources,
        "fold_consumers": fold_runtime.event_consumers,
    }
    return RightResourceFoldEventSchedule(
        descriptor=descriptor,
        grouping=grouping,
        program=program,
        resource_stage=resource_stage,
        grouped_body=grouped_body,
        fold_finalize=fold_finalize,
        resource_buffer=resource_buffer,
        realization=realization,
        resource_runtime_inputs=resource_runtime,
        fold_runtime_inputs=fold_runtime,
        reuse_runtime_inputs=reuse_runtime,
        program_fingerprint=_fingerprint(program_payload),
        runtime_fingerprint=_fingerprint(runtime_payload),
    )


def _right_resource_grouping(
    relation: RelationPlan,
    descriptor: RightResourcePipelineDescriptor,
) -> RightResourceGrouping:
    edges_by_resource: dict[tuple[int, int], list[int]] = {}
    for edge in np.flatnonzero(relation.edge_valid.reshape(-1)):
        route_slot = int(relation.route_slot[edge])
        partition = descriptor.edge_partition_by_slot[route_slot]
        resource = int(relation.destination_item[edge])
        edges_by_resource.setdefault((partition, resource), []).append(int(edge))
    groups = tuple(sorted(edges_by_resource))
    task_groups = tuple(
        (partition, resource, tuple(edges[begin : begin + descriptor.edge_capacity_per_task]))
        for (partition, resource), edges in ((group, edges_by_resource[group]) for group in groups)
        for begin in range(0, len(edges), descriptor.edge_capacity_per_task)
    )
    offsets = [0]
    edges = []
    for _, _, task_edges in task_groups:
        edges.extend(task_edges)
        offsets.append(len(edges))
    return RightResourceGrouping(
        resource_partition=tuple(partition for partition, _, _ in task_groups),
        resource_item=tuple(resource for _, resource, _ in task_groups),
        resource_edge_offsets=tuple(offsets),
        resource_edges=tuple(edges),
    )


def _body_to_fold_pairs(
    relation: RelationPlan,
    grouping: RightResourceGrouping,
    descriptor: RightResourcePipelineDescriptor,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    pairs = set()
    for task in range(grouping.task_count):
        left = grouping.resource_edge_offsets[task]
        right = grouping.resource_edge_offsets[task + 1]
        for edge in grouping.resource_edges[left:right]:
            route_slot = int(relation.route_slot[edge])
            partition = descriptor.edge_partition_by_slot[route_slot]
            pairs.add(((task,), (int(relation.source_item[edge]), partition)))
    return tuple(sorted(pairs))


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
