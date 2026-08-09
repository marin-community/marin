# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Event Tensor attachment for a generic streaming Contract/Fold schedule.

This module is below tensor semantics and above CUDA synchronization.  It
derives the logical producer/consumer and buffer-reuse edges implemented by a
Q-resident streaming contraction skeleton.  Backend-specific barrier IDs and
pipeline classes remain physical allocation choices.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import ceil

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
    MemoryVisibility,
    TaskAxis,
    TaskDependence,
    TaskFamily,
    TaskRelation,
    derive_event_tensor_plan,
    verify_event_dataflow_program,
)


@dataclass(frozen=True)
class StreamingContractFoldDescriptor:
    """Target-independent facts for one resident-left streaming skeleton.

    This descriptor is produced after tensor semantics have been decomposed
    into two Contracts around an ordered Fold.  It deliberately carries no
    attention operation, mask, normalized-exponential, or model identity.
    Those semantics determine the task graph before this schedule boundary;
    Event Tensor construction only needs task names, extents, payload widths,
    and the selected bounded pipeline shape.
    """

    first_contract_name: str
    fold_update_name: str
    second_contract_name: str
    finalize_name: str
    fold_extent: int
    resident_tile_size: int
    streamed_tile_size: int
    pipeline_depth: int
    resident_reduction_dimension: int
    streamed_reduction_dimension: int
    output_dimension: int
    element_bytes: int

    def __post_init__(self) -> None:
        names = (
            self.first_contract_name,
            self.fold_update_name,
            self.second_contract_name,
            self.finalize_name,
        )
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("streaming task-family names must be non-empty and distinct")
        dimensions = (
            self.fold_extent,
            self.resident_tile_size,
            self.streamed_tile_size,
            self.pipeline_depth,
            self.resident_reduction_dimension,
            self.streamed_reduction_dimension,
            self.output_dimension,
            self.element_bytes,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("streaming extents, tile sizes, dimensions, and element width must be positive")


@dataclass(frozen=True)
class StreamingWorkerAssignment:
    """Finite worker decomposition selected for one physical tile skeleton."""

    transfer_warps: int
    transfer_warpgroup_threads: int
    matrix_warpgroups: int
    threads_per_warpgroup: int

    @property
    def matrix_warps(self) -> int:
        """Return the number of warps participating in matrix consumption."""
        return self.matrix_warpgroups * (self.threads_per_warpgroup // 32)

    @property
    def cta_threads(self) -> int:
        """Return the complete CTA size, including the transfer warpgroup."""
        return (self.matrix_warpgroups + 1) * self.threads_per_warpgroup

    @property
    def scheduler_arrival_threads(self) -> int:
        """Return pairwise matrix-warpgroup handshake participants."""
        return 2 * self.threads_per_warpgroup if self.matrix_warpgroups > 1 else 0


@dataclass(frozen=True)
class StreamingPipelineDataflow:
    """Kernel-local task graph and bounded buffers for two tile generations."""

    program: EventDataflowProgram
    query_stage: TaskFamily
    key_stage: TaskFamily
    value_stage: TaskFamily
    first_contract: TaskFamily
    fold_update: TaskFamily
    second_contract: TaskFamily
    finalize: TaskFamily
    partition_count: int
    generation_count: int
    query_buffer: BoundedBufferPlan
    key_buffer: BoundedBufferPlan
    value_buffer: BoundedBufferPlan


@dataclass(frozen=True)
class StreamingPhysicalEventSchedule:
    """Verified readiness configuration consumed by a physical GPU skeleton."""

    dataflow: StreamingPipelineDataflow
    audit: EventRealizationAudit
    workers: StreamingWorkerAssignment
    query_stages: int
    key_stages: int
    value_stages: int
    barriers_per_stage: int
    query_transaction_bytes: int
    key_transaction_bytes: int
    value_transaction_bytes: int
    fingerprint: str

    @property
    def query_barrier_slots(self) -> int:
        """Return full/empty barrier storage required for Q."""
        return self.query_stages * self.barriers_per_stage

    @property
    def key_barrier_slots(self) -> int:
        """Return full/empty barrier storage required for K."""
        return self.key_stages * self.barriers_per_stage

    @property
    def value_barrier_slots(self) -> int:
        """Return full/empty barrier storage required for V."""
        return self.value_stages * self.barriers_per_stage


def verify_streaming_event_backend_parameters(
    schedule: StreamingPhysicalEventSchedule,
    *,
    query_stages: int,
    key_stages: int,
    value_stages: int,
    barriers_per_stage: int,
    transfer_warps: int,
    matrix_warpgroups: int,
    scheduler_arrival_threads: int,
) -> None:
    """Reject a physical emitter whose synchronization constants drifted.

    This is the source-audit boundary: the backend supplies the values it will
    emit, and the schedule-level Event Tensor derivation remains the source of
    truth. Concrete barrier identifiers are intentionally outside this check.
    """
    actual = {
        "query_stages": query_stages,
        "key_stages": key_stages,
        "value_stages": value_stages,
        "barriers_per_stage": barriers_per_stage,
        "transfer_warps": transfer_warps,
        "matrix_warpgroups": matrix_warpgroups,
        "scheduler_arrival_threads": scheduler_arrival_threads,
    }
    expected = {
        "query_stages": schedule.query_stages,
        "key_stages": schedule.key_stages,
        "value_stages": schedule.value_stages,
        "barriers_per_stage": schedule.barriers_per_stage,
        "transfer_warps": schedule.workers.transfer_warps,
        "matrix_warpgroups": schedule.workers.matrix_warpgroups,
        "scheduler_arrival_threads": schedule.workers.scheduler_arrival_threads,
    }
    mismatches = tuple(
        f"{name}: expected {expected[name]}, found {actual[name]}" for name in expected if actual[name] != expected[name]
    )
    if mismatches:
        raise ValueError("streaming Event Tensor/backend mismatch: " + "; ".join(mismatches))


def derive_streaming_physical_event_schedule(
    descriptor: StreamingContractFoldDescriptor,
) -> StreamingPhysicalEventSchedule:
    """Derive the synchronization contract for a Q-resident streaming skeleton.

    The task graph is a kernel-local template.  Two query-tile generations are
    sufficient to expose Q reuse, while every K/V partition is represented so
    pipeline-slot generations and the final producer tail are exact.
    """
    query_tile = descriptor.resident_tile_size
    key_tile = descriptor.streamed_tile_size
    partition_count = ceil(descriptor.fold_extent / key_tile)
    pipeline_depth = descriptor.pipeline_depth

    dataflow = _derive_pipeline_dataflow(
        descriptor,
        partition_count=partition_count,
        pipeline_depth=pipeline_depth,
        generation_count=2,
    )
    matrix_warpgroups = query_tile // 64
    if matrix_warpgroups <= 0 or query_tile % 64:
        raise ValueError("the physical worker decomposition requires query tiles divisible by 64")
    workers = StreamingWorkerAssignment(
        transfer_warps=1,
        transfer_warpgroup_threads=128,
        matrix_warpgroups=matrix_warpgroups,
        threads_per_warpgroup=128,
    )
    audit = _realization_audit(dataflow, workers)
    payload = {
        "partition_count": partition_count,
        "pipeline_depth": pipeline_depth,
        "query_tile": query_tile,
        "key_tile": key_tile,
        "workers": {
            "transfer_warps": workers.transfer_warps,
            "matrix_warpgroups": workers.matrix_warpgroups,
            "cta_threads": workers.cta_threads,
        },
        "events": tuple((entry.plan_name, entry.kind.value, entry.mechanism) for entry in audit.entries),
        "buffers": {
            "query": (dataflow.query_buffer.capacity, dataflow.query_buffer.generations),
            "key": (dataflow.key_buffer.capacity, dataflow.key_buffer.generations),
            "value": (dataflow.value_buffer.capacity, dataflow.value_buffer.generations),
        },
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return StreamingPhysicalEventSchedule(
        dataflow=dataflow,
        audit=audit,
        workers=workers,
        query_stages=1,
        key_stages=pipeline_depth,
        value_stages=pipeline_depth,
        barriers_per_stage=2,
        query_transaction_bytes=query_tile * descriptor.resident_reduction_dimension * descriptor.element_bytes,
        key_transaction_bytes=key_tile * descriptor.streamed_reduction_dimension * descriptor.element_bytes,
        value_transaction_bytes=key_tile * descriptor.output_dimension * descriptor.element_bytes,
        fingerprint=fingerprint,
    )


def _derive_pipeline_dataflow(
    descriptor: StreamingContractFoldDescriptor,
    *,
    partition_count: int,
    pipeline_depth: int,
    generation_count: int,
) -> StreamingPipelineDataflow:
    generation_axis = TaskAxis("generation", generation_count)
    partition_axis = TaskAxis("fold_partition", partition_count)
    generation_domain = (generation_axis,)
    partition_domain = (generation_axis, partition_axis)
    query_stage = TaskFamily("query_stage", generation_domain, placement="transfer_workers")
    key_stage = TaskFamily("key_stage", partition_domain, placement="transfer_workers")
    value_stage = TaskFamily("value_stage", partition_domain, placement="transfer_workers")
    first_contract = TaskFamily(descriptor.first_contract_name, partition_domain, placement="matrix_workers")
    fold_update = TaskFamily(descriptor.fold_update_name, partition_domain, placement="matrix_workers")
    second_contract = TaskFamily(descriptor.second_contract_name, partition_domain, placement="matrix_workers")
    finalize = TaskFamily(descriptor.finalize_name, generation_domain, placement="matrix_workers")
    visibility = MemoryVisibility(EventMemoryScope.CTA)

    pointwise = tuple(
        ((generation, partition), (generation, partition))
        for generation in range(generation_count)
        for partition in range(partition_count)
    )
    query_to_first_contract = TaskDependence(
        TaskRelation.from_pairs(
            query_stage,
            first_contract,
            tuple(
                ((generation,), (generation, partition))
                for generation in range(generation_count)
                for partition in range(partition_count)
            ),
        ),
        visibility,
    )
    key_to_first_contract = TaskDependence(TaskRelation.from_pairs(key_stage, first_contract, pointwise), visibility)
    first_contract_to_fold = TaskDependence(TaskRelation.from_pairs(first_contract, fold_update, pointwise), visibility)
    fold_to_second_contract = TaskDependence(
        TaskRelation.from_pairs(fold_update, second_contract, pointwise), visibility
    )
    value_to_second_contract = TaskDependence(
        TaskRelation.from_pairs(value_stage, second_contract, pointwise), visibility
    )
    second_contract_to_next_first_contract = TaskDependence(
        TaskRelation.from_pairs(
            second_contract,
            first_contract,
            tuple(
                ((generation, partition), (generation, partition + 1))
                for generation in range(generation_count)
                for partition in range(partition_count - 1)
            ),
        ),
        visibility,
    )
    second_contract_to_finalize = TaskDependence(
        TaskRelation.from_pairs(
            second_contract,
            finalize,
            tuple(((generation, partition_count - 1), (generation,)) for generation in range(generation_count)),
        ),
        visibility,
    )
    initial_dependences = (
        query_to_first_contract,
        key_to_first_contract,
        first_contract_to_fold,
        fold_to_second_contract,
        value_to_second_contract,
        second_contract_to_next_first_contract,
        second_contract_to_finalize,
    )
    families = (query_stage, key_stage, value_stage, first_contract, fold_update, second_contract, finalize)
    initial_program = _event_program(families, initial_dependences)

    query_buffer = derive_bounded_buffer_plan(
        name="query_pipeline",
        program=initial_program,
        producer=query_stage,
        uses=(query_to_first_contract.relation,),
        capacity=1,
        slot_for={(generation,): 0 for generation in range(generation_count)},
        generation_for={(generation,): generation for generation in range(generation_count)},
        visibility=visibility,
    )

    def linear_index(coordinate: tuple[int, ...]) -> int:
        generation, partition = coordinate
        return generation * partition_count + partition

    pipeline_coordinates = key_stage.coordinates
    slots = {coordinate: linear_index(coordinate) % pipeline_depth for coordinate in pipeline_coordinates}
    generations = {coordinate: linear_index(coordinate) // pipeline_depth for coordinate in pipeline_coordinates}
    key_buffer = derive_bounded_buffer_plan(
        name="key_pipeline",
        program=initial_program,
        producer=key_stage,
        uses=(key_to_first_contract.relation,),
        capacity=pipeline_depth,
        slot_for=slots,
        generation_for=generations,
        visibility=visibility,
    )
    value_buffer = derive_bounded_buffer_plan(
        name="value_pipeline",
        program=initial_program,
        producer=value_stage,
        uses=(value_to_second_contract.relation,),
        capacity=pipeline_depth,
        slot_for=slots,
        generation_for=generations,
        visibility=visibility,
    )
    dependences = (
        *initial_dependences,
        *query_buffer.reuse_dependences,
        *key_buffer.reuse_dependences,
        *value_buffer.reuse_dependences,
    )
    complete_program = _event_program(families, dependences)
    verify_event_dataflow_program(complete_program)
    return StreamingPipelineDataflow(
        program=complete_program,
        query_stage=query_stage,
        key_stage=key_stage,
        value_stage=value_stage,
        first_contract=first_contract,
        fold_update=fold_update,
        second_contract=second_contract,
        finalize=finalize,
        partition_count=partition_count,
        generation_count=generation_count,
        query_buffer=query_buffer,
        key_buffer=key_buffer,
        value_buffer=value_buffer,
    )


def _event_program(
    families: tuple[TaskFamily, ...],
    dependences: tuple[TaskDependence, ...],
) -> EventDataflowProgram:
    plans = tuple(
        derive_event_tensor_plan(
            dependence,
            name=f"event_{index}_{dependence.relation.source.name}_to_{dependence.relation.target.name}",
            generation_policy=(
                EventGenerationPolicy.PHASED
                if dependence.relation.target.name.endswith("_stage")
                else EventGenerationPolicy.PER_INVOCATION
            ),
        )
        for index, dependence in enumerate(dependences)
    )
    return EventDataflowProgram(families, dependences, plans)


def _realization_audit(
    dataflow: StreamingPipelineDataflow,
    workers: StreamingWorkerAssignment,
) -> EventRealizationAudit:
    reuse_relations = {
        dependence.relation
        for buffer in (dataflow.query_buffer, dataflow.key_buffer, dataflow.value_buffer)
        for dependence in buffer.reuse_dependences
    }
    realizations = []
    for plan in dataflow.program.event_plans:
        relation = plan.required_dependence.relation
        endpoints = (relation.source, relation.target)
        if relation in reuse_relations:
            buffer_name = {
                dataflow.query_stage.name: "Q",
                dataflow.key_stage.name: "K",
                dataflow.value_stage.name: "V",
            }[relation.target.name]
            realizations.append(
                physical_event_realization(
                    plan,
                    mechanism=f"{buffer_name} pipeline empty barrier with phased slot generation",
                    reason="the derived last consumer must release a bounded slot before reuse",
                )
            )
        elif relation.source in (dataflow.query_stage, dataflow.key_stage, dataflow.value_stage):
            payload = {
                dataflow.query_stage.name: "Q",
                dataflow.key_stage.name: "K",
                dataflow.value_stage.name: "V",
            }[relation.source.name]
            realizations.append(
                physical_event_realization(
                    plan,
                    mechanism=f"{payload} pipeline full barrier",
                    reason="matrix consumers require acquire visibility after asynchronous tile movement",
                )
            )
        elif endpoints == (dataflow.second_contract, dataflow.first_contract) and workers.matrix_warpgroups > 1:
            realizations.append(
                physical_event_realization(
                    plan,
                    mechanism="pairwise matrix-warpgroup scheduler barrier",
                    reason="the selected overlap schedule hands ordered Fold state between matrix warpgroups",
                )
            )
        else:
            realizations.append(
                erased_event_realization(
                    plan,
                    kind=EventRealizationKind.ERASED_PROGRAM_ORDER,
                    mechanism="matrix-owner program order",
                    reason="the selected skeleton executes this dependence in one ordered tile body",
                )
            )
    return verify_event_realizations(dataflow.program, tuple(realizations))
