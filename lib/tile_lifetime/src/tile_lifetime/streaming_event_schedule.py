# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Event Tensor attachment for a generic streaming Contract/Fold schedule.

This module is below tensor semantics and above CUDA synchronization.  It
derives the logical producer/consumer and buffer-reuse edges implemented by a
resident-left streaming contraction skeleton. Backend-specific barrier IDs and
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
    into two Contracts around an ordered Fold. It deliberately carries no
    tensor-operation or model identity. Those semantics determine the task
    graph before this schedule boundary; Event Tensor construction only needs
    task names, extents, payload widths, and the selected bounded pipeline
    shape.
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
    resident_input_stage: TaskFamily
    first_streamed_input_stage: TaskFamily
    second_streamed_input_stage: TaskFamily
    first_contract: TaskFamily
    fold_update: TaskFamily
    second_contract: TaskFamily
    finalize: TaskFamily
    partition_count: int
    generation_count: int
    resident_input_buffer: BoundedBufferPlan
    first_streamed_input_buffer: BoundedBufferPlan
    second_streamed_input_buffer: BoundedBufferPlan


@dataclass(frozen=True)
class StreamingPhysicalEventSchedule:
    """Verified readiness configuration consumed by a physical GPU skeleton."""

    dataflow: StreamingPipelineDataflow
    audit: EventRealizationAudit
    workers: StreamingWorkerAssignment
    resident_input_stages: int
    first_streamed_input_stages: int
    second_streamed_input_stages: int
    barriers_per_stage: int
    resident_input_transaction_bytes: int
    first_streamed_input_transaction_bytes: int
    second_streamed_input_transaction_bytes: int
    fingerprint: str

    @property
    def resident_input_barrier_slots(self) -> int:
        """Return full/empty barrier storage for the resident input."""
        return self.resident_input_stages * self.barriers_per_stage

    @property
    def first_streamed_input_barrier_slots(self) -> int:
        """Return full/empty barrier storage for the first streamed input."""
        return self.first_streamed_input_stages * self.barriers_per_stage

    @property
    def second_streamed_input_barrier_slots(self) -> int:
        """Return full/empty barrier storage for the second streamed input."""
        return self.second_streamed_input_stages * self.barriers_per_stage


def verify_streaming_event_backend_parameters(
    schedule: StreamingPhysicalEventSchedule,
    *,
    resident_input_stages: int,
    first_streamed_input_stages: int,
    second_streamed_input_stages: int,
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
        "resident_input_stages": resident_input_stages,
        "first_streamed_input_stages": first_streamed_input_stages,
        "second_streamed_input_stages": second_streamed_input_stages,
        "barriers_per_stage": barriers_per_stage,
        "transfer_warps": transfer_warps,
        "matrix_warpgroups": matrix_warpgroups,
        "scheduler_arrival_threads": scheduler_arrival_threads,
    }
    expected = {
        "resident_input_stages": schedule.resident_input_stages,
        "first_streamed_input_stages": schedule.first_streamed_input_stages,
        "second_streamed_input_stages": schedule.second_streamed_input_stages,
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
    """Derive synchronization for a resident-left streaming skeleton.

    The task graph is a kernel-local template. Two resident-tile generations
    expose reuse, while every streamed partition is represented so pipeline
    generations and the final producer tail are exact.
    """
    resident_tile = descriptor.resident_tile_size
    streamed_tile = descriptor.streamed_tile_size
    partition_count = ceil(descriptor.fold_extent / streamed_tile)
    pipeline_depth = descriptor.pipeline_depth

    dataflow = _derive_pipeline_dataflow(
        descriptor,
        partition_count=partition_count,
        pipeline_depth=pipeline_depth,
        generation_count=2,
    )
    matrix_warpgroups = resident_tile // 64
    if matrix_warpgroups <= 0 or resident_tile % 64:
        raise ValueError("the physical worker decomposition requires resident tiles divisible by 64")
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
        "resident_tile": resident_tile,
        "streamed_tile": streamed_tile,
        "workers": {
            "transfer_warps": workers.transfer_warps,
            "matrix_warpgroups": workers.matrix_warpgroups,
            "cta_threads": workers.cta_threads,
        },
        "events": tuple((entry.plan_name, entry.kind.value, entry.mechanism) for entry in audit.entries),
        "buffers": {
            "resident_input": (
                dataflow.resident_input_buffer.capacity,
                dataflow.resident_input_buffer.generations,
            ),
            "first_streamed_input": (
                dataflow.first_streamed_input_buffer.capacity,
                dataflow.first_streamed_input_buffer.generations,
            ),
            "second_streamed_input": (
                dataflow.second_streamed_input_buffer.capacity,
                dataflow.second_streamed_input_buffer.generations,
            ),
        },
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return StreamingPhysicalEventSchedule(
        dataflow=dataflow,
        audit=audit,
        workers=workers,
        resident_input_stages=1,
        first_streamed_input_stages=pipeline_depth,
        second_streamed_input_stages=pipeline_depth,
        barriers_per_stage=2,
        resident_input_transaction_bytes=(
            resident_tile * descriptor.resident_reduction_dimension * descriptor.element_bytes
        ),
        first_streamed_input_transaction_bytes=(
            streamed_tile * descriptor.streamed_reduction_dimension * descriptor.element_bytes
        ),
        second_streamed_input_transaction_bytes=(streamed_tile * descriptor.output_dimension * descriptor.element_bytes),
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
    resident_input_stage = TaskFamily("resident_input_stage", generation_domain, placement="transfer_workers")
    first_streamed_input_stage = TaskFamily("first_streamed_input_stage", partition_domain, placement="transfer_workers")
    second_streamed_input_stage = TaskFamily(
        "second_streamed_input_stage", partition_domain, placement="transfer_workers"
    )
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
    resident_input_to_first_contract = TaskDependence(
        TaskRelation.from_pairs(
            resident_input_stage,
            first_contract,
            tuple(
                ((generation,), (generation, partition))
                for generation in range(generation_count)
                for partition in range(partition_count)
            ),
        ),
        visibility,
    )
    first_streamed_input_to_first_contract = TaskDependence(
        TaskRelation.from_pairs(first_streamed_input_stage, first_contract, pointwise), visibility
    )
    first_contract_to_fold = TaskDependence(TaskRelation.from_pairs(first_contract, fold_update, pointwise), visibility)
    fold_to_second_contract = TaskDependence(
        TaskRelation.from_pairs(fold_update, second_contract, pointwise), visibility
    )
    second_streamed_input_to_second_contract = TaskDependence(
        TaskRelation.from_pairs(second_streamed_input_stage, second_contract, pointwise), visibility
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
        resident_input_to_first_contract,
        first_streamed_input_to_first_contract,
        first_contract_to_fold,
        fold_to_second_contract,
        second_streamed_input_to_second_contract,
        second_contract_to_next_first_contract,
        second_contract_to_finalize,
    )
    families = (
        resident_input_stage,
        first_streamed_input_stage,
        second_streamed_input_stage,
        first_contract,
        fold_update,
        second_contract,
        finalize,
    )
    initial_program = _event_program(families, initial_dependences)

    resident_input_buffer = derive_bounded_buffer_plan(
        name="resident_input_pipeline",
        program=initial_program,
        producer=resident_input_stage,
        uses=(resident_input_to_first_contract.relation,),
        capacity=1,
        slot_for={(generation,): 0 for generation in range(generation_count)},
        generation_for={(generation,): generation for generation in range(generation_count)},
        visibility=visibility,
    )

    def linear_index(coordinate: tuple[int, ...]) -> int:
        generation, partition = coordinate
        return generation * partition_count + partition

    pipeline_coordinates = first_streamed_input_stage.coordinates
    slots = {coordinate: linear_index(coordinate) % pipeline_depth for coordinate in pipeline_coordinates}
    generations = {coordinate: linear_index(coordinate) // pipeline_depth for coordinate in pipeline_coordinates}
    first_streamed_input_buffer = derive_bounded_buffer_plan(
        name="first_streamed_input_pipeline",
        program=initial_program,
        producer=first_streamed_input_stage,
        uses=(first_streamed_input_to_first_contract.relation,),
        capacity=pipeline_depth,
        slot_for=slots,
        generation_for=generations,
        visibility=visibility,
    )
    second_streamed_input_buffer = derive_bounded_buffer_plan(
        name="second_streamed_input_pipeline",
        program=initial_program,
        producer=second_streamed_input_stage,
        uses=(second_streamed_input_to_second_contract.relation,),
        capacity=pipeline_depth,
        slot_for=slots,
        generation_for=generations,
        visibility=visibility,
    )
    dependences = (
        *initial_dependences,
        *resident_input_buffer.reuse_dependences,
        *first_streamed_input_buffer.reuse_dependences,
        *second_streamed_input_buffer.reuse_dependences,
    )
    complete_program = _event_program(families, dependences)
    verify_event_dataflow_program(complete_program)
    return StreamingPipelineDataflow(
        program=complete_program,
        resident_input_stage=resident_input_stage,
        first_streamed_input_stage=first_streamed_input_stage,
        second_streamed_input_stage=second_streamed_input_stage,
        first_contract=first_contract,
        fold_update=fold_update,
        second_contract=second_contract,
        finalize=finalize,
        partition_count=partition_count,
        generation_count=generation_count,
        resident_input_buffer=resident_input_buffer,
        first_streamed_input_buffer=first_streamed_input_buffer,
        second_streamed_input_buffer=second_streamed_input_buffer,
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
        for buffer in (
            dataflow.resident_input_buffer,
            dataflow.first_streamed_input_buffer,
            dataflow.second_streamed_input_buffer,
        )
        for dependence in buffer.reuse_dependences
    }
    realizations = []
    for plan in dataflow.program.event_plans:
        relation = plan.required_dependence.relation
        endpoints = (relation.source, relation.target)
        if relation in reuse_relations:
            buffer_name = {
                dataflow.resident_input_stage.name: "resident input",
                dataflow.first_streamed_input_stage.name: "first streamed input",
                dataflow.second_streamed_input_stage.name: "second streamed input",
            }[relation.target.name]
            realizations.append(
                physical_event_realization(
                    plan,
                    mechanism=f"{buffer_name} pipeline empty barrier with phased slot generation",
                    reason="the derived last consumer must release a bounded slot before reuse",
                )
            )
        elif relation.source in (
            dataflow.resident_input_stage,
            dataflow.first_streamed_input_stage,
            dataflow.second_streamed_input_stage,
        ):
            payload = {
                dataflow.resident_input_stage.name: "resident input",
                dataflow.first_streamed_input_stage.name: "first streamed input",
                dataflow.second_streamed_input_stage.name: "second streamed input",
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
