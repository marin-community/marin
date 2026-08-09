# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Event Tensor attachment for a clustered grouped-Contract skeleton.

This module describes the synchronization interface of a generic physical
Contract template.  It deliberately does not know how rows were grouped, what
an expert is, or which model requested the Contract.  A preceding segmented
schedule decides which output tiles exist; this layer decomposes one such tile
into cooperative operand transfers, ordered matrix work, and output
finalization.

Logical Event Tensor indegrees and physical transaction completion are kept
separate.  In particular, several cooperative TMA owners may notify one
logical operand-ready event even when the selected GPU primitive realizes that
event with one transaction barrier plus an expected-byte count.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum

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
    EventTensorPlan,
    MemoryVisibility,
    TaskAxis,
    TaskDependence,
    TaskFamily,
    TaskRelation,
    derive_event_tensor_plan,
    verify_event_dataflow_program,
)


class GroupedContractReleasePoint(StrEnum):
    """The point after which a bounded physical resource may be reused."""

    MATRIX_OPERAND_CONSUMED = "matrix_operand_consumed"
    ACCUMULATOR_READ_COMPLETE = "accumulator_read_complete"


@dataclass(frozen=True)
class GroupedContractWorkerAssignment:
    """Generic cluster roles selected for one grouped-Contract tile."""

    cluster_ctas: int
    transfer_warpgroups_per_cta: int
    matrix_warpgroups: int
    epilogue_warpgroups_per_cta: int
    matrix_owner_cta: int

    def __post_init__(self) -> None:
        if self.cluster_ctas <= 0:
            raise ValueError("a grouped Contract requires at least one cluster CTA")
        if self.transfer_warpgroups_per_cta <= 0:
            raise ValueError("each CTA requires at least one transfer warpgroup")
        if self.matrix_warpgroups <= 0:
            raise ValueError("a grouped Contract requires at least one matrix warpgroup")
        if self.epilogue_warpgroups_per_cta <= 0:
            raise ValueError("each CTA requires at least one epilogue warpgroup")
        if not 0 <= self.matrix_owner_cta < self.cluster_ctas:
            raise ValueError("the matrix-owner CTA is outside the cluster")

    @property
    def cooperative_transfer_owners(self) -> int:
        """Return owners contributing one cluster-wide operand stage."""
        return self.cluster_ctas * self.transfer_warpgroups_per_cta

    @property
    def matrix_owners(self) -> int:
        """Return independently notifying matrix owners."""
        return self.matrix_warpgroups

    @property
    def epilogue_owners(self) -> int:
        """Return owners that release one accumulator generation."""
        return self.cluster_ctas * self.epilogue_warpgroups_per_cta


@dataclass(frozen=True)
class GroupedContractSynchronizationDescriptor:
    """Physical-template facts needed to derive intra-Contract readiness.

    ``operand_bytes_per_transfer_owner`` describes byte completion for one
    cooperative owner's operand payload.  It is not an Event Tensor count.
    """

    workers: GroupedContractWorkerAssignment
    load_pipeline_stages: int
    operand_bytes_per_transfer_owner: int
    operand_release_point: GroupedContractReleasePoint
    output_release_point: GroupedContractReleasePoint
    uses_scale_pipeline: bool = False
    scale_bytes_per_transfer_owner: int = 0

    def __post_init__(self) -> None:
        if self.load_pipeline_stages <= 0:
            raise ValueError("grouped Contract load-pipeline stages must be positive")
        if self.operand_bytes_per_transfer_owner <= 0:
            raise ValueError("operand bytes per transfer owner must be positive")
        if self.operand_release_point is not GroupedContractReleasePoint.MATRIX_OPERAND_CONSUMED:
            raise ValueError("the initial grouped-Contract template releases operands after matrix consumption")
        if self.output_release_point is not GroupedContractReleasePoint.ACCUMULATOR_READ_COMPLETE:
            raise ValueError("the initial grouped-Contract template releases output storage after accumulator reads")
        if self.uses_scale_pipeline != (self.scale_bytes_per_transfer_owner > 0):
            raise ValueError("scale-pipeline presence and scale payload bytes must agree")


@dataclass(frozen=True)
class GroupedContractPipelineDataflow:
    """Kernel-local task graph for repeated grouped-Contract output tiles."""

    program: EventDataflowProgram
    accumulator_acquire: TaskFamily
    operand_stage: TaskFamily
    scale_stage: TaskFamily | None
    matrix_issue: TaskFamily
    epilogue: TaskFamily
    operand_buffer: BoundedBufferPlan
    scale_buffer: BoundedBufferPlan | None
    reduction_partition_count: int
    generation_count: int


@dataclass(frozen=True)
class GroupedContractPhysicalEventSchedule:
    """Verified logical events and physical realization requirements."""

    descriptor: GroupedContractSynchronizationDescriptor
    dataflow: GroupedContractPipelineDataflow
    audit: EventRealizationAudit
    operand_ready_count: int
    operand_release_count: int
    output_ready_count: int
    output_release_count: int
    operand_transaction_bytes: int
    scale_transaction_bytes: int
    transaction_completion_enabled: bool
    fingerprint: str


def derive_grouped_contract_physical_event_schedule(
    descriptor: GroupedContractSynchronizationDescriptor,
    *,
    reduction_partition_count: int,
    generation_count: int = 2,
) -> GroupedContractPhysicalEventSchedule:
    """Derive intra-Contract Event Tensors from a generic worker template."""
    if reduction_partition_count <= 0:
        raise ValueError("grouped Contract reduction partitions must be positive")
    if generation_count <= 0:
        raise ValueError("grouped Contract generations must be positive")

    dataflow = _derive_grouped_contract_dataflow(
        descriptor,
        reduction_partition_count=reduction_partition_count,
        generation_count=generation_count,
    )
    audit = _realization_audit(dataflow)
    operand_ready = _plan(dataflow, dataflow.operand_stage, dataflow.matrix_issue)
    operand_reuse = _plan(dataflow, dataflow.matrix_issue, dataflow.operand_stage)
    output_ready = _plan(dataflow, dataflow.matrix_issue, dataflow.epilogue)
    output_release = _plan(dataflow, dataflow.epilogue, dataflow.accumulator_acquire)
    operand_ready_count = _nonzero_uniform_count(operand_ready)
    operand_release_count = _nonzero_uniform_count(operand_reuse)
    output_ready_count = _nonzero_uniform_count(output_ready)
    output_release_count = _nonzero_uniform_count(output_release)
    operand_transaction_bytes = (
        descriptor.workers.cooperative_transfer_owners * descriptor.operand_bytes_per_transfer_owner
    )
    scale_transaction_bytes = descriptor.workers.cooperative_transfer_owners * descriptor.scale_bytes_per_transfer_owner
    fingerprint_payload = {
        "workers": {
            "cluster_ctas": descriptor.workers.cluster_ctas,
            "transfer_warpgroups_per_cta": descriptor.workers.transfer_warpgroups_per_cta,
            "matrix_warpgroups": descriptor.workers.matrix_warpgroups,
            "epilogue_warpgroups_per_cta": descriptor.workers.epilogue_warpgroups_per_cta,
            "matrix_owner_cta": descriptor.workers.matrix_owner_cta,
        },
        "pipeline": {
            "stages": descriptor.load_pipeline_stages,
            "reduction_partitions": reduction_partition_count,
            "generations": generation_count,
        },
        "logical_counts": {
            "operand_ready": operand_ready_count,
            "operand_release": operand_release_count,
            "output_ready": output_ready_count,
            "output_release": output_release_count,
        },
        "transaction_bytes": {
            "operand": operand_transaction_bytes,
            "scale": scale_transaction_bytes,
        },
        "release_points": {
            "operand": descriptor.operand_release_point.value,
            "output": descriptor.output_release_point.value,
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return GroupedContractPhysicalEventSchedule(
        descriptor=descriptor,
        dataflow=dataflow,
        audit=audit,
        operand_ready_count=operand_ready_count,
        operand_release_count=operand_release_count,
        output_ready_count=output_ready_count,
        output_release_count=output_release_count,
        operand_transaction_bytes=operand_transaction_bytes,
        scale_transaction_bytes=scale_transaction_bytes,
        transaction_completion_enabled=True,
        fingerprint=fingerprint,
    )


def verify_grouped_contract_backend_parameters(
    schedule: GroupedContractPhysicalEventSchedule,
    *,
    cluster_ctas: int,
    load_pipeline_stages: int,
    operand_release_count: int,
    output_release_count: int,
) -> None:
    """Reject a grouped-Contract wrapper whose synchronization ABI drifted."""
    expected = {
        "cluster_ctas": schedule.descriptor.workers.cluster_ctas,
        "load_pipeline_stages": schedule.descriptor.load_pipeline_stages,
        "operand_release_count": schedule.operand_release_count,
        "output_release_count": schedule.output_release_count,
    }
    actual = {
        "cluster_ctas": cluster_ctas,
        "load_pipeline_stages": load_pipeline_stages,
        "operand_release_count": operand_release_count,
        "output_release_count": output_release_count,
    }
    mismatches = tuple(
        f"{name}: expected {expected[name]}, found {actual[name]}" for name in expected if expected[name] != actual[name]
    )
    if mismatches:
        raise ValueError("grouped Contract Event Tensor/backend mismatch: " + "; ".join(mismatches))


def _derive_grouped_contract_dataflow(
    descriptor: GroupedContractSynchronizationDescriptor,
    *,
    reduction_partition_count: int,
    generation_count: int,
) -> GroupedContractPipelineDataflow:
    generation_axis = TaskAxis("generation", generation_count)
    partition_axis = TaskAxis("reduction_partition", reduction_partition_count)
    transfer_owner_axis = TaskAxis("transfer_owner", descriptor.workers.cooperative_transfer_owners)
    epilogue_owner_axis = TaskAxis("epilogue_owner", descriptor.workers.epilogue_owners)
    generation_domain = (generation_axis,)
    reduction_domain = (generation_axis, partition_axis)
    transfer_domain = (generation_axis, partition_axis, transfer_owner_axis)

    accumulator_acquire = TaskFamily("contract_accumulator_acquire", generation_domain, placement="matrix_workers")
    operand_stage = TaskFamily("contract_operand_stage", transfer_domain, placement="transfer_workers")
    scale_stage = (
        TaskFamily("contract_scale_stage", transfer_domain, placement="transfer_workers")
        if descriptor.uses_scale_pipeline
        else None
    )
    matrix_issue = TaskFamily("contract_matrix_issue", reduction_domain, placement="matrix_workers")
    epilogue = TaskFamily(
        "contract_epilogue",
        (generation_axis, epilogue_owner_axis),
        placement="epilogue_workers",
    )
    visibility = MemoryVisibility(EventMemoryScope.CLUSTER)

    acquire_to_first_matrix = TaskDependence(
        TaskRelation.from_pairs(
            accumulator_acquire,
            matrix_issue,
            tuple(((generation,), (generation, 0)) for generation in range(generation_count)),
        ),
        visibility,
    )
    operand_to_matrix = TaskDependence(
        TaskRelation.from_pairs(
            operand_stage,
            matrix_issue,
            tuple(
                ((generation, partition, owner), (generation, partition))
                for generation in range(generation_count)
                for partition in range(reduction_partition_count)
                for owner in range(descriptor.workers.cooperative_transfer_owners)
            ),
        ),
        visibility,
    )
    matrix_order = TaskDependence(
        TaskRelation.from_pairs(
            matrix_issue,
            matrix_issue,
            tuple(
                ((generation, partition), (generation, partition + 1))
                for generation in range(generation_count)
                for partition in range(reduction_partition_count - 1)
            ),
        ),
        visibility,
    )
    final_matrix_to_epilogue = TaskDependence(
        TaskRelation.from_pairs(
            matrix_issue,
            epilogue,
            tuple(
                ((generation, reduction_partition_count - 1), (generation, owner))
                for generation in range(generation_count)
                for owner in range(descriptor.workers.epilogue_owners)
            ),
        ),
        visibility,
    )
    epilogue_to_next_acquire = TaskDependence(
        TaskRelation.from_pairs(
            epilogue,
            accumulator_acquire,
            tuple(
                ((generation, owner), (generation + 1,))
                for generation in range(generation_count - 1)
                for owner in range(descriptor.workers.epilogue_owners)
            ),
        ),
        visibility,
    )
    scale_to_matrix = (
        TaskDependence(
            TaskRelation.from_pairs(
                scale_stage,
                matrix_issue,
                tuple(
                    ((generation, partition, owner), (generation, partition))
                    for generation in range(generation_count)
                    for partition in range(reduction_partition_count)
                    for owner in range(descriptor.workers.cooperative_transfer_owners)
                ),
            ),
            visibility,
        )
        if scale_stage is not None
        else None
    )
    initial_dependences = tuple(
        dependence
        for dependence in (
            acquire_to_first_matrix,
            operand_to_matrix,
            scale_to_matrix,
            matrix_order,
            final_matrix_to_epilogue,
            epilogue_to_next_acquire,
        )
        if dependence is not None
    )
    families = tuple(
        family
        for family in (accumulator_acquire, operand_stage, scale_stage, matrix_issue, epilogue)
        if family is not None
    )
    initial_program = _event_program(families, initial_dependences)

    def linear_partition(coordinate: tuple[int, ...]) -> int:
        generation, partition, _ = coordinate
        return generation * reduction_partition_count + partition

    operand_coordinates = operand_stage.coordinates
    transfer_owner_count = descriptor.workers.cooperative_transfer_owners
    operand_slots = {
        coordinate: (
            coordinate[2] * descriptor.load_pipeline_stages
            + linear_partition(coordinate) % descriptor.load_pipeline_stages
        )
        for coordinate in operand_coordinates
    }
    operand_generations = {
        coordinate: linear_partition(coordinate) // descriptor.load_pipeline_stages for coordinate in operand_coordinates
    }
    operand_buffer = derive_bounded_buffer_plan(
        name="contract_operand_pipeline",
        program=initial_program,
        producer=operand_stage,
        uses=(operand_to_matrix.relation,),
        capacity=transfer_owner_count * descriptor.load_pipeline_stages,
        slot_for=operand_slots,
        generation_for=operand_generations,
        visibility=visibility,
    )
    scale_buffer = None
    if scale_stage is not None:
        assert scale_to_matrix is not None
        scale_buffer = derive_bounded_buffer_plan(
            name="contract_scale_pipeline",
            program=initial_program,
            producer=scale_stage,
            uses=(scale_to_matrix.relation,),
            capacity=transfer_owner_count * descriptor.load_pipeline_stages,
            slot_for=operand_slots,
            generation_for=operand_generations,
            visibility=visibility,
        )
    reuse_dependences = (
        *operand_buffer.reuse_dependences,
        *(scale_buffer.reuse_dependences if scale_buffer is not None else ()),
    )
    complete_program = _event_program(families, (*initial_dependences, *reuse_dependences))
    verify_event_dataflow_program(complete_program)
    return GroupedContractPipelineDataflow(
        program=complete_program,
        accumulator_acquire=accumulator_acquire,
        operand_stage=operand_stage,
        scale_stage=scale_stage,
        matrix_issue=matrix_issue,
        epilogue=epilogue,
        operand_buffer=operand_buffer,
        scale_buffer=scale_buffer,
        reduction_partition_count=reduction_partition_count,
        generation_count=generation_count,
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
                if dependence.relation.target.name
                in {"contract_operand_stage", "contract_scale_stage", "contract_accumulator_acquire"}
                and dependence.relation.source.name in {"contract_matrix_issue", "contract_epilogue"}
                else EventGenerationPolicy.PER_INVOCATION
            ),
        )
        for index, dependence in enumerate(dependences)
    )
    return EventDataflowProgram(families, dependences, plans)


def _realization_audit(dataflow: GroupedContractPipelineDataflow) -> EventRealizationAudit:
    reuse_relations = {
        dependence.relation
        for buffer in (dataflow.operand_buffer, dataflow.scale_buffer)
        if buffer is not None
        for dependence in buffer.reuse_dependences
    }
    realizations = []
    for plan in dataflow.program.event_plans:
        relation = plan.required_dependence.relation
        endpoints = (relation.source, relation.target)
        if relation in reuse_relations:
            realizations.append(
                physical_event_realization(
                    plan,
                    mechanism="phased operand-stage release barrier",
                    reason="the matrix owner must release a bounded transfer slot before reuse",
                )
            )
        elif relation.source in {dataflow.operand_stage, dataflow.scale_stage}:
            realizations.append(
                physical_event_realization(
                    plan,
                    mechanism="cluster transaction-completion barrier",
                    reason="matrix work requires acquire visibility after every cooperative asynchronous transfer",
                )
            )
        elif endpoints == (dataflow.matrix_issue, dataflow.epilogue):
            realizations.append(
                physical_event_realization(
                    plan,
                    mechanism="cluster accumulator-ready barrier",
                    reason="epilogue owners consume accumulator storage produced by one matrix owner",
                )
            )
        elif endpoints == (dataflow.epilogue, dataflow.accumulator_acquire):
            realizations.append(
                physical_event_realization(
                    plan,
                    mechanism="phased cluster accumulator-release barrier",
                    reason="every epilogue owner must finish reading accumulator storage before reuse",
                )
            )
        else:
            realizations.append(
                erased_event_realization(
                    plan,
                    kind=EventRealizationKind.ERASED_PROGRAM_ORDER,
                    mechanism="matrix-owner program order",
                    reason="the selected skeleton executes this dependence in one ordered matrix task body",
                )
            )
    return verify_event_realizations(dataflow.program, tuple(realizations))


def _plan(
    dataflow: GroupedContractPipelineDataflow,
    source: TaskFamily,
    target: TaskFamily,
) -> EventTensorPlan:
    plans = tuple(
        plan
        for plan in dataflow.program.event_plans
        if plan.notify_relation.source == source and plan.trigger_relation.target == target
    )
    if len(plans) != 1:
        raise ValueError(f"expected one Event Tensor from {source.name} to {target.name}, found {len(plans)}")
    return plans[0]


def _nonzero_uniform_count(plan: EventTensorPlan) -> int:
    values = {count.value for count in plan.initial_count.counts if count.value > 0}
    if len(values) != 1:
        raise ValueError(f"Event Tensor {plan.name} does not have one nonzero cardinality: {sorted(values)}")
    return next(iter(values))
