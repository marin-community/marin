# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose runtime segment readiness with a grouped-Contract tile schedule."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from tile_lifetime.event_buffering import (
    EventRealizationAudit,
    EventRealizationKind,
    erased_event_realization,
    verify_event_realizations,
)
from tile_lifetime.event_dataflow import EventMemoryScope, EventTensorRuntimeInputs, event_tensor_runtime_inputs
from tile_lifetime.event_dataflow_adapters import (
    SegmentedContractTaskDataflow,
    relation_segmented_contract_task_dataflow,
)
from tile_lifetime.grouped_contract_event_schedule import (
    GroupedContractPhysicalEventSchedule,
    GroupedContractSynchronizationDescriptor,
    derive_grouped_contract_physical_event_schedule,
)
from tile_lifetime.relation import RelationPlan


@dataclass(frozen=True)
class SegmentedGroupedContractEventSchedule:
    """Hierarchical Event Tensor plan around one generic grouped Contract.

    ``segment_readiness`` decides when an output tile may start from a runtime
    relation. ``contract_pipeline`` decides readiness inside the selected
    physical tile template. Keeping the two plans separate prevents relation
    indegrees from being mistaken for cluster transaction-barrier counts.
    """

    segment_readiness: SegmentedContractTaskDataflow
    segment_runtime_inputs: EventTensorRuntimeInputs
    segment_realization: EventRealizationAudit
    contract_pipeline: GroupedContractPhysicalEventSchedule
    program_fingerprint: str
    runtime_fingerprint: str


def derive_same_stream_segmented_grouped_contract_schedule(
    relation: RelationPlan,
    *,
    output_tile_count: int,
    descriptor: GroupedContractSynchronizationDescriptor,
    reduction_partition_count: int,
    generation_count: int = 2,
) -> SegmentedGroupedContractEventSchedule:
    """Derive a device-stream realization around a grouped Contract.

    This first executable candidate intentionally coarsens outer relation
    readiness to one same-stream boundary. It does not claim producer/consumer
    overlap. The inner grouped-Contract schedule remains independently derived
    from its worker and bounded-buffer decomposition.
    """
    segment_readiness = relation_segmented_contract_task_dataflow(
        relation,
        output_tile_count=output_tile_count,
        visibility_scope=EventMemoryScope.DEVICE,
    )
    if len(segment_readiness.program.event_plans) != 1:
        raise ValueError("segmented grouped Contract requires one outer readiness plan")
    outer_plan = segment_readiness.program.event_plans[0]
    segment_realization = verify_event_realizations(
        segment_readiness.program,
        (
            erased_event_realization(
                outer_plan,
                kind=EventRealizationKind.ERASED_STREAM_ORDER,
                mechanism="same JAX device stream",
                reason="grouping and padding complete before the grouped Contract launch on the same stream",
            ),
        ),
    )
    segment_runtime_inputs = event_tensor_runtime_inputs(outer_plan)
    contract_pipeline = derive_grouped_contract_physical_event_schedule(
        descriptor,
        reduction_partition_count=reduction_partition_count,
        generation_count=generation_count,
    )
    program_payload = {
        "outer": {
            "producer_axes": tuple(axis.extent for axis in outer_plan.notify_relation.source.axes),
            "consumer_axes": tuple(axis.extent for axis in outer_plan.trigger_relation.target.axes),
            "output_tiles": output_tile_count,
            "memory_scope": outer_plan.memory_scope.value,
            "realization": EventRealizationKind.ERASED_STREAM_ORDER.value,
        },
        "inner": contract_pipeline.fingerprint,
    }
    runtime_payload = {
        "counts": segment_runtime_inputs.event_initial_counts,
        "source_offsets": segment_runtime_inputs.event_source_offsets,
        "sources": segment_runtime_inputs.event_sources,
        "trigger_offsets": segment_runtime_inputs.event_trigger_offsets,
        "consumers": segment_runtime_inputs.event_consumers,
    }
    program_fingerprint = hashlib.sha256(
        json.dumps(program_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    runtime_fingerprint = hashlib.sha256(
        json.dumps(runtime_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return SegmentedGroupedContractEventSchedule(
        segment_readiness=segment_readiness,
        segment_runtime_inputs=segment_runtime_inputs,
        segment_realization=segment_realization,
        contract_pipeline=contract_pipeline,
        program_fingerprint=program_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
    )
