# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from tile_lifetime.event_buffering import EventRealizationKind
from tile_lifetime.streaming_event_schedule import (
    StreamingContractFoldDescriptor,
    derive_streaming_physical_event_schedule,
    verify_streaming_event_backend_parameters,
)


def _program(*, query_tile: int = 128, key_length: int = 512, pipeline_depth: int = 3):
    return StreamingContractFoldDescriptor(
        first_contract_name="first_contract",
        fold_update_name="state_update",
        second_contract_name="second_contract",
        finalize_name="finalize",
        fold_extent=key_length,
        resident_tile_size=query_tile,
        streamed_tile_size=64,
        pipeline_depth=pipeline_depth,
        resident_reduction_dimension=128,
        streamed_reduction_dimension=128,
        output_dimension=128,
        element_bytes=2,
    )


def test_streaming_event_schedule_derives_distinct_q_k_v_lifetimes() -> None:
    schedule = derive_streaming_physical_event_schedule(_program())

    assert schedule.query_barrier_slots == 2
    assert schedule.key_barrier_slots == 6
    assert schedule.value_barrier_slots == 6
    assert schedule.dataflow.partition_count == 8
    assert schedule.workers.matrix_warpgroups == 2
    assert schedule.workers.matrix_warps == 8
    assert schedule.workers.cta_threads == 384
    assert schedule.workers.scheduler_arrival_threads == 256

    query_last_consumers = dict(schedule.dataflow.query_buffer.last_consumers)
    key_last_consumers = dict(schedule.dataflow.key_buffer.last_consumers)
    value_last_consumers = dict(schedule.dataflow.value_buffer.last_consumers)
    assert {consumer.family for consumer in query_last_consumers.values()} == {schedule.dataflow.first_contract.name}
    assert {consumer.family for consumer in key_last_consumers.values()} == {schedule.dataflow.first_contract.name}
    assert {consumer.family for consumer in value_last_consumers.values()} == {schedule.dataflow.second_contract.name}
    assert query_last_consumers[(0,)].coordinate == (0, 7)

    physical_mechanisms = {entry.mechanism for entry in schedule.audit.physical}
    assert physical_mechanisms == {
        "Q pipeline full barrier",
        "K pipeline full barrier",
        "V pipeline full barrier",
        "Q pipeline empty barrier with phased slot generation",
        "K pipeline empty barrier with phased slot generation",
        "V pipeline empty barrier with phased slot generation",
        "pairwise matrix-warpgroup scheduler barrier",
    }


def test_pipeline_depth_mutation_regenerates_buffer_events() -> None:
    depth_two = derive_streaming_physical_event_schedule(_program(pipeline_depth=2))
    depth_three = derive_streaming_physical_event_schedule(_program(pipeline_depth=3))

    assert depth_two.key_stages == depth_two.value_stages == 2
    assert depth_three.key_stages == depth_three.value_stages == 3
    assert depth_two.key_barrier_slots == 4
    assert depth_three.key_barrier_slots == 6
    assert depth_two.dataflow.key_buffer.slots != depth_three.dataflow.key_buffer.slots
    assert depth_two.dataflow.value_buffer.generations != depth_three.dataflow.value_buffer.generations
    assert depth_two.fingerprint != depth_three.fingerprint


def test_worker_mutation_changes_scheduler_event_realization() -> None:
    one_warpgroup = derive_streaming_physical_event_schedule(_program(query_tile=64))
    two_warpgroups = derive_streaming_physical_event_schedule(_program(query_tile=128))

    assert one_warpgroup.workers.matrix_warpgroups == 1
    assert one_warpgroup.workers.cta_threads == 256
    assert one_warpgroup.workers.scheduler_arrival_threads == 0
    assert two_warpgroups.workers.matrix_warpgroups == 2
    assert two_warpgroups.workers.cta_threads == 384
    assert two_warpgroups.workers.scheduler_arrival_threads == 256

    def scheduler_edges(schedule):
        plan_names = {
            plan.name
            for plan in schedule.dataflow.program.event_plans
            if plan.notify_relation.source == schedule.dataflow.second_contract
            and plan.trigger_relation.target == schedule.dataflow.first_contract
        }
        return tuple(entry for entry in schedule.audit.entries if entry.plan_name in plan_names)

    one_scheduler_edges = scheduler_edges(one_warpgroup)
    two_scheduler_edges = scheduler_edges(two_warpgroups)
    assert len(one_scheduler_edges) == len(two_scheduler_edges) == 1
    assert one_scheduler_edges[0].kind is EventRealizationKind.ERASED_PROGRAM_ORDER
    assert two_scheduler_edges[0].kind is EventRealizationKind.PHYSICAL


def test_backend_source_audit_rejects_stale_synchronization_constants() -> None:
    schedule = derive_streaming_physical_event_schedule(_program(pipeline_depth=3))

    verify_streaming_event_backend_parameters(
        schedule,
        query_stages=1,
        key_stages=3,
        value_stages=3,
        barriers_per_stage=2,
        transfer_warps=1,
        matrix_warpgroups=2,
        scheduler_arrival_threads=256,
    )
    with pytest.raises(ValueError, match="key_stages: expected 3, found 2"):
        verify_streaming_event_backend_parameters(
            schedule,
            query_stages=1,
            key_stages=2,
            value_stages=3,
            barriers_per_stage=2,
            transfer_warps=1,
            matrix_warpgroups=2,
            scheduler_arrival_threads=256,
        )
