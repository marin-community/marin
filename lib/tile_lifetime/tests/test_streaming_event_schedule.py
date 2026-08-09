# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from tile_lifetime.event_buffering import EventRealizationKind
from tile_lifetime.streaming_event_schedule import (
    StreamingContractFoldDescriptor,
    derive_streaming_physical_event_schedule,
    verify_streaming_event_backend_parameters,
)


def _descriptor(*, resident_tile: int = 128, fold_extent: int = 512, pipeline_depth: int = 3):
    return StreamingContractFoldDescriptor(
        first_contract_name="first_contract",
        fold_update_name="state_update",
        second_contract_name="second_contract",
        finalize_name="finalize",
        fold_extent=fold_extent,
        resident_tile_size=resident_tile,
        streamed_tile_size=64,
        pipeline_depth=pipeline_depth,
        resident_reduction_dimension=128,
        streamed_reduction_dimension=128,
        output_dimension=128,
        element_bytes=2,
    )


def test_streaming_event_schedule_derives_distinct_input_lifetimes() -> None:
    schedule = derive_streaming_physical_event_schedule(_descriptor())

    assert schedule.resident_input_barrier_slots == 2
    assert schedule.first_streamed_input_barrier_slots == 6
    assert schedule.second_streamed_input_barrier_slots == 6
    assert schedule.dataflow.partition_count == 8
    assert schedule.workers.matrix_warpgroups == 2
    assert schedule.workers.matrix_warps == 8
    assert schedule.workers.cta_threads == 384
    assert schedule.workers.scheduler_arrival_threads == 256

    resident_last_consumers = dict(schedule.dataflow.resident_input_buffer.last_consumers)
    first_streamed_last_consumers = dict(schedule.dataflow.first_streamed_input_buffer.last_consumers)
    second_streamed_last_consumers = dict(schedule.dataflow.second_streamed_input_buffer.last_consumers)
    assert {consumer.family for consumer in resident_last_consumers.values()} == {schedule.dataflow.first_contract.name}
    assert {consumer.family for consumer in first_streamed_last_consumers.values()} == {
        schedule.dataflow.first_contract.name
    }
    assert {consumer.family for consumer in second_streamed_last_consumers.values()} == {
        schedule.dataflow.second_contract.name
    }
    assert resident_last_consumers[(0,)].coordinate == (0, 7)

    physical_mechanisms = {entry.mechanism for entry in schedule.audit.physical}
    assert physical_mechanisms == {
        "resident input pipeline full barrier",
        "first streamed input pipeline full barrier",
        "second streamed input pipeline full barrier",
        "resident input pipeline empty barrier with phased slot generation",
        "first streamed input pipeline empty barrier with phased slot generation",
        "second streamed input pipeline empty barrier with phased slot generation",
        "pairwise matrix-warpgroup scheduler barrier",
    }


def test_pipeline_depth_mutation_regenerates_buffer_events() -> None:
    depth_two = derive_streaming_physical_event_schedule(_descriptor(pipeline_depth=2))
    depth_three = derive_streaming_physical_event_schedule(_descriptor(pipeline_depth=3))

    assert depth_two.first_streamed_input_stages == depth_two.second_streamed_input_stages == 2
    assert depth_three.first_streamed_input_stages == depth_three.second_streamed_input_stages == 3
    assert depth_two.first_streamed_input_barrier_slots == 4
    assert depth_three.first_streamed_input_barrier_slots == 6
    assert depth_two.dataflow.first_streamed_input_buffer.slots != depth_three.dataflow.first_streamed_input_buffer.slots
    assert (
        depth_two.dataflow.second_streamed_input_buffer.generations
        != depth_three.dataflow.second_streamed_input_buffer.generations
    )
    assert depth_two.fingerprint != depth_three.fingerprint


def test_worker_mutation_changes_scheduler_event_realization() -> None:
    one_warpgroup = derive_streaming_physical_event_schedule(_descriptor(resident_tile=64))
    two_warpgroups = derive_streaming_physical_event_schedule(_descriptor(resident_tile=128))

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
    schedule = derive_streaming_physical_event_schedule(_descriptor(pipeline_depth=3))

    verify_streaming_event_backend_parameters(
        schedule,
        resident_input_stages=1,
        first_streamed_input_stages=3,
        second_streamed_input_stages=3,
        barriers_per_stage=2,
        transfer_warps=1,
        matrix_warpgroups=2,
        scheduler_arrival_threads=256,
    )
    with pytest.raises(ValueError, match="first_streamed_input_stages: expected 3, found 2"):
        verify_streaming_event_backend_parameters(
            schedule,
            resident_input_stages=1,
            first_streamed_input_stages=2,
            second_streamed_input_stages=3,
            barriers_per_stage=2,
            transfer_warps=1,
            matrix_warpgroups=2,
            scheduler_arrival_threads=256,
        )
