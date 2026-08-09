# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from tile_lifetime.event_buffering import EventRealizationKind
from tile_lifetime.ir import DType
from tile_lifetime.streaming_attention import (
    StreamingTileSchedule,
    apply_causal_score_mask,
    build_attention_tensor_program,
    derive_streaming_attention,
    scaled_score_map,
)
from tile_lifetime.streaming_event_schedule import (
    derive_streaming_physical_event_schedule,
    verify_streaming_event_backend_parameters,
)


def _program(*, query_tile: int = 128, key_length: int = 512, pipeline_depth: int = 3):
    semantic = build_attention_tensor_program(
        batch_size=1,
        query_length=256,
        key_length=key_length,
        query_heads=4,
        key_value_heads=2,
        key_dimension=128,
        value_dimension=128,
        score_map=apply_causal_score_mask(scaled_score_map(128**-0.5)),
        input_dtype=DType.BF16,
    )
    return derive_streaming_attention(
        semantic,
        schedule=StreamingTileSchedule(
            query_tile_size=query_tile,
            key_value_tile_size=64,
            pipeline_depth=pipeline_depth,
        ),
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
    assert {consumer.family for consumer in query_last_consumers.values()} == {schedule.dataflow.qk_contract.name}
    assert {consumer.family for consumer in key_last_consumers.values()} == {schedule.dataflow.qk_contract.name}
    assert {consumer.family for consumer in value_last_consumers.values()} == {schedule.dataflow.pv_contract.name}
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
    one_scheduler_edges = tuple(entry for entry in one_warpgroup.audit.entries if entry.plan_name.endswith("pv_to_qk"))
    two_scheduler_edges = tuple(entry for entry in two_warpgroups.audit.entries if entry.plan_name.endswith("pv_to_qk"))
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
