# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from tile_lifetime.event_buffering import (
    EventRealizationKind,
    derive_bounded_buffer_plan,
    erased_event_realization,
    physical_event_realization,
    verify_event_realizations,
)
from tile_lifetime.event_dataflow import (
    EventDataflowError,
    EventMemoryScope,
    derive_event_tensor_plan,
    verify_event_dataflow_program,
)
from tile_lifetime.event_dataflow_adapters import (
    relation_segmented_contract_task_dataflow,
    streaming_contract_fold_event_descriptor,
    streaming_fold_task_dataflow,
)
from tile_lifetime.event_dataflow_examples import relation_segment_dependence
from tile_lifetime.ir import DType
from tile_lifetime.relation import build_relation_plan
from tile_lifetime.streaming_attention import (
    StreamingTileSchedule,
    apply_causal_score_mask,
    build_attention_tensor_program,
    derive_streaming_attention,
    scaled_score_map,
)
from tile_lifetime.streaming_event_schedule import derive_streaming_physical_event_schedule


def _streaming_program(*, key_length: int = 9, query_tile_size: int = 3):
    source = build_attention_tensor_program(
        batch_size=2,
        query_length=7,
        key_length=key_length,
        query_heads=4,
        key_value_heads=2,
        key_dimension=8,
        value_dimension=8,
        score_map=apply_causal_score_mask(scaled_score_map(0.5)),
        input_dtype=DType.BF16,
    )
    return derive_streaming_attention(
        source,
        schedule=StreamingTileSchedule(
            query_tile_size=query_tile_size,
            key_value_tile_size=4,
            pipeline_depth=2,
        ),
    )


def test_streaming_program_mechanically_derives_contract_fold_task_graph() -> None:
    streaming = _streaming_program()

    adapted = streaming_fold_task_dataflow(streaming)

    verify_event_dataflow_program(adapted.program)
    assert adapted.qk_contract.name == streaming.qk.name
    assert adapted.pv_contract.name == streaming.pv.name
    assert adapted.finalize.name == streaming.finalize.name
    assert adapted.row_tile_count == 24
    assert adapted.fold_partition_count == 3
    assert adapted.pipeline_depth == streaming.schedule.pipeline_depth
    assert tuple(plan.initial_count.counts[0].value for plan in adapted.program.event_plans) == (1, 1, 1, 1, 3, 0)
    assert all("attention" not in family.name for family in adapted.program.task_families)
    assert adapted.key_value_buffer.capacity == adapted.row_tile_count * adapted.pipeline_depth
    assert len(adapted.key_value_buffer.reuse_dependences) == 1

    last_consumers = dict(adapted.key_value_buffer.last_consumers)
    assert set(item.family for item in last_consumers.values()) == {streaming.pv.name}
    reuse = adapted.key_value_buffer.reuse_dependences[0].relation
    assert ((0, 0), (0, 2)) in {(pair.source, pair.target) for pair in reuse.pairs}
    assert ((0, 1), (0, 3)) not in {(pair.source, pair.target) for pair in reuse.pairs}


def test_fold_domain_mutation_changes_events_without_a_workload_dispatch() -> None:
    primary = streaming_fold_task_dataflow(_streaming_program(key_length=9))
    mutated = streaming_fold_task_dataflow(_streaming_program(key_length=17))

    assert primary.fold_partition_count == 3
    assert mutated.fold_partition_count == 5
    assert primary.program.task_families[0].name == mutated.program.task_families[0].name
    primary_finalize = next(
        plan for plan in primary.program.event_plans if plan.trigger_relation.target == primary.finalize
    )
    mutated_finalize = next(
        plan for plan in mutated.program.event_plans if plan.trigger_relation.target == mutated.finalize
    )
    assert primary_finalize.initial_count.counts[0].value == 3
    assert mutated_finalize.initial_count.counts[0].value == 5


def test_attention_program_erases_to_generic_streaming_event_descriptor() -> None:
    semantic_program = _streaming_program(key_length=9, query_tile_size=64)
    descriptor = streaming_contract_fold_event_descriptor(semantic_program)
    schedule = derive_streaming_physical_event_schedule(descriptor)

    assert descriptor.fold_extent == 9
    assert descriptor.resident_tile_size == 64
    assert descriptor.streamed_tile_size == 4
    assert descriptor.resident_reduction_dimension == 8
    assert descriptor.streamed_reduction_dimension == 8
    assert descriptor.output_dimension == 8
    assert descriptor.element_bytes == 2
    assert schedule.dataflow.partition_count == 3
    assert all("attention" not in family.name for family in schedule.dataflow.program.task_families)


def test_moe_style_relation_plan_directly_drives_runtime_event_readiness() -> None:
    selected_experts = np.asarray([[0, 2], [1, 2], [2, 3], [0, 2]], dtype=np.int32)
    route_weights = np.full(selected_experts.shape, 0.5, dtype=np.float32)
    destination_items = np.arange(4, dtype=np.int32)
    relation = build_relation_plan(
        selected_experts,
        route_weights,
        destination_rank_by_item=destination_items // 2,
        destination_local_item_by_item=destination_items % 2,
        padding_quantum=4,
    )

    dependence = relation_segment_dependence(relation, visibility_scope=EventMemoryScope.CTA)
    event_plan = derive_event_tensor_plan(dependence, name="runtime_segment_readiness")

    assert tuple(count.value for count in event_plan.initial_count.counts) == tuple(
        int(count) for count in relation.group_count
    )
    assert len(event_plan.notify_relation.pairs) == relation.route_count
    assert dependence.relation.source.axes[0].extent == relation.route_count
    assert dependence.relation.target.axes[0].extent == relation.destination_count


def test_relation_plan_drives_every_segmented_contract_output_tile() -> None:
    selected_experts = np.asarray([[0, 2], [1, 2], [2, 3], [0, 2]], dtype=np.int32)
    route_weights = np.full(selected_experts.shape, 0.5, dtype=np.float32)
    destinations = np.arange(5, dtype=np.int32)
    relation = build_relation_plan(
        selected_experts,
        route_weights,
        destination_rank_by_item=np.zeros(5, dtype=np.int32),
        destination_local_item_by_item=destinations,
        padding_quantum=1,
    )

    dataflow = relation_segmented_contract_task_dataflow(relation, output_tile_count=3)

    verify_event_dataflow_program(dataflow.program)
    counts = dataflow.program.event_plans[0].initial_count.as_mapping()
    for segment, count in enumerate(relation.group_count):
        for output_tile in range(3):
            assert counts[(segment, output_tile)] == int(count)
    assert counts[(4, 0)] == 0


def test_event_realization_audit_distinguishes_erased_and_physical_edges() -> None:
    adapted = streaming_fold_task_dataflow(_streaming_program())
    realizations = []
    for plan in adapted.program.event_plans:
        endpoints = (plan.notify_relation.source.name, plan.trigger_relation.target.name)
        if endpoints in {
            (adapted.qk_contract.name, adapted.fold_partial.name),
            (adapted.fold_partial.name, adapted.pv_contract.name),
        }:
            realizations.append(
                erased_event_realization(
                    plan,
                    kind=EventRealizationKind.ERASED_PROGRAM_ORDER,
                    mechanism="one sequential matrix/reduction owner",
                    reason="the selected physical template performs these stages in one ordered task body",
                )
            )
        else:
            realizations.append(
                physical_event_realization(
                    plan,
                    mechanism="CTA barrier or generation semaphore",
                    reason="the edge crosses workers or protects circular storage reuse",
                )
            )

    audit = verify_event_realizations(adapted.program, tuple(realizations))

    assert len(audit.erased) == 2
    assert len(audit.physical) == len(adapted.program.event_plans) - 2
    bad = list(realizations)
    erased = next(index for index, value in enumerate(bad) if value.kind is not EventRealizationKind.PHYSICAL)
    bad[erased] = erased_event_realization(
        adapted.program.event_plans[erased],
        kind=EventRealizationKind.ERASED_PROGRAM_ORDER,
        mechanism="invalid order",
        reason="test rejection",
        ordering=type(adapted.program.event_plans[erased].scheduled_dependence).from_pairs(
            adapted.program.event_plans[erased].notify_relation.source,
            adapted.program.event_plans[erased].trigger_relation.target,
            (),
        ),
    )
    with pytest.raises(EventDataflowError, match="omits scheduled edges"):
        verify_event_realizations(adapted.program, tuple(bad))


def test_bounded_buffer_rejects_reusing_a_slot_in_the_same_generation() -> None:
    adapted = streaming_fold_task_dataflow(_streaming_program())
    stage_to_consumers = tuple(
        dependence.relation
        for dependence in adapted.program.dependences
        if dependence.relation.source == adapted.key_value_stage
    )
    coordinates = adapted.key_value_stage.coordinates
    with pytest.raises(EventDataflowError, match="generations must be contiguous"):
        derive_bounded_buffer_plan(
            name="invalid_reuse",
            program=adapted.program,
            producer=adapted.key_value_stage,
            uses=stage_to_consumers,
            capacity=adapted.key_value_buffer.capacity,
            slot_for={
                coordinate: coordinate[0] * adapted.pipeline_depth + coordinate[1] % adapted.pipeline_depth
                for coordinate in coordinates
            },
            generation_for={coordinate: 0 for coordinate in coordinates},
            visibility=adapted.program.dependences[0].visibility,
        )
