# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from tile_lifetime.event_dataflow import EventMemoryScope, derive_event_tensor_plan, verify_event_dataflow_program
from tile_lifetime.event_dataflow_adapters import streaming_fold_task_dataflow
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


def _streaming_program(*, key_length: int = 9):
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
        schedule=StreamingTileSchedule(query_tile_size=3, key_value_tile_size=4, pipeline_depth=2),
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
    assert tuple(plan.initial_count.counts[0].value for plan in adapted.program.event_plans) == (1, 1, 3)
    assert all("attention" not in family.name for family in adapted.program.task_families)


def test_fold_domain_mutation_changes_events_without_a_workload_dispatch() -> None:
    primary = streaming_fold_task_dataflow(_streaming_program(key_length=9))
    mutated = streaming_fold_task_dataflow(_streaming_program(key_length=17))

    assert primary.fold_partition_count == 3
    assert mutated.fold_partition_count == 5
    assert primary.program.task_families[0].name == mutated.program.task_families[0].name
    assert primary.program.event_plans[-1].initial_count.counts[0].value == 3
    assert mutated.program.event_plans[-1].initial_count.counts[0].value == 5


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
