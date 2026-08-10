# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from shuttle.ir import DType
from tile_lifetime.cuda_event_workload_codegen import (
    evaluate_segmented_contract_event,
    evaluate_streaming_contract_fold_event,
    generate_segmented_contract_event_ffi,
    generate_streaming_contract_fold_event_ffi,
)
from tile_lifetime.event_buffering import EventRealizationKind
from tile_lifetime.event_dataflow_adapters import (
    relation_segmented_contract_task_dataflow,
    streaming_fold_task_dataflow,
)
from tile_lifetime.relation import build_relation_plan
from tile_lifetime.streaming_attention import (
    StreamingTileSchedule,
    build_attention_tensor_program,
    derive_streaming_attention,
    scaled_score_map,
)


def _relation():
    destination = np.asarray([[0, 2], [1, 2], [2, 0], [1, 2]], dtype=np.int32)
    weight = np.ones_like(destination, dtype=np.float32)
    items = np.arange(4, dtype=np.int32)
    return build_relation_plan(
        destination,
        weight,
        destination_rank_by_item=np.zeros(4, dtype=np.int32),
        destination_local_item_by_item=items,
        padding_quantum=1,
    )


def _streaming(*, key_length: int = 6, pipeline_depth: int = 2):
    semantic = build_attention_tensor_program(
        batch_size=1,
        query_length=4,
        key_length=key_length,
        query_heads=1,
        key_value_heads=1,
        key_dimension=3,
        value_dimension=2,
        score_map=scaled_score_map(0.5),
        input_dtype=DType.FP32,
    )
    return derive_streaming_attention(
        semantic,
        schedule=StreamingTileSchedule(query_tile_size=2, key_value_tile_size=2, pipeline_depth=pipeline_depth),
    )


def test_segmented_contract_codegen_consumes_payload_and_relation_csr() -> None:
    relation = _relation()
    dataflow = relation_segmented_contract_task_dataflow(relation, output_tile_count=1)
    generated = generate_segmented_contract_event_ffi(
        dataflow,
        relation,
        reduction_dimension=3,
        output_dimension=2,
        target_name="shuttle.segmented_contract_test",
    )
    source = np.arange(12, dtype=np.float32).reshape(4, 3)
    weights = np.arange(24, dtype=np.float32).reshape(4, 3, 2) / 8

    output = evaluate_segmented_contract_event(relation, source, weights)

    expected = np.concatenate(
        [
            source[relation.grouped_source_item[begin:end]] @ weights[segment]
            for segment, (begin, end) in enumerate(
                zip(relation.destination_edge_offsets[:-1], relation.destination_edge_offsets[1:], strict=True)
            )
        ],
        axis=0,
    )
    np.testing.assert_array_equal(output, expected)
    assert generated.event_audit.entries[0].kind is EventRealizationKind.ERASED_PROGRAM_ORDER
    assert "event_counts[segment]" in generated.ffi.source
    assert "edge_sources[edge]" in generated.ffi.source
    assert "fmaf(" in generated.ffi.source


def test_streaming_codegen_preserves_physical_stage_and_last_consumer_reuse() -> None:
    streaming = _streaming()
    dataflow = streaming_fold_task_dataflow(streaming)
    generated = generate_streaming_contract_fold_event_ffi(
        dataflow,
        query_tile_size=2,
        key_value_tile_size=2,
        reduction_dimension=3,
        value_dimension=2,
        score_scale=0.5,
        target_name="shuttle.streaming_contract_fold_test",
    )
    rng = np.random.default_rng(7)
    query = rng.normal(size=(2, 2, 3)).astype(np.float32)
    key = rng.normal(size=(2, 3, 2, 3)).astype(np.float32)
    value = rng.normal(size=(2, 3, 2, 2)).astype(np.float32)
    valid = np.ones((2, 2, 3, 2), dtype=np.int32)
    valid[0, 0, 2] = 0

    output = evaluate_streaming_contract_fold_event(query, key, value, valid, score_scale=0.5)
    materialized_score = np.einsum("rqd,rpkd->rqpk", query, key) * np.float32(0.5)
    materialized_score = np.where(valid, materialized_score, -np.inf)
    probability = np.exp(materialized_score - np.max(materialized_score, axis=(2, 3), keepdims=True))
    probability /= np.sum(probability, axis=(2, 3), keepdims=True)
    expected = np.einsum("rqpk,rpkv->rqv", probability, value)

    np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)
    assert sum(entry.kind is EventRealizationKind.PHYSICAL for entry in generated.event_audit.entries) == 3
    assert "__syncthreads();" in generated.ffi.source
    assert "slot_generation[slot] = generation + 1" in generated.ffi.source
    assert "__shared__ int generation_valid;" in generated.ffi.source
    assert "if (threadIdx.x == 0 && slot_generation[slot] != generation) return;" not in generated.ffi.source
    generation_check = generated.ffi.source.index("generation_valid = slot_generation[slot] == generation")
    uniform_return = generated.ffi.source.index("if (!generation_valid) return;", generation_check)
    assert "__syncthreads();" in generated.ffi.source[generation_check:uniform_return]
    assert "QK Contract, online normalized-exp Fold, and PV Contract" in " ".join(generated.physical_schedule)


def test_streaming_pipeline_mutations_change_generated_program() -> None:
    primary_dataflow = streaming_fold_task_dataflow(_streaming(key_length=6, pipeline_depth=2))
    depth_mutation_dataflow = streaming_fold_task_dataflow(_streaming(key_length=6, pipeline_depth=3))
    partition_mutation_dataflow = streaming_fold_task_dataflow(_streaming(key_length=8, pipeline_depth=2))

    def generate(dataflow, name: str):
        return generate_streaming_contract_fold_event_ffi(
            dataflow,
            query_tile_size=2,
            key_value_tile_size=2,
            reduction_dimension=3,
            value_dimension=2,
            score_scale=0.5,
            target_name=name,
        )

    primary = generate(primary_dataflow, "shuttle.streaming_primary")
    depth_mutation = generate(depth_mutation_dataflow, "shuttle.streaming_depth_mutation")
    partition_mutation = generate(partition_mutation_dataflow, "shuttle.streaming_partition_mutation")

    assert primary.ffi.plan_fingerprint != depth_mutation.ffi.plan_fingerprint
    assert primary.ffi.plan_fingerprint != partition_mutation.ffi.plan_fingerprint
    assert "constexpr int kPipelineDepth = 2;" in primary.ffi.source
    assert "constexpr int kPipelineDepth = 3;" in depth_mutation.ffi.source
    assert "constexpr int kPartitionCount = 3;" in primary.ffi.source
    assert "constexpr int kPartitionCount = 4;" in partition_mutation.ffi.source
