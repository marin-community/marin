# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib

import numpy as np
import pytest

from tile_lifetime import (
    DType,
    RoutedAttentionOrientation,
    RoutedAttentionPlanConfig,
    StreamingTileSchedule,
    apply_causal_score_mask,
    apply_tanh_softcap,
    bounded_kv_reuse_plan,
    build_attention_tensor_program,
    build_routed_attention_relation,
    compile_bounded_kv_major_candidate,
    compile_routed_attention_candidates,
    compile_routed_streaming_attention_candidates,
    derive_streaming_attention,
    execute_kv_major_attention,
    execute_query_major_attention,
    finalize_attention_partial,
    make_causal_block_relation,
    merge_attention_partials,
    query_major_block_index_plan,
    scaled_score_map,
    summarize_attention_partial,
)
from tile_lifetime.routed_attention import routed_attention_reference


def _inputs():
    rng = np.random.default_rng(17)
    query = rng.normal(size=(3, 4, 4, 8)).astype(np.float32)
    key = rng.normal(size=(3, 4, 2, 8)).astype(np.float32)
    value = rng.normal(size=(3, 4, 2, 5)).astype(np.float32)
    selected_blocks = np.array(
        [
            [0, -1, -1],
            [0, 1, -1],
            [0, 2, 1],
        ],
        dtype=np.int32,
    )
    edge_valid = selected_blocks >= 0
    return query, key, value, selected_blocks, edge_valid


def test_query_and_kv_major_attention_match_independent_masked_reference() -> None:
    query, key, value, selected_blocks, edge_valid = _inputs()
    scale = 0.5

    reference = routed_attention_reference(
        query,
        key,
        value,
        selected_blocks,
        edge_valid=edge_valid,
        scale=scale,
        causal=True,
        sequence_length=10,
    )
    query_major = execute_query_major_attention(
        query,
        key,
        value,
        selected_blocks,
        edge_valid=edge_valid,
        scale=scale,
        causal=True,
        sequence_length=10,
    )
    kv_major = execute_kv_major_attention(
        query,
        key,
        value,
        selected_blocks,
        edge_valid=edge_valid,
        scale=scale,
        causal=True,
        padding_quantum=2,
        sequence_length=10,
    )

    np.testing.assert_allclose(query_major, reference, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(kv_major, reference, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(query_major.reshape(-1, 4, 5)[10:], 0)
    np.testing.assert_array_equal(
        kv_major,
        execute_kv_major_attention(
            query,
            key,
            value,
            selected_blocks,
            edge_valid=edge_valid,
            scale=scale,
            causal=True,
            padding_quantum=2,
            sequence_length=10,
        ),
    )


def test_causal_block_relation_is_deterministic_historical_and_ragged() -> None:
    selected, edge_valid = make_causal_block_relation(
        sequence_length=10,
        block_size=4,
        selected_blocks=3,
    )
    repeated, repeated_valid = make_causal_block_relation(
        sequence_length=10,
        block_size=4,
        selected_blocks=3,
    )

    np.testing.assert_array_equal(selected, repeated)
    np.testing.assert_array_equal(edge_valid, repeated_valid)
    assert np.count_nonzero(edge_valid, axis=1).tolist() == [1, 2, 3]
    for query_block in range(selected.shape[0]):
        chosen = selected[query_block, edge_valid[query_block]]
        assert chosen[0] == 0
        assert chosen[-1] == query_block
        assert np.all(chosen <= query_block)


def test_primary_relation_matches_preserved_h100_fixture() -> None:
    selected, edge_valid = make_causal_block_relation(
        sequence_length=16_384,
        block_size=128,
        selected_blocks=8,
    )
    dense_relation = np.zeros((128, 128), dtype=np.bool_)
    for query_block in range(selected.shape[0]):
        dense_relation[query_block, selected[query_block, edge_valid[query_block]]] = True

    assert np.count_nonzero(edge_valid) == 996
    assert hashlib.sha256(dense_relation.tobytes(order="C")).hexdigest() == (
        "b2a57606e303f8af4da0c8002ddea162f86625725696bca7f18b8072a8143427"
    )


def test_attention_partial_merge_is_stable_and_tree_reassociable() -> None:
    query, key, value, _, _ = _inputs()
    query_block = query[2]
    partials = [
        summarize_attention_partial(query_block, key[kv_block], value[kv_block], scale=0.5) for kv_block in range(3)
    ]

    stable = merge_attention_partials(merge_attention_partials(partials[0], partials[1]), partials[2])
    tree = merge_attention_partials(partials[0], merge_attention_partials(partials[1], partials[2]))

    np.testing.assert_allclose(
        finalize_attention_partial(stable),
        finalize_attention_partial(tree),
        rtol=2e-6,
        atol=2e-6,
    )


def test_kv_major_relation_reuses_generic_grouping_and_inverse_mapping() -> None:
    _, _, _, selected_blocks, edge_valid = _inputs()

    relation = build_routed_attention_relation(
        selected_blocks,
        edge_valid=edge_valid,
        kv_rank_by_block=np.array([0, 1, 1], dtype=np.int32),
        kv_local_block_by_block=np.array([0, 0, 1], dtype=np.int32),
        padding_quantum=2,
    )

    assert relation.source_item_count == 3
    assert relation.route_count == 6
    assert relation.group_count.tolist() == [3, 2, 1]
    assert relation.group_padded_count.tolist() == [4, 2, 2]
    assert relation.destination_rank_count == 2
    assert relation.destination_edge_offsets.tolist() == [0, 3, 5, 6]
    assert relation.grouped_source_item.tolist() == [0, 1, 2, 1, 2, 2]
    assert relation.grouped_route_slot.tolist() == [0, 0, 0, 1, 2, 1]
    assert relation.inverse_dispatch(relation.dispatch(np.arange(3)[:, None])).shape == (3, 3, 1)


def test_non_monotone_relation_preserves_source_slots_and_both_orientations() -> None:
    query, key, value, _, _ = _inputs()
    selected_blocks = np.array(
        [
            [0, -1, -1],
            [1, 0, -1],
            [2, 0, 1],
        ],
        dtype=np.int32,
    )
    edge_valid = selected_blocks >= 0
    relation = build_routed_attention_relation(selected_blocks, edge_valid=edge_valid, padding_quantum=2)
    index_plan = query_major_block_index_plan(relation)

    assert index_plan.block_count.tolist() == [1, 2, 3]
    assert index_plan.block_index.tolist() == [[0, 0, 0], [1, 0, 0], [2, 0, 1]]
    assert relation.grouped_source_item.tolist() == [0, 1, 2, 1, 2, 2]

    query_major = execute_query_major_attention(
        query,
        key,
        value,
        selected_blocks,
        edge_valid=edge_valid,
        scale=0.5,
        causal=True,
        sequence_length=10,
    )
    kv_major = execute_kv_major_attention(
        query,
        key,
        value,
        selected_blocks,
        edge_valid=edge_valid,
        scale=0.5,
        causal=True,
        padding_quantum=2,
        sequence_length=10,
    )
    np.testing.assert_allclose(query_major, kv_major, rtol=2e-6, atol=2e-6)


def test_routed_attention_rejects_duplicate_selected_blocks() -> None:
    query, key, value, _, _ = _inputs()
    selected_blocks = np.array([[0, 0], [0, 1], [1, 2]], dtype=np.int32)

    with pytest.raises(ValueError, match="duplicate selected KV blocks"):
        execute_query_major_attention(
            query,
            key,
            value,
            selected_blocks,
            scale=0.5,
            causal=False,
        )


def test_routed_attention_rejects_query_rows_without_selected_keys() -> None:
    query, key, value, _, _ = _inputs()
    selected_blocks = np.full((3, 1), -1, dtype=np.int32)

    with pytest.raises(ValueError, match="no valid selected keys"):
        execute_query_major_attention(
            query,
            key,
            value,
            selected_blocks,
            edge_valid=np.zeros_like(selected_blocks, dtype=np.bool_),
            scale=0.5,
            causal=False,
        )


def test_routed_attention_candidates_derive_readiness_and_materialization_from_one_relation() -> None:
    _, _, _, selected_blocks, edge_valid = _inputs()
    relation = build_routed_attention_relation(
        selected_blocks,
        edge_valid=edge_valid,
        kv_rank_by_block=np.zeros(3, dtype=np.int32),
        kv_local_block_by_block=np.arange(3, dtype=np.int32),
        padding_quantum=2,
    )
    config = RoutedAttentionPlanConfig(
        query_block_size=4,
        key_value_block_size=4,
        query_heads=4,
        key_value_heads=2,
        head_dimension=8,
        value_dimension=5,
        buffer_depth=2,
        transfer_workers=1,
        matrix_workers=2,
        reduction_workers=1,
    )

    query_major, kv_major = compile_routed_attention_candidates(relation, config)

    assert query_major.orientation is RoutedAttentionOrientation.QUERY_MAJOR
    assert kv_major.orientation is RoutedAttentionOrientation.KV_MAJOR
    assert query_major.event("query_selected_edges_complete").required_arrivals == (1, 2, 3)
    assert kv_major.event("kv_incident_queries_complete").required_arrivals == (3, 2, 1)
    assert kv_major.event("query_partials_ready").required_arrivals == (1, 2, 3)
    assert query_major.partial_state_materialization_bytes == 0
    assert kv_major.partial_state_materialization_bytes == 6 * 4 * 4 * 7 * 4
    assert query_major.sequence_squared_materialization_bytes == 0
    assert kv_major.sequence_squared_materialization_bytes == 0
    assert "sequence_squared_materialization_bytes: 0" in kv_major.dump()

    bounded = compile_bounded_kv_major_candidate(relation, config)
    assert bounded.orientation is RoutedAttentionOrientation.KV_MAJOR_SLOT_WAVES
    assert bounded.partial_state_materialization_bytes == 0
    assert bounded.online_state_materialization_bytes == 3 * 4 * 4 * 7 * 4
    assert bounded.event("slot_0_state_ready").required_arrivals == 3
    assert bounded.event("slot_1_state_ready").required_arrivals == 2
    assert bounded.event("slot_2_state_ready").required_arrivals == 2
    assert bounded.event("slot_3_state_ready").required_arrivals == 1
    assert bounded.event("slot_2_kv_groups_complete").required_arrivals == (0, 1, 0)
    assert "no atomic accumulation" in bounded.numerical_policy


def test_bounded_kv_reuse_groups_nonmonotone_relation_without_duplicate_writers() -> None:
    selected_blocks = np.array(
        [
            [0, -1, -1],
            [1, 0, -1],
            [0, 2, 1],
            [1, 3, 0],
        ],
        dtype=np.int32,
    )
    relation = build_routed_attention_relation(selected_blocks, edge_valid=selected_blocks >= 0)

    paired = bounded_kv_reuse_plan(relation, query_capacity_per_task=2)
    scalar = bounded_kv_reuse_plan(relation, query_capacity_per_task=1)

    assert paired.edge_count == scalar.edge_count == relation.route_count
    assert paired.task_count < scalar.task_count
    assert paired.waves[0].key_value_block.tolist() == [0, 1]
    assert paired.waves[0].query_block.tolist() == [[0, 2], [1, 3]]
    assert paired.waves[0].query_count.tolist() == [2, 2]
    for wave in paired.waves:
        valid_queries = wave.query_block[wave.query_block >= 0]
        assert len(valid_queries) == len(set(valid_queries.tolist()))


@pytest.mark.parametrize("softcap", [None, 0.7])
def test_relation_candidates_are_derived_from_generic_streaming_semantics(softcap: float | None) -> None:
    _, _, _, selected_blocks, edge_valid = _inputs()
    relation = build_routed_attention_relation(selected_blocks, edge_valid=edge_valid)
    score_map = apply_causal_score_mask(scaled_score_map(0.5))
    if softcap is not None:
        score_map = apply_tanh_softcap(score_map, softcap)
    source = build_attention_tensor_program(
        batch_size=1,
        query_length=12,
        key_length=12,
        query_heads=4,
        key_value_heads=2,
        key_dimension=8,
        value_dimension=5,
        score_map=score_map,
        input_dtype=DType.BF16,
    )
    streamed = derive_streaming_attention(
        source,
        schedule=StreamingTileSchedule(query_tile_size=4, key_value_tile_size=4, pipeline_depth=2),
    )
    config = RoutedAttentionPlanConfig(
        query_block_size=4,
        key_value_block_size=4,
        query_heads=4,
        key_value_heads=2,
        head_dimension=8,
        value_dimension=5,
        buffer_depth=2,
        transfer_workers=1,
        matrix_workers=2,
        reduction_workers=1,
    )

    compilation = compile_routed_streaming_attention_candidates(streamed, relation, config)

    assert compilation.program.score_map.expression == score_map.expression
    assert compilation.relation is relation
    assert [candidate.orientation for candidate in compilation.candidates] == [
        RoutedAttentionOrientation.QUERY_MAJOR,
        RoutedAttentionOrientation.KV_MAJOR,
    ]
