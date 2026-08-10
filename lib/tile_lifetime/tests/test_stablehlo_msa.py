# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from shuttle.experimental.stablehlo_import import CompositeAttributes, import_stablehlo
from tile_lifetime.msa_frontend import MSA_INPUT_NAMES, MSADebugConfig, export_debug_msa, msa_region
from tile_lifetime.msa_recovery import (
    NaturalProjectedRoutedAttentionCompilation,
    recover_projected_routed_attention_program,
)
from tile_lifetime.pipeline import (
    compile_stablehlo_projected_routed_attention_program,
    recover_stablehlo_projected_routed_attention_program,
)
from tile_lifetime.routed_attention import (
    IndexDomainRestriction,
    ProjectedBlockSelectionProgram,
    SelectionOutputOrder,
    SelectionTieBreak,
    UnderfilledSelectionPolicy,
    build_grouped_routed_attention_relation,
    execute_projected_block_selection,
)
from tile_lifetime.routed_attention_plan import (
    RoutedAttentionOrientation,
    RoutedAttentionPlanConfig,
    compile_routed_streaming_attention_candidates,
)
from tile_lifetime.semantic_erasure import semantic_erasure_errors
from tile_lifetime.sm100_projected_routed_lowering import lower_sm100_projected_routed_candidates
from tile_lifetime.sm100_routed_lowering import SM100RelationOrientation
from tile_lifetime.sm100_selection_lowering import SM100SelectionStrategy
from tile_lifetime.streaming_attention import StreamingTileSchedule, derive_streaming_attention

FIXTURE = Path(__file__).parent / "fixtures" / "stablehlo" / "projected_routed_attention_v1_14_1.mlir.bc.b64"


def _fixture_artifact() -> bytes:
    return base64.b64decode(FIXTURE.read_text())


def _inputs(config: MSADebugConfig, seed: int) -> tuple[np.ndarray, ...]:
    generator = np.random.default_rng(seed)
    query_hidden = generator.normal(size=(config.query_length, config.hidden_dimension)).astype(np.float32)
    key_value_hidden = generator.normal(size=(config.key_value_length, config.hidden_dimension)).astype(np.float32)
    query_weight = generator.normal(size=(config.hidden_dimension, config.query_heads * config.head_dimension)).astype(
        np.float32
    )
    key_weight = generator.normal(size=(config.hidden_dimension, config.key_value_heads * config.head_dimension)).astype(
        np.float32
    )
    value_weight = generator.normal(size=key_weight.shape).astype(np.float32)
    left_index_weight = generator.normal(
        size=(config.hidden_dimension, config.key_value_heads * config.index_dimension)
    ).astype(np.float32)
    right_index_weight = generator.normal(size=(config.hidden_dimension, config.index_dimension)).astype(np.float32)
    return (
        query_hidden,
        key_value_hidden,
        query_weight,
        key_weight,
        value_weight,
        left_index_weight,
        right_index_weight,
    )


def _physical_config(config: MSADebugConfig) -> RoutedAttentionPlanConfig:
    return RoutedAttentionPlanConfig(
        query_block_size=1,
        key_value_block_size=config.block_size,
        query_heads=config.query_heads,
        key_value_heads=config.key_value_heads,
        head_dimension=config.head_dimension,
        value_dimension=config.head_dimension,
        buffer_depth=2,
        transfer_workers=1,
        matrix_workers=2,
        reduction_workers=1,
    )


def _selected_attention_reference(
    config: MSADebugConfig,
    query_hidden: np.ndarray,
    key_value_hidden: np.ndarray,
    query_weight: np.ndarray,
    key_weight: np.ndarray,
    value_weight: np.ndarray,
    selected: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    query_hidden_bf16 = jnp.asarray(query_hidden, dtype=jnp.bfloat16)
    key_value_hidden_bf16 = jnp.asarray(key_value_hidden, dtype=jnp.bfloat16)
    query = np.asarray(
        jnp.matmul(
            query_hidden_bf16,
            jnp.asarray(query_weight, dtype=jnp.bfloat16),
            preferred_element_type=jnp.float32,
        )
        .astype(jnp.bfloat16)
        .reshape(config.query_length, config.query_heads, config.head_dimension),
        dtype=np.float32,
    )
    key = np.asarray(
        jnp.matmul(
            key_value_hidden_bf16,
            jnp.asarray(key_weight, dtype=jnp.bfloat16),
            preferred_element_type=jnp.float32,
        )
        .astype(jnp.bfloat16)
        .reshape(config.key_value_length, config.key_value_heads, config.head_dimension),
        dtype=np.float32,
    )
    value = np.asarray(
        jnp.matmul(
            key_value_hidden_bf16,
            jnp.asarray(value_weight, dtype=jnp.bfloat16),
            preferred_element_type=jnp.float32,
        )
        .astype(jnp.bfloat16)
        .reshape(config.key_value_length, config.key_value_heads, config.head_dimension),
        dtype=np.float32,
    )
    heads_per_group = config.query_heads // config.key_value_heads
    output = np.empty_like(query)
    for query_offset in range(config.query_length):
        query_position = config.query_position_offset + query_offset
        for query_head in range(config.query_heads):
            group = query_head // heads_per_group
            blocks = selected[query_offset, group, valid[query_offset, group]]
            key_positions = np.concatenate(
                [np.arange(block * config.block_size, (block + 1) * config.block_size) for block in blocks]
            )
            key_positions = key_positions[key_positions <= query_position]
            score = query[query_offset, query_head] @ key[key_positions, group].T
            score *= np.float32(config.attention_scale)
            probability = np.exp(score - np.max(score))
            probability /= np.sum(probability)
            output[query_offset, query_head] = probability @ value[key_positions, group]
    return output


def test_frozen_projected_relation_fixture_matches_current_jax_structure() -> None:
    current = import_stablehlo(export_debug_msa(), input_names=MSA_INPUT_NAMES)
    frozen = import_stablehlo(_fixture_artifact(), input_names=MSA_INPUT_NAMES)

    current_structure = [
        (operation.kind, tuple(current.value(value).shape for value in operation.outputs))
        for operation in current.operations
    ]
    frozen_structure = [
        (operation.kind, tuple(frozen.value(value).shape for value in operation.outputs))
        for operation in frozen.operations
    ]
    assert frozen_structure == current_structure
    assert len(frozen.operations) == 125
    top_k = next(operation for operation in frozen.operations if operation.kind == "composite")
    assert top_k.attributes == CompositeAttributes(
        name="chlo.top_k",
        attributes=(("k", "2 : i64"),),
        version=1,
    )
    assert [operation.attributes.reducer for operation in frozen.operations if operation.kind == "reduce"] == [
        "maximum",
        "maximum",
        "add",
    ]


def test_projected_relation_names_erase_before_schedule_synthesis() -> None:
    recovered = recover_stablehlo_projected_routed_attention_program(
        _fixture_artifact(),
        input_names=MSA_INPUT_NAMES,
    )

    assert recovered.generic_operation_kinds == (
        "Contract",
        "Contract",
        "Map",
        "DomainRestriction",
        "Fold",
        "Selection",
        "Relation",
        "RelationPlan",
        "Contract",
        "Map",
        "Fold",
        "Map",
        "Map",
        "Fold",
        "Contract",
        "Map",
    )
    assert semantic_erasure_errors(recovered.semantic_erasure_report) == ()
    assert all("msa" not in key.lower() for key in recovered.semantic_erasure_report.scheduling_keys)
    assert recovered.relation_selection.force_local_block
    assert recovered.relation_selection.group_count == 2
    assert recovered.relation_selection.source_count == 8
    assert recovered.relation_selection.resolved_right_count == 16
    assert recovered.relation_selection.left_position_offset == 8
    assert recovered.relation_selection.projection_output_dtype == "bf16"
    assert recovered.relation_selection.right_block_size == 4
    semantics = recovered.relation_selection.selection_semantics
    assert semantics.tie_break is SelectionTieBreak.RIGHT_INDEX_ASCENDING
    assert semantics.output_order is SelectionOutputOrder.SCORE_DESCENDING
    assert semantics.underfilled_policy is UnderfilledSelectionPolicy.EXPLICIT_INVALID_SLOTS
    assert semantics.invalid_index == -1
    assert any("underfilled=explicit_invalid_slots" in key for key in recovered.semantic_erasure_report.scheduling_keys)


def test_projected_selection_preserves_source_order_ties_and_explicit_invalid_slots() -> None:
    program = ProjectedBlockSelectionProgram(
        source_input="left",
        left_weight_input="left_weight",
        right_weight_input="right_weight",
        source_count=4,
        source_feature_count=1,
        group_count=1,
        relation_feature_count=1,
        right_block_size=1,
        selected_count=3,
        score_scale=1.0,
        token_restriction=IndexDomainRestriction(
            left_axis="query_position",
            right_axis="key_position",
            predicate="left_greater_equal_right",
        ),
        force_local_block=True,
        projection_output_dtype="fp32",
        right_source_input="right",
        right_source_feature_count=1,
        right_count=4,
    )
    selection = execute_projected_block_selection(
        program,
        {
            "left": np.ones((4, 1), dtype=np.float32),
            "right": np.ones((4, 1), dtype=np.float32),
            "left_weight": np.ones((1, 1), dtype=np.float32),
            "right_weight": np.ones((1, 1), dtype=np.float32),
        },
    )

    # The forced local block remains first, then equal scores select lower
    # right identities. Rows with too few causal blocks retain rectangular
    # slots, but those slots have no destination identity.
    np.testing.assert_array_equal(
        selection.indices[:, 0],
        np.array(
            [
                [0, -1, -1],
                [1, 0, -1],
                [2, 0, 1],
                [3, 0, 1],
            ],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(selection.valid, selection.indices >= 0)
    np.testing.assert_array_equal(selection.invalid, selection.indices == -1)

    relation = build_grouped_routed_attention_relation(selection.indices, edge_valid=selection.valid)
    assert relation.route_count == 9
    np.testing.assert_array_equal(relation.edge_valid.reshape(selection.valid.shape), selection.valid)
    assert np.all(relation.route_to_destination_row[~selection.valid.reshape(-1)] == -1)


def test_natural_projected_relation_matches_independent_selected_attention() -> None:
    config = MSADebugConfig()
    inputs = _inputs(config, seed=37)
    query_hidden, key_value_hidden, query_weight, key_weight, value_weight, left_weight, right_weight = inputs
    recovered = recover_projected_routed_attention_program(
        import_stablehlo(_fixture_artifact(), input_names=MSA_INPUT_NAMES)
    )
    query_hidden_bf16 = np.asarray(jnp.asarray(query_hidden, dtype=jnp.bfloat16), dtype=np.float32)
    key_value_hidden_bf16 = np.asarray(jnp.asarray(key_value_hidden, dtype=jnp.bfloat16), dtype=np.float32)
    selection = execute_projected_block_selection(
        recovered.relation_selection,
        {
            "query_hidden": query_hidden_bf16,
            "key_value_hidden": key_value_hidden_bf16,
            "left_index_weight": left_weight,
            "right_index_weight": right_weight,
        },
    )
    selected = selection.indices
    valid = selection.valid

    natural = np.asarray(
        msa_region(config)(
            jnp.asarray(query_hidden, dtype=jnp.bfloat16),
            jnp.asarray(key_value_hidden, dtype=jnp.bfloat16),
            jnp.asarray(query_weight, dtype=jnp.bfloat16),
            jnp.asarray(key_weight, dtype=jnp.bfloat16),
            jnp.asarray(value_weight, dtype=jnp.bfloat16),
            left_weight,
            right_weight,
        ),
        dtype=np.float32,
    )
    reference = _selected_attention_reference(
        config,
        query_hidden,
        key_value_hidden,
        query_weight,
        key_weight,
        value_weight,
        selected,
        valid,
    )

    for query_offset in range(config.query_length):
        query_position = config.query_position_offset + query_offset
        local_block = query_position // config.block_size
        assert np.all(np.any((selected[query_offset] == local_block) & valid[query_offset], axis=-1))
    error = np.abs(natural - reference)
    assert float(np.max(error)) <= 0.0625
    assert float(np.mean(error)) <= 0.005


def test_projected_runtime_selection_builds_grouped_dual_orientation_relation() -> None:
    config = MSADebugConfig()
    query_hidden, key_value_hidden, _, _, _, left_weight, right_weight = _inputs(config, seed=71)
    query_hidden_bf16 = np.asarray(jnp.asarray(query_hidden, dtype=jnp.bfloat16), dtype=np.float32)
    key_value_hidden_bf16 = np.asarray(jnp.asarray(key_value_hidden, dtype=jnp.bfloat16), dtype=np.float32)
    compilation = compile_stablehlo_projected_routed_attention_program(
        _fixture_artifact(),
        input_names=MSA_INPUT_NAMES,
        runtime_inputs={
            "query_hidden": query_hidden_bf16,
            "key_value_hidden": key_value_hidden_bf16,
            "left_index_weight": left_weight,
            "right_index_weight": right_weight,
        },
        schedule=StreamingTileSchedule(query_tile_size=1, key_value_tile_size=4, pipeline_depth=2),
        config=_physical_config(config),
    )

    assert compilation.relation.source_item_count == config.query_length
    assert compilation.relation.route_slots == config.key_value_heads * config.selected_blocks
    assert compilation.relation.destination_count == config.block_count
    assert compilation.relation.route_count == int(np.count_nonzero(compilation.edge_valid))
    local_block = (np.arange(config.query_length, dtype=np.int32) + config.query_position_offset) // config.block_size
    np.testing.assert_array_equal(
        compilation.selected_right_blocks[..., 0],
        np.broadcast_to(local_block[:, None], (config.query_length, config.key_value_heads)),
    )
    assert np.all(compilation.selected_right_blocks[~compilation.edge_valid] == -1)
    for row, valid in zip(
        compilation.selected_right_blocks.reshape(-1, config.selected_blocks),
        compilation.edge_valid.reshape(-1, config.selected_blocks),
        strict=True,
    ):
        assert np.unique(row[valid]).size == np.count_nonzero(valid)
    assert [candidate.orientation for candidate in compilation.scheduled.candidates] == [
        RoutedAttentionOrientation.QUERY_MAJOR,
        RoutedAttentionOrientation.KV_MAJOR,
    ]
    assert compilation.streaming_program.source == compilation.recovered.tensor_program


def test_symmetric_projected_relation_is_a_semantic_shape_mutation() -> None:
    config = MSADebugConfig(query_length=16, key_value_length=16)
    artifact = export_debug_msa(config)
    query_hidden, key_value_hidden, query_weight, key_weight, value_weight, left_weight, right_weight = _inputs(
        config, seed=83
    )
    recovered = recover_stablehlo_projected_routed_attention_program(artifact, input_names=MSA_INPUT_NAMES)
    query_hidden_bf16 = np.asarray(jnp.asarray(query_hidden, dtype=jnp.bfloat16), dtype=np.float32)
    key_value_hidden_bf16 = np.asarray(jnp.asarray(key_value_hidden, dtype=jnp.bfloat16), dtype=np.float32)
    selection = execute_projected_block_selection(
        recovered.relation_selection,
        {
            "query_hidden": query_hidden_bf16,
            "key_value_hidden": key_value_hidden_bf16,
            "left_index_weight": left_weight,
            "right_index_weight": right_weight,
        },
    )
    selected = selection.indices
    valid = selection.valid
    natural = np.asarray(
        msa_region(config)(
            jnp.asarray(query_hidden, dtype=jnp.bfloat16),
            jnp.asarray(key_value_hidden, dtype=jnp.bfloat16),
            jnp.asarray(query_weight, dtype=jnp.bfloat16),
            jnp.asarray(key_weight, dtype=jnp.bfloat16),
            jnp.asarray(value_weight, dtype=jnp.bfloat16),
            left_weight,
            right_weight,
        ),
        dtype=np.float32,
    )
    reference = _selected_attention_reference(
        config,
        query_hidden,
        key_value_hidden,
        query_weight,
        key_weight,
        value_weight,
        selected,
        valid,
    )

    assert recovered.relation_selection.left_position_offset == 0
    assert recovered.relation_selection.resolved_right_count == recovered.relation_selection.source_count
    assert (
        recovered.generic_operation_kinds
        == recover_stablehlo_projected_routed_attention_program(
            _fixture_artifact(), input_names=MSA_INPUT_NAMES
        ).generic_operation_kinds
    )
    assert not np.all(valid[0])
    error = np.abs(natural - reference)
    assert float(np.max(error)) <= 0.0625
    assert float(np.mean(error)) <= 0.005


def test_primary_asymmetric_prefill_shape_recovers_without_a_workload_node() -> None:
    config = MSADebugConfig(
        query_length=256,
        key_value_length=16384,
        hidden_dimension=128,
        query_heads=32,
        key_value_heads=8,
        head_dimension=128,
        index_dimension=128,
        block_size=128,
        selected_blocks=16,
    )
    recovered = recover_stablehlo_projected_routed_attention_program(
        export_debug_msa(config),
        input_names=MSA_INPUT_NAMES,
    )

    selection = recovered.relation_selection
    assert selection.source_count == 256
    assert selection.resolved_right_count == 16384
    assert selection.left_position_offset == 16128
    assert selection.group_count == 8
    assert selection.relation_feature_count == 128
    assert selection.projection_output_dtype == "bf16"
    assert selection.right_block_count == 128
    assert selection.selected_count == 16
    assert semantic_erasure_errors(recovered.semantic_erasure_report) == ()


def test_primary_asymmetric_prefill_enumerates_joint_sm100_candidates() -> None:
    config = MSADebugConfig(
        query_length=256,
        key_value_length=16384,
        hidden_dimension=128,
        query_heads=32,
        key_value_heads=8,
        head_dimension=128,
        index_dimension=128,
        block_size=128,
        selected_blocks=16,
    )
    recovered = recover_stablehlo_projected_routed_attention_program(
        export_debug_msa(config),
        input_names=MSA_INPUT_NAMES,
    )
    absolute_query = np.arange(config.query_length, dtype=np.int32) + config.query_position_offset
    local_block = absolute_query // config.block_size
    selected = local_block[:, None, None] - np.arange(config.selected_blocks - 1, -1, -1, dtype=np.int32)
    selected = np.broadcast_to(selected, (config.query_length, config.key_value_heads, config.selected_blocks)).copy()
    edge_valid = selected >= 0
    relation = build_grouped_routed_attention_relation(selected, edge_valid=edge_valid)
    streaming = derive_streaming_attention(
        recovered.tensor_program,
        schedule=StreamingTileSchedule(query_tile_size=1, key_value_tile_size=128, pipeline_depth=2),
    )
    scheduled = compile_routed_streaming_attention_candidates(streaming, relation, _physical_config(config))
    natural = NaturalProjectedRoutedAttentionCompilation(
        recovered=recovered,
        selected_right_blocks=selected,
        edge_valid=edge_valid,
        relation=relation,
        streaming_program=streaming,
        scheduled=scheduled,
    )

    lowered = lower_sm100_projected_routed_candidates(natural)

    assert len(lowered.candidates) == 4
    materialized = [
        candidate
        for candidate in lowered.candidates
        if candidate.selection.schedule.strategy is SM100SelectionStrategy.MATERIALIZED_BLOCK_SCORES
    ]
    fused = [
        candidate
        for candidate in lowered.candidates
        if candidate.selection.schedule.strategy is SM100SelectionStrategy.FUSED_STREAMING_TOP_K
    ]
    assert {candidate.selection.token_score_materialization_bytes for candidate in materialized} == {134_217_728}
    assert {candidate.selection.token_score_materialization_bytes for candidate in fused} == {0}
    assert {candidate.streaming.schedule.orientation for candidate in lowered.candidates} == {
        SM100RelationOrientation.LEFT_MAJOR,
        SM100RelationOrientation.RIGHT_MAJOR,
    }
    position_domain = lowered.candidates[0].position_domain
    assert position_domain.bottom_right_aligned
    assert position_domain.left_position(0) == 16128
    assert position_domain.left_position(255) == 16383
    assert position_domain.right_position(0) == 0
    assert position_domain.right_position(16383) == 16383
    assert position_domain.allows(0, 16128)
    assert not position_domain.allows(0, 16129)
