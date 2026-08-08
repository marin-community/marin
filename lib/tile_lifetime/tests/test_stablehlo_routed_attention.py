# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from tile_lifetime import (
    RoutedAttentionOrientation,
    RoutedAttentionPlanConfig,
    StreamingTileSchedule,
    compile_stablehlo_routed_attention_program,
    execute_query_major_attention,
    execute_relation_selection,
    recover_stablehlo_routed_attention_program,
    semantic_erasure_errors,
)
from tile_lifetime.routed_attention_frontend import (
    ROUTED_ATTENTION_INPUT_NAMES,
    RoutedAttentionDebugConfig,
    export_debug_routed_attention,
    routed_attention_region,
)
from tile_lifetime.stablehlo_import import CompositeAttributes, GatherAttributes, import_stablehlo

FIXTURE = Path(__file__).parent / "fixtures" / "stablehlo" / "routed_attention_v1_14_1.mlir.bc.b64"


def _fixture_artifact() -> bytes:
    return base64.b64decode(FIXTURE.read_text())


def _physical_config(config: RoutedAttentionDebugConfig) -> RoutedAttentionPlanConfig:
    return RoutedAttentionPlanConfig(
        query_block_size=config.block_size,
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


def _semantic_inputs(config: RoutedAttentionDebugConfig, seed: int):
    generator = np.random.default_rng(seed)
    query = generator.normal(size=(1, config.sequence, config.query_heads, config.head_dimension)).astype(np.float32)
    key = generator.normal(size=(1, config.sequence, config.key_value_heads, config.head_dimension)).astype(np.float32)
    value = generator.normal(size=key.shape).astype(np.float32)
    query_metadata = generator.normal(size=(config.block_count, config.router_dimension)).astype(np.float32)
    key_value_metadata = generator.normal(size=query_metadata.shape).astype(np.float32)
    return query, key, value, query_metadata, key_value_metadata


def test_natural_routed_attention_export_contains_selection_and_tensor_algebra() -> None:
    graph = import_stablehlo(_fixture_artifact(), input_names=ROUTED_ATTENTION_INPUT_NAMES)

    composites = tuple(operation for operation in graph.operations if operation.kind == "composite")
    gathers = tuple(operation for operation in graph.operations if operation.kind == "gather")
    assert len(graph.operations) == 82
    assert composites[0].attributes == CompositeAttributes(
        name="chlo.top_k",
        attributes=(("k", "2 : i64"),),
        version=1,
    )
    assert len(gathers) == 2
    assert all(isinstance(operation.attributes, GatherAttributes) for operation in gathers)
    assert all("routed_attention_frontend.py" in operation.source_location for operation in (*composites, *gathers))
    assert [operation.kind for operation in graph.operations].count("dot_general") == 3
    assert [operation.attributes.reducer for operation in graph.operations if operation.kind == "reduce"] == [
        "maximum",
        "add",
    ]


def test_frozen_fixture_matches_current_natural_jax_export_structure() -> None:
    current = import_stablehlo(export_debug_routed_attention(), input_names=ROUTED_ATTENTION_INPUT_NAMES)
    frozen = import_stablehlo(_fixture_artifact(), input_names=ROUTED_ATTENTION_INPUT_NAMES)

    frozen_operations = [
        (operation.kind, tuple(frozen.value(value).shape for value in operation.outputs))
        for operation in frozen.operations
    ]
    assert frozen_operations == [
        (operation.kind, tuple(current.value(value).shape for value in operation.outputs))
        for operation in current.operations
    ]


def test_public_frontend_erases_names_before_relation_scheduling() -> None:
    recovered = recover_stablehlo_routed_attention_program(
        _fixture_artifact(),
        input_names=ROUTED_ATTENTION_INPUT_NAMES,
    )

    assert recovered.generic_operation_kinds == (
        "Contract",
        "DomainRestriction",
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
    assert recovered.semantic_erasure_report.source_semantics == (
        "top_k",
        "selected_exact_attention",
        "normalized_exponential",
        "causal_predicate",
    )
    assert recovered.source_operation_ids == tuple(range(82))
    assert all("attention" not in key for key in recovered.semantic_erasure_report.scheduling_keys)


def test_runtime_relation_mutation_changes_edges_without_changing_generated_body() -> None:
    config = RoutedAttentionDebugConfig()
    recovered = recover_stablehlo_routed_attention_program(
        _fixture_artifact(),
        input_names=ROUTED_ATTENTION_INPUT_NAMES,
    )
    _, _, _, left_a, right_a = _semantic_inputs(config, 11)
    _, _, _, left_b, right_b = _semantic_inputs(config, 29)
    schedule = StreamingTileSchedule(
        query_tile_size=config.block_size,
        key_value_tile_size=config.block_size,
        pipeline_depth=2,
    )

    compiled_a = compile_stablehlo_routed_attention_program(
        _fixture_artifact(),
        input_names=ROUTED_ATTENTION_INPUT_NAMES,
        runtime_inputs={"query_metadata": left_a, "key_value_metadata": right_a},
        schedule=schedule,
        config=_physical_config(config),
    )
    compiled_b = compile_stablehlo_routed_attention_program(
        _fixture_artifact(),
        input_names=ROUTED_ATTENTION_INPUT_NAMES,
        runtime_inputs={"query_metadata": left_b, "key_value_metadata": right_b},
        schedule=schedule,
        config=_physical_config(config),
    )

    assert not np.array_equal(compiled_a.relation.destination_item, compiled_b.relation.destination_item)
    assert compiled_a.streaming_program.source == compiled_b.streaming_program.source
    assert [candidate.orientation for candidate in compiled_a.scheduled.candidates] == [
        RoutedAttentionOrientation.QUERY_MAJOR,
        RoutedAttentionOrientation.KV_MAJOR,
    ]
    assert [candidate.orientation for candidate in compiled_b.scheduled.candidates] == [
        RoutedAttentionOrientation.QUERY_MAJOR,
        RoutedAttentionOrientation.KV_MAJOR,
    ]
    assert recovered.relation_selection == compiled_a.recovered.relation_selection
    assert recovered.relation_selection == compiled_b.recovered.relation_selection


def test_natural_jax_output_matches_relation_driven_online_fold() -> None:
    config = RoutedAttentionDebugConfig()
    query, key, value, query_metadata, key_value_metadata = _semantic_inputs(config, 41)
    recovered = recover_stablehlo_routed_attention_program(
        _fixture_artifact(),
        input_names=ROUTED_ATTENTION_INPUT_NAMES,
    )
    selected, valid = execute_relation_selection(
        recovered.relation_selection,
        {
            "query_metadata": query_metadata,
            "key_value_metadata": key_value_metadata,
        },
    )

    natural = np.asarray(
        routed_attention_region(config)(
            jnp.asarray(query, dtype=jnp.bfloat16),
            jnp.asarray(key, dtype=jnp.bfloat16),
            jnp.asarray(value, dtype=jnp.bfloat16),
            query_metadata,
            key_value_metadata,
        ),
        dtype=np.float32,
    )
    generated_reference = execute_query_major_attention(
        query.reshape(config.block_count, config.block_size, config.query_heads, config.head_dimension),
        key.reshape(config.block_count, config.block_size, config.key_value_heads, config.head_dimension),
        value.reshape(config.block_count, config.block_size, config.key_value_heads, config.head_dimension),
        selected,
        edge_valid=valid,
        scale=config.scale,
        causal=True,
    ).reshape(natural.shape)
    error = np.abs(natural - generated_reference)

    assert float(np.max(error)) <= 0.016
    assert float(np.mean(error)) <= 0.002
