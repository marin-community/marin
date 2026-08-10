# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from tile_lifetime.stablehlo_algebra_import import (
    ImportedContractNode,
    ImportedDomainRestrictionNode,
    ImportedFoldNode,
    ImportedMapNode,
    import_stablehlo_algebra,
)
from tile_lifetime.stablehlo_import import import_stablehlo
from tile_lifetime.streaming_attention_backward_reference import (
    STREAMING_ATTENTION_BACKWARD_INPUT_NAMES,
    StreamingAttentionBackwardDebugConfig,
    export_debug_streaming_attention_training,
)


def _import(scale: float = 0.5):
    graph = import_stablehlo(
        export_debug_streaming_attention_training(
            StreamingAttentionBackwardDebugConfig(
                batch=1,
                query_length=4,
                key_length=4,
                query_heads=4,
                key_value_heads=2,
                head_dimension=4,
                scale=scale,
            )
        ),
        input_names=STREAMING_ATTENTION_BACKWARD_INPUT_NAMES,
    )
    return graph, import_stablehlo_algebra(graph)


def test_generic_import_preserves_every_source_value_dependency_and_operation() -> None:
    graph, imported = _import()

    assert imported.inputs == graph.inputs
    assert imported.outputs == graph.outputs
    assert tuple(value.source_value_id for value in imported.values) == tuple(value.id for value in graph.values)
    assert tuple(operation.source_operation_id for operation in imported.operations) == tuple(
        operation.id for operation in graph.operations
    )
    for source_value, imported_value in zip(graph.values, imported.values, strict=True):
        producer = graph.producer(source_value.id)
        assert imported_value.producer_node_id == (producer.id if producer is not None else None)
        assert imported_value.consumer_node_ids == tuple(consumer.id for consumer in graph.consumers(source_value.id))
        assert (imported_value.shape, imported_value.dtype) == (source_value.shape, source_value.dtype)


def test_generic_import_classifies_algebra_without_workload_dispatch() -> None:
    graph, imported = _import()

    assert sum(isinstance(operation, ImportedContractNode) for operation in imported.operations) == sum(
        operation.kind == "dot_general" for operation in graph.operations
    )
    assert sum(isinstance(operation, ImportedFoldNode) for operation in imported.operations) == sum(
        operation.kind == "reduce" for operation in graph.operations
    )
    assert sum(isinstance(operation, ImportedDomainRestrictionNode) for operation in imported.operations) == sum(
        operation.kind == "select" for operation in graph.operations
    )
    assert all(
        isinstance(operation, ImportedMapNode)
        for operation in imported.operations
        if operation.source_kind in {"convert", "reshape", "transpose", "broadcast_in_dim"}
    )


def test_scale_mutation_changes_source_algebra_without_changing_importer() -> None:
    _first_graph, first = _import(0.5)
    _second_graph, second = _import(0.375)

    assert tuple(type(operation) for operation in first.operations) == tuple(
        type(operation) for operation in second.operations
    )
    assert tuple(operation.source_kind for operation in first.operations) == tuple(
        operation.source_kind for operation in second.operations
    )
    assert tuple(operation.attributes for operation in first.operations) != tuple(
        operation.attributes for operation in second.operations
    )
