# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover generic row-statistic reverse algebra emitted by JAX AD."""

from __future__ import annotations

from dataclasses import dataclass

from tile_lifetime.ir import DType
from tile_lifetime.row_normalization_training import (
    RowNormalizationAxisFoldPrograms,
    RowStatisticKind,
    build_row_normalization_axis_fold_programs,
)
from tile_lifetime.stablehlo_import import ReductionAttributes, StableHLOGraph, StableHLOOperation


class StableHLORowNormalizationBackwardError(ValueError):
    """A structural mismatch in a candidate JAX-differentiated row program."""


@dataclass(frozen=True)
class RecoveredStableHLORowNormalizationBackward:
    """Generic Fold/Map structure recovered from a natural JAX VJP graph."""

    graph: StableHLOGraph
    input: int
    feature_scale: int
    output_cotangent: int
    input_cotangent: int
    feature_scale_cotangent: int
    inverse_scale: int
    rows: int
    hidden: int
    source_dtype: DType
    statistic_kind: RowStatisticKind
    row_fold_operations: tuple[int, ...]
    feature_fold_operations: tuple[int, ...]


@dataclass(frozen=True)
class StableHLORowNormalizationBackwardCompilation:
    """Natural JAX reverse graph and its generic executable axis Folds."""

    recovered: RecoveredStableHLORowNormalizationBackward
    programs: RowNormalizationAxisFoldPrograms


def recover_stablehlo_row_normalization_backward(
    graph: StableHLOGraph,
) -> RecoveredStableHLORowNormalizationBackward:
    """Recover row-statistic reverse structure without using source names.

    JAX is the source of the reverse program. This pass assigns generic axis and
    value roles, validates the required Fold dataflow, and leaves physical Fold
    fusion to the normal Shuttle lowering.
    """
    if len(graph.inputs) != 3 or len(graph.outputs) != 2:
        raise StableHLORowNormalizationBackwardError("expected three inputs and two cotangent outputs")
    matrix_inputs = tuple(value_id for value_id in graph.inputs if len(graph.value(value_id).shape) == 2)
    vector_inputs = tuple(value_id for value_id in graph.inputs if len(graph.value(value_id).shape) == 1)
    if len(matrix_inputs) != 2 or len(vector_inputs) != 1:
        raise StableHLORowNormalizationBackwardError("expected two matrix inputs and one feature vector")
    matrix_shapes = {graph.value(value_id).shape for value_id in matrix_inputs}
    if len(matrix_shapes) != 1:
        raise StableHLORowNormalizationBackwardError("primal input and output cotangent shapes differ")
    rows, hidden = graph.value(matrix_inputs[0]).shape
    feature_scale = vector_inputs[0]
    if graph.value(feature_scale).shape != (hidden,):
        raise StableHLORowNormalizationBackwardError("feature vector does not match the hidden axis")

    inverse_operations = tuple(operation for operation in graph.operations if operation.kind == "rsqrt")
    if len(inverse_operations) != 1 or len(inverse_operations[0].outputs) != 1:
        raise StableHLORowNormalizationBackwardError("expected one inverse-statistic rsqrt")
    inverse_operation = inverse_operations[0]
    inverse_ancestors = _ancestor_values(graph, inverse_operation.inputs)
    primal_inputs = tuple(value_id for value_id in matrix_inputs if value_id in inverse_ancestors)
    if len(primal_inputs) != 1:
        raise StableHLORowNormalizationBackwardError("inverse statistic does not identify one primal matrix")
    primal = primal_inputs[0]
    output_cotangent = next(value_id for value_id in matrix_inputs if value_id != primal)

    matrix_outputs = tuple(value_id for value_id in graph.outputs if graph.value(value_id).shape == (rows, hidden))
    vector_outputs = tuple(value_id for value_id in graph.outputs if graph.value(value_id).shape == (hidden,))
    if len(matrix_outputs) != 1 or len(vector_outputs) != 1:
        raise StableHLORowNormalizationBackwardError("cotangent output shapes do not match the primal inputs")
    input_cotangent = matrix_outputs[0]
    feature_scale_cotangent = vector_outputs[0]
    input_dependencies = _ancestor_values(graph, (input_cotangent,))
    if not {primal, feature_scale, output_cotangent}.issubset(input_dependencies):
        raise StableHLORowNormalizationBackwardError("input cotangent omits a required primal or cotangent input")
    scale_dependencies = _ancestor_values(graph, (feature_scale_cotangent,))
    if not {primal, output_cotangent}.issubset(scale_dependencies):
        raise StableHLORowNormalizationBackwardError("feature-scale cotangent omits required inputs")

    inverse_operation_ids = _ancestor_operation_ids(graph, inverse_operation.inputs)
    centered = any(
        operation.kind == "subtract" and operation.id in inverse_operation_ids for operation in graph.operations
    )
    statistic_kind = RowStatisticKind.CENTERED_SECOND_MOMENT if centered else RowStatisticKind.UNCENTERED_SECOND_MOMENT
    row_folds = _matrix_sum_folds(graph, rows=rows, hidden=hidden, dimension=1)
    feature_folds = _matrix_sum_folds(graph, rows=rows, hidden=hidden, dimension=0)
    minimum_row_folds = 3 if centered else 2
    if len(row_folds) < minimum_row_folds:
        raise StableHLORowNormalizationBackwardError("reverse graph lacks the required row Fold structure")
    if not feature_folds:
        raise StableHLORowNormalizationBackwardError("reverse graph lacks the feature-cotangent Fold")
    if any(operation.kind == "dot_general" for operation in graph.operations):
        raise StableHLORowNormalizationBackwardError("row-normalization reverse region unexpectedly contains a Contract")

    source_dtype = graph.value(primal).dtype
    if graph.value(output_cotangent).dtype is not source_dtype:
        raise StableHLORowNormalizationBackwardError("primal and output cotangent dtypes differ")
    return RecoveredStableHLORowNormalizationBackward(
        graph=graph,
        input=primal,
        feature_scale=feature_scale,
        output_cotangent=output_cotangent,
        input_cotangent=input_cotangent,
        feature_scale_cotangent=feature_scale_cotangent,
        inverse_scale=inverse_operation.outputs[0],
        rows=rows,
        hidden=hidden,
        source_dtype=source_dtype,
        statistic_kind=statistic_kind,
        row_fold_operations=tuple(operation.id for operation in row_folds),
        feature_fold_operations=tuple(operation.id for operation in feature_folds),
    )


def compile_stablehlo_row_normalization_backward(
    graph: StableHLOGraph,
    *,
    threads: int = 256,
) -> StableHLORowNormalizationBackwardCompilation:
    """Lower a JAX-differentiated row program into generic axis Folds."""
    recovered = recover_stablehlo_row_normalization_backward(graph)
    programs = build_row_normalization_axis_fold_programs(
        rows=recovered.rows,
        hidden=recovered.hidden,
        source_dtype=recovered.source_dtype,
        statistic_kind=recovered.statistic_kind,
        threads=threads,
    )
    return StableHLORowNormalizationBackwardCompilation(recovered=recovered, programs=programs)


def _ancestor_values(graph: StableHLOGraph, roots: tuple[int, ...]) -> frozenset[int]:
    visited: set[int] = set()
    pending = list(roots)
    while pending:
        value_id = pending.pop()
        if value_id in visited:
            continue
        visited.add(value_id)
        producer = graph.producer(value_id)
        if producer is not None:
            pending.extend(producer.inputs)
    return frozenset(visited)


def _ancestor_operation_ids(graph: StableHLOGraph, roots: tuple[int, ...]) -> frozenset[int]:
    operation_ids: set[int] = set()
    pending = list(roots)
    visited_values: set[int] = set()
    while pending:
        value_id = pending.pop()
        if value_id in visited_values:
            continue
        visited_values.add(value_id)
        producer = graph.producer(value_id)
        if producer is None:
            continue
        operation_ids.add(producer.id)
        pending.extend(producer.inputs)
    return frozenset(operation_ids)


def _matrix_sum_folds(
    graph: StableHLOGraph,
    *,
    rows: int,
    hidden: int,
    dimension: int,
) -> tuple[StableHLOOperation, ...]:
    folds: list[StableHLOOperation] = []
    for operation in graph.operations:
        attributes = operation.attributes
        if operation.kind != "reduce" or not isinstance(attributes, ReductionAttributes):
            continue
        if attributes.reducer != "add" or attributes.dimensions != (dimension,):
            continue
        if graph.value(operation.inputs[0]).shape != (rows, hidden):
            continue
        folds.append(operation)
    return tuple(folds)
