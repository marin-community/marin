# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover generic row-statistic reverse algebra emitted by JAX AD."""

from __future__ import annotations

import re
from dataclasses import dataclass

from shuttle.ir import DType
from shuttle.stablehlo_import import (
    ConstantAttributes,
    ReductionAttributes,
    StableHLOGraph,
    StableHLOOperation,
)
from tile_lifetime.cuda_axis_fold_codegen import (
    AxisFoldDirection,
    AxisFoldInput,
    AxisFoldInputLayout,
    AxisFoldOutputKind,
    AxisFoldPipeline,
    AxisFoldPipelineSchedule,
    AxisFoldPipelineStage,
    AxisFoldProgram,
    AxisFoldReassociation,
    AxisFoldReduction,
    AxisFoldTiledReductionStrategy,
    GeneratedCudaAxisFoldFfi,
    generate_cuda_axis_fold_pipeline_ffi,
)
from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.row_normalization_training import (
    RowNormalizationAxisFoldPrograms,
    RowStatisticKind,
    build_row_normalization_axis_fold_programs,
)
from tile_lifetime.tensor_program import (
    ScalarExpression,
    ScalarExpressionKind,
    scalar_binary,
    scalar_constant,
    scalar_input,
    scalar_unary,
)

SCALAR_LITERAL = re.compile(r"dense<([^>]+)>")


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
    epsilon: float
    row_fold_operations: tuple[int, ...]
    feature_fold_operations: tuple[int, ...]


@dataclass(frozen=True)
class StableHLORowNormalizationBackwardCompilation:
    """Natural JAX reverse graph and its generic executable axis Folds."""

    recovered: RecoveredStableHLORowNormalizationBackward
    programs: RowNormalizationAxisFoldPrograms


@dataclass(frozen=True)
class StableHLORowNormalizationBackwardFfiCompilation:
    """Whole-entry replacement by a generated generic axis-Fold pipeline."""

    recovered: RecoveredStableHLORowNormalizationBackward
    pipeline: AxisFoldPipeline
    generated: GeneratedCudaAxisFoldFfi
    numerical_policy: NumericalPolicy
    input_bindings: tuple[tuple[str, int], ...]
    output_bindings: tuple[tuple[str, int], ...]
    replaced_operation_ids: tuple[int, ...]


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
    epsilon = _inverse_epsilon(graph, inverse_operation)
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
        epsilon=epsilon,
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


def compile_stablehlo_row_normalization_backward_ffi(
    graph: StableHLOGraph,
    *,
    target_name: str,
    numerical_policy: NumericalPolicy,
    threads: int = 256,
    feature_groups_per_block: int = 1,
    feature_outputs_per_group: int = 1,
    feature_tiled_reduction_strategy: AxisFoldTiledReductionStrategy = AxisFoldTiledReductionStrategy.BARRIER_TREE,
    pipeline_schedule: AxisFoldPipelineSchedule = AxisFoldPipelineSchedule.SEPARATE_STAGES,
) -> StableHLORowNormalizationBackwardFfiCompilation:
    """Replace one natural row-statistic VJP with generated Folds.

    The frontend name is only a recovery boundary. The physical generator sees
    generic row reductions, element Maps, and a feature reduction. Centered
    statistics add a row-mean Fold before the variance Fold; they do not select
    a different workload kernel.
    """
    recovered = recover_stablehlo_row_normalization_backward(graph)
    if numerical_policy is NumericalPolicy.BITWISE_EXACT:
        raise StableHLORowNormalizationBackwardError(
            "parallel axis-Fold replacement changes source reduction association; " "ALLOW_ROUNDING_REORDER is required"
        )
    if recovered.statistic_kind is RowStatisticKind.CENTERED_SECOND_MOMENT:
        if pipeline_schedule is not AxisFoldPipelineSchedule.SEPARATE_STAGES:
            raise StableHLORowNormalizationBackwardError(
                "centered row statistics currently require the separate-stage Fold schedule"
            )
        pipeline = _centered_second_moment_backward_pipeline(
            recovered,
            threads=threads,
            feature_groups_per_block=feature_groups_per_block,
            feature_outputs_per_group=feature_outputs_per_group,
            feature_tiled_reduction_strategy=feature_tiled_reduction_strategy,
        )
    else:
        pipeline = _uncentered_second_moment_backward_pipeline(
            recovered,
            threads=threads,
            feature_groups_per_block=feature_groups_per_block,
            feature_outputs_per_group=feature_outputs_per_group,
            feature_tiled_reduction_strategy=feature_tiled_reduction_strategy,
        )
    generated = generate_cuda_axis_fold_pipeline_ffi(
        pipeline,
        target_name=target_name,
        schedule=pipeline_schedule,
    )
    return StableHLORowNormalizationBackwardFfiCompilation(
        recovered=recovered,
        pipeline=pipeline,
        generated=generated,
        numerical_policy=numerical_policy,
        input_bindings=(
            ("primal", recovered.input),
            ("feature_scale", recovered.feature_scale),
            ("output_cotangent", recovered.output_cotangent),
        ),
        output_bindings=(
            ("input_cotangent", recovered.input_cotangent),
            ("feature_scale_cotangent", recovered.feature_scale_cotangent),
        ),
        replaced_operation_ids=tuple(operation.id for operation in graph.operations),
    )


def _uncentered_second_moment_backward_pipeline(
    recovered: RecoveredStableHLORowNormalizationBackward,
    *,
    threads: int,
    feature_groups_per_block: int,
    feature_outputs_per_group: int,
    feature_tiled_reduction_strategy: AxisFoldTiledReductionStrategy,
) -> AxisFoldPipeline:
    source_dtype = recovered.source_dtype
    output_dtype = recovered.graph.value(recovered.input_cotangent).dtype
    scale_output_dtype = recovered.graph.value(recovered.feature_scale_cotangent).dtype
    primal = scalar_input("primal")
    feature_scale = scalar_input("feature_scale")
    output_cotangent = scalar_input("output_cotangent")
    inverse_scale = scalar_input("inverse_scale")
    hidden = scalar_constant(float(recovered.hidden))
    local = _multiply(output_cotangent, feature_scale)

    inverse_program = AxisFoldProgram(
        rows=recovered.rows,
        columns=recovered.hidden,
        inputs=(AxisFoldInput("primal", source_dtype, AxisFoldInputLayout.ELEMENT),),
        reductions=(AxisFoldReduction("sum_square", _multiply(primal, primal)),),
        reduction_axis=AxisFoldDirection.COLUMNS,
        output_kind=AxisFoldOutputKind.REDUCED,
        output_expression=scalar_unary(
            ScalarExpressionKind.RSQRT,
            _add(_divide(scalar_input("sum_square"), hidden), scalar_constant(recovered.epsilon)),
        ),
        output_dtype=DType.FP32,
        threads=threads,
        reassociation=AxisFoldReassociation.DETERMINISTIC_TREE,
    )
    input_cotangent_program = AxisFoldProgram(
        rows=recovered.rows,
        columns=recovered.hidden,
        inputs=(
            AxisFoldInput("primal", source_dtype, AxisFoldInputLayout.ELEMENT),
            AxisFoldInput(
                "feature_scale",
                recovered.graph.value(recovered.feature_scale).dtype,
                AxisFoldInputLayout.COLUMN,
            ),
            AxisFoldInput(
                "output_cotangent",
                recovered.graph.value(recovered.output_cotangent).dtype,
                AxisFoldInputLayout.ELEMENT,
            ),
            AxisFoldInput("inverse_scale", DType.FP32, AxisFoldInputLayout.ROW),
        ),
        reductions=(AxisFoldReduction("correlation", _multiply(local, primal)),),
        reduction_axis=AxisFoldDirection.COLUMNS,
        output_kind=AxisFoldOutputKind.ELEMENT,
        output_expression=_multiply(
            inverse_scale,
            _subtract(
                local,
                _multiply(
                    primal,
                    _multiply(
                        _multiply(inverse_scale, inverse_scale),
                        _divide(scalar_input("correlation"), hidden),
                    ),
                ),
            ),
        ),
        output_dtype=output_dtype,
        threads=threads,
        reassociation=AxisFoldReassociation.DETERMINISTIC_TREE,
    )
    feature_scale_cotangent_program = AxisFoldProgram(
        rows=recovered.rows,
        columns=recovered.hidden,
        inputs=(
            AxisFoldInput("primal", source_dtype, AxisFoldInputLayout.ELEMENT),
            AxisFoldInput(
                "output_cotangent",
                recovered.graph.value(recovered.output_cotangent).dtype,
                AxisFoldInputLayout.ELEMENT,
            ),
            AxisFoldInput("inverse_scale", DType.FP32, AxisFoldInputLayout.ROW),
        ),
        reductions=(
            AxisFoldReduction(
                "scale_cotangent_sum",
                _multiply(_multiply(output_cotangent, primal), inverse_scale),
            ),
        ),
        reduction_axis=AxisFoldDirection.ROWS,
        output_kind=AxisFoldOutputKind.REDUCED,
        output_expression=scalar_input("scale_cotangent_sum"),
        output_dtype=scale_output_dtype,
        threads=threads,
        groups_per_block=feature_groups_per_block,
        outputs_per_group=feature_outputs_per_group,
        tiled_reduction_strategy=feature_tiled_reduction_strategy,
        reassociation=AxisFoldReassociation.DETERMINISTIC_TREE,
    )
    return AxisFoldPipeline(
        stages=(
            AxisFoldPipelineStage("inverse_scale", inverse_program, expose_output=False),
            AxisFoldPipelineStage("input_cotangent", input_cotangent_program, expose_output=True),
            AxisFoldPipelineStage(
                "feature_scale_cotangent",
                feature_scale_cotangent_program,
                expose_output=True,
            ),
        )
    )


def _centered_second_moment_backward_pipeline(
    recovered: RecoveredStableHLORowNormalizationBackward,
    *,
    threads: int,
    feature_groups_per_block: int,
    feature_outputs_per_group: int,
    feature_tiled_reduction_strategy: AxisFoldTiledReductionStrategy,
) -> AxisFoldPipeline:
    source_dtype = recovered.source_dtype
    output_dtype = recovered.graph.value(recovered.input_cotangent).dtype
    scale_output_dtype = recovered.graph.value(recovered.feature_scale_cotangent).dtype
    primal = scalar_input("primal")
    feature_scale = scalar_input("feature_scale")
    output_cotangent = scalar_input("output_cotangent")
    row_mean = scalar_input("row_mean")
    inverse_scale = scalar_input("inverse_scale")
    hidden = scalar_constant(float(recovered.hidden))
    centered = _subtract(primal, row_mean)
    standardized = _multiply(centered, inverse_scale)
    local = _multiply(output_cotangent, feature_scale)

    row_mean_program = AxisFoldProgram(
        rows=recovered.rows,
        columns=recovered.hidden,
        inputs=(AxisFoldInput("primal", source_dtype, AxisFoldInputLayout.ELEMENT),),
        reductions=(AxisFoldReduction("row_sum", primal),),
        reduction_axis=AxisFoldDirection.COLUMNS,
        output_kind=AxisFoldOutputKind.REDUCED,
        output_expression=_divide(scalar_input("row_sum"), hidden),
        output_dtype=DType.FP32,
        threads=threads,
        reassociation=AxisFoldReassociation.DETERMINISTIC_TREE,
    )
    inverse_program = AxisFoldProgram(
        rows=recovered.rows,
        columns=recovered.hidden,
        inputs=(
            AxisFoldInput("primal", source_dtype, AxisFoldInputLayout.ELEMENT),
            AxisFoldInput("row_mean", DType.FP32, AxisFoldInputLayout.ROW),
        ),
        reductions=(AxisFoldReduction("centered_sum_square", _multiply(centered, centered)),),
        reduction_axis=AxisFoldDirection.COLUMNS,
        output_kind=AxisFoldOutputKind.REDUCED,
        output_expression=scalar_unary(
            ScalarExpressionKind.RSQRT,
            _add(
                _divide(scalar_input("centered_sum_square"), hidden),
                scalar_constant(recovered.epsilon),
            ),
        ),
        output_dtype=DType.FP32,
        threads=threads,
        reassociation=AxisFoldReassociation.DETERMINISTIC_TREE,
    )
    input_cotangent_program = AxisFoldProgram(
        rows=recovered.rows,
        columns=recovered.hidden,
        inputs=(
            AxisFoldInput("primal", source_dtype, AxisFoldInputLayout.ELEMENT),
            AxisFoldInput(
                "feature_scale",
                recovered.graph.value(recovered.feature_scale).dtype,
                AxisFoldInputLayout.COLUMN,
            ),
            AxisFoldInput(
                "output_cotangent",
                recovered.graph.value(recovered.output_cotangent).dtype,
                AxisFoldInputLayout.ELEMENT,
            ),
            AxisFoldInput("row_mean", DType.FP32, AxisFoldInputLayout.ROW),
            AxisFoldInput("inverse_scale", DType.FP32, AxisFoldInputLayout.ROW),
        ),
        reductions=(
            AxisFoldReduction("correlation", _multiply(local, standardized)),
            AxisFoldReduction("local_sum", local),
        ),
        reduction_axis=AxisFoldDirection.COLUMNS,
        output_kind=AxisFoldOutputKind.ELEMENT,
        output_expression=_multiply(
            inverse_scale,
            _subtract(
                _subtract(
                    local,
                    _multiply(standardized, _divide(scalar_input("correlation"), hidden)),
                ),
                _divide(scalar_input("local_sum"), hidden),
            ),
        ),
        output_dtype=output_dtype,
        threads=threads,
        reassociation=AxisFoldReassociation.DETERMINISTIC_TREE,
    )
    feature_scale_cotangent_program = AxisFoldProgram(
        rows=recovered.rows,
        columns=recovered.hidden,
        inputs=(
            AxisFoldInput("primal", source_dtype, AxisFoldInputLayout.ELEMENT),
            AxisFoldInput(
                "output_cotangent",
                recovered.graph.value(recovered.output_cotangent).dtype,
                AxisFoldInputLayout.ELEMENT,
            ),
            AxisFoldInput("row_mean", DType.FP32, AxisFoldInputLayout.ROW),
            AxisFoldInput("inverse_scale", DType.FP32, AxisFoldInputLayout.ROW),
        ),
        reductions=(
            AxisFoldReduction(
                "scale_cotangent_sum",
                _multiply(output_cotangent, standardized),
            ),
        ),
        reduction_axis=AxisFoldDirection.ROWS,
        output_kind=AxisFoldOutputKind.REDUCED,
        output_expression=scalar_input("scale_cotangent_sum"),
        output_dtype=scale_output_dtype,
        threads=threads,
        groups_per_block=feature_groups_per_block,
        outputs_per_group=feature_outputs_per_group,
        tiled_reduction_strategy=feature_tiled_reduction_strategy,
        reassociation=AxisFoldReassociation.DETERMINISTIC_TREE,
    )
    return AxisFoldPipeline(
        stages=(
            AxisFoldPipelineStage("row_mean", row_mean_program, expose_output=False),
            AxisFoldPipelineStage("inverse_scale", inverse_program, expose_output=False),
            AxisFoldPipelineStage("input_cotangent", input_cotangent_program, expose_output=True),
            AxisFoldPipelineStage(
                "feature_scale_cotangent",
                feature_scale_cotangent_program,
                expose_output=True,
            ),
        )
    )


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


def _inverse_epsilon(graph: StableHLOGraph, inverse_operation: StableHLOOperation) -> float:
    input_producer = graph.producer(inverse_operation.inputs[0])
    if input_producer is None or input_producer.kind != "add" or len(input_producer.inputs) != 2:
        raise StableHLORowNormalizationBackwardError("inverse statistic must consume a binary epsilon add")
    constants = tuple(
        value
        for value in (_constant_expression(graph, value_id) for value_id in input_producer.inputs)
        if value is not None
    )
    if len(constants) != 1 or constants[0] < 0:
        raise StableHLORowNormalizationBackwardError("inverse statistic must contain one non-negative scalar epsilon")
    return constants[0]


def _constant_expression(graph: StableHLOGraph, value_id: int) -> float | None:
    producer = graph.producer(value_id)
    if producer is None:
        return None
    if producer.kind == "constant":
        attributes = producer.attributes
        if not isinstance(attributes, ConstantAttributes):
            return None
        match = SCALAR_LITERAL.search(attributes.literal)
        if match is None:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None
    if producer.kind in {"broadcast_in_dim", "convert", "reshape", "transpose"} and len(producer.inputs) == 1:
        return _constant_expression(graph, producer.inputs[0])
    return None


def _add(left: ScalarExpression, right: ScalarExpression) -> ScalarExpression:
    return scalar_binary(ScalarExpressionKind.ADD, left, right)


def _subtract(left: ScalarExpression, right: ScalarExpression) -> ScalarExpression:
    return scalar_binary(ScalarExpressionKind.SUBTRACT, left, right)


def _multiply(left: ScalarExpression, right: ScalarExpression) -> ScalarExpression:
    return scalar_binary(ScalarExpressionKind.MULTIPLY, left, right)


def _divide(left: ScalarExpression, right: ScalarExpression) -> ScalarExpression:
    return scalar_binary(ScalarExpressionKind.DIVIDE, left, right)
