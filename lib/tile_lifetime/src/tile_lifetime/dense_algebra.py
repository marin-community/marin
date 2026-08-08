# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Erase dense frontend names into generic Contract/Map/Fold algebra."""

from tile_lifetime.ir import LinearOp, ResidualAddOp, RMSNormOp, TensorGraph, TensorValue
from tile_lifetime.plan import SemanticLoweringStep
from tile_lifetime.semantic_erasure import ErasedTensorProgram, build_tensor_erasure_report
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    FoldPrimitive,
    FoldReducer,
    MapPrimitive,
    ProgramValue,
    ScalarExpressionKind,
    TensorAxis,
    TensorProgram,
    scalar_binary,
    scalar_constant,
    scalar_input,
    scalar_unary,
)


class DenseSemanticErasureError(ValueError):
    """The recovered dense graph is outside the generic prototype vocabulary."""


def erase_dense_semantics(graph: TensorGraph) -> ErasedTensorProgram:
    """Lower named dense frontend operations to a closed generic tensor program."""
    values: dict[int, ProgramValue] = {}
    inputs: list[ProgramValue] = []
    operations: list[ContractPrimitive | MapPrimitive | FoldPrimitive] = []
    lowering_steps: list[SemanticLoweringStep] = []
    next_axis_id = 0

    def new_axis(extent: int, label: str | None = None) -> TensorAxis:
        nonlocal next_axis_id
        axis = TensorAxis(next_axis_id, extent, label)
        next_axis_id += 1
        return axis

    def bind(value: TensorValue, axes: tuple[TensorAxis, ...], *, external: bool) -> ProgramValue:
        existing = values.get(value.id)
        if existing is not None:
            if existing.axes != axes:
                raise DenseSemanticErasureError(
                    f"value {value.name!r} has inconsistent logical axes {existing.axes} and {axes}"
                )
            return existing
        if tuple(axis.extent for axis in axes) != value.shape:
            raise DenseSemanticErasureError(
                f"logical axes {tuple(axis.extent for axis in axes)} do not match {value.name!r} shape {value.shape}"
            )
        program_value = ProgramValue(value.name, axes, value.dtype)
        values[value.id] = program_value
        if external:
            inputs.append(program_value)
        return program_value

    def external(value: TensorValue, axes: tuple[TensorAxis, ...] | None = None) -> ProgramValue:
        existing = values.get(value.id)
        if existing is not None:
            if axes is not None and existing.axes != axes:
                raise DenseSemanticErasureError(f"value {value.name!r} requires an unsupported logical-axis unification")
            return existing
        if axes is None:
            axes = tuple(new_axis(extent) for extent in value.shape)
        return bind(value, axes, external=True)

    def produced(value: TensorValue, axes: tuple[TensorAxis, ...]) -> ProgramValue:
        return bind(value, axes, external=False)

    for operation in graph.operations:
        if isinstance(operation, LinearOp):
            input_value = external(operation.input)
            if len(input_value.axes) != 2:
                raise DenseSemanticErasureError("the generic H100 contraction prototype requires rank-two inputs")
            row_axis, reduction_axis = input_value.axes
            output_axis = new_axis(operation.output.shape[1])
            weight = external(operation.weight, (reduction_axis, output_axis))
            output = produced(operation.output, (row_axis, output_axis))
            operations.append(
                ContractPrimitive(
                    name=f"contract.{len(operations)}",
                    inputs=(input_value, weight),
                    output=output,
                    reduction_axes=(reduction_axis,),
                    accumulation_dtype=operation.accumulation_dtype,
                )
            )
            lowering_steps.append(SemanticLoweringStep(type(operation).__name__, ("Contract",)))
            continue

        if isinstance(operation, ResidualAddOp):
            left = external(operation.left)
            right = external(operation.right, left.axes)
            output = produced(operation.output, left.axes)
            operations.append(
                MapPrimitive(
                    name=f"map.{len(operations)}",
                    inputs=(left, right),
                    output=output,
                    expression=scalar_binary(
                        ScalarExpressionKind.ADD,
                        scalar_input(left.name),
                        scalar_input(right.name),
                    ),
                )
            )
            lowering_steps.append(SemanticLoweringStep(type(operation).__name__, ("Map",)))
            continue

        if isinstance(operation, RMSNormOp):
            source = external(operation.input)
            reduction_axis = source.axes[operation.axis]
            gamma = external(operation.gamma, (reduction_axis,))
            row_axes = tuple(axis for axis in source.axes if axis != reduction_axis)
            square = ProgramValue(f"value.{operation.id}.square", source.axes, operation.reduction_dtype)
            summed = ProgramValue(f"value.{operation.id}.fold", row_axes, operation.reduction_dtype)
            inverse_scale = ProgramValue(f"value.{operation.id}.row_scalar", row_axes, operation.reduction_dtype)
            feature_scaled = ProgramValue(f"value.{operation.id}.feature_scaled", source.axes, source.dtype)
            output = produced(operation.output, source.axes)
            operations.extend(
                (
                    MapPrimitive(
                        name=f"map.{len(operations)}",
                        inputs=(source,),
                        output=square,
                        expression=scalar_binary(
                            ScalarExpressionKind.MULTIPLY,
                            scalar_input(source.name),
                            scalar_input(source.name),
                        ),
                    ),
                    FoldPrimitive(
                        name=f"fold.{len(operations) + 1}",
                        input=square,
                        output=summed,
                        reduction_axes=(reduction_axis,),
                        reducer=FoldReducer.SUM,
                        accumulation_dtype=operation.reduction_dtype,
                    ),
                    MapPrimitive(
                        name=f"map.{len(operations) + 2}",
                        inputs=(summed,),
                        output=inverse_scale,
                        expression=scalar_unary(
                            ScalarExpressionKind.RSQRT,
                            scalar_binary(
                                ScalarExpressionKind.ADD,
                                scalar_binary(
                                    ScalarExpressionKind.DIVIDE,
                                    scalar_input(summed.name),
                                    scalar_constant(reduction_axis.extent),
                                ),
                                scalar_constant(operation.epsilon),
                            ),
                        ),
                    ),
                    MapPrimitive(
                        name=f"map.{len(operations) + 3}",
                        inputs=(source, gamma),
                        output=feature_scaled,
                        expression=scalar_binary(
                            ScalarExpressionKind.MULTIPLY,
                            scalar_input(source.name),
                            scalar_input(gamma.name),
                        ),
                    ),
                    MapPrimitive(
                        name=f"map.{len(operations) + 4}",
                        inputs=(feature_scaled, inverse_scale),
                        output=output,
                        expression=scalar_binary(
                            ScalarExpressionKind.MULTIPLY,
                            scalar_input(feature_scaled.name),
                            scalar_input(inverse_scale.name),
                        ),
                    ),
                )
            )
            lowering_steps.append(SemanticLoweringStep(type(operation).__name__, ("Map", "Fold", "Map")))
            continue

        raise DenseSemanticErasureError(
            f"dense semantic erasure does not support frontend operation {type(operation).__name__}"
        )

    outputs = tuple(
        values[value.id]
        for value in graph.values
        if value.id in values and graph.producer(value) is not None and not graph.consumers(value)
    )
    program = TensorProgram(inputs=tuple(inputs), operations=tuple(operations), outputs=outputs)
    source_semantics = tuple(dict.fromkeys(type(operation).__name__ for operation in graph.operations))
    report = build_tensor_erasure_report(
        program,
        source_semantics=source_semantics,
        lowering_steps=tuple(lowering_steps),
    )
    return ErasedTensorProgram(program=program, report=report)
