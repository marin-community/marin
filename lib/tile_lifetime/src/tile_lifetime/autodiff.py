# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reverse-mode construction for generic Shuttle tensor algebra."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tile_lifetime.tensor_program import (
    ContractPrimitive,
    FoldPrimitive,
    FoldReducer,
    MapPrimitive,
    ProgramValue,
    ScalarExpression,
    ScalarExpressionKind,
    TensorAxis,
    TensorPrimitive,
    TensorProgram,
    primitive_inputs,
    scalar_binary,
    scalar_constant,
    scalar_expression_inputs,
    scalar_input,
    scalar_select,
    scalar_unary,
)


@dataclass(frozen=True)
class DifferentiatedTensorProgram:
    """A source program followed by its compiler-generated reverse program."""

    source: TensorProgram
    program: TensorProgram
    output_cotangents: tuple[ProgramValue, ...]
    input_gradients: tuple[ProgramValue, ...]


@dataclass(frozen=True)
class BackwardTensorProgram:
    """A standalone reverse program plus its explicit save/recompute policy."""

    source: DifferentiatedTensorProgram
    program: TensorProgram
    saved_values: tuple[ProgramValue, ...]
    recomputed_operations: tuple[TensorPrimitive, ...]


def differentiate_scalar_expression(expression: ScalarExpression, input_name: str) -> ScalarExpression:
    """Differentiate a scalar Map AST with respect to one named input."""

    kind = expression.kind
    if kind is ScalarExpressionKind.INPUT:
        return scalar_constant(1.0 if expression.input_name == input_name else 0.0)
    if kind is ScalarExpressionKind.CONSTANT:
        return scalar_constant(0.0)
    operands = expression.operands
    derivatives = tuple(differentiate_scalar_expression(operand, input_name) for operand in operands)
    if kind is ScalarExpressionKind.ADD:
        return _add(derivatives[0], derivatives[1])
    if kind is ScalarExpressionKind.SUBTRACT:
        return _subtract(derivatives[0], derivatives[1])
    if kind is ScalarExpressionKind.MULTIPLY:
        return _add(_multiply(derivatives[0], operands[1]), _multiply(operands[0], derivatives[1]))
    if kind is ScalarExpressionKind.DIVIDE:
        numerator = _subtract(
            _multiply(derivatives[0], operands[1]),
            _multiply(operands[0], derivatives[1]),
        )
        return _divide(numerator, _multiply(operands[1], operands[1]))
    if kind is ScalarExpressionKind.EXP:
        return _multiply(scalar_unary(ScalarExpressionKind.EXP, operands[0]), derivatives[0])
    if kind is ScalarExpressionKind.LOG:
        return _divide(derivatives[0], operands[0])
    if kind is ScalarExpressionKind.RSQRT:
        inverse_root = scalar_unary(ScalarExpressionKind.RSQRT, operands[0])
        inverse_cube = _multiply(inverse_root, _multiply(inverse_root, inverse_root))
        return _multiply(_multiply(scalar_constant(-0.5), inverse_cube), derivatives[0])
    if kind is ScalarExpressionKind.TANH:
        hyperbolic_tangent = scalar_unary(ScalarExpressionKind.TANH, operands[0])
        slope = _subtract(scalar_constant(1.0), _multiply(hyperbolic_tangent, hyperbolic_tangent))
        return _multiply(slope, derivatives[0])
    if kind is ScalarExpressionKind.LESS_EQUAL:
        return scalar_constant(0.0)
    assert kind is ScalarExpressionKind.SELECT
    return scalar_select(operands[0], derivatives[1], derivatives[2])


def scalar_expression_vjp(
    expression: ScalarExpression,
    *,
    input_name: str,
    cotangent_name: str,
) -> ScalarExpression:
    """Return the pointwise vector-Jacobian product for one scalar input."""

    return _multiply(
        scalar_input(cotangent_name),
        differentiate_scalar_expression(expression, input_name),
    )


def differentiate_tensor_program(
    source: TensorProgram,
    *,
    with_respect_to: tuple[str, ...],
) -> DifferentiatedTensorProgram:
    """Append a reverse program for Contract, Map, and sum-Fold primitives.

    The first prototype recomputes source intermediates by retaining the forward
    operations in the differentiated program. Saved-value versus recomputation
    is intentionally left to a later materialization planner.
    """

    value_by_name = {value.name: value for value in source.inputs}
    for operation in source.operations:
        value_by_name[operation.output.name] = operation.output
    requested = tuple(_required_value(value_by_name, name) for name in with_respect_to)
    counter = 0
    backward_operations: list[ContractPrimitive | MapPrimitive | FoldPrimitive] = []
    contributions: dict[str, list[ProgramValue]] = {}

    def fresh_value(prefix: str, axes: tuple[TensorAxis, ...], template: ProgramValue) -> ProgramValue:
        nonlocal counter
        counter += 1
        return ProgramValue(name=f"autodiff.{counter}.{prefix}", axes=axes, dtype=template.dtype)

    output_cotangents = tuple(
        ProgramValue(name=f"cotangent.{output.name}", axes=output.axes, dtype=output.dtype) for output in source.outputs
    )
    for output, cotangent in zip(source.outputs, output_cotangents, strict=True):
        contributions.setdefault(output.name, []).append(cotangent)

    def aggregate(value: ProgramValue) -> ProgramValue | None:
        pending = contributions.get(value.name, [])
        if not pending:
            return None
        accumulated = pending[0]
        for contribution in pending[1:]:
            output = fresh_value(f"sum_d_{value.name}", value.axes, value)
            backward_operations.append(
                MapPrimitive(
                    name=f"sum gradients for {value.name}",
                    inputs=(accumulated, contribution),
                    output=output,
                    expression=scalar_binary(
                        ScalarExpressionKind.ADD,
                        scalar_input(accumulated.name),
                        scalar_input(contribution.name),
                    ),
                )
            )
            accumulated = output
        contributions[value.name] = [accumulated]
        return accumulated

    for operation in reversed(source.operations):
        output_cotangent = aggregate(operation.output)
        if output_cotangent is None:
            continue
        if isinstance(operation, MapPrimitive):
            _differentiate_map(
                operation,
                output_cotangent,
                contributions,
                backward_operations,
                fresh_value,
            )
            continue
        if isinstance(operation, FoldPrimitive):
            if operation.reducer is not FoldReducer.SUM:
                raise NotImplementedError("maximum-Fold adjoints require an explicit tie policy")
            input_cotangent = fresh_value(f"d_{operation.input.name}", operation.input.axes, operation.input)
            backward_operations.append(
                MapPrimitive(
                    name=f"broadcast adjoint of {operation.name}",
                    inputs=(output_cotangent, operation.input),
                    output=input_cotangent,
                    expression=scalar_binary(
                        ScalarExpressionKind.ADD,
                        scalar_input(output_cotangent.name),
                        scalar_binary(
                            ScalarExpressionKind.MULTIPLY,
                            scalar_constant(0.0),
                            scalar_input(operation.input.name),
                        ),
                    ),
                )
            )
            contributions.setdefault(operation.input.name, []).append(input_cotangent)
            continue
        assert isinstance(operation, ContractPrimitive)
        if operation.input_index_maps:
            raise NotImplementedError("Contract adjoints with index maps are not implemented")
        for input_index, operand in enumerate(operation.inputs):
            other_operands = tuple(value for index, value in enumerate(operation.inputs) if index != input_index)
            adjoint_inputs = (output_cotangent, *other_operands)
            available_axes = {axis for value in adjoint_inputs for axis in value.axes}
            if not set(operand.axes) <= available_axes:
                raise NotImplementedError(f"cannot recover all axes of Contract operand {operand.name!r}")
            reduction_axes = tuple(axis for axis in _ordered_axes(adjoint_inputs) if axis not in operand.axes)
            input_cotangent = fresh_value(f"d_{operand.name}", operand.axes, operand)
            backward_operations.append(
                ContractPrimitive(
                    name=f"adjoint of {operation.name} for {operand.name}",
                    inputs=adjoint_inputs,
                    output=input_cotangent,
                    reduction_axes=reduction_axes,
                    accumulation_dtype=operation.accumulation_dtype,
                )
            )
            contributions.setdefault(operand.name, []).append(input_cotangent)

    input_gradients: list[ProgramValue] = []
    for value in requested:
        gradient = aggregate(value)
        if gradient is None:
            raise ValueError(f"requested input {value.name!r} has no differentiable path to a program output")
        input_gradients.append(gradient)
    differentiated = TensorProgram(
        inputs=(*source.inputs, *output_cotangents),
        operations=(*source.operations, *backward_operations),
        outputs=tuple(input_gradients),
    )
    return DifferentiatedTensorProgram(
        source=source,
        program=differentiated,
        output_cotangents=output_cotangents,
        input_gradients=tuple(input_gradients),
    )


def extract_backward_tensor_program(
    differentiated: DifferentiatedTensorProgram,
    *,
    saved_values: tuple[str, ...] = (),
) -> BackwardTensorProgram:
    """Extract a standalone reverse program under an explicit saved-value policy.

    Forward intermediates named in ``saved_values`` become reverse-program
    inputs. Any other forward intermediate required by the generated adjoint is
    recomputed from source inputs. This keeps save-versus-recompute as an
    inspectable compiler decision instead of embedding it in a workload
    backward kernel.
    """

    source = differentiated.source
    source_value_by_name = {value.name: value for value in source.inputs}
    producer_by_name: dict[str, TensorPrimitive] = {}
    for operation in source.operations:
        source_value_by_name[operation.output.name] = operation.output
        producer_by_name[operation.output.name] = operation

    duplicate_saved_values = sorted(name for name in set(saved_values) if saved_values.count(name) > 1)
    if duplicate_saved_values:
        raise ValueError(f"saved values must be unique: {duplicate_saved_values}")
    unknown_saved_values = sorted(set(saved_values) - set(producer_by_name))
    if unknown_saved_values:
        raise ValueError(f"saved values must be forward intermediates: {unknown_saved_values}")
    saved = tuple(source_value_by_name[name] for name in saved_values)
    saved_names = set(saved_values)

    backward_operations = differentiated.program.operations[len(source.operations) :]
    required_recompute_names: set[str] = set()

    def require_forward_value(value: ProgramValue) -> None:
        if value.name in saved_names or value.name not in producer_by_name:
            return
        if value.name in required_recompute_names:
            return
        required_recompute_names.add(value.name)
        for operand in primitive_inputs(producer_by_name[value.name]):
            require_forward_value(operand)

    source_names = set(source_value_by_name)
    for operation in backward_operations:
        for operand in primitive_inputs(operation):
            if operand.name in source_names:
                require_forward_value(operand)

    recomputed_operations = tuple(
        operation for operation in source.operations if operation.output.name in required_recompute_names
    )
    operations = (*recomputed_operations, *backward_operations)
    produced_names = {operation.output.name for operation in operations}
    required_input_names = {
        value.name
        for operation in operations
        for value in primitive_inputs(operation)
        if value.name not in produced_names
    }
    candidate_inputs = (*source.inputs, *saved, *differentiated.output_cotangents)
    inputs = tuple(value for value in candidate_inputs if value.name in required_input_names)
    supplied_names = {value.name for value in inputs}
    if supplied_names != required_input_names:
        missing = sorted(required_input_names - supplied_names)
        raise ValueError(f"backward extraction left unavailable values {missing}")

    program = TensorProgram(
        inputs=inputs,
        operations=operations,
        outputs=differentiated.input_gradients,
    )
    return BackwardTensorProgram(
        source=differentiated,
        program=program,
        saved_values=saved,
        recomputed_operations=recomputed_operations,
    )


def _differentiate_map(
    operation: MapPrimitive,
    output_cotangent: ProgramValue,
    contributions: dict[str, list[ProgramValue]],
    backward_operations: list[ContractPrimitive | MapPrimitive | FoldPrimitive],
    fresh_value: Callable[[str, tuple[TensorAxis, ...], ProgramValue], ProgramValue],
) -> None:
    input_by_name = {value.name: value for value in operation.inputs}
    for input_value in operation.inputs:
        derivative = differentiate_scalar_expression(operation.expression, input_value.name)
        if _is_constant(derivative, 0.0):
            continue
        expression = _multiply(scalar_input(output_cotangent.name), derivative)
        referenced_names = scalar_expression_inputs(expression)
        pointwise_inputs = tuple(
            value for value in (*operation.inputs, output_cotangent) if value.name in referenced_names
        )
        if set(referenced_names) != {value.name for value in pointwise_inputs}:
            missing = sorted(set(referenced_names) - set(input_by_name) - {output_cotangent.name})
            raise ValueError(f"Map derivative references unavailable inputs {missing}")
        pointwise = fresh_value(f"pointwise_d_{input_value.name}", operation.output.axes, input_value)
        backward_operations.append(
            MapPrimitive(
                name=f"pointwise adjoint of {operation.name} for {input_value.name}",
                inputs=pointwise_inputs,
                output=pointwise,
                expression=expression,
            )
        )
        reduction_axes = tuple(axis for axis in operation.output.axes if axis not in input_value.axes)
        if reduction_axes:
            input_cotangent = fresh_value(f"d_{input_value.name}", input_value.axes, input_value)
            backward_operations.append(
                FoldPrimitive(
                    name=f"broadcast adjoint Fold for {input_value.name}",
                    input=pointwise,
                    output=input_cotangent,
                    reduction_axes=reduction_axes,
                    reducer=FoldReducer.SUM,
                    accumulation_dtype=input_value.dtype,
                )
            )
        else:
            input_cotangent = pointwise
        contributions.setdefault(input_value.name, []).append(input_cotangent)


def _required_value(values: dict[str, ProgramValue], name: str) -> ProgramValue:
    try:
        return values[name]
    except KeyError as exc:
        raise ValueError(f"unknown differentiation input {name!r}") from exc


def _ordered_axes(values: tuple[ProgramValue, ...]) -> tuple[TensorAxis, ...]:
    result: list[TensorAxis] = []
    for value in values:
        for axis in value.axes:
            if axis not in result:
                result.append(axis)
    return tuple(result)


def _is_constant(expression: ScalarExpression, value: float) -> bool:
    return expression.kind is ScalarExpressionKind.CONSTANT and expression.constant == value


def _add(left: ScalarExpression, right: ScalarExpression) -> ScalarExpression:
    if _is_constant(left, 0.0):
        return right
    if _is_constant(right, 0.0):
        return left
    return scalar_binary(ScalarExpressionKind.ADD, left, right)


def _subtract(left: ScalarExpression, right: ScalarExpression) -> ScalarExpression:
    if _is_constant(right, 0.0):
        return left
    return scalar_binary(ScalarExpressionKind.SUBTRACT, left, right)


def _multiply(left: ScalarExpression, right: ScalarExpression) -> ScalarExpression:
    if _is_constant(left, 0.0) or _is_constant(right, 0.0):
        return scalar_constant(0.0)
    if _is_constant(left, 1.0):
        return right
    if _is_constant(right, 1.0):
        return left
    return scalar_binary(ScalarExpressionKind.MULTIPLY, left, right)


def _divide(left: ScalarExpression, right: ScalarExpression) -> ScalarExpression:
    if _is_constant(left, 0.0):
        return scalar_constant(0.0)
    if _is_constant(right, 1.0):
        return left
    return scalar_binary(ScalarExpressionKind.DIVIDE, left, right)
