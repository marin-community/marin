# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Anonymous two-Contract/one-Map training algebra for generated backends."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from shuttle.ir import DType
from tile_lifetime.autodiff import DifferentiatedTensorProgram, differentiate_tensor_program
from tile_lifetime.cuda_map_fold_codegen import evaluate_scalar_expression
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    MapPrimitive,
    ProgramValue,
    ScalarExpression,
    ScalarExpressionKind,
    TensorAxis,
    TensorProgram,
    scalar_binary,
    scalar_constant,
    scalar_input,
    scalar_unary,
)


class ContractMapNumericalPolicy(StrEnum):
    """Finite-precision Contract schedules admitted by the backend."""

    SOURCE_ORDERED = "source_ordered"
    FAST = "fast"


@dataclass(frozen=True)
class ContractMapBackendProgram:
    """Validated forward algebra and its mechanically derived reverse."""

    source: TensorProgram
    differentiated: DifferentiatedTensorProgram
    numerical_policy: ContractMapNumericalPolicy
    activation: ProgramValue
    first_weight: ProgramValue
    second_weight: ProgramValue
    preactivation: ProgramValue
    hidden: ProgramValue
    output: ProgramValue
    output_cotangent: ProgramValue
    input_adjoint: ProgramValue
    first_weight_adjoint: ProgramValue
    second_weight_adjoint: ProgramValue
    semantic_fingerprint: str

    @property
    def rows(self) -> int:
        return self.activation.shape[0]

    @property
    def reduction(self) -> int:
        return self.activation.shape[1]

    @property
    def features(self) -> int:
        return self.first_weight.shape[1]

    @property
    def scalar_expression(self) -> ScalarExpression:
        operation = self.source.operations[1]
        assert isinstance(operation, MapPrimitive)
        return operation.expression


@dataclass(frozen=True)
class ContractMapReverseLoweringPlan:
    """Closed five-operation reverse graph accepted by CUDA lowering."""

    hidden_adjoint_contract: ContractPrimitive
    second_weight_adjoint_contract: ContractPrimitive
    pointwise_adjoint: MapPrimitive
    input_adjoint_contract: ContractPrimitive
    first_weight_adjoint_contract: ContractPrimitive

    @property
    def operations(self) -> tuple[ContractPrimitive | MapPrimitive, ...]:
        return (
            self.hidden_adjoint_contract,
            self.second_weight_adjoint_contract,
            self.pointwise_adjoint,
            self.input_adjoint_contract,
            self.first_weight_adjoint_contract,
        )


@dataclass(frozen=True)
class ContractMapForwardReference:
    """Forward output and explicit values retained by the reverse."""

    output: np.ndarray
    preactivation: np.ndarray
    hidden: np.ndarray


@dataclass(frozen=True)
class ContractMapReverseReference:
    """Adjoints for all three differentiable source inputs."""

    input_adjoint: np.ndarray
    first_weight_adjoint: np.ndarray
    second_weight_adjoint: np.ndarray
    preactivation_adjoint: np.ndarray


def sigmoid_product_expression(name: str = "z") -> ScalarExpression:
    """Return ``z * sigmoid(z)`` with explicit left-to-right AST order."""
    value = scalar_input(name)
    negative = scalar_binary(ScalarExpressionKind.MULTIPLY, scalar_constant(-1.0), value)
    denominator = scalar_binary(
        ScalarExpressionKind.ADD,
        scalar_constant(1.0),
        scalar_unary(ScalarExpressionKind.EXP, negative),
    )
    sigmoid = scalar_binary(ScalarExpressionKind.DIVIDE, scalar_constant(1.0), denominator)
    return scalar_binary(ScalarExpressionKind.MULTIPLY, value, sigmoid)


def tanh_product_expression(name: str = "z") -> ScalarExpression:
    """Return ``z * tanh(z)`` with explicit operand order."""
    value = scalar_input(name)
    return scalar_binary(
        ScalarExpressionKind.MULTIPLY,
        value,
        scalar_unary(ScalarExpressionKind.TANH, value),
    )


def cubic_mix_expression(name: str = "z") -> ScalarExpression:
    """Return ``z + ((z * z) * z)`` with an explicit cubic fold order."""
    value = scalar_input(name)
    square = scalar_binary(ScalarExpressionKind.MULTIPLY, value, value)
    cube = scalar_binary(ScalarExpressionKind.MULTIPLY, square, value)
    return scalar_binary(ScalarExpressionKind.ADD, value, cube)


def build_contract_map_backend_program(
    *,
    rows: int,
    reduction: int,
    features: int,
    scalar_expression: ScalarExpression,
    numerical_policy: ContractMapNumericalPolicy,
) -> ContractMapBackendProgram:
    """Build anonymous ``x @ w0 -> Map -> @ w1`` algebra and its VJP."""
    if min(rows, reduction, features) <= 0:
        raise ValueError("Contract/Map dimensions must be positive")
    row_axis = TensorAxis(0, rows)
    reduction_axis = TensorAxis(1, reduction)
    feature_axis = TensorAxis(2, features)
    activation = ProgramValue("value.0", (row_axis, reduction_axis), DType.BF16)
    first_weight = ProgramValue("value.1", (reduction_axis, feature_axis), DType.BF16)
    second_weight = ProgramValue("value.2", (feature_axis, reduction_axis), DType.BF16)
    preactivation = ProgramValue("value.3", (row_axis, feature_axis), DType.BF16)
    hidden = ProgramValue("value.4", (row_axis, feature_axis), DType.BF16)
    output = ProgramValue("value.5", (row_axis, reduction_axis), DType.BF16)
    expression = _rename_single_input(scalar_expression, preactivation.name)
    source = TensorProgram(
        inputs=(activation, first_weight, second_weight),
        operations=(
            ContractPrimitive(
                name="operation.0",
                inputs=(activation, first_weight),
                output=preactivation,
                reduction_axes=(reduction_axis,),
                accumulation_dtype=DType.FP32,
            ),
            MapPrimitive(
                name="operation.1",
                inputs=(preactivation,),
                output=hidden,
                expression=expression,
            ),
            ContractPrimitive(
                name="operation.2",
                inputs=(hidden, second_weight),
                output=output,
                reduction_axes=(feature_axis,),
                accumulation_dtype=DType.FP32,
            ),
        ),
        outputs=(output,),
    )
    return form_contract_map_backend_program(source, numerical_policy=numerical_policy)


def form_contract_map_backend_program(
    source: TensorProgram,
    *,
    numerical_policy: ContractMapNumericalPolicy,
) -> ContractMapBackendProgram:
    """Validate a structural two-Contract/Map graph and derive its reverse."""
    if type(numerical_policy) is not ContractMapNumericalPolicy:
        raise TypeError("numerical_policy must be a ContractMapNumericalPolicy")
    if len(source.inputs) != 3 or len(source.operations) != 3 or len(source.outputs) != 1:
        raise ValueError("backend requires three inputs and Contract, Map, Contract operations")
    first_contract, scalar_map, second_contract = source.operations
    if not isinstance(first_contract, ContractPrimitive) or not isinstance(scalar_map, MapPrimitive):
        raise ValueError("backend requires Contract, Map, Contract operation order")
    if not isinstance(second_contract, ContractPrimitive):
        raise ValueError("backend requires Contract, Map, Contract operation order")
    activation, first_weight, second_weight = source.inputs
    preactivation = first_contract.output
    hidden = scalar_map.output
    output = second_contract.output
    _validate_forward_structure(
        activation,
        first_weight,
        second_weight,
        first_contract,
        scalar_map,
        second_contract,
        output,
    )
    if source.outputs != (output,):
        raise ValueError("backend return must be the second Contract result")
    differentiated = differentiate_tensor_program(
        source,
        with_respect_to=(activation.name, first_weight.name, second_weight.name),
    )
    input_adjoint, first_weight_adjoint, second_weight_adjoint = differentiated.input_gradients
    fingerprint = _semantic_fingerprint(source, numerical_policy)
    program = ContractMapBackendProgram(
        source=source,
        differentiated=differentiated,
        numerical_policy=numerical_policy,
        activation=activation,
        first_weight=first_weight,
        second_weight=second_weight,
        preactivation=preactivation,
        hidden=hidden,
        output=output,
        output_cotangent=differentiated.output_cotangents[0],
        input_adjoint=input_adjoint,
        first_weight_adjoint=first_weight_adjoint,
        second_weight_adjoint=second_weight_adjoint,
        semantic_fingerprint=fingerprint,
    )
    contract_map_reverse_lowering_plan(program)
    return program


def contract_map_reverse_lowering_plan(program: ContractMapBackendProgram) -> ContractMapReverseLoweringPlan:
    """Parse and validate the authoritative five-operation reverse program."""
    differentiated = program.differentiated
    if differentiated.source != program.source:
        raise ValueError("differentiated program must retain the exact source program")
    if differentiated.output_cotangents != (program.output_cotangent,):
        raise ValueError("reverse lowering requires one source-output cotangent")
    if differentiated.input_gradients != (
        program.input_adjoint,
        program.first_weight_adjoint,
        program.second_weight_adjoint,
    ):
        raise ValueError("reverse lowering gradients must preserve source-input order")
    expected_inputs = (*program.source.inputs, program.output_cotangent)
    if differentiated.program.inputs != expected_inputs:
        raise ValueError("differentiated program inputs must be source inputs followed by the output cotangent")
    source_operation_count = len(program.source.operations)
    if differentiated.program.operations[:source_operation_count] != program.source.operations:
        raise ValueError("differentiated program must retain the exact source-operation prefix")
    reverse_operations = differentiated.program.operations[source_operation_count:]
    if len(reverse_operations) != 5:
        raise ValueError("Contract/Map reverse lowering requires exactly five differentiated operations")
    hidden_adjoint, second_weight_adjoint, pointwise_adjoint, input_adjoint, first_weight_adjoint = reverse_operations
    if not isinstance(hidden_adjoint, ContractPrimitive):
        raise ValueError("reverse operation 0 must be the hidden-adjoint Contract")
    if not isinstance(second_weight_adjoint, ContractPrimitive):
        raise ValueError("reverse operation 1 must be the second-weight-adjoint Contract")
    if not isinstance(pointwise_adjoint, MapPrimitive):
        raise ValueError("reverse operation 2 must be the pointwise adjoint Map")
    if not isinstance(input_adjoint, ContractPrimitive):
        raise ValueError("reverse operation 3 must be the input-adjoint Contract")
    if not isinstance(first_weight_adjoint, ContractPrimitive):
        raise ValueError("reverse operation 4 must be the first-weight-adjoint Contract")
    plan = ContractMapReverseLoweringPlan(
        hidden_adjoint_contract=hidden_adjoint,
        second_weight_adjoint_contract=second_weight_adjoint,
        pointwise_adjoint=pointwise_adjoint,
        input_adjoint_contract=input_adjoint,
        first_weight_adjoint_contract=first_weight_adjoint,
    )
    _validate_reverse_lowering_plan(program, plan)
    return plan


def execute_contract_map_source_ordered_forward(
    program: ContractMapBackendProgram,
    activation: np.ndarray,
    first_weight: np.ndarray,
    second_weight: np.ndarray,
) -> ContractMapForwardReference:
    """Evaluate the literal FP32 reduction order with BF16 boundaries."""
    activation = _bf16_input(program.activation, activation)
    first_weight = _bf16_input(program.first_weight, first_weight)
    second_weight = _bf16_input(program.second_weight, second_weight)
    preactivation = _ordered_bf16_contract(activation, first_weight)
    hidden = _pointwise_bf16(program.scalar_expression, program.preactivation.name, preactivation)
    output = _ordered_bf16_contract(hidden, second_weight)
    return ContractMapForwardReference(output=output, preactivation=preactivation, hidden=hidden)


def execute_contract_map_source_ordered_reverse(
    program: ContractMapBackendProgram,
    activation: np.ndarray,
    first_weight: np.ndarray,
    second_weight: np.ndarray,
    saved: ContractMapForwardReference,
    output_cotangent: np.ndarray,
) -> ContractMapReverseReference:
    """Evaluate the mechanically derived VJP using literal reduction order."""
    activation = _bf16_input(program.activation, activation)
    first_weight = _bf16_input(program.first_weight, first_weight)
    second_weight = _bf16_input(program.second_weight, second_weight)
    cotangent = _bf16_input(program.output, output_cotangent)
    hidden_adjoint = _ordered_bf16_contract(cotangent, second_weight.T)
    derivative_map = _derived_map_adjoint(program)
    if derivative_map.inputs[0] != program.preactivation or len(derivative_map.inputs) != 2:
        raise ValueError("mechanical Map adjoint has an incompatible input boundary")
    cotangent_name = derivative_map.inputs[1].name
    preactivation_adjoint = np.empty_like(saved.preactivation)
    for index in np.ndindex(saved.preactivation.shape):
        preactivation_adjoint[index] = evaluate_scalar_expression(
            derivative_map.expression,
            {
                program.preactivation.name: float(saved.preactivation[index]),
                cotangent_name: float(hidden_adjoint[index]),
            },
        )
    preactivation_adjoint = round_float32_to_bfloat16(preactivation_adjoint)
    input_adjoint = _ordered_bf16_contract(preactivation_adjoint, first_weight.T)
    first_weight_adjoint = _ordered_bf16_contract(activation.T, preactivation_adjoint)
    second_weight_adjoint = _ordered_bf16_contract(saved.hidden.T, cotangent)
    return ContractMapReverseReference(
        input_adjoint=input_adjoint,
        first_weight_adjoint=first_weight_adjoint,
        second_weight_adjoint=second_weight_adjoint,
        preactivation_adjoint=preactivation_adjoint,
    )


def round_float32_to_bfloat16(value: np.ndarray) -> np.ndarray:
    """Round FP32 to BF16 RNE while retaining a NumPy FP32 container."""
    contiguous = np.ascontiguousarray(value, dtype=np.float32)
    bits = contiguous.view(np.uint32)
    is_nan = (bits & np.uint32(0x7F800000) == np.uint32(0x7F800000)) & (bits & np.uint32(0x007FFFFF) != 0)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = (bits + rounding_bias) & np.uint32(0xFFFF0000)
    rounded = np.where(is_nan, (bits & np.uint32(0xFFFF0000)) | np.uint32(0x00400000), rounded)
    return rounded.view(np.float32)


def _validate_forward_structure(
    activation: ProgramValue,
    first_weight: ProgramValue,
    second_weight: ProgramValue,
    first_contract: ContractPrimitive,
    scalar_map: MapPrimitive,
    second_contract: ContractPrimitive,
    output: ProgramValue,
) -> None:
    values = (
        activation,
        first_weight,
        second_weight,
        first_contract.output,
        scalar_map.output,
        output,
    )
    if any(value.dtype is not DType.BF16 for value in values):
        raise ValueError("bounded Contract/Map backend requires BF16 input and output boundaries")
    if any(axis.extent <= 0 for value in values for axis in value.axes):
        raise ValueError("bounded Contract/Map backend requires positive axis extents")
    if any(contract.accumulation_dtype is not DType.FP32 for contract in (first_contract, second_contract)):
        raise ValueError("bounded Contract/Map backend requires FP32 Contract accumulation")
    if first_contract.input_index_maps or second_contract.input_index_maps:
        raise ValueError("bounded Contract/Map backend does not support Contract index maps")
    if any(len(value.axes) != 2 for value in (activation, first_weight, second_weight, output)):
        raise ValueError("bounded Contract/Map backend requires rank-two values")
    row_axis, reduction_axis = activation.axes
    if first_weight.axes[0] != reduction_axis:
        raise ValueError("first Contract weight must share the activation reduction axis")
    feature_axis = first_weight.axes[1]
    if first_contract.inputs != (activation, first_weight) or first_contract.reduction_axes != (reduction_axis,):
        raise ValueError("first Contract has incompatible operands or reduction axes")
    if first_contract.output.axes != (row_axis, feature_axis):
        raise ValueError("first Contract must produce row-by-feature preactivation")
    if scalar_map.inputs != (first_contract.output,) or scalar_map.output.axes != first_contract.output.axes:
        raise ValueError("the intermediate Map must be unary and pointwise")
    if second_weight.axes != (feature_axis, reduction_axis):
        raise ValueError("second Contract weight must map feature back to reduction")
    if second_contract.inputs != (scalar_map.output, second_weight):
        raise ValueError("second Contract must consume the Map result and second weight")
    if second_contract.reduction_axes != (feature_axis,) or output.axes != activation.axes:
        raise ValueError("second Contract must map row-by-feature back to the activation shape")


def _validate_reverse_lowering_plan(
    program: ContractMapBackendProgram,
    plan: ContractMapReverseLoweringPlan,
) -> None:
    row_axis, reduction_axis = program.activation.axes
    feature_axis = program.first_weight.axes[1]
    hidden_adjoint = plan.hidden_adjoint_contract
    second_weight_adjoint = plan.second_weight_adjoint_contract
    pointwise_adjoint = plan.pointwise_adjoint
    input_adjoint = plan.input_adjoint_contract
    first_weight_adjoint = plan.first_weight_adjoint_contract

    if hidden_adjoint.inputs != (program.output_cotangent, program.second_weight):
        raise ValueError("hidden-adjoint Contract operands do not match the differentiated graph")
    _require_adjoint_contract(
        hidden_adjoint,
        output_axes=program.hidden.axes,
        reduction_axes=(reduction_axis,),
        context="hidden-adjoint Contract",
    )
    if second_weight_adjoint.inputs != (program.output_cotangent, program.hidden):
        raise ValueError("second-weight-adjoint Contract operands do not match the differentiated graph")
    if second_weight_adjoint.output != program.second_weight_adjoint:
        raise ValueError("second-weight-adjoint Contract output is not the requested source-input gradient")
    _require_adjoint_contract(
        second_weight_adjoint,
        output_axes=program.second_weight.axes,
        reduction_axes=(row_axis,),
        context="second-weight-adjoint Contract",
    )
    if pointwise_adjoint.inputs != (program.preactivation, hidden_adjoint.output):
        raise ValueError("pointwise adjoint Map must consume preactivation and the hidden-adjoint Contract")
    if pointwise_adjoint.output.axes != program.preactivation.axes:
        raise ValueError("pointwise adjoint Map output axes must match preactivation axes")
    if pointwise_adjoint.output.dtype is not DType.BF16:
        raise ValueError("pointwise adjoint Map output must be BF16")
    if input_adjoint.inputs != (pointwise_adjoint.output, program.first_weight):
        raise ValueError("input-adjoint Contract operands do not match the differentiated graph")
    if input_adjoint.output != program.input_adjoint:
        raise ValueError("input-adjoint Contract output is not the requested source-input gradient")
    _require_adjoint_contract(
        input_adjoint,
        output_axes=program.activation.axes,
        reduction_axes=(feature_axis,),
        context="input-adjoint Contract",
    )
    if first_weight_adjoint.inputs != (pointwise_adjoint.output, program.activation):
        raise ValueError("first-weight-adjoint Contract operands do not match the differentiated graph")
    if first_weight_adjoint.output != program.first_weight_adjoint:
        raise ValueError("first-weight-adjoint Contract output is not the requested source-input gradient")
    _require_adjoint_contract(
        first_weight_adjoint,
        output_axes=program.first_weight.axes,
        reduction_axes=(row_axis,),
        context="first-weight-adjoint Contract",
    )
    if program.differentiated.program.outputs != (
        input_adjoint.output,
        first_weight_adjoint.output,
        second_weight_adjoint.output,
    ):
        raise ValueError("differentiated returns must be the three authoritative adjoint Contract outputs")


def _require_adjoint_contract(
    operation: ContractPrimitive,
    *,
    output_axes: tuple[TensorAxis, ...],
    reduction_axes: tuple[TensorAxis, ...],
    context: str,
) -> None:
    if len(operation.inputs) != 2:
        raise ValueError(f"{context} must have exactly two operands")
    if operation.output.axes != output_axes:
        raise ValueError(f"{context} has incompatible output axes")
    if operation.reduction_axes != reduction_axes:
        raise ValueError(f"{context} has incompatible reduction axes")
    if operation.accumulation_dtype is not DType.FP32:
        raise ValueError(f"{context} must accumulate in FP32")
    if operation.input_index_maps:
        raise ValueError(f"{context} does not support Contract index maps")
    if any(value.dtype is not DType.BF16 for value in (*operation.inputs, operation.output)):
        raise ValueError(f"{context} requires BF16 operand and output boundaries")


def _semantic_fingerprint(source: TensorProgram, policy: ContractMapNumericalPolicy) -> str:
    first_contract, scalar_map, second_contract = source.operations
    assert isinstance(first_contract, ContractPrimitive)
    assert isinstance(scalar_map, MapPrimitive)
    assert isinstance(second_contract, ContractPrimitive)
    expression = _normalized_expression(scalar_map.expression, {scalar_map.inputs[0].name: "v0"})
    record = {
        "algebra": ("contract", "map", "contract"),
        "shape": {
            "rows": source.inputs[0].shape[0],
            "reduction": source.inputs[0].shape[1],
            "features": source.inputs[1].shape[1],
        },
        "dtypes": {
            "inputs": tuple(value.dtype.value for value in source.inputs),
            "outputs": tuple(value.dtype.value for value in source.outputs),
            "accumulators": (first_contract.accumulation_dtype.value, second_contract.accumulation_dtype.value),
        },
        "map": expression,
        "policy": policy.value,
    }
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _normalized_expression(expression: ScalarExpression, names: dict[str, str]) -> dict[str, object]:
    record: dict[str, object] = {"kind": expression.kind.value}
    if expression.input_name is not None:
        try:
            record["input"] = names[expression.input_name]
        except KeyError as error:
            raise ValueError(f"Map expression references unknown input {expression.input_name!r}") from error
    if expression.constant is not None:
        record["constant"] = expression.constant
    if expression.operands:
        record["operands"] = [_normalized_expression(operand, names) for operand in expression.operands]
    return record


def _rename_single_input(expression: ScalarExpression, name: str) -> ScalarExpression:
    input_names = _expression_input_names(expression)
    if len(input_names) != 1:
        raise ValueError("the bounded intermediate Map requires exactly one scalar input")
    return _rename_expression(expression, {next(iter(input_names)): name})


def _rename_expression(expression: ScalarExpression, names: dict[str, str]) -> ScalarExpression:
    if expression.kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        try:
            return scalar_input(names[expression.input_name])
        except KeyError as error:
            raise ValueError(f"scalar expression references unknown input {expression.input_name!r}") from error
    if expression.kind is ScalarExpressionKind.CONSTANT:
        assert expression.constant is not None
        return scalar_constant(expression.constant)
    return ScalarExpression(
        kind=expression.kind,
        operands=tuple(_rename_expression(operand, names) for operand in expression.operands),
    )


def _expression_input_names(expression: ScalarExpression) -> set[str]:
    if expression.kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        return {expression.input_name}
    return {name for operand in expression.operands for name in _expression_input_names(operand)}


def _derived_map_adjoint(program: ContractMapBackendProgram) -> MapPrimitive:
    return contract_map_reverse_lowering_plan(program).pointwise_adjoint


def _bf16_input(value: ProgramValue, array: np.ndarray) -> np.ndarray:
    if np.shape(array) != value.shape:
        raise ValueError(f"input shape {np.shape(array)} does not match {value.shape}")
    return round_float32_to_bfloat16(np.asarray(array, dtype=np.float32))


def _ordered_bf16_contract(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape[1] != rhs.shape[0]:
        raise ValueError("ordered Contract requires compatible rank-two operands")
    output = np.zeros((lhs.shape[0], rhs.shape[1]), dtype=np.float32)
    for row in range(lhs.shape[0]):
        for column in range(rhs.shape[1]):
            accumulator = np.float32(0.0)
            for reduction in range(lhs.shape[1]):
                product = np.float32(lhs[row, reduction] * rhs[reduction, column])
                accumulator = np.float32(accumulator + product)
            output[row, column] = accumulator
    return round_float32_to_bfloat16(output)


def _pointwise_bf16(expression: ScalarExpression, input_name: str, values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float32)
    for index in np.ndindex(values.shape):
        output[index] = evaluate_scalar_expression(expression, {input_name: float(values[index])})
    return round_float32_to_bfloat16(output)
