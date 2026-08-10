# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic two-Contract scalar-Map training programs and reference execution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from tile_lifetime.cast_scalar_program import CastScalarProgram, evaluate_cast_scalar_program
from tile_lifetime.xla_low_rank_gated_product import (
    LowRankGatedProductForwardPlan,
    LowRankGatedProductReversePlan,
    RankTwoContractPlan,
)

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\](?:\{(?P<layout>[0-9,]+)\})?")


class ContractMapChainValue(StrEnum):
    """Logical values available to generated scalar Maps in one chain."""

    INPUT = "input"
    FIRST_CONTRACT_OUTPUT = "first_contract_output"
    SECOND_CONTRACT_OUTPUT = "second_contract_output"
    OUTPUT_COTANGENT = "output_cotangent"
    SECOND_CONTRACT_INPUT_ADJOINT = "second_contract_input_adjoint"
    FIRST_CONTRACT_INPUT_ADJOINT = "first_contract_input_adjoint"


@dataclass(frozen=True)
class RankTwoContractShape:
    """Row-major Contract dimensions and explicit physical number types."""

    rows: int
    reduction: int
    features: int
    input_dtype: str = "bf16"
    accumulation_dtype: str = "f32"
    output_dtype: str = "bf16"
    output_rounding: str = "round_to_nearest_even"

    def __post_init__(self) -> None:
        if min(self.rows, self.reduction, self.features) <= 0:
            raise ValueError("Contract dimensions must be positive")
        if (self.input_dtype, self.accumulation_dtype, self.output_dtype) != ("bf16", "f32", "bf16"):
            raise ValueError("bounded Contract/Map chains require BF16/FP32/BF16 Contracts")
        if self.output_rounding != "round_to_nearest_even":
            raise ValueError("bounded Contract/Map chains require BF16 round-to-nearest-even outputs")


@dataclass(frozen=True)
class ContractMapChainBufferLayout:
    """One CUDA buffer's rank-two shape and XLA minor-to-major layout."""

    name: str
    shape: tuple[int, int]
    minor_to_major: tuple[int, int]
    dtype: str = "bf16"

    def __post_init__(self) -> None:
        if min(self.shape) <= 0:
            raise ValueError(f"Contract/Map buffer {self.name!r} must have positive dimensions")
        if sorted(self.minor_to_major) != [0, 1]:
            raise ValueError(f"Contract/Map buffer {self.name!r} layout must be a rank-two permutation")
        if self.dtype != "bf16":
            raise ValueError(f"Contract/Map buffer {self.name!r} must preserve the BF16 boundary")

    @property
    def hlo_shape(self) -> str:
        """Render direct HLO syntax, whose layout order is minor-to-major."""
        dimensions = ",".join(str(value) for value in self.shape)
        layout = ",".join(str(value) for value in self.minor_to_major)
        return f"{self.dtype}[{dimensions}]{{{layout}}}"


@dataclass(frozen=True)
class ContractMapChainPhysicalAbi:
    """CUDA indexing contract shared by direct HLO and JAX FFI call sites."""

    forward_inputs: tuple[ContractMapChainBufferLayout, ...]
    forward_outputs: tuple[ContractMapChainBufferLayout, ...]
    reverse_inputs: tuple[ContractMapChainBufferLayout, ...]
    reverse_outputs: tuple[ContractMapChainBufferLayout, ...]


@dataclass(frozen=True)
class BoundCastScalarMap:
    """A source-ordered scalar AST with explicit generic chain-value bindings."""

    program: CastScalarProgram
    inputs: tuple[ContractMapChainValue, ...]

    def __post_init__(self) -> None:
        scalar_inputs = self.program.inputs
        if len(scalar_inputs) != len(self.inputs):
            raise ValueError("scalar Map bindings must cover every distinct AST input")
        names = tuple(value.input_name for value in scalar_inputs)
        if len(set(names)) != len(names):
            raise ValueError("bounded scalar Maps require one pointwise index relation per input name")
        if any(
            value.input_index is None or value.input_index.row_offset != 0 or value.input_index.feature_offset != 0
            for value in scalar_inputs
        ):
            raise ValueError("bounded Contract/Map chains currently require pointwise scalar Maps")


@dataclass(frozen=True)
class TwoContractMapTrainingProgram:
    """Two Contracts and JAX-owned forward/reverse scalar Map programs."""

    first_contract: RankTwoContractShape
    hidden_map: BoundCastScalarMap
    second_contract: RankTwoContractShape
    output_map: BoundCastScalarMap
    second_output_vjp_map: BoundCastScalarMap
    hidden_vjp_map: BoundCastScalarMap
    input_vjp_map: BoundCastScalarMap
    first_weight_adjoint_minor_to_major: tuple[int, int] = (1, 0)
    second_weight_adjoint_minor_to_major: tuple[int, int] = (1, 0)
    numerical_policy: str = "source_ordered"

    def __post_init__(self) -> None:
        first = self.first_contract
        second = self.second_contract
        if second.rows != first.rows or second.reduction != first.features or second.features != first.reduction:
            raise ValueError("two-Contract chain must map [row,input] through [row,rank] back to [row,input]")
        expected = {
            "hidden": (ContractMapChainValue.FIRST_CONTRACT_OUTPUT,),
            "output": (ContractMapChainValue.INPUT, ContractMapChainValue.SECOND_CONTRACT_OUTPUT),
            "second output VJP": (
                ContractMapChainValue.INPUT,
                ContractMapChainValue.SECOND_CONTRACT_OUTPUT,
                ContractMapChainValue.OUTPUT_COTANGENT,
            ),
            "hidden VJP": (
                ContractMapChainValue.SECOND_CONTRACT_INPUT_ADJOINT,
                ContractMapChainValue.FIRST_CONTRACT_OUTPUT,
            ),
            "input VJP": (
                ContractMapChainValue.FIRST_CONTRACT_INPUT_ADJOINT,
                ContractMapChainValue.SECOND_CONTRACT_OUTPUT,
                ContractMapChainValue.OUTPUT_COTANGENT,
            ),
        }
        actual = {
            "hidden": self.hidden_map.inputs,
            "output": self.output_map.inputs,
            "second output VJP": self.second_output_vjp_map.inputs,
            "hidden VJP": self.hidden_vjp_map.inputs,
            "input VJP": self.input_vjp_map.inputs,
        }
        for name, bindings in actual.items():
            if bindings != expected[name]:
                raise ValueError(f"{name} scalar Map has incompatible chain-value bindings")
        for name, layout in (
            ("first weight adjoint", self.first_weight_adjoint_minor_to_major),
            ("second weight adjoint", self.second_weight_adjoint_minor_to_major),
        ):
            if sorted(layout) != [0, 1]:
                raise ValueError(f"{name} layout must be a rank-two minor-to-major permutation")
        if self.numerical_policy != "source_ordered":
            raise ValueError("bounded Contract/Map chains currently preserve only source-ordered Maps")


@dataclass(frozen=True)
class TwoContractMapForwardResult:
    """Forward result plus the BF16 boundaries retained for JAX's VJP."""

    output: np.ndarray
    first_contract_output: np.ndarray
    hidden: np.ndarray
    second_contract_output: np.ndarray


@dataclass(frozen=True)
class TwoContractMapReverseResult:
    """BF16 input and weight adjoints before the surrounding FP32 conversion."""

    input_adjoint: np.ndarray
    first_weight_adjoint: np.ndarray
    second_weight_adjoint: np.ndarray


def contract_map_chain_physical_abi(program: TwoContractMapTrainingProgram) -> ContractMapChainPhysicalAbi:
    """Return the exact rank-two buffers indexed by the generated CUDA body."""
    first = program.first_contract
    row_major = (1, 0)
    activation = (first.rows, first.reduction)
    first_weight = (first.reduction, first.features)
    second_weight = (first.features, first.reduction)
    rank_value = (first.rows, first.features)

    def buffer(
        name: str,
        shape: tuple[int, int],
        layout: tuple[int, int] = row_major,
    ) -> ContractMapChainBufferLayout:
        return ContractMapChainBufferLayout(name=name, shape=shape, minor_to_major=layout)

    forward_inputs = (
        buffer("activation", activation),
        buffer("first_weight", first_weight),
        buffer("second_weight", second_weight),
    )
    forward_outputs = (
        buffer("output", activation),
        buffer("first_contract_output", rank_value),
        buffer("hidden", rank_value),
        buffer("second_contract_output", activation),
    )
    reverse_inputs = (
        *forward_inputs,
        forward_outputs[1],
        forward_outputs[2],
        forward_outputs[3],
        buffer("output_cotangent", activation),
    )
    reverse_outputs = (
        buffer("input_adjoint", activation),
        buffer("first_weight_adjoint", first_weight, program.first_weight_adjoint_minor_to_major),
        buffer("second_weight_adjoint", second_weight, program.second_weight_adjoint_minor_to_major),
    )
    return ContractMapChainPhysicalAbi(
        forward_inputs=forward_inputs,
        forward_outputs=forward_outputs,
        reverse_inputs=reverse_inputs,
        reverse_outputs=reverse_outputs,
    )


def form_two_contract_map_training_program(
    forward: LowRankGatedProductForwardPlan,
    reverse: LowRankGatedProductReversePlan,
) -> TwoContractMapTrainingProgram:
    """Adapt structurally recovered Contract/Map records to the generic family."""
    if reverse.primal != forward:
        raise ValueError("reverse plan must be JAX's VJP of the selected forward realization")
    first = _contract_shape(forward.down_contract)
    second = _contract_shape(forward.up_contract)
    if _contract_shape(reverse.up_input_adjoint) != RankTwoContractShape(
        first.rows,
        second.features,
        second.reduction,
    ):
        raise ValueError("second-Contract input adjoint has incompatible dimensions")
    if _contract_shape(reverse.down_input_adjoint) != RankTwoContractShape(
        first.rows,
        first.features,
        first.reduction,
    ):
        raise ValueError("first-Contract input adjoint has incompatible dimensions")
    _validate_weight_adjoint(
        reverse.down_weight_adjoint, rows=first.reduction, reduction=first.rows, features=first.features
    )
    _validate_weight_adjoint(
        reverse.up_weight_adjoint,
        rows=second.reduction,
        reduction=second.rows,
        features=second.features,
    )
    return TwoContractMapTrainingProgram(
        first_contract=first,
        hidden_map=BoundCastScalarMap(
            forward.hidden_map,
            (ContractMapChainValue.FIRST_CONTRACT_OUTPUT,),
        ),
        second_contract=second,
        output_map=BoundCastScalarMap(
            forward.output_map,
            (ContractMapChainValue.INPUT, ContractMapChainValue.SECOND_CONTRACT_OUTPUT),
        ),
        second_output_vjp_map=BoundCastScalarMap(
            reverse.up_input_map,
            (
                ContractMapChainValue.INPUT,
                ContractMapChainValue.SECOND_CONTRACT_OUTPUT,
                ContractMapChainValue.OUTPUT_COTANGENT,
            ),
        ),
        hidden_vjp_map=BoundCastScalarMap(
            reverse.hidden_vjp_map,
            (
                ContractMapChainValue.SECOND_CONTRACT_INPUT_ADJOINT,
                ContractMapChainValue.FIRST_CONTRACT_OUTPUT,
            ),
        ),
        input_vjp_map=BoundCastScalarMap(
            reverse.residual_vjp_map,
            (
                ContractMapChainValue.FIRST_CONTRACT_INPUT_ADJOINT,
                ContractMapChainValue.SECOND_CONTRACT_OUTPUT,
                ContractMapChainValue.OUTPUT_COTANGENT,
            ),
        ),
        first_weight_adjoint_minor_to_major=_layout(reverse.down_weight_adjoint.output.shape),
        second_weight_adjoint_minor_to_major=_layout(reverse.up_weight_adjoint.output.shape),
    )


def execute_two_contract_map_forward(
    program: TwoContractMapTrainingProgram,
    activation: np.ndarray,
    first_weight: np.ndarray,
    second_weight: np.ndarray,
) -> TwoContractMapForwardResult:
    """Execute the exact bounded forward numerical contract on CPU."""
    first = program.first_contract
    _require_shape("activation", activation, (first.rows, first.reduction))
    _require_shape("first_weight", first_weight, (first.reduction, first.features))
    _require_shape("second_weight", second_weight, (first.features, first.reduction))
    activation = round_float32_to_bfloat16(activation)
    first_weight = round_float32_to_bfloat16(first_weight)
    second_weight = round_float32_to_bfloat16(second_weight)
    first_output = ordered_bf16_contract(activation, first_weight)
    hidden = _execute_map(
        program.hidden_map,
        {ContractMapChainValue.FIRST_CONTRACT_OUTPUT: first_output},
        shape=first_output.shape,
    )
    second_output = ordered_bf16_contract(hidden, second_weight)
    output = _execute_map(
        program.output_map,
        {
            ContractMapChainValue.INPUT: activation,
            ContractMapChainValue.SECOND_CONTRACT_OUTPUT: second_output,
        },
        shape=activation.shape,
    )
    return TwoContractMapForwardResult(output, first_output, hidden, second_output)


def execute_two_contract_map_reverse(
    program: TwoContractMapTrainingProgram,
    activation: np.ndarray,
    first_weight: np.ndarray,
    second_weight: np.ndarray,
    saved: TwoContractMapForwardResult,
    output_cotangent: np.ndarray,
) -> TwoContractMapReverseResult:
    """Execute JAX-owned reverse Maps and generic Contract adjoints on CPU."""
    first = program.first_contract
    _require_shape("output_cotangent", output_cotangent, (first.rows, first.reduction))
    activation = round_float32_to_bfloat16(activation)
    first_weight = round_float32_to_bfloat16(first_weight)
    second_weight = round_float32_to_bfloat16(second_weight)
    cotangent = round_float32_to_bfloat16(output_cotangent)
    second_output_adjoint = _execute_map(
        program.second_output_vjp_map,
        {
            ContractMapChainValue.INPUT: activation,
            ContractMapChainValue.SECOND_CONTRACT_OUTPUT: saved.second_contract_output,
            ContractMapChainValue.OUTPUT_COTANGENT: cotangent,
        },
        shape=cotangent.shape,
    )
    second_input_adjoint = ordered_bf16_contract(second_output_adjoint, second_weight.T)
    first_output_adjoint = _execute_map(
        program.hidden_vjp_map,
        {
            ContractMapChainValue.SECOND_CONTRACT_INPUT_ADJOINT: second_input_adjoint,
            ContractMapChainValue.FIRST_CONTRACT_OUTPUT: saved.first_contract_output,
        },
        shape=second_input_adjoint.shape,
    )
    first_input_adjoint = ordered_bf16_contract(first_output_adjoint, first_weight.T)
    input_adjoint = _execute_map(
        program.input_vjp_map,
        {
            ContractMapChainValue.FIRST_CONTRACT_INPUT_ADJOINT: first_input_adjoint,
            ContractMapChainValue.SECOND_CONTRACT_OUTPUT: saved.second_contract_output,
            ContractMapChainValue.OUTPUT_COTANGENT: cotangent,
        },
        shape=first_input_adjoint.shape,
    )
    first_weight_adjoint = ordered_bf16_contract(activation.T, first_output_adjoint)
    second_weight_adjoint = ordered_bf16_contract(saved.hidden.T, second_output_adjoint)
    return TwoContractMapReverseResult(input_adjoint, first_weight_adjoint, second_weight_adjoint)


def ordered_bf16_contract(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Contract in a fixed FP32 left-to-right order, then round once to BF16."""
    lhs = np.asarray(lhs, dtype=np.float32)
    rhs = np.asarray(rhs, dtype=np.float32)
    if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape[1] != rhs.shape[0]:
        raise ValueError("ordered Contract requires compatible rank-two operands")
    output = np.zeros((lhs.shape[0], rhs.shape[1]), dtype=np.float32)
    for row in range(lhs.shape[0]):
        for feature in range(rhs.shape[1]):
            accumulator = np.float32(0.0)
            for reduction in range(lhs.shape[1]):
                product = np.float32(lhs[row, reduction] * rhs[reduction, feature])
                accumulator = np.float32(accumulator + product)
            output[row, feature] = accumulator
    return round_float32_to_bfloat16(output)


def round_float32_to_bfloat16(value: np.ndarray) -> np.ndarray:
    """Round FP32 to BF16 RNE while retaining a NumPy FP32 container."""
    contiguous = np.ascontiguousarray(value, dtype=np.float32)
    bits = contiguous.view(np.uint32)
    is_nan = (bits & np.uint32(0x7F800000) == np.uint32(0x7F800000)) & (bits & np.uint32(0x007FFFFF) != 0)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = (bits + rounding_bias) & np.uint32(0xFFFF0000)
    rounded = np.where(is_nan, (bits & np.uint32(0xFFFF0000)) | np.uint32(0x00400000), rounded)
    return rounded.view(np.float32)


def _execute_map(
    scalar_map: BoundCastScalarMap,
    values: dict[ContractMapChainValue, np.ndarray],
    *,
    shape: tuple[int, int],
) -> np.ndarray:
    if set(values) != set(scalar_map.inputs):
        raise ValueError("scalar Map values do not match its declared chain bindings")
    for role, value in values.items():
        _require_shape(role.value, value, shape)
    output = np.empty(shape, dtype=np.float32)
    input_leaves = scalar_map.program.inputs
    for row in range(shape[0]):
        for feature in range(shape[1]):
            bindings = {
                leaf.input_name: float(values[role][row, feature])
                for leaf, role in zip(input_leaves, scalar_map.inputs, strict=True)
                if leaf.input_name is not None
            }
            output[row, feature] = evaluate_cast_scalar_program(scalar_map.program, bindings)
    return output


def _contract_shape(contract: RankTwoContractPlan) -> RankTwoContractShape:
    lhs_dtype, lhs = _shape(contract.lhs.shape)
    rhs_dtype, rhs = _shape(contract.rhs.shape)
    output_dtype, output = _shape(contract.output.shape)
    if (lhs_dtype, rhs_dtype, output_dtype) != ("bf16", "bf16", "bf16"):
        raise ValueError("recovered Contracts must preserve BF16 operand and output boundaries")
    if len(lhs) != 2 or len(rhs) != 2 or len(output) != 2:
        raise ValueError("bounded Contract/Map generation requires rank-two physical Contracts")
    if contract.lhs_contracting_dimension == 1 and contract.rhs_contracting_dimension == 0:
        rows, reduction = lhs
        rhs_reduction, features = rhs
    elif contract.lhs_contracting_dimension == 1 and contract.rhs_contracting_dimension == 1:
        rows, reduction = lhs
        features, rhs_reduction = rhs
    else:
        raise ValueError("bounded Contract/Map generation does not support these contracting dimensions")
    if rhs_reduction != reduction or output != (rows, features):
        raise ValueError("recovered Contract shapes and contracting dimensions disagree")
    return RankTwoContractShape(rows, reduction, features)


def _validate_weight_adjoint(contract: RankTwoContractPlan, *, rows: int, reduction: int, features: int) -> None:
    lhs_dtype, lhs = _shape(contract.lhs.shape)
    rhs_dtype, rhs = _shape(contract.rhs.shape)
    output_dtype, output = _shape(contract.output.shape)
    if (lhs_dtype, rhs_dtype, output_dtype) != ("bf16", "bf16", "bf16"):
        raise ValueError("weight-adjoint Contracts must preserve the BF16 boundary before FP32 conversion")
    if contract.lhs_contracting_dimension != 0 or contract.rhs_contracting_dimension != 1:
        raise ValueError("weight-adjoint Contract must reduce the logical row axis")
    if lhs != (reduction, rows) or rhs != (features, reduction) or output != (rows, features):
        raise ValueError("weight-adjoint Contract shapes disagree with the generic chain")


def _shape(shape: str) -> tuple[str, tuple[int, ...]]:
    match = _ARRAY_SHAPE.match(shape)
    if match is None:
        raise ValueError(f"expected a physical array shape, found {shape!r}")
    dims = tuple(int(value) for value in match.group("dims").split(",") if value)
    return match.group("dtype"), dims


def _layout(shape: str) -> tuple[int, int]:
    match = _ARRAY_SHAPE.match(shape)
    if match is None:
        raise ValueError(f"expected a physical array shape, found {shape!r}")
    layout = match.group("layout")
    if layout is None:
        raise ValueError(f"expected an explicit physical layout, found {shape!r}")
    minor_to_major = tuple(int(value) for value in layout.split(","))
    if len(minor_to_major) != 2 or sorted(minor_to_major) != [0, 1]:
        raise ValueError(f"expected a rank-two physical layout, found {shape!r}")
    return minor_to_major


def _require_shape(name: str, value: np.ndarray, shape: tuple[int, int]) -> None:
    if np.shape(value) != shape:
        raise ValueError(f"{name} must have shape {shape}, found {np.shape(value)}")
