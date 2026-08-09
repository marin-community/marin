# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small backend-independent tensor semantics used by synthesis prototypes."""

import json
from dataclasses import dataclass
from enum import StrEnum

from tile_lifetime.ir import DType


@dataclass(frozen=True)
class TensorAxis:
    """One logical dimension with stable identity."""

    id: int
    extent: int
    label: str | None = None


@dataclass(frozen=True)
class ProgramValue:
    """A named tensor value indexed by logical axes."""

    name: str
    axes: tuple[TensorAxis, ...]
    dtype: DType

    def __post_init__(self) -> None:
        if len(set(self.axes)) != len(self.axes):
            raise ValueError(f"value {self.name!r} repeats a logical axis")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(axis.extent for axis in self.axes)


class ScalarExpressionKind(StrEnum):
    """Scalar operations allowed in a pointwise map."""

    INPUT = "input"
    CONSTANT = "constant"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    EXP = "exp"
    LOG = "log"
    RSQRT = "rsqrt"
    TANH = "tanh"
    LESS_EQUAL = "less_equal"
    SELECT = "select"


@dataclass(frozen=True)
class ScalarExpression:
    """A scalar expression evaluated pointwise after logical-axis broadcasting."""

    kind: ScalarExpressionKind
    operands: tuple["ScalarExpression", ...] = ()
    input_name: str | None = None
    constant: float | bool | None = None

    def __post_init__(self) -> None:
        if self.kind is ScalarExpressionKind.INPUT:
            if self.input_name is None or self.operands or self.constant is not None:
                raise ValueError("input scalar expressions require only an input name")
            return
        if self.kind is ScalarExpressionKind.CONSTANT:
            if self.constant is None or self.operands or self.input_name is not None:
                raise ValueError("constant scalar expressions require only a literal")
            return
        expected_arity = {
            ScalarExpressionKind.ADD: 2,
            ScalarExpressionKind.SUBTRACT: 2,
            ScalarExpressionKind.MULTIPLY: 2,
            ScalarExpressionKind.DIVIDE: 2,
            ScalarExpressionKind.EXP: 1,
            ScalarExpressionKind.LOG: 1,
            ScalarExpressionKind.RSQRT: 1,
            ScalarExpressionKind.TANH: 1,
            ScalarExpressionKind.LESS_EQUAL: 2,
            ScalarExpressionKind.SELECT: 3,
        }[self.kind]
        if len(self.operands) != expected_arity or self.input_name is not None or self.constant is not None:
            raise ValueError(f"{self.kind.value} scalar expressions require {expected_arity} operands")


def serialize_scalar_expression(expression: ScalarExpression) -> str:
    """Serialize one scalar AST for transport through a physical-plan record."""

    def encode(node: ScalarExpression) -> dict[str, object]:
        encoded: dict[str, object] = {"kind": node.kind.value}
        if node.input_name is not None:
            encoded["input"] = node.input_name
        if node.constant is not None:
            encoded["constant"] = node.constant
        if node.operands:
            encoded["operands"] = [encode(operand) for operand in node.operands]
        return encoded

    return json.dumps(encode(expression), sort_keys=True, separators=(",", ":"))


def deserialize_scalar_expression(serialized: str) -> ScalarExpression:
    """Restore a scalar AST emitted by :func:`serialize_scalar_expression`."""

    def decode(encoded: object) -> ScalarExpression:
        if not isinstance(encoded, dict) or not isinstance(encoded.get("kind"), str):
            raise ValueError("serialized scalar expression must be an object with a string kind")
        kind = ScalarExpressionKind(encoded["kind"])
        operands_value = encoded.get("operands", ())
        if not isinstance(operands_value, (list, tuple)):
            raise ValueError("serialized scalar-expression operands must be a sequence")
        operands = tuple(decode(operand) for operand in operands_value)
        input_name = encoded.get("input")
        constant = encoded.get("constant")
        if input_name is not None and not isinstance(input_name, str):
            raise ValueError("serialized scalar-expression inputs must be strings")
        if constant is not None and not isinstance(constant, (bool, int, float)):
            raise ValueError("serialized scalar-expression constants must be numeric or Boolean")
        return ScalarExpression(
            kind=kind,
            operands=operands,
            input_name=input_name,
            constant=constant,
        )

    return decode(json.loads(serialized))


def scalar_input(name: str) -> ScalarExpression:
    """Reference one pointwise-map input."""
    return ScalarExpression(kind=ScalarExpressionKind.INPUT, input_name=name)


def scalar_constant(value: float | bool) -> ScalarExpression:
    """Construct one scalar literal."""
    return ScalarExpression(kind=ScalarExpressionKind.CONSTANT, constant=value)


def scalar_unary(kind: ScalarExpressionKind, operand: ScalarExpression) -> ScalarExpression:
    """Construct a unary scalar expression."""
    if kind not in (
        ScalarExpressionKind.EXP,
        ScalarExpressionKind.LOG,
        ScalarExpressionKind.RSQRT,
        ScalarExpressionKind.TANH,
    ):
        raise ValueError(f"{kind.value} is not unary")
    return ScalarExpression(kind=kind, operands=(operand,))


def scalar_binary(
    kind: ScalarExpressionKind,
    left: ScalarExpression,
    right: ScalarExpression,
) -> ScalarExpression:
    """Construct a binary scalar expression."""
    if kind not in {
        ScalarExpressionKind.ADD,
        ScalarExpressionKind.SUBTRACT,
        ScalarExpressionKind.MULTIPLY,
        ScalarExpressionKind.DIVIDE,
        ScalarExpressionKind.LESS_EQUAL,
    }:
        raise ValueError(f"{kind.value} is not binary")
    return ScalarExpression(kind=kind, operands=(left, right))


def scalar_select(
    predicate: ScalarExpression,
    when_true: ScalarExpression,
    when_false: ScalarExpression,
) -> ScalarExpression:
    """Construct a pointwise selection."""
    return ScalarExpression(
        kind=ScalarExpressionKind.SELECT,
        operands=(predicate, when_true, when_false),
    )


@dataclass(frozen=True)
class ContractPrimitive:
    """A multilinear contraction with explicit key and reduction axes."""

    name: str
    inputs: tuple[ProgramValue, ...]
    output: ProgramValue
    reduction_axes: tuple[TensorAxis, ...]
    accumulation_dtype: DType
    input_index_maps: tuple[tuple["AxisIndexMap", ...], ...] = ()

    def __post_init__(self) -> None:
        if len(self.inputs) < 2:
            raise ValueError("a contraction requires at least two inputs")
        input_axes = {axis for value in self.inputs for axis in value.axes}
        if not set(self.reduction_axes) <= input_axes:
            raise ValueError("contraction reduction axes must occur in its inputs")
        if set(self.output.axes) & set(self.reduction_axes):
            raise ValueError("contraction output cannot retain a reduction axis")
        if self.input_index_maps and len(self.input_index_maps) != len(self.inputs):
            raise ValueError("contraction index-map groups must match its inputs")
        for value, mappings in zip(self.inputs, self.input_index_maps, strict=False):
            mapped_operand_axes: set[TensorAxis] = set()
            for mapping in mappings:
                if mapping.domain_axis not in self.output.axes:
                    raise ValueError("contraction index-map domain axes must occur in the output")
                if mapping.operand_axis not in value.axes:
                    raise ValueError("contraction index-map operand axes must occur in the mapped input")
                if mapping.operand_axis in mapped_operand_axes:
                    raise ValueError("a contraction input axis can be indexed only once")
                mapped_operand_axes.add(mapping.operand_axis)

    def index_maps_for_input(self, input_index: int) -> tuple["AxisIndexMap", ...]:
        """Return the logical indexing relations applied to one operand."""
        if not self.input_index_maps:
            return ()
        return self.input_index_maps[input_index]


@dataclass(frozen=True)
class AxisIndexMap:
    """Map one output-domain index onto an operand axis.

    The initial affine integer form is deliberately small: it covers grouped
    head sharing and block-oriented relations without naming either workload.
    For a domain index ``i``, the operand index is
    ``((i // divisor) + offset) % modulus`` when a modulus is present.
    """

    domain_axis: TensorAxis
    operand_axis: TensorAxis
    divisor: int = 1
    offset: int = 0
    modulus: int | None = None

    def __post_init__(self) -> None:
        if self.divisor <= 0:
            raise ValueError("axis index-map divisors must be positive")
        if self.modulus is not None and self.modulus <= 0:
            raise ValueError("axis index-map moduli must be positive")
        mapped_min = self.offset
        mapped_max = (self.domain_axis.extent - 1) // self.divisor + self.offset
        if self.modulus is None and (mapped_min < 0 or mapped_max >= self.operand_axis.extent):
            raise ValueError("axis index map can address outside the operand axis")
        if self.modulus is not None and self.modulus > self.operand_axis.extent:
            raise ValueError("axis index-map modulus exceeds the operand axis")

    def indices(self) -> tuple[int, ...]:
        """Materialize the finite index relation for reference execution."""
        values = tuple(index // self.divisor + self.offset for index in range(self.domain_axis.extent))
        if self.modulus is None:
            return values
        return tuple(value % self.modulus for value in values)


@dataclass(frozen=True)
class MapPrimitive:
    """A pointwise map over explicitly broadcast tensor inputs."""

    name: str
    inputs: tuple[ProgramValue, ...]
    output: ProgramValue
    expression: ScalarExpression

    def __post_init__(self) -> None:
        input_names = {value.name for value in self.inputs}
        referenced_names = scalar_expression_inputs(self.expression)
        if referenced_names != input_names:
            raise ValueError(
                f"map {self.name!r} expression inputs {sorted(referenced_names)} do not match "
                f"declared inputs {sorted(input_names)}"
            )
        output_axes = set(self.output.axes)
        if any(not set(value.axes) <= output_axes for value in self.inputs):
            raise ValueError(f"map {self.name!r} inputs do not broadcast to its output axes")


class FoldReducer(StrEnum):
    """Associative reduction operators supported by the semantic IR."""

    MAXIMUM = "maximum"
    SUM = "sum"


@dataclass(frozen=True)
class FoldPrimitive:
    """An unordered reduction over one or more logical axes."""

    name: str
    input: ProgramValue
    output: ProgramValue
    reduction_axes: tuple[TensorAxis, ...]
    reducer: FoldReducer
    accumulation_dtype: DType

    def __post_init__(self) -> None:
        if not self.reduction_axes or not set(self.reduction_axes) <= set(self.input.axes):
            raise ValueError("fold axes must be a nonempty subset of input axes")
        expected_axes = tuple(axis for axis in self.input.axes if axis not in self.reduction_axes)
        if self.output.axes != expected_axes:
            raise ValueError("fold output axes must equal the unreduced input axes in order")


TensorPrimitive = ContractPrimitive | MapPrimitive | FoldPrimitive


@dataclass(frozen=True)
class TensorProgram:
    """A closed acyclic tensor program composed only of Contract, Map, and Fold."""

    inputs: tuple[ProgramValue, ...]
    operations: tuple[TensorPrimitive, ...]
    outputs: tuple[ProgramValue, ...]

    def __post_init__(self) -> None:
        available = {value.name: value for value in self.inputs}
        if len(available) != len(self.inputs):
            raise ValueError("tensor program input names must be unique")
        for operation in self.operations:
            operation_inputs = primitive_inputs(operation)
            missing = tuple(value.name for value in operation_inputs if value.name not in available)
            if missing:
                raise ValueError(f"operation {operation.name!r} reads unavailable values {missing}")
            if operation.output.name in available:
                raise ValueError(f"tensor value {operation.output.name!r} has multiple definitions")
            available[operation.output.name] = operation.output
        missing_outputs = tuple(value.name for value in self.outputs if value.name not in available)
        if missing_outputs:
            raise ValueError(f"tensor program outputs are unavailable: {missing_outputs}")


def primitive_inputs(operation: TensorPrimitive) -> tuple[ProgramValue, ...]:
    """Return the tensor operands consumed by one primitive."""
    if isinstance(operation, FoldPrimitive):
        return (operation.input,)
    return operation.inputs


def scalar_expression_inputs(expression: ScalarExpression) -> set[str]:
    """Return the names referenced by one scalar expression."""
    if expression.kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        return {expression.input_name}
    names: set[str] = set()
    for operand in expression.operands:
        names.update(scalar_expression_inputs(operand))
    return names
