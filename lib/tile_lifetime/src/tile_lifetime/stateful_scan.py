# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic ordered-state semantics and affine chunk summaries."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from shuttle.ir import DType
from tile_lifetime.plan import ScanNumericalContract


class ScanValueRole(StrEnum):
    """Semantic role of a value in an ordered scan body."""

    INPUT = "input"
    STATE = "state"
    TEMPORARY = "temporary"
    OUTPUT = "output"
    SUMMARY = "summary"


class ScanPrimitiveKind(StrEnum):
    """Small operation families allowed inside a scan body."""

    MAP = "map"
    CONTRACT = "contract"
    FOLD = "fold"


class TensorExpressionKind(StrEnum):
    """Tensor-algebra operations used by generic scan recovery."""

    INPUT = "input"
    UNARY = "unary"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    CONTRACT = "contract"


@dataclass(frozen=True)
class LogicalAxis:
    """Stable identity and extent for one logical program dimension."""

    id: int
    extent: int
    label: str | None = None


@dataclass(frozen=True)
class TensorExpression:
    """A small indexed tensor expression independent of model names."""

    kind: TensorExpressionKind
    axes: tuple[LogicalAxis, ...]
    inputs: tuple["TensorExpression", ...] = ()
    name: str | None = None
    operation: str | None = None
    reduction_axes: tuple[LogicalAxis, ...] = ()

    def __post_init__(self) -> None:
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("tensor expression axes must be unique")
        if self.kind is TensorExpressionKind.INPUT:
            if self.name is None or self.inputs:
                raise ValueError("input expressions require a name and no inputs")
            return
        if self.name is not None:
            raise ValueError("only input expressions may carry a name")
        if self.kind is TensorExpressionKind.UNARY and (len(self.inputs) != 1 or self.operation is None):
            raise ValueError("unary expressions require one input and an operation")
        if self.kind in (TensorExpressionKind.ADD, TensorExpressionKind.SUBTRACT, TensorExpressionKind.MULTIPLY):
            if len(self.inputs) != 2:
                raise ValueError(f"{self.kind.value} expressions require two inputs")
        if self.kind is TensorExpressionKind.CONTRACT and not self.inputs:
            raise ValueError("contraction expressions require at least one input")


def input_expression(name: str, axes: tuple[LogicalAxis, ...]) -> TensorExpression:
    """Construct one named tensor input."""
    return TensorExpression(kind=TensorExpressionKind.INPUT, axes=axes, name=name)


def unary_expression(operation: str, value: TensorExpression) -> TensorExpression:
    """Construct a tile-local unary expression."""
    return TensorExpression(
        kind=TensorExpressionKind.UNARY,
        axes=value.axes,
        inputs=(value,),
        operation=operation,
    )


def binary_expression(
    kind: TensorExpressionKind,
    left: TensorExpression,
    right: TensorExpression,
    axes: tuple[LogicalAxis, ...],
) -> TensorExpression:
    """Construct a broadcast-aware binary expression."""
    if kind not in (TensorExpressionKind.ADD, TensorExpressionKind.SUBTRACT, TensorExpressionKind.MULTIPLY):
        raise ValueError(f"unsupported binary expression kind {kind.value}")
    return TensorExpression(kind=kind, axes=axes, inputs=(left, right))


def contract_expression(
    *inputs: TensorExpression,
    axes: tuple[LogicalAxis, ...],
    reduction_axes: tuple[LogicalAxis, ...] = (),
) -> TensorExpression:
    """Construct an indexed contraction or outer product."""
    return TensorExpression(
        kind=TensorExpressionKind.CONTRACT,
        axes=axes,
        inputs=inputs,
        reduction_axes=reduction_axes,
    )


@dataclass(frozen=True)
class ScanValue:
    """A typed logical value used by a stateful scan."""

    name: str
    axes: tuple[LogicalAxis, ...]
    dtype: DType
    role: ScanValueRole


@dataclass(frozen=True)
class ScanPrimitive:
    """One map, contraction, or reduction in a scan or chunk program."""

    name: str
    kind: ScanPrimitiveKind
    inputs: tuple[str, ...]
    output: str
    equation: str
    reduction_axes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChunkAlgebra:
    """How a chunk summarizes, applies, and emits one ordered state transform."""

    summary_values: tuple[str, ...]
    summarize: tuple[ScanPrimitive, ...]
    apply: tuple[ScanPrimitive, ...]
    emit: tuple[ScanPrimitive, ...]
    compose: tuple[ScanPrimitive, ...] = ()
    bounded_representation_closed_under_compose: bool = False


@dataclass(frozen=True)
class StatefulScan:
    """A dependency-free semantic description of an ordered state program."""

    name: str
    ordered_axis: LogicalAxis
    values: tuple[ScanValue, ...]
    state_input: str
    state_output: str
    scan_inputs: tuple[str, ...]
    scan_outputs: tuple[str, ...]
    update: tuple[ScanPrimitive, ...]
    read: tuple[ScanPrimitive, ...]
    numerical_contract: ScanNumericalContract
    chunk_algebra: ChunkAlgebra | None = None

    def __post_init__(self) -> None:
        _validate_stateful_scan(self)

    def value(self, name: str) -> ScanValue:
        """Return one named value from the scan body."""
        matches = tuple(value for value in self.values if value.name == name)
        if len(matches) != 1:
            raise KeyError(f"expected one scan value named {name!r}, found {len(matches)}")
        return matches[0]


@dataclass(frozen=True)
class AffineStateTransform:
    """A batched affine state transform ``S_out = P @ S_in + H``."""

    transition: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True)
class AffineChunkSummary:
    """Final and prefix affine transforms for one ordered chunk."""

    final: AffineStateTransform
    prefixes: tuple[AffineStateTransform, ...]


def compose_affine_transforms(
    earlier: AffineStateTransform,
    later: AffineStateTransform,
) -> AffineStateTransform:
    """Return ``later(earlier(S))`` for batched matrix-valued state."""
    _validate_affine_transform(earlier)
    _validate_affine_transform(later)
    if earlier.transition.shape != later.transition.shape or earlier.bias.shape != later.bias.shape:
        raise ValueError("affine state transforms must have identical shapes")

    transition = np.matmul(later.transition, earlier.transition)
    bias = np.matmul(later.transition, earlier.bias) + later.bias
    return AffineStateTransform(transition=transition, bias=bias)


def apply_affine_transform(transform: AffineStateTransform, state: np.ndarray) -> np.ndarray:
    """Apply a batched affine transform to a matrix-valued state."""
    _validate_affine_transform(transform)
    expected = transform.bias.shape
    if state.shape != expected:
        raise ValueError(f"state shape {state.shape} does not match affine bias shape {expected}")
    return np.matmul(transform.transition, state) + transform.bias


def summarize_affine_sequence(transitions: np.ndarray, biases: np.ndarray) -> AffineChunkSummary:
    """Build ordered prefix summaries from per-item affine state transforms."""
    if transitions.ndim < 3 or biases.ndim != transitions.ndim:
        raise ValueError("affine sequence transitions and biases must be batched matrices")
    if transitions.shape[-1] != transitions.shape[-2]:
        raise ValueError("affine sequence transitions must be square")
    if transitions.shape[:-2] != biases.shape[:-2]:
        raise ValueError("affine sequence transition and bias domains must match")
    if transitions.shape[-1] != biases.shape[-2]:
        raise ValueError("affine sequence transition must act on the first state dimension")

    prefix_shape = (*transitions.shape[:-3], transitions.shape[-2], transitions.shape[-1])
    bias_shape = (*biases.shape[:-3], biases.shape[-2], biases.shape[-1])
    identity = np.broadcast_to(np.eye(transitions.shape[-1], dtype=transitions.dtype), prefix_shape).copy()
    zero = np.zeros(bias_shape, dtype=biases.dtype)
    prefix = AffineStateTransform(transition=identity, bias=zero)
    prefixes: list[AffineStateTransform] = []
    for position in range(transitions.shape[-3]):
        item = AffineStateTransform(
            transition=transitions[..., position, :, :],
            bias=biases[..., position, :, :],
        )
        prefix = compose_affine_transforms(prefix, item)
        prefixes.append(prefix)
    return AffineChunkSummary(final=prefix, prefixes=tuple(prefixes))


def explain_stateful_scan(scan: StatefulScan) -> str:
    """Render a compact, backend-independent semantic dump."""
    state = scan.value(scan.state_input)
    lines = [
        f"StatefulScan {scan.name}",
        f"  ordered axis: {scan.ordered_axis.label or scan.ordered_axis.id} [{scan.ordered_axis.extent}]",
        f"  state: {scan.state_input} {tuple(axis.extent for axis in state.axes)} {state.dtype.value}",
        f"  numerical contract: {scan.numerical_contract.value}",
        "  update:",
    ]
    lines.extend(f"    {primitive.output} = {primitive.equation}" for primitive in scan.update)
    lines.append("  read:")
    lines.extend(f"    {primitive.output} = {primitive.equation}" for primitive in scan.read)
    if scan.chunk_algebra is not None:
        lines.extend(
            (
                "  chunk algebra:",
                f"    summary: {', '.join(scan.chunk_algebra.summary_values)}",
                "    bounded representation closed under compose: "
                f"{str(scan.chunk_algebra.bounded_representation_closed_under_compose).lower()}",
            )
        )
    return "\n".join(lines)


def _validate_stateful_scan(scan: StatefulScan) -> None:
    if scan.ordered_axis.extent <= 0:
        raise ValueError("ordered scan extent must be positive")

    axis_extents: dict[int, int] = {}
    for value in scan.values:
        for axis in value.axes:
            if axis.extent <= 0:
                raise ValueError(f"axis {axis.label or axis.id!r} must have positive extent")
            previous = axis_extents.setdefault(axis.id, axis.extent)
            if previous != axis.extent:
                raise ValueError(f"logical axis {axis.id} has inconsistent extents {previous} and {axis.extent}")

    value_by_name = {value.name: value for value in scan.values}
    if len(value_by_name) != len(scan.values):
        raise ValueError("stateful scan value names must be unique")

    required_names = (scan.state_input, scan.state_output, *scan.scan_inputs, *scan.scan_outputs)
    missing = tuple(name for name in required_names if name not in value_by_name)
    if missing:
        raise ValueError(f"stateful scan references unknown values: {missing}")

    for state_name in (scan.state_input, scan.state_output):
        if value_by_name[state_name].role is not ScanValueRole.STATE:
            raise ValueError(f"scan state {state_name!r} must have role=state")
        if scan.ordered_axis in value_by_name[state_name].axes:
            raise ValueError(f"scan state {state_name!r} cannot carry the ordered axis")

    for input_name in scan.scan_inputs:
        if scan.ordered_axis not in value_by_name[input_name].axes:
            raise ValueError(f"scan input {input_name!r} must carry the ordered axis")
    for output_name in scan.scan_outputs:
        if scan.ordered_axis not in value_by_name[output_name].axes:
            raise ValueError(f"scan output {output_name!r} must carry the ordered axis")

    primitives = (*scan.update, *scan.read)
    for primitive in primitives:
        unknown = tuple(name for name in (*primitive.inputs, primitive.output) if name not in value_by_name)
        if unknown:
            raise ValueError(f"primitive {primitive.name!r} references unknown values: {unknown}")

    if not any(primitive.output == scan.state_output for primitive in scan.update):
        raise ValueError("scan update must produce the declared state output")
    produced_outputs = {primitive.output for primitive in scan.read}
    missing_outputs = tuple(name for name in scan.scan_outputs if name not in produced_outputs)
    if missing_outputs:
        raise ValueError(f"scan read does not produce declared outputs: {missing_outputs}")


def _validate_affine_transform(transform: AffineStateTransform) -> None:
    if transform.transition.ndim < 2 or transform.bias.ndim != transform.transition.ndim:
        raise ValueError("affine transition and bias must be batched matrices")
    if transform.transition.shape[-1] != transform.transition.shape[-2]:
        raise ValueError("affine transition must be square in its final two dimensions")
    if transform.transition.shape[:-2] != transform.bias.shape[:-2]:
        raise ValueError("affine transition and bias batch dimensions must match")
    if transform.transition.shape[-1] != transform.bias.shape[-2]:
        raise ValueError("affine transition must act on the first state dimension")
