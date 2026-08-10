# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral programs for tiled Fold finalization.

The semantic program is a small structured reduction, not a workload tag.  It
defines an optional row-scalar reduction, a scalar weight expression, a vector
contribution/update, an optional weight denominator, and a final expression.
The physical record separately describes the logical axes, addressing, tiles,
vector width, staging, and deterministic partial order.
"""

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from shuttle.ir import DType
from tile_lifetime.tensor_program import (
    ScalarExpression,
    ScalarExpressionKind,
    TensorAxis,
    scalar_binary,
    scalar_constant,
    scalar_expression_inputs,
    scalar_input,
    scalar_select,
    scalar_unary,
)


class FoldPartialOrder(StrEnum):
    """Logical order in which partial contributions enter the Fold."""

    ASCENDING = "ascending"


class FoldReassociationPolicy(StrEnum):
    """Finite-precision freedom available to a physical reduction."""

    SOURCE_ORDERED = "source_ordered"
    DETERMINISTIC_TREE = "deterministic_tree"


class FoldScalarReduction(StrEnum):
    """Optional row-scalar reduction performed before vector accumulation."""

    NONE = "none"
    MAXIMUM = "maximum"


class FoldDenominatorPolicy(StrEnum):
    """Optional scalar denominator accumulated beside the vector state."""

    NONE = "none"
    SUM_WEIGHTS = "sum_weights"


class FoldPartialAddressing(StrEnum):
    """How a backend resolves one logical ``(partial, row)`` value vector."""

    DENSE = "dense_partial_row"
    INDEXED = "indexed_row"


class FoldPhysicalAxis(StrEnum):
    """Axis roles appearing in a physical Fold input layout."""

    PARTIAL = "partial"
    ROW = "row"
    FEATURE = "feature"
    SOURCE = "source"


class FoldFeatureLayout(StrEnum):
    """Physical ordering of feature coordinates inside one stored row."""

    CONTIGUOUS = "contiguous"
    STG128_LANE_PERMUTED = "stg128_lane_permuted"


@dataclass(frozen=True)
class TiledFoldAxes:
    """Stable logical identities and extents for one tiled finalization."""

    partial: TensorAxis
    row: TensorAxis
    feature: TensorAxis

    def __post_init__(self) -> None:
        if len({self.partial.id, self.row.id, self.feature.id}) != 3:
            raise ValueError("tiled Fold axes must have distinct identities")
        if min(self.partial.extent, self.row.extent, self.feature.extent) <= 0:
            raise ValueError("tiled Fold axis extents must be positive")


@dataclass(frozen=True)
class TiledFoldInputLayout:
    """Physical axis order for dense or row-indexed partial storage."""

    addressing: FoldPartialAddressing
    value_axis_order: tuple[FoldPhysicalAxis, ...]
    scalar_axis_order: tuple[FoldPhysicalAxis, FoldPhysicalAxis]
    index_axis_order: tuple[FoldPhysicalAxis, ...] = ()
    feature_layout: FoldFeatureLayout = FoldFeatureLayout.CONTIGUOUS

    def __post_init__(self) -> None:
        if set(self.scalar_axis_order) != {FoldPhysicalAxis.PARTIAL, FoldPhysicalAxis.ROW}:
            raise ValueError("partial scalar layout must contain the partial and row axes")
        if self.addressing is FoldPartialAddressing.DENSE:
            if set(self.value_axis_order) != {
                FoldPhysicalAxis.PARTIAL,
                FoldPhysicalAxis.ROW,
                FoldPhysicalAxis.FEATURE,
            }:
                raise ValueError("dense partial values must contain partial, row, and feature axes")
            if self.index_axis_order:
                raise ValueError("dense partial storage cannot have an index layout")
        else:
            if self.value_axis_order != (FoldPhysicalAxis.SOURCE, FoldPhysicalAxis.FEATURE):
                raise ValueError("indexed values must be source-row major with contiguous features")
            if set(self.index_axis_order) != {FoldPhysicalAxis.PARTIAL, FoldPhysicalAxis.ROW}:
                raise ValueError("indexed partial metadata must contain row and partial axes")
        if self.value_axis_order[-1] is not FoldPhysicalAxis.FEATURE:
            raise ValueError("the initial coalesced vector loader requires contiguous features")


@dataclass(frozen=True)
class TiledFoldFinalizeSchedule:
    """Workload-neutral physical choices for one Fold-finalization skeleton."""

    axes: TiledFoldAxes
    partial_addressing: FoldPartialAddressing
    row_tile: int
    feature_tile: int
    vector_bytes: int
    shared_stages: int
    threads: int
    partial_lanes: int
    shared_buffers: int = 2
    input_layout: TiledFoldInputLayout | None = None
    partial_order: FoldPartialOrder = FoldPartialOrder.ASCENDING

    def __post_init__(self) -> None:
        if self.input_layout is None:
            if self.partial_addressing is FoldPartialAddressing.DENSE:
                layout = TiledFoldInputLayout(
                    addressing=self.partial_addressing,
                    value_axis_order=(
                        FoldPhysicalAxis.PARTIAL,
                        FoldPhysicalAxis.ROW,
                        FoldPhysicalAxis.FEATURE,
                    ),
                    scalar_axis_order=(FoldPhysicalAxis.PARTIAL, FoldPhysicalAxis.ROW),
                )
            else:
                layout = TiledFoldInputLayout(
                    addressing=self.partial_addressing,
                    value_axis_order=(FoldPhysicalAxis.SOURCE, FoldPhysicalAxis.FEATURE),
                    scalar_axis_order=(FoldPhysicalAxis.ROW, FoldPhysicalAxis.PARTIAL),
                    index_axis_order=(FoldPhysicalAxis.ROW, FoldPhysicalAxis.PARTIAL),
                )
            object.__setattr__(self, "input_layout", layout)
        elif self.input_layout.addressing is not self.partial_addressing:
            raise ValueError("Fold schedule addressing and input layout must agree")
        if (
            min(
                self.row_tile,
                self.feature_tile,
                self.vector_bytes,
                self.shared_stages,
                self.shared_buffers,
                self.threads,
                self.partial_lanes,
            )
            <= 0
        ):
            raise ValueError("tiled Fold schedule parameters must be positive")
        if self.vector_bytes & (self.vector_bytes - 1):
            raise ValueError("tiled Fold vector width must be a power of two")
        if self.partial_lanes & (self.partial_lanes - 1):
            raise ValueError("tiled Fold partial-lane count must be a power of two")
        if self.partial_lanes > self.threads:
            raise ValueError("partial reduction lanes cannot exceed the worker count")
        if self.shared_buffers <= 0 or self.shared_buffers & (self.shared_buffers - 1):
            raise ValueError("tiled Fold shared-buffer count must be a power of two")


@dataclass(frozen=True)
class TiledFoldFinalizeSemantics:
    """Scalar ASTs and reduction structure interpreted by a tiled skeleton."""

    scalar_reduction: FoldScalarReduction
    denominator: FoldDenominatorPolicy
    weight_expression: ScalarExpression
    contribution_expression: ScalarExpression
    update_expression: ScalarExpression
    finalize_expression: ScalarExpression
    partial_value_dtype: DType
    partial_scalar_dtype: DType
    accumulation_dtype: DType
    output_dtype: DType
    reassociation: FoldReassociationPolicy

    def __post_init__(self) -> None:
        if self.partial_scalar_dtype is not DType.FP32 or self.accumulation_dtype is not DType.FP32:
            raise ValueError("tiled Fold scalar inputs and accumulation must be FP32")
        if self.partial_value_dtype not in {DType.BF16, DType.FP32}:
            raise ValueError("tiled Fold partial values must be BF16 or FP32")
        if self.output_dtype not in {DType.BF16, DType.FP32}:
            raise ValueError("tiled Fold output must be BF16 or FP32")

        weight_inputs = scalar_expression_inputs(self.weight_expression)
        allowed_weight_inputs = {"partial_scalar", "valid"}
        if self.scalar_reduction is FoldScalarReduction.MAXIMUM:
            allowed_weight_inputs.add("reduced_scalar")
        if not weight_inputs <= allowed_weight_inputs or "valid" not in weight_inputs:
            raise ValueError(
                "weight expression must use validity and only available row-scalar inputs; "
                f"found {sorted(weight_inputs)}"
            )
        _require_expression_inputs(
            self.contribution_expression,
            {"partial_value", "weight"},
            "contribution",
        )
        _require_expression_inputs(self.update_expression, {"state", "contribution"}, "update")
        expected_finalize = {"state"}
        if self.denominator is FoldDenominatorPolicy.SUM_WEIGHTS:
            expected_finalize.add("denominator")
        _require_expression_inputs(self.finalize_expression, expected_finalize, "finalize")


@dataclass(frozen=True)
class TiledFoldFinalizeProgram:
    """Generic Fold semantics and physical choices consumed by one skeleton."""

    semantics: TiledFoldFinalizeSemantics
    schedule: TiledFoldFinalizeSchedule

    def __post_init__(self) -> None:
        if self.semantics.reassociation is FoldReassociationPolicy.SOURCE_ORDERED and self.schedule.partial_lanes != 1:
            raise ValueError("source-ordered Fold finalization requires one logical partial lane")


def normalized_exponential_fold_program(
    schedule: TiledFoldFinalizeSchedule,
    *,
    partial_value_dtype: DType,
    output_dtype: DType,
    output_scale: float = 1.0,
) -> TiledFoldFinalizeProgram:
    """Build normalized weighted-vector merge from generic reduction structure."""
    partial_scalar = scalar_input("partial_scalar")
    reduced_scalar = scalar_input("reduced_scalar")
    valid = scalar_input("valid")
    weight = scalar_select(
        valid,
        scalar_unary(
            ScalarExpressionKind.EXP,
            scalar_binary(ScalarExpressionKind.SUBTRACT, partial_scalar, reduced_scalar),
        ),
        scalar_constant(0.0),
    )
    state = scalar_input("state")
    contribution = scalar_input("contribution")
    return TiledFoldFinalizeProgram(
        semantics=TiledFoldFinalizeSemantics(
            scalar_reduction=FoldScalarReduction.MAXIMUM,
            denominator=FoldDenominatorPolicy.SUM_WEIGHTS,
            weight_expression=weight,
            contribution_expression=scalar_binary(
                ScalarExpressionKind.MULTIPLY,
                scalar_input("partial_value"),
                scalar_input("weight"),
            ),
            update_expression=scalar_binary(ScalarExpressionKind.ADD, state, contribution),
            finalize_expression=scalar_binary(
                ScalarExpressionKind.MULTIPLY,
                scalar_binary(
                    ScalarExpressionKind.DIVIDE,
                    scalar_input("state"),
                    scalar_input("denominator"),
                ),
                scalar_constant(output_scale),
            ),
            partial_value_dtype=partial_value_dtype,
            partial_scalar_dtype=DType.FP32,
            accumulation_dtype=DType.FP32,
            output_dtype=output_dtype,
            reassociation=FoldReassociationPolicy.DETERMINISTIC_TREE,
        ),
        schedule=schedule,
    )


def deterministic_weighted_sum_fold_program(
    schedule: TiledFoldFinalizeSchedule,
    *,
    partial_value_dtype: DType,
    output_dtype: DType,
) -> TiledFoldFinalizeProgram:
    """Build a source-ordered weighted vector sum from generic scalar ASTs."""
    state = scalar_input("state")
    contribution = scalar_input("contribution")
    return TiledFoldFinalizeProgram(
        semantics=TiledFoldFinalizeSemantics(
            scalar_reduction=FoldScalarReduction.NONE,
            denominator=FoldDenominatorPolicy.NONE,
            weight_expression=scalar_select(
                scalar_input("valid"),
                scalar_input("partial_scalar"),
                scalar_constant(0.0),
            ),
            contribution_expression=scalar_binary(
                ScalarExpressionKind.MULTIPLY,
                scalar_input("partial_value"),
                scalar_input("weight"),
            ),
            update_expression=scalar_binary(ScalarExpressionKind.ADD, state, contribution),
            finalize_expression=scalar_input("state"),
            partial_value_dtype=partial_value_dtype,
            partial_scalar_dtype=DType.FP32,
            accumulation_dtype=DType.FP32,
            output_dtype=output_dtype,
            reassociation=FoldReassociationPolicy.SOURCE_ORDERED,
        ),
        schedule=schedule,
    )


def evaluate_tiled_fold_finalize(
    program: TiledFoldFinalizeProgram,
    partial_values: np.ndarray,
    partial_scalars: np.ndarray,
    *,
    partial_valid: np.ndarray,
) -> np.ndarray:
    """Execute canonical dense Fold inputs in ascending partial order.

    ``partial_values`` has shape ``[partial, row, feature]`` and
    ``partial_scalars`` and ``partial_valid`` have shape ``[partial, row]``.
    Indexed physical inputs are gathered into these logical coordinates only
    for this NumPy reference; a backend may resolve row indirection in its tile
    loader.
    """
    axes = program.schedule.axes
    expected_values = (axes.partial.extent, axes.row.extent, axes.feature.extent)
    expected_scalars = expected_values[:2]
    if partial_values.shape != expected_values:
        raise ValueError(f"partial values must have logical shape {expected_values}, found {partial_values.shape}")
    if partial_scalars.shape != expected_scalars:
        raise ValueError(f"partial scalars must have logical shape {expected_scalars}, found {partial_scalars.shape}")
    valid = np.asarray(partial_valid)
    if valid.shape != expected_scalars or valid.dtype != np.bool_:
        raise ValueError("partial validity must be a Boolean tensor over the partial and row axes")

    values = np.asarray(partial_values, dtype=np.float32)
    scalars = np.asarray(partial_scalars, dtype=np.float32)
    semantics = program.semantics
    reduced_scalar = _reduce_scalars(semantics.scalar_reduction, scalars, valid)
    weights = _evaluate_weights(semantics, scalars, reduced_scalar, valid)
    denominator = _accumulate_denominator(semantics.denominator, weights)

    state = np.zeros(values.shape[1:], dtype=np.float32)
    for partial in range(values.shape[0]):
        for row in range(values.shape[1]):
            if not valid[partial, row]:
                continue
            for feature in range(values.shape[2]):
                contribution = _evaluate_expression(
                    semantics.contribution_expression,
                    {
                        "partial_value": float(values[partial, row, feature]),
                        "weight": float(weights[partial, row]),
                    },
                )
                state[row, feature] = np.float32(
                    _evaluate_expression(
                        semantics.update_expression,
                        {"state": float(state[row, feature]), "contribution": float(contribution)},
                    )
                )

    result = np.empty_like(state)
    for row in range(values.shape[1]):
        for feature in range(values.shape[2]):
            inputs = {"state": float(state[row, feature])}
            if denominator is not None:
                inputs["denominator"] = float(denominator[row])
            result[row, feature] = np.float32(_evaluate_expression(semantics.finalize_expression, inputs))
    return _cast_output(result, semantics.output_dtype)


def _reduce_scalars(
    reduction: FoldScalarReduction,
    scalars: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray | None:
    if reduction is FoldScalarReduction.NONE:
        return None
    usable = valid & np.isfinite(scalars)
    if np.any(~np.any(usable, axis=0)):
        empty_rows = np.flatnonzero(~np.any(usable, axis=0))
        raise ValueError(f"maximum Fold rows have no valid finite partials: {empty_rows.tolist()}")
    state = np.full(scalars.shape[1], -np.inf, dtype=np.float32)
    for partial in range(scalars.shape[0]):
        state = np.where(usable[partial], np.maximum(state, scalars[partial]), state).astype(np.float32)
    return state


def _evaluate_weights(
    semantics: TiledFoldFinalizeSemantics,
    scalars: np.ndarray,
    reduced_scalar: np.ndarray | None,
    valid: np.ndarray,
) -> np.ndarray:
    weights = np.empty_like(scalars)
    for partial in range(scalars.shape[0]):
        for row in range(scalars.shape[1]):
            inputs: dict[str, float | bool] = {
                "partial_scalar": float(scalars[partial, row]),
                "valid": bool(valid[partial, row]),
            }
            if reduced_scalar is not None:
                inputs["reduced_scalar"] = float(reduced_scalar[row])
            weights[partial, row] = np.float32(_evaluate_expression(semantics.weight_expression, inputs))
    return weights


def _accumulate_denominator(policy: FoldDenominatorPolicy, weights: np.ndarray) -> np.ndarray | None:
    if policy is FoldDenominatorPolicy.NONE:
        return None
    denominator = np.zeros(weights.shape[1], dtype=np.float32)
    for partial in range(weights.shape[0]):
        denominator = (denominator + weights[partial]).astype(np.float32)
    if np.any(denominator <= 0):
        empty_rows = np.flatnonzero(denominator <= 0)
        raise ValueError(f"weight denominator is empty for rows {empty_rows.tolist()}")
    return denominator


def _evaluate_expression(expression: ScalarExpression, inputs: dict[str, float | bool]) -> float | bool:
    kind = expression.kind
    if kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        return inputs[expression.input_name]
    if kind is ScalarExpressionKind.CONSTANT:
        assert expression.constant is not None
        return expression.constant
    if kind is ScalarExpressionKind.SELECT:
        predicate = _evaluate_expression(expression.operands[0], inputs)
        selected = expression.operands[1] if bool(predicate) else expression.operands[2]
        return _evaluate_expression(selected, inputs)
    values = tuple(_evaluate_expression(operand, inputs) for operand in expression.operands)
    if kind is ScalarExpressionKind.ADD:
        return float(values[0]) + float(values[1])
    if kind is ScalarExpressionKind.SUBTRACT:
        return float(values[0]) - float(values[1])
    if kind is ScalarExpressionKind.MULTIPLY:
        return float(values[0]) * float(values[1])
    if kind is ScalarExpressionKind.DIVIDE:
        return float(values[0]) / float(values[1])
    if kind is ScalarExpressionKind.EXP:
        return math.exp(float(values[0]))
    if kind is ScalarExpressionKind.RSQRT:
        return 1.0 / math.sqrt(float(values[0]))
    if kind is ScalarExpressionKind.TANH:
        return math.tanh(float(values[0]))
    if kind is ScalarExpressionKind.LESS_EQUAL:
        return float(values[0]) <= float(values[1])
    raise AssertionError(f"unhandled scalar expression kind {kind}")


def _cast_output(values: np.ndarray, dtype: DType) -> np.ndarray:
    if dtype is DType.FP32:
        return values.astype(np.float32, copy=False)
    if dtype is not DType.BF16:
        raise AssertionError(f"unsupported tiled Fold output dtype {dtype}")

    # NumPy has no built-in BF16 dtype. Return FP32 values rounded to exact BF16
    # representable values so reference execution remains portable.
    bits = values.astype(np.float32, copy=False).view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding_bias) & np.uint32(0xFFFF0000)).view(np.float32)


def _require_expression_inputs(expression: ScalarExpression, expected: set[str], label: str) -> None:
    observed = scalar_expression_inputs(expression)
    if observed != expected:
        raise ValueError(f"{label} expression inputs must be {sorted(expected)}, found {sorted(observed)}")
