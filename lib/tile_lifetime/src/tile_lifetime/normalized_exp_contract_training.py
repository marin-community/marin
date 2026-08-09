# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic Contract and normalized-exponential Fold training semantics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tile_lifetime.autodiff import differentiate_scalar_expression
from tile_lifetime.ir import DType
from tile_lifetime.streaming_attention import execute_tensor_program
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    FoldPrimitive,
    FoldReducer,
    MapPrimitive,
    ProgramValue,
    ScalarExpression,
    ScalarExpressionKind,
    TensorAxis,
    TensorProgram,
    scalar_binary,
    scalar_constant,
    scalar_expression_inputs,
    scalar_input,
    scalar_select,
    scalar_unary,
)


@dataclass(frozen=True)
class IndexedFoldSelection:
    """Select one coordinate of a Fold axis for every unreduced row."""

    source: ProgramValue
    fold_axis: TensorAxis
    indices: ProgramValue
    row_validity: ProgramValue
    output: ProgramValue

    def __post_init__(self) -> None:
        if self.fold_axis not in self.source.axes:
            raise ValueError("indexed selection Fold axis is absent from its source")
        row_axes = tuple(axis for axis in self.source.axes if axis != self.fold_axis)
        if self.indices.axes != row_axes or self.indices.dtype is not DType.INT32:
            raise ValueError("indexed selection requires one INT32 coordinate per unreduced row")
        if self.row_validity.axes != row_axes or self.row_validity.dtype is not DType.BOOL:
            raise ValueError("indexed selection validity must be Boolean over the unreduced rows")
        if self.output.axes != row_axes or self.output.dtype is not self.source.dtype:
            raise ValueError("indexed selection output must preserve the source dtype and unreduced axes")


@dataclass(frozen=True)
class NormalizedExpContractTrainingProgram:
    """Decomposed Contract/Fold semantics and JAX-owned reverse inputs.

    This record is a semantic composition, not a physical skeleton or workload
    dispatch key. Physical selection must inspect the component primitives,
    axes, layouts, and numerical policy.
    """

    forward: TensorProgram
    selection: IndexedFoldSelection
    output: TensorProgram
    reverse: TensorProgram
    score_contract: ContractPrimitive
    score_map: MapPrimitive
    restricted_score_map: MapPrimitive
    maximum_fold: FoldPrimitive
    exponential_map: MapPrimitive
    sum_fold: FoldPrimitive
    log_normalizer_map: MapPrimitive
    reverse_score_map: MapPrimitive
    input_reverse_contract: ContractPrimitive
    operand_reverse_contract: ContractPrimitive
    fold_validity: ProgramValue
    output_cotangent: ProgramValue
    state_cotangent: ProgramValue
    selected_mask: ProgramValue


@dataclass(frozen=True)
class NormalizedExpContractTrainingExecution:
    """Materialized reference outputs for one semantic training program."""

    output: np.ndarray
    log_normalizer: np.ndarray
    input_cotangent: np.ndarray
    operand_cotangent: np.ndarray
    score_cotangent: np.ndarray


def build_normalized_exp_contract_training_program(
    *,
    rows: int,
    reduction: int,
    fold_extent: int,
    score_expression: ScalarExpression | None = None,
) -> NormalizedExpContractTrainingProgram:
    """Build a generic streamed-score training program before physical tiling."""
    if min(rows, reduction, fold_extent) <= 0:
        raise ValueError("normalized-exp Contract extents must be positive")
    row = TensorAxis(1, rows, "row")
    reduction_axis = TensorAxis(2, reduction, "contract_reduction")
    fold_axis = TensorAxis(3, fold_extent, "fold")

    input_value = ProgramValue("contract.left", (row, reduction_axis), DType.FP32)
    operand = ProgramValue("contract.right", (reduction_axis, fold_axis), DType.FP32)
    fold_validity = ProgramValue("fold.valid", (fold_axis,), DType.BOOL)
    selected_indices = ProgramValue("selection.indices", (row,), DType.INT32)
    row_validity = ProgramValue("row.valid", (row,), DType.BOOL)
    raw_score = ProgramValue("score.raw", (row, fold_axis), DType.FP32)
    mapped_score = ProgramValue("score.mapped", raw_score.axes, DType.FP32)
    restricted_score = ProgramValue("score.restricted", raw_score.axes, DType.FP32)
    row_maximum = ProgramValue("state.maximum", (row,), DType.FP32)
    centered_score = ProgramValue("score.centered", raw_score.axes, DType.FP32)
    exponential = ProgramValue("score.exponential", raw_score.axes, DType.FP32)
    row_sum_exp = ProgramValue("state.sum_exp", (row,), DType.FP32)
    log_normalizer = ProgramValue("state.log_normalizer", (row,), DType.FP32)
    visible_log_normalizer = ProgramValue("normalized_exp.log_normalizer", (row,), DType.FP32)
    selected_score = ProgramValue("selection.output", (row,), DType.FP32)
    output_value = ProgramValue("normalized_exp.output", (row,), DType.FP32)

    score_contract = ContractPrimitive(
        "score Contract",
        (input_value, operand),
        raw_score,
        (reduction_axis,),
        DType.FP32,
    )
    expression = score_expression or scalar_input(raw_score.name)
    if scalar_expression_inputs(expression) != {raw_score.name}:
        raise ValueError("score Map expression must read exactly the raw Contract result")
    score_map = MapPrimitive("score Map", (raw_score,), mapped_score, expression)
    restricted_score_map = MapPrimitive(
        "fold-domain restriction",
        (mapped_score, fold_validity),
        restricted_score,
        scalar_select(
            scalar_input(fold_validity.name),
            scalar_input(mapped_score.name),
            scalar_constant(float("-inf")),
        ),
    )
    maximum_fold = FoldPrimitive(
        "normalized-exp maximum Fold",
        restricted_score,
        row_maximum,
        (fold_axis,),
        FoldReducer.MAXIMUM,
        DType.FP32,
    )
    centered_map = MapPrimitive(
        "center scores",
        (restricted_score, row_maximum),
        centered_score,
        scalar_binary(
            ScalarExpressionKind.SUBTRACT,
            scalar_input(restricted_score.name),
            scalar_input(row_maximum.name),
        ),
    )
    exponential_map = MapPrimitive(
        "normalized exponential",
        (centered_score,),
        exponential,
        scalar_unary(ScalarExpressionKind.EXP, scalar_input(centered_score.name)),
    )
    sum_fold = FoldPrimitive(
        "normalized-exp sum Fold",
        exponential,
        row_sum_exp,
        (fold_axis,),
        FoldReducer.SUM,
        DType.FP32,
    )
    log_normalizer_map = MapPrimitive(
        "finalize normalized-exp state",
        (row_sum_exp, row_maximum),
        log_normalizer,
        scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_unary(ScalarExpressionKind.LOG, scalar_input(row_sum_exp.name)),
            scalar_input(row_maximum.name),
        ),
    )
    forward = TensorProgram(
        inputs=(input_value, operand, fold_validity),
        operations=(
            score_contract,
            score_map,
            restricted_score_map,
            maximum_fold,
            centered_map,
            exponential_map,
            sum_fold,
            log_normalizer_map,
        ),
        outputs=(raw_score, mapped_score, restricted_score, log_normalizer),
    )
    selection = IndexedFoldSelection(
        mapped_score,
        fold_axis,
        selected_indices,
        row_validity,
        selected_score,
    )
    output_map = MapPrimitive(
        "subtract indexed score",
        (log_normalizer, selected_score, row_validity),
        output_value,
        scalar_select(
            scalar_input(row_validity.name),
            scalar_binary(
                ScalarExpressionKind.SUBTRACT,
                scalar_input(log_normalizer.name),
                scalar_input(selected_score.name),
            ),
            scalar_constant(0.0),
        ),
    )
    visible_log_normalizer_map = MapPrimitive(
        "mask normalized-exp state output",
        (log_normalizer, row_validity),
        visible_log_normalizer,
        scalar_select(
            scalar_input(row_validity.name),
            scalar_input(log_normalizer.name),
            scalar_constant(0.0),
        ),
    )
    output_program = TensorProgram(
        inputs=(log_normalizer, selected_score, row_validity),
        operations=(output_map, visible_log_normalizer_map),
        outputs=(output_value, visible_log_normalizer),
    )

    output_cotangent = ProgramValue("cotangent.output", (row,), DType.FP32)
    state_cotangent = ProgramValue("cotangent.log_normalizer", (row,), DType.FP32)
    selected_mask = ProgramValue("selection.mask", raw_score.axes, DType.BOOL)
    score_cotangent = ProgramValue("cotangent.score", raw_score.axes, DType.FP32)
    probability = scalar_unary(
        ScalarExpressionKind.EXP,
        scalar_binary(
            ScalarExpressionKind.SUBTRACT,
            scalar_input(restricted_score.name),
            scalar_input(log_normalizer.name),
        ),
    )
    effective_output_cotangent = scalar_select(
        scalar_input(row_validity.name),
        scalar_input(output_cotangent.name),
        scalar_constant(0.0),
    )
    effective_state_cotangent = scalar_select(
        scalar_input(row_validity.name),
        scalar_input(state_cotangent.name),
        scalar_constant(0.0),
    )
    state_scale = scalar_binary(
        ScalarExpressionKind.ADD,
        effective_output_cotangent,
        effective_state_cotangent,
    )
    selected_contribution = scalar_select(
        scalar_input(selected_mask.name),
        effective_output_cotangent,
        scalar_constant(0.0),
    )
    raw_score_cotangent = scalar_binary(
        ScalarExpressionKind.SUBTRACT,
        scalar_binary(ScalarExpressionKind.MULTIPLY, probability, state_scale),
        selected_contribution,
    )
    score_derivative = differentiate_scalar_expression(expression, raw_score.name)
    reverse_expression = scalar_binary(
        ScalarExpressionKind.MULTIPLY,
        raw_score_cotangent,
        score_derivative,
    )
    reverse_value_by_name = {
        value.name: value
        for value in (
            raw_score,
            restricted_score,
            log_normalizer,
            output_cotangent,
            state_cotangent,
            selected_mask,
            row_validity,
        )
    }
    reverse_inputs = tuple(
        reverse_value_by_name[name]
        for name in reverse_value_by_name
        if name in scalar_expression_inputs(reverse_expression)
    )
    reverse_score_map = MapPrimitive(
        "generated normalized-exp score reverse Map",
        reverse_inputs,
        score_cotangent,
        reverse_expression,
    )
    input_cotangent = ProgramValue("cotangent.contract.left", input_value.axes, DType.FP32)
    operand_cotangent = ProgramValue("cotangent.contract.right", operand.axes, DType.FP32)
    input_reverse_contract = ContractPrimitive(
        "input reverse Contract",
        (score_cotangent, operand),
        input_cotangent,
        (fold_axis,),
        DType.FP32,
    )
    operand_reverse_contract = ContractPrimitive(
        "operand reverse Contract",
        (input_value, score_cotangent),
        operand_cotangent,
        (row,),
        DType.FP32,
    )
    reverse_external_inputs = tuple(dict.fromkeys((input_value, operand, *reverse_inputs)))
    reverse = TensorProgram(
        inputs=reverse_external_inputs,
        operations=(reverse_score_map, input_reverse_contract, operand_reverse_contract),
        outputs=(input_cotangent, operand_cotangent, score_cotangent),
    )
    return NormalizedExpContractTrainingProgram(
        forward=forward,
        selection=selection,
        output=output_program,
        reverse=reverse,
        score_contract=score_contract,
        score_map=score_map,
        restricted_score_map=restricted_score_map,
        maximum_fold=maximum_fold,
        exponential_map=exponential_map,
        sum_fold=sum_fold,
        log_normalizer_map=log_normalizer_map,
        reverse_score_map=reverse_score_map,
        input_reverse_contract=input_reverse_contract,
        operand_reverse_contract=operand_reverse_contract,
        fold_validity=fold_validity,
        output_cotangent=output_cotangent,
        state_cotangent=state_cotangent,
        selected_mask=selected_mask,
    )


def execute_normalized_exp_contract_training(
    program: NormalizedExpContractTrainingProgram,
    *,
    left: np.ndarray,
    right: np.ndarray,
    selected_indices: np.ndarray,
    row_validity: np.ndarray,
    fold_validity: np.ndarray,
    output_cotangent: np.ndarray,
    state_cotangent: np.ndarray,
) -> NormalizedExpContractTrainingExecution:
    """Execute the decomposed semantics as an independent NumPy reference."""
    score_contract = program.score_contract
    left_value, right_value = score_contract.inputs
    forward = execute_tensor_program(
        program.forward,
        {
            left_value.name: np.asarray(left),
            right_value.name: np.asarray(right),
            program.fold_validity.name: np.asarray(fold_validity),
        },
    )
    mapped_score = forward[program.score_map.output.name]
    log_normalizer = forward[program.log_normalizer_map.output.name]
    selection = program.selection
    indices = np.asarray(selected_indices, dtype=np.int32)
    valid_rows = np.asarray(row_validity, dtype=np.bool_)
    valid_fold = np.asarray(fold_validity, dtype=np.bool_)
    if indices.shape != selection.indices.shape or valid_rows.shape != selection.row_validity.shape:
        raise ValueError("indexed-selection runtime arrays do not match their semantic domains")
    if valid_fold.shape != program.fold_validity.shape:
        raise ValueError("Fold validity runtime array does not match its semantic domain")
    if not np.any(valid_fold):
        raise ValueError("normalized-exp Fold domain has no valid coordinates")
    if np.any(valid_rows & ((indices < 0) | (indices >= valid_fold.shape[0]))):
        raise ValueError("valid rows contain an out-of-domain selected Fold coordinate")
    safe_indices = np.clip(indices, 0, valid_fold.shape[0] - 1)
    if np.any(valid_rows & ~valid_fold[safe_indices]):
        raise ValueError("valid rows select a restricted Fold coordinate")
    selected_score = np.take_along_axis(mapped_score, safe_indices[:, None], axis=1)[:, 0]
    visible_outputs = execute_tensor_program(
        program.output,
        {
            program.log_normalizer_map.output.name: log_normalizer,
            selection.output.name: selected_score,
            selection.row_validity.name: valid_rows,
        },
    )
    output = visible_outputs[program.output.outputs[0].name]
    visible_log_normalizer = visible_outputs[program.output.outputs[1].name]
    selected_mask = np.zeros(mapped_score.shape, dtype=np.bool_)
    selected_mask[np.arange(mapped_score.shape[0]), safe_indices] = valid_rows
    reverse_values = {
        left_value.name: np.asarray(left),
        right_value.name: np.asarray(right),
        program.score_contract.output.name: forward[program.score_contract.output.name],
        program.restricted_score_map.output.name: forward[program.restricted_score_map.output.name],
        program.log_normalizer_map.output.name: log_normalizer,
        program.output_cotangent.name: np.asarray(output_cotangent),
        program.state_cotangent.name: np.asarray(state_cotangent),
        program.selected_mask.name: selected_mask,
        selection.row_validity.name: valid_rows,
    }
    reverse = execute_tensor_program(
        program.reverse,
        {value.name: reverse_values[value.name] for value in program.reverse.inputs},
    )
    return NormalizedExpContractTrainingExecution(
        output=output,
        log_normalizer=visible_log_normalizer,
        input_cotangent=reverse[program.input_reverse_contract.output.name],
        operand_cotangent=reverse[program.operand_reverse_contract.output.name],
        score_cotangent=reverse[program.reverse_score_map.output.name],
    )


def tanh_soft_cap_score_expression(raw_score_name: str, cap: float) -> ScalarExpression:
    """Return a generic finite score Map mutation used by synthesis tests."""
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("score soft cap must be finite and positive")
    return scalar_binary(
        ScalarExpressionKind.MULTIPLY,
        scalar_constant(cap),
        scalar_unary(
            ScalarExpressionKind.TANH,
            scalar_binary(
                ScalarExpressionKind.DIVIDE,
                scalar_input(raw_score_name),
                scalar_constant(cap),
            ),
        ),
    )
