# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive exact streaming attention from generic Contract/Map/Fold semantics."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from tile_lifetime.ir import DType, ScaledDotProductAttentionOp
from tile_lifetime.tensor_program import (
    AxisIndexMap,
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
    scalar_input,
    scalar_select,
    scalar_unary,
)


class AttentionScoreAxis(StrEnum):
    """Logical score axes available to extra map inputs."""

    BATCH = "batch"
    HEAD = "head"
    QUERY = "query"
    KEY = "key"


@dataclass(frozen=True)
class ScoreInputSpec:
    """One external tensor broadcast into the score map."""

    name: str
    axes: tuple[AttentionScoreAxis, ...]
    dtype: DType


@dataclass(frozen=True)
class ScoreMapSpec:
    """A generic scalar expression and the tensors it references."""

    expression: ScalarExpression
    inputs: tuple[ScoreInputSpec, ...] = ()

    def __post_init__(self) -> None:
        names = [value.name for value in self.inputs]
        if len(set(names)) != len(names):
            raise ValueError("score-map input names must be unique")


def scaled_score_map(scale: float) -> ScoreMapSpec:
    """Multiply raw QK scores by a finite scalar."""
    if not np.isfinite(scale):
        raise ValueError("attention score scale must be finite")
    return ScoreMapSpec(
        expression=scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            scalar_input("score.raw"),
            scalar_constant(scale),
        )
    )


def add_score_bias(
    score_map: ScoreMapSpec,
    *,
    name: str = "score.bias",
    axes: tuple[AttentionScoreAxis, ...] = (
        AttentionScoreAxis.HEAD,
        AttentionScoreAxis.QUERY,
        AttentionScoreAxis.KEY,
    ),
) -> ScoreMapSpec:
    """Add an arbitrary broadcast-compatible score bias."""
    return _extend_score_map(
        score_map,
        ScoreInputSpec(name=name, axes=axes, dtype=DType.FP32),
        scalar_binary(ScalarExpressionKind.ADD, score_map.expression, scalar_input(name)),
    )


def apply_tanh_softcap(score_map: ScoreMapSpec, cap: float) -> ScoreMapSpec:
    """Replace scores with ``cap * tanh(score / cap)``."""
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("score softcap must be finite and positive")
    divided = scalar_binary(
        ScalarExpressionKind.DIVIDE,
        score_map.expression,
        scalar_constant(cap),
    )
    expression = scalar_binary(
        ScalarExpressionKind.MULTIPLY,
        scalar_constant(cap),
        scalar_unary(ScalarExpressionKind.TANH, divided),
    )
    return ScoreMapSpec(expression=expression, inputs=score_map.inputs)


def apply_arbitrary_score_mask(
    score_map: ScoreMapSpec,
    *,
    name: str = "score.mask",
) -> ScoreMapSpec:
    """Select negative infinity where an arbitrary score mask is false."""
    mask = ScoreInputSpec(
        name=name,
        axes=(
            AttentionScoreAxis.BATCH,
            AttentionScoreAxis.HEAD,
            AttentionScoreAxis.QUERY,
            AttentionScoreAxis.KEY,
        ),
        dtype=DType.BOOL,
    )
    expression = scalar_select(
        scalar_input(name),
        score_map.expression,
        scalar_constant(float("-inf")),
    )
    return _extend_score_map(score_map, mask, expression)


def apply_causal_score_mask(score_map: ScoreMapSpec) -> ScoreMapSpec:
    """Mask keys whose integer position is greater than the query position."""
    query_position = ScoreInputSpec(
        name="query.position",
        axes=(AttentionScoreAxis.QUERY,),
        dtype=DType.INT32,
    )
    key_position = ScoreInputSpec(
        name="key.position",
        axes=(AttentionScoreAxis.KEY,),
        dtype=DType.INT32,
    )
    predicate = scalar_binary(
        ScalarExpressionKind.LESS_EQUAL,
        scalar_input(key_position.name),
        scalar_input(query_position.name),
    )
    expression = scalar_select(predicate, score_map.expression, scalar_constant(float("-inf")))
    return ScoreMapSpec(
        expression=expression,
        inputs=(*score_map.inputs, query_position, key_position),
    )


def _extend_score_map(
    score_map: ScoreMapSpec,
    value: ScoreInputSpec,
    expression: ScalarExpression,
) -> ScoreMapSpec:
    if any(existing.name == value.name for existing in score_map.inputs):
        raise ValueError(f"score-map input {value.name!r} is already defined")
    return ScoreMapSpec(expression=expression, inputs=(*score_map.inputs, value))


def build_attention_tensor_program(
    *,
    batch_size: int,
    query_length: int,
    key_length: int,
    query_heads: int,
    key_value_heads: int,
    key_dimension: int,
    value_dimension: int,
    score_map: ScoreMapSpec,
    input_dtype: DType = DType.BF16,
    accumulation_dtype: DType = DType.FP32,
) -> TensorProgram:
    """Build materialized attention exclusively from Contract, Map, and Fold."""
    extents = (
        batch_size,
        query_length,
        key_length,
        query_heads,
        key_value_heads,
        key_dimension,
        value_dimension,
    )
    if any(extent <= 0 for extent in extents):
        raise ValueError("attention dimensions must be positive")
    if query_heads % key_value_heads:
        raise ValueError("query heads must be an integer multiple of key/value heads")

    batch = TensorAxis(0, batch_size, AttentionScoreAxis.BATCH.value)
    query_token = TensorAxis(1, query_length, AttentionScoreAxis.QUERY.value)
    key_token = TensorAxis(2, key_length, AttentionScoreAxis.KEY.value)
    query_head = TensorAxis(3, query_heads, AttentionScoreAxis.HEAD.value)
    key_value_head = TensorAxis(4, key_value_heads, "key_value_head")
    key_feature = TensorAxis(5, key_dimension, "key_feature")
    value_feature = TensorAxis(6, value_dimension, "value_feature")
    score_axes = (batch, query_head, query_token, key_token)

    query = ProgramValue("query", (batch, query_token, query_head, key_feature), input_dtype)
    key = ProgramValue("key", (batch, key_token, key_value_head, key_feature), input_dtype)
    value = ProgramValue("value", (batch, key_token, key_value_head, value_feature), input_dtype)
    raw_score = ProgramValue("score.raw", score_axes, accumulation_dtype)
    mapped_score = ProgramValue("score.mapped", score_axes, accumulation_dtype)
    row_axes = (batch, query_head, query_token)
    row_max = ProgramValue("score.row_max", row_axes, accumulation_dtype)
    centered_score = ProgramValue("score.centered", score_axes, accumulation_dtype)
    exponentials = ProgramValue("score.exponential", score_axes, accumulation_dtype)
    row_sum_exp = ProgramValue("score.row_sum_exp", row_axes, accumulation_dtype)
    weighted_value = ProgramValue(
        "attention.weighted_value",
        (batch, query_token, query_head, value_feature),
        accumulation_dtype,
    )
    output = ProgramValue("attention.output", weighted_value.axes, input_dtype)

    axis_by_label = {
        AttentionScoreAxis.BATCH: batch,
        AttentionScoreAxis.HEAD: query_head,
        AttentionScoreAxis.QUERY: query_token,
        AttentionScoreAxis.KEY: key_token,
    }
    score_inputs = tuple(
        ProgramValue(spec.name, tuple(axis_by_label[axis] for axis in spec.axes), spec.dtype)
        for spec in score_map.inputs
    )
    score_map_inputs = (raw_score, *score_inputs)
    head_group_size = query_heads // key_value_heads
    key_head_map = AxisIndexMap(
        domain_axis=query_head,
        operand_axis=key_value_head,
        divisor=head_group_size,
    )
    centered_expression = scalar_binary(
        ScalarExpressionKind.SUBTRACT,
        scalar_input(mapped_score.name),
        scalar_input(row_max.name),
    )
    finalize_expression = scalar_binary(
        ScalarExpressionKind.DIVIDE,
        scalar_input(weighted_value.name),
        scalar_input(row_sum_exp.name),
    )
    operations = (
        ContractPrimitive(
            name="qk",
            inputs=(query, key),
            output=raw_score,
            reduction_axes=(key_feature,),
            accumulation_dtype=accumulation_dtype,
            input_index_maps=((), (key_head_map,)),
        ),
        MapPrimitive(
            name="score_map",
            inputs=score_map_inputs,
            output=mapped_score,
            expression=score_map.expression,
        ),
        FoldPrimitive(
            name="row_max",
            input=mapped_score,
            output=row_max,
            reduction_axes=(key_token,),
            reducer=FoldReducer.MAXIMUM,
            accumulation_dtype=accumulation_dtype,
        ),
        MapPrimitive(
            name="center_scores",
            inputs=(mapped_score, row_max),
            output=centered_score,
            expression=centered_expression,
        ),
        MapPrimitive(
            name="exp_scores",
            inputs=(centered_score,),
            output=exponentials,
            expression=scalar_unary(ScalarExpressionKind.EXP, scalar_input(centered_score.name)),
        ),
        FoldPrimitive(
            name="row_sum_exp",
            input=exponentials,
            output=row_sum_exp,
            reduction_axes=(key_token,),
            reducer=FoldReducer.SUM,
            accumulation_dtype=accumulation_dtype,
        ),
        ContractPrimitive(
            name="pv",
            inputs=(exponentials, value),
            output=weighted_value,
            reduction_axes=(key_token,),
            accumulation_dtype=accumulation_dtype,
            input_index_maps=((), (key_head_map,)),
        ),
        MapPrimitive(
            name="normalize",
            inputs=(weighted_value, row_sum_exp),
            output=output,
            expression=finalize_expression,
        ),
    )
    return TensorProgram(
        inputs=(query, key, value, *score_inputs),
        operations=operations,
        outputs=(output,),
    )


class StreamingAttentionStage(StrEnum):
    """Backend-neutral pipeline roles needed by a streaming implementation."""

    LOAD_QUERY = "load_query"
    LOAD_KEY_VALUE = "load_key_value"
    QK_CONTRACT = "qk_contract"
    SCORE_MAP = "score_map"
    ONLINE_FOLD_UPDATE = "online_fold_update"
    PV_CONTRACT = "pv_contract"
    FINALIZE = "finalize"


@dataclass(frozen=True)
class StreamingTileSchedule:
    """Target-independent tiling and producer/consumer role contract."""

    query_tile_size: int
    key_value_tile_size: int
    pipeline_depth: int
    stages: tuple[StreamingAttentionStage, ...] = (
        StreamingAttentionStage.LOAD_QUERY,
        StreamingAttentionStage.LOAD_KEY_VALUE,
        StreamingAttentionStage.QK_CONTRACT,
        StreamingAttentionStage.SCORE_MAP,
        StreamingAttentionStage.ONLINE_FOLD_UPDATE,
        StreamingAttentionStage.PV_CONTRACT,
        StreamingAttentionStage.FINALIZE,
    )

    def __post_init__(self) -> None:
        if self.query_tile_size <= 0 or self.key_value_tile_size <= 0 or self.pipeline_depth <= 0:
            raise ValueError("streaming tile sizes and pipeline depth must be positive")


@dataclass(frozen=True)
class OnlineAttentionState:
    """Derived bounded state replacing materialized scores and probabilities."""

    row_max: ProgramValue
    row_sum_exp: ProgramValue
    weighted_value_accumulator: ProgramValue


@dataclass(frozen=True)
class StreamingAttentionProgram:
    """Derived streaming program ready for a Triton, CuTe, or CPU emitter."""

    source: TensorProgram
    qk: ContractPrimitive
    score_map: MapPrimitive
    pv: ContractPrimitive
    finalize: MapPrimitive
    state: OnlineAttentionState
    schedule: StreamingTileSchedule
    materialized_values: tuple[ProgramValue, ...]


@dataclass(frozen=True)
class StreamingAttentionExecution:
    """Materialized output and bounded normalized-exponential state."""

    output: np.ndarray
    row_max: np.ndarray
    row_sum_exp: np.ndarray


def streaming_attention_from_semantic_operation(
    operation: ScaledDotProductAttentionOp,
    *,
    schedule: StreamingTileSchedule,
) -> StreamingAttentionProgram:
    """Lower a recovered semantic attention operation through tensor algebra."""
    score_map = scaled_score_map(operation.scale)
    if operation.causal:
        score_map = apply_causal_score_mask(score_map)
    source = build_attention_tensor_program(
        batch_size=operation.query.shape[0],
        query_length=operation.query.shape[1],
        key_length=operation.key.shape[1],
        query_heads=operation.query_heads,
        key_value_heads=operation.key_value_heads,
        key_dimension=operation.head_dimension,
        value_dimension=operation.value.shape[-1],
        score_map=score_map,
        input_dtype=operation.query.dtype,
        accumulation_dtype=operation.accumulation_dtype,
    )
    return derive_streaming_attention(source, schedule=schedule)


def derive_streaming_attention(
    program: TensorProgram,
    *,
    schedule: StreamingTileSchedule,
) -> StreamingAttentionProgram:
    """Recognize a cascaded softmax reduction and synthesize bounded online state."""
    if len(program.operations) != 8:
        raise ValueError("streaming attention expects an eight-operation Contract/Map/Fold normal form")
    qk, score_map, row_max, center, exponentiate, row_sum, pv, finalize = program.operations
    if not isinstance(qk, ContractPrimitive) or not isinstance(pv, ContractPrimitive):
        raise ValueError("streaming attention requires QK and PV contractions")
    if not isinstance(score_map, MapPrimitive) or not isinstance(center, MapPrimitive):
        raise ValueError("streaming attention requires explicit score maps")
    if not isinstance(exponentiate, MapPrimitive) or not isinstance(finalize, MapPrimitive):
        raise ValueError("streaming attention requires explicit exponentiation and finalization maps")
    if not isinstance(row_max, FoldPrimitive) or row_max.reducer is not FoldReducer.MAXIMUM:
        raise ValueError("streaming attention requires a row-maximum fold")
    if not isinstance(row_sum, FoldPrimitive) or row_sum.reducer is not FoldReducer.SUM:
        raise ValueError("streaming attention requires a row-sum fold")
    if qk.output not in score_map.inputs or row_max.input != score_map.output:
        raise ValueError("score map must consume QK and feed the maximum fold")
    if row_max.reduction_axes != row_sum.reduction_axes or qk.reduction_axes == row_max.reduction_axes:
        raise ValueError("softmax folds must share a key axis distinct from the QK reduction axis")
    _expect_expression(center.expression, ScalarExpressionKind.SUBTRACT, (score_map.output.name, row_max.output.name))
    _expect_expression(exponentiate.expression, ScalarExpressionKind.EXP, (center.output.name,))
    if row_sum.input != exponentiate.output or pv.inputs[0] != exponentiate.output:
        raise ValueError("softmax exponentials must feed both the sum fold and PV contraction")
    if pv.reduction_axes != row_sum.reduction_axes:
        raise ValueError("PV must reduce the same key axis as softmax")
    _expect_expression(finalize.expression, ScalarExpressionKind.DIVIDE, (pv.output.name, row_sum.output.name))
    if finalize.output not in program.outputs:
        raise ValueError("normalized attention value must be a tensor-program output")

    state = OnlineAttentionState(
        row_max=row_max.output,
        row_sum_exp=row_sum.output,
        weighted_value_accumulator=pv.output,
    )
    return StreamingAttentionProgram(
        source=program,
        qk=qk,
        score_map=score_map,
        pv=pv,
        finalize=finalize,
        state=state,
        schedule=schedule,
        materialized_values=(finalize.output,),
    )


def _expect_expression(
    expression: ScalarExpression,
    kind: ScalarExpressionKind,
    input_names: tuple[str, ...],
) -> None:
    if expression.kind is not kind:
        raise ValueError(f"expected {kind.value} expression, found {expression.kind.value}")
    actual_names = tuple(operand.input_name for operand in expression.operands)
    if actual_names != input_names:
        raise ValueError(f"expected {kind.value} inputs {input_names}, found {actual_names}")


def execute_tensor_program(
    program: TensorProgram,
    inputs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Execute a small semantic tensor program as a materialized NumPy reference."""
    values = _validated_inputs(program, inputs)
    for operation in program.operations:
        if isinstance(operation, ContractPrimitive):
            arrays = tuple(values[value.name] for value in operation.inputs)
            values[operation.output.name] = _contract(operation, arrays)
        elif isinstance(operation, MapPrimitive):
            bindings = {value.name: values[value.name] for value in operation.inputs}
            values[operation.output.name] = _evaluate_map(operation, bindings)
        else:
            assert isinstance(operation, FoldPrimitive)
            array = values[operation.input.name].astype(np.float32)
            axes = tuple(operation.input.axes.index(axis) for axis in operation.reduction_axes)
            reducer = np.max if operation.reducer is FoldReducer.MAXIMUM else np.sum
            values[operation.output.name] = reducer(array, axis=axes)
    return {output.name: values[output.name] for output in program.outputs}


def execute_streaming_attention(
    program: StreamingAttentionProgram,
    inputs: dict[str, np.ndarray],
) -> np.ndarray:
    """Execute derived online-softmax state without sequence-squared materialization."""
    return execute_streaming_attention_with_state(program, inputs).output


def execute_streaming_attention_with_state(
    program: StreamingAttentionProgram,
    inputs: dict[str, np.ndarray],
) -> StreamingAttentionExecution:
    """Execute streaming attention and retain its generic Fold state."""
    values = _validated_inputs(program.source, inputs)
    query_value, key_value = program.qk.inputs
    value_value = program.pv.inputs[1]
    query = values[query_value.name]
    key = values[key_value.name]
    value = values[value_value.name]
    query_axis = next(axis for axis in query_value.axes if axis.label == AttentionScoreAxis.QUERY.value)
    key_axis = next(axis for axis in key_value.axes if axis.label == AttentionScoreAxis.KEY.value)
    row_axes = program.state.row_max.axes
    accumulator_axes = program.state.weighted_value_accumulator.axes
    output = np.empty(program.finalize.output.shape, dtype=np.float32)
    saved_row_max = np.empty(program.state.row_max.shape, dtype=np.float32)
    saved_row_sum_exp = np.empty(program.state.row_sum_exp.shape, dtype=np.float32)

    for query_start in range(0, query_axis.extent, program.schedule.query_tile_size):
        query_stop = min(query_start + program.schedule.query_tile_size, query_axis.extent)
        axis_slices = {query_axis: slice(query_start, query_stop)}
        query_tile = _slice_array(query, query_value, axis_slices)
        tile_row_shape = tuple(query_stop - query_start if axis == query_axis else axis.extent for axis in row_axes)
        row_max = np.full(tile_row_shape, -np.inf, dtype=np.float32)
        row_sum_exp = np.zeros(tile_row_shape, dtype=np.float32)
        accumulator_shape = tuple(
            query_stop - query_start if axis == query_axis else axis.extent for axis in accumulator_axes
        )
        accumulator = np.zeros(accumulator_shape, dtype=np.float32)

        for key_start in range(0, key_axis.extent, program.schedule.key_value_tile_size):
            key_stop = min(key_start + program.schedule.key_value_tile_size, key_axis.extent)
            tile_slices = {
                query_axis: slice(query_start, query_stop),
                key_axis: slice(key_start, key_stop),
            }
            key_tile = _slice_array(key, key_value, tile_slices)
            value_tile = _slice_array(value, value_value, tile_slices)
            raw_score = _contract(program.qk, (query_tile, key_tile))
            score_bindings = {program.qk.output.name: raw_score}
            for score_input in program.score_map.inputs[1:]:
                score_bindings[score_input.name] = _slice_array(values[score_input.name], score_input, tile_slices)
            scores = _evaluate_map(program.score_map, score_bindings)

            key_reduction_axis = program.qk.output.axes.index(key_axis)
            tile_max = np.max(scores, axis=key_reduction_axis)
            next_max = np.maximum(row_max, tile_max)
            previous_scale = np.zeros_like(row_sum_exp)
            populated = row_sum_exp > 0
            previous_scale[populated] = np.exp(row_max[populated] - next_max[populated])
            centered = np.full(scores.shape, -np.inf, dtype=np.float32)
            finite_rows = np.isfinite(next_max)
            next_max_in_scores = _align_array(next_max, row_axes, program.qk.output.axes)
            finite_rows_in_scores = _align_array(finite_rows, row_axes, program.qk.output.axes)
            np.subtract(scores, next_max_in_scores, out=centered, where=finite_rows_in_scores)
            probabilities = np.exp(centered)
            tile_sum = np.sum(probabilities, axis=key_reduction_axis)
            tile_weighted = _contract(program.pv, (probabilities, value_tile))
            previous_scale_in_accumulator = _align_array(previous_scale, row_axes, accumulator_axes)
            accumulator = previous_scale_in_accumulator * accumulator + tile_weighted
            row_sum_exp = previous_scale * row_sum_exp + tile_sum
            row_max = next_max

        if np.any(row_sum_exp <= 0):
            rows = np.argwhere(row_sum_exp <= 0)
            raise ValueError(f"attention rows have no valid keys at tile-local indices {rows.tolist()}")
        normalized = _evaluate_map(
            program.finalize,
            {
                program.state.weighted_value_accumulator.name: accumulator,
                program.state.row_sum_exp.name: row_sum_exp,
            },
        )
        output_slices = tuple(
            slice(query_start, query_stop) if axis == query_axis else slice(None)
            for axis in program.finalize.output.axes
        )
        output[output_slices] = normalized
        row_slices = tuple(
            slice(query_start, query_stop) if axis == query_axis else slice(None) for axis in program.state.row_max.axes
        )
        saved_row_max[row_slices] = row_max
        saved_row_sum_exp[row_slices] = row_sum_exp
    return StreamingAttentionExecution(
        output=output,
        row_max=saved_row_max,
        row_sum_exp=saved_row_sum_exp,
    )


def _validated_inputs(program: TensorProgram, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    expected_names = {value.name for value in program.inputs}
    if set(inputs) != expected_names:
        raise ValueError(f"tensor program expected inputs {sorted(expected_names)}, got {sorted(inputs)}")
    values: dict[str, np.ndarray] = {}
    for value in program.inputs:
        array = np.asarray(inputs[value.name])
        if array.shape != value.shape:
            raise ValueError(f"input {value.name!r} has shape {array.shape}, expected {value.shape}")
        values[value.name] = array
    return values


def _contract(operation: ContractPrimitive, arrays: tuple[np.ndarray, ...]) -> np.ndarray:
    arguments: list[object] = []
    for input_index, (value, array) in enumerate(zip(operation.inputs, arrays, strict=True)):
        axes = list(value.axes)
        for mapping in operation.index_maps_for_input(input_index):
            operand_position = axes.index(mapping.operand_axis)
            array = np.take(array, mapping.indices(), axis=operand_position)
            axes[operand_position] = mapping.domain_axis
        arguments.extend((array.astype(np.float32), [axis.id for axis in axes]))
    arguments.append([axis.id for axis in operation.output.axes])
    return np.einsum(*arguments, optimize=True, dtype=np.float32)


def _evaluate_map(operation: MapPrimitive, bindings: dict[str, np.ndarray]) -> np.ndarray:
    value_by_name = {value.name: value for value in operation.inputs}

    def evaluate(expression: ScalarExpression) -> np.ndarray | float | bool:
        if expression.kind is ScalarExpressionKind.INPUT:
            assert expression.input_name is not None
            value = value_by_name[expression.input_name]
            return _align_array(bindings[value.name], value.axes, operation.output.axes)
        if expression.kind is ScalarExpressionKind.CONSTANT:
            assert expression.constant is not None
            return expression.constant
        operands = tuple(evaluate(operand) for operand in expression.operands)
        if expression.kind is ScalarExpressionKind.ADD:
            return np.add(*operands)
        if expression.kind is ScalarExpressionKind.SUBTRACT:
            return np.subtract(*operands)
        if expression.kind is ScalarExpressionKind.MULTIPLY:
            return np.multiply(*operands)
        if expression.kind is ScalarExpressionKind.DIVIDE:
            return np.divide(*operands)
        if expression.kind is ScalarExpressionKind.EXP:
            return np.exp(operands[0])
        if expression.kind is ScalarExpressionKind.LOG:
            return np.log(operands[0])
        if expression.kind is ScalarExpressionKind.RSQRT:
            return np.reciprocal(np.sqrt(operands[0]))
        if expression.kind is ScalarExpressionKind.TANH:
            return np.tanh(operands[0])
        if expression.kind is ScalarExpressionKind.LESS_EQUAL:
            return np.less_equal(*operands)
        assert expression.kind is ScalarExpressionKind.SELECT
        return np.where(operands[0], operands[1], operands[2])

    return np.asarray(evaluate(operation.expression))


def _align_array(
    array: np.ndarray,
    source_axes: tuple[TensorAxis, ...],
    target_axes: tuple[TensorAxis, ...],
) -> np.ndarray:
    if not set(source_axes) <= set(target_axes):
        raise ValueError("cannot broadcast an array onto unrelated logical axes")
    ordered_source_axes = tuple(axis for axis in target_axes if axis in source_axes)
    if ordered_source_axes != source_axes:
        permutation = tuple(source_axes.index(axis) for axis in ordered_source_axes)
        array = np.transpose(array, permutation)
    actual_extent = dict(zip(ordered_source_axes, array.shape, strict=True))
    shape = tuple(actual_extent.get(axis, 1) for axis in target_axes)
    return np.reshape(array, shape)


def _slice_array(
    array: np.ndarray,
    value: ProgramValue,
    axis_slices: dict[TensorAxis, slice],
) -> np.ndarray:
    slices = tuple(axis_slices.get(axis, slice(None)) for axis in value.axes)
    return array[slices]
