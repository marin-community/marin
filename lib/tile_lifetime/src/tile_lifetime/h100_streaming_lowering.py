# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free legalization of streaming tensor programs for SM90."""

import math
from dataclasses import dataclass

from tile_lifetime.streaming_attention import StreamingAttentionProgram
from tile_lifetime.tensor_program import ScalarExpression, ScalarExpressionKind


@dataclass(frozen=True)
class LoweredScoreMap:
    """Physical parameters recovered from scalar score-map dataflow."""

    scale: float
    causal: bool
    softcap: float | None


@dataclass(frozen=True)
class H100StreamingSchedule:
    """Finite SM90 schedule selected for the extracted physical skeleton."""

    tile_m: int
    tile_n: int
    stages: int
    threads: int = 384
    pack_gqa: bool = True
    q_in_registers: bool = False
    intra_warpgroup_overlap: bool = True
    pv_register_source: bool = True


@dataclass(frozen=True)
class H100StreamingLowering:
    """CUDA-independent contract consumed by the optional CuTe emitter."""

    score_map: LoweredScoreMap
    schedule: H100StreamingSchedule
    head_group_size: int
    output_scale: float


def lower_h100_streaming_program(program: StreamingAttentionProgram) -> H100StreamingLowering:
    """Legalize generic Contract/Map/Fold structure without importing CUDA."""
    query_value, key_value = program.qk.inputs
    value_value = program.pv.inputs[1]
    query_shape = query_value.shape
    key_shape = key_value.shape
    value_shape = value_value.shape
    if query_value.dtype.value != "bf16" or key_value.dtype.value != "bf16" or value_value.dtype.value != "bf16":
        raise ValueError("the initial SM90 skeleton accepts BF16 contraction operands")
    if query_shape[0] != key_shape[0] or query_shape[1] != key_shape[1]:
        raise ValueError("the initial SM90 skeleton requires equal dense query and key sequence domains")
    if query_shape[-1] not in (64, 128) or value_shape[-1] not in (64, 128):
        raise ValueError("the initial SM90 skeleton supports head/value dimensions 64 or 128")

    schedule = H100StreamingSchedule(
        tile_m=program.schedule.query_tile_size,
        tile_n=program.schedule.key_value_tile_size,
        stages=program.schedule.pipeline_depth,
    )
    if schedule.tile_m not in (64, 128) or schedule.tile_n not in (64, 128):
        raise ValueError("the extracted SM90 schedule supports 64/128 query and KV tiles")
    if schedule.stages not in (1, 2, 3):
        raise ValueError("the extracted SM90 schedule supports one to three pipeline stages")
    return H100StreamingLowering(
        score_map=lower_score_map(program),
        schedule=schedule,
        head_group_size=_head_group_size(program),
        output_scale=_lower_output_scale(program),
    )


def lower_score_map(program: StreamingAttentionProgram) -> LoweredScoreMap:
    """Recover scale, causal selection, and optional tanh from scalar dataflow."""
    causal = False
    softcap: float | None = None
    scale: float | None = None

    def literal(expression: ScalarExpression) -> float | bool | None:
        return expression.constant if expression.kind is ScalarExpressionKind.CONSTANT else None

    def input_name(expression: ScalarExpression) -> str | None:
        return expression.input_name if expression.kind is ScalarExpressionKind.INPUT else None

    def visit(expression: ScalarExpression) -> None:
        nonlocal causal, softcap, scale
        if expression.kind is ScalarExpressionKind.SELECT:
            predicate, selected, rejected = expression.operands
            if literal(rejected) != float("-inf"):
                raise ValueError("the SM90 skeleton only lowers masks that select negative infinity")
            if predicate.kind is not ScalarExpressionKind.LESS_EQUAL:
                raise ValueError("tensor-valued masks require an auxiliary-tensor emitter")
            left, right = predicate.operands
            if (input_name(left), input_name(right)) != ("key.position", "query.position"):
                raise ValueError("the built-in causal schedule requires key.position <= query.position")
            causal = True
            visit(selected)
            return
        if expression.kind is ScalarExpressionKind.MULTIPLY:
            left, right = expression.operands
            left_literal = literal(left)
            right_literal = literal(right)
            candidate_tanh = right if left_literal is not None else left
            candidate_cap = left_literal if left_literal is not None else right_literal
            if candidate_tanh.kind is ScalarExpressionKind.TANH and candidate_cap is not None:
                divided = candidate_tanh.operands[0]
                if divided.kind is not ScalarExpressionKind.DIVIDE or literal(divided.operands[1]) != candidate_cap:
                    raise ValueError("softcap must have the form cap * tanh(score / cap)")
                softcap = float(candidate_cap)
                visit(divided.operands[0])
                return
            raw_name = input_name(left) or input_name(right)
            candidate_scale = right_literal if input_name(left) is not None else left_literal
            if raw_name != program.qk.output.name or candidate_scale is None:
                raise ValueError("QK score must be multiplied by one scalar")
            scale = float(candidate_scale)
            return
        raise ValueError(f"unsupported SM90 score-map expression {expression.kind.value}")

    visit(program.score_map.expression)
    if scale is None:
        raise ValueError("score map does not contain a QK scale")
    return LoweredScoreMap(scale=scale, causal=causal, softcap=softcap)


def _head_group_size(program: StreamingAttentionProgram) -> int:
    key_maps = program.qk.index_maps_for_input(1)
    value_maps = program.pv.index_maps_for_input(1)
    if len(key_maps) != 1 or value_maps != key_maps:
        raise ValueError("SM90 GQA requires one shared Q-head to KV-head operand index map")
    mapping = key_maps[0]
    if mapping.offset != 0 or mapping.modulus is not None:
        raise ValueError("SM90 packed GQA supports floor-division head maps without offset or modulus")
    if mapping.domain_axis.extent != mapping.operand_axis.extent * mapping.divisor:
        raise ValueError("GQA floor-division map does not exactly cover query heads")
    return mapping.divisor


def _lower_output_scale(program: StreamingAttentionProgram) -> float:
    """Lower ``weighted / denominator`` with an optional scalar output map."""
    expression = program.finalize.expression
    output_scale = 1.0
    if expression.kind is ScalarExpressionKind.MULTIPLY:
        left, right = expression.operands
        if left.kind is ScalarExpressionKind.CONSTANT:
            scale_expression, expression = left, right
        elif right.kind is ScalarExpressionKind.CONSTANT:
            scale_expression, expression = right, left
        else:
            raise ValueError("the SM90 finalizer only supports multiplication by a scalar constant")
        assert scale_expression.constant is not None
        output_scale = float(scale_expression.constant)
    if expression.kind is not ScalarExpressionKind.DIVIDE:
        raise ValueError("the SM90 finalizer must divide the weighted accumulator by the normalized-exp sum")
    numerator, denominator = expression.operands
    expected_inputs = (
        program.state.weighted_value_accumulator.name,
        program.state.row_sum_exp.name,
    )
    actual_inputs = (numerator.input_name, denominator.input_name)
    if actual_inputs != expected_inputs:
        raise ValueError(f"the SM90 finalizer expected inputs {expected_inputs}, found {actual_inputs}")
    if not math.isfinite(output_scale):
        raise ValueError("the SM90 finalizer output scale must be finite")
    return output_scale
