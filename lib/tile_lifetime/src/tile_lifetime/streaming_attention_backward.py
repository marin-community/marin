# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive streaming reverse programs from generic normalized-exp tensor algebra."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from tile_lifetime.autodiff import differentiate_scalar_expression
from tile_lifetime.ir import DType
from tile_lifetime.streaming_attention import (
    AttentionScoreAxis,
    StreamingAttentionExecution,
    StreamingAttentionProgram,
    _align_array,
    _contract,
    _evaluate_map,
    _slice_array,
    _validated_inputs,
)
from tile_lifetime.tensor_program import MapPrimitive, ProgramValue, ScalarExpressionKind, scalar_expression_inputs


class StreamingAttentionBackwardStage(StrEnum):
    """Generic physical stages needed by the reverse streamed Fold."""

    LOAD_QUERY_AND_STATE = "load_query_and_state"
    LOAD_KEY_VALUE = "load_key_value"
    RECOMPUTE_QK = "recompute_qk"
    RECOMPUTE_PROBABILITY = "recompute_probability"
    DV_CONTRACT = "dv_contract"
    DP_CONTRACT = "dp_contract"
    SCORE_MAP_VJP = "score_map_vjp"
    DQ_CONTRACT = "dq_contract"
    DK_CONTRACT = "dk_contract"


class StreamingAttentionBackwardReassociation(StrEnum):
    """Finite-precision order used for gradient accumulation."""

    DETERMINISTIC_TREE = "deterministic_tree"


class StreamingAttentionBackwardProvenance(StrEnum):
    """Origin of the visible generic reverse algebra."""

    REFERENCE_SYMBOLIC_VJP = "reference_symbolic_vjp"
    JAX_VJP_HLO_RECOVERY = "jax_vjp_hlo_recovery"


class StreamingAttentionBackwardDomainTraversal(StrEnum):
    """Physical traversal selected from the score-domain restriction."""

    FULL = "full"
    LOWER_TRIANGULAR = "lower_triangular"


class StreamingAttentionBackwardFoldOrder(StrEnum):
    """Explicit finite-precision order for the key/value gradient Fold."""

    QUERY_ROW_MAJOR_MAPPED_HEAD_MINOR_TREE = "query_row_major_mapped_head_minor_tree"


@dataclass(frozen=True)
class StreamingAttentionBackwardTileSchedule:
    """A deterministic reverse schedule over generic Contract/Fold axes.

    ``query_heads_per_key_value_tile`` is derived from the QK input index map.
    It coalesces all query rows that consume the same K/V tile into one physical
    Contract.  This is an axis-relation transformation, not an attention-name
    dispatch.
    """

    query_tile_size: int
    key_value_tile_size: int
    query_heads_per_key_value_tile: int
    domain_traversal: StreamingAttentionBackwardDomainTraversal
    key_value_fold_order: StreamingAttentionBackwardFoldOrder

    def __post_init__(self) -> None:
        extents = (
            self.query_tile_size,
            self.key_value_tile_size,
            self.query_heads_per_key_value_tile,
        )
        if any(extent <= 0 for extent in extents):
            raise ValueError("streaming backward tile extents must be positive")


@dataclass(frozen=True)
class StreamingAttentionBackwardWorkEstimate:
    """Static Contract counts and temporary footprint for one reverse schedule."""

    logical_query_key_tile_pairs: int
    fully_restricted_tile_pairs: int
    query_gradient_contract_invocations: int
    full_domain_query_gradient_contract_invocations: int
    key_value_gradient_contract_invocations: int
    scalar_head_key_value_contract_invocations: int
    full_domain_scalar_head_key_value_contract_invocations: int
    packed_query_rows: int
    peak_score_tile_elements: int
    peak_query_tile_elements: int
    key_value_gradient_accumulator_elements: int

    @property
    def key_value_contract_invocation_reduction(self) -> float:
        return self.scalar_head_key_value_contract_invocations / self.key_value_gradient_contract_invocations

    @property
    def key_value_contract_invocation_reduction_from_full_scalar(self) -> float:
        return self.full_domain_scalar_head_key_value_contract_invocations / self.key_value_gradient_contract_invocations


@dataclass(frozen=True)
class StreamingAttentionBackwardProgram:
    """A reverse stream with visible scalar and contraction algebra.

    Accepted end-to-end paths recover this structure from JAX VJP HLO.  The
    local symbolic derivation remains a component oracle for physical schedule
    work until that recovery boundary is connected.
    """

    forward: StreamingAttentionProgram
    output_cotangent: ProgramValue
    score_map_vjp: MapPrimitive
    output_scale: float
    stages: tuple[StreamingAttentionBackwardStage, ...]
    materialized_values: tuple[ProgramValue, ...]
    provenance: StreamingAttentionBackwardProvenance
    accumulation_dtype: DType = DType.FP32
    reassociation: StreamingAttentionBackwardReassociation = StreamingAttentionBackwardReassociation.DETERMINISTIC_TREE


@dataclass(frozen=True)
class StreamingAttentionBackwardExecution:
    """Input cotangents produced by the generic reverse stream."""

    query_cotangent: np.ndarray
    key_cotangent: np.ndarray
    value_cotangent: np.ndarray


def derive_streaming_attention_backward(
    forward: StreamingAttentionProgram,
) -> StreamingAttentionBackwardProgram:
    """Build a reference reverse program used to validate JAX-VJP recovery."""
    query, key = forward.qk.inputs
    value = forward.pv.inputs[1]
    output_cotangent = ProgramValue(
        "cotangent.streaming_output",
        forward.finalize.output.axes,
        DType.FP32,
    )
    score_map_derivative = differentiate_scalar_expression(
        forward.score_map.expression,
        forward.qk.output.name,
    )
    derivative_input_names = scalar_expression_inputs(score_map_derivative)
    score_map_vjp = MapPrimitive(
        name="generated score Map derivative",
        inputs=tuple(value for value in forward.score_map.inputs if value.name in derivative_input_names),
        output=forward.score_map.output,
        expression=score_map_derivative,
    )
    output_scale = _normalized_output_scale(forward)
    cotangents = (
        ProgramValue("cotangent.query", query.axes, DType.FP32),
        ProgramValue("cotangent.key", key.axes, DType.FP32),
        ProgramValue("cotangent.value", value.axes, DType.FP32),
    )
    return StreamingAttentionBackwardProgram(
        forward=forward,
        output_cotangent=output_cotangent,
        score_map_vjp=score_map_vjp,
        output_scale=output_scale,
        stages=tuple(StreamingAttentionBackwardStage),
        materialized_values=cotangents,
        provenance=StreamingAttentionBackwardProvenance.REFERENCE_SYMBOLIC_VJP,
    )


def derive_streaming_attention_backward_tile_schedule(
    program: StreamingAttentionBackwardProgram,
    *,
    query_tile_size: int,
    key_value_tile_size: int,
    domain_traversal: StreamingAttentionBackwardDomainTraversal,
) -> StreamingAttentionBackwardTileSchedule:
    """Coalesce query heads that share one operand through a Contract index map."""
    query, key = program.forward.qk.inputs
    query_head = next(axis for axis in query.axes if axis.label == AttentionScoreAxis.HEAD.value)
    key_value_head = next(axis for axis in key.axes if axis.label == "key_value_head")
    matching_maps = tuple(
        index_map
        for index_map in program.forward.qk.index_maps_for_input(1)
        if index_map.domain_axis == query_head and index_map.operand_axis == key_value_head
    )
    if len(matching_maps) != 1:
        raise ValueError("streaming backward requires one query-head to K/V-head index relation")
    index_map = matching_maps[0]
    if query_head.extent != key_value_head.extent * index_map.divisor:
        raise ValueError("query-head index relation must partition query heads evenly")
    return StreamingAttentionBackwardTileSchedule(
        query_tile_size=query_tile_size,
        key_value_tile_size=key_value_tile_size,
        query_heads_per_key_value_tile=index_map.divisor,
        domain_traversal=domain_traversal,
        key_value_fold_order=StreamingAttentionBackwardFoldOrder.QUERY_ROW_MAJOR_MAPPED_HEAD_MINOR_TREE,
    )


def estimate_streaming_attention_backward_work(
    program: StreamingAttentionBackwardProgram,
    schedule: StreamingAttentionBackwardTileSchedule,
) -> StreamingAttentionBackwardWorkEstimate:
    """Count physical reverse Contracts without executing a target backend."""
    query, key = program.forward.qk.inputs
    query_axis = next(axis for axis in query.axes if axis.label == AttentionScoreAxis.QUERY.value)
    key_axis = next(axis for axis in key.axes if axis.label == AttentionScoreAxis.KEY.value)
    batch_axis = next(axis for axis in query.axes if axis.label == AttentionScoreAxis.BATCH.value)
    query_head = next(axis for axis in query.axes if axis.label == AttentionScoreAxis.HEAD.value)
    query_feature = next(axis for axis in query.axes if axis.label == "key_feature")
    key_value_head = next(axis for axis in key.axes if axis.label == "key_value_head")
    expected_head_group = query_head.extent // key_value_head.extent
    if schedule.query_heads_per_key_value_tile != expected_head_group:
        raise ValueError("work estimate schedule disagrees with the Contract head index relation")
    if query_axis.extent % schedule.query_tile_size or key_axis.extent % schedule.key_value_tile_size:
        raise ValueError("work estimates require tile-aligned query and key domains")
    query_tiles = query_axis.extent // schedule.query_tile_size
    key_tiles = key_axis.extent // schedule.key_value_tile_size
    all_tile_pairs = query_tiles * key_tiles
    if schedule.domain_traversal is StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR:
        valid_tile_pairs = sum(
            max(0, query_tiles - (key_start // schedule.query_tile_size))
            for key_start in range(0, key_axis.extent, schedule.key_value_tile_size)
        )
    else:
        valid_tile_pairs = all_tile_pairs
    if valid_tile_pairs <= 0 or valid_tile_pairs > all_tile_pairs:
        raise ValueError("domain traversal produced an invalid Q/K tile count")

    batch_count = batch_axis.extent
    logical_tile_pairs = batch_count * query_head.extent * valid_tile_pairs
    fully_restricted = batch_count * query_head.extent * (all_tile_pairs - valid_tile_pairs)
    query_contracts = logical_tile_pairs * 3
    full_domain_query_contracts = batch_count * query_head.extent * all_tile_pairs * 3
    grouped_key_value_contracts = batch_count * key_value_head.extent * valid_tile_pairs * 4
    scalar_key_value_contracts = logical_tile_pairs * 4
    full_domain_scalar_key_value_contracts = batch_count * query_head.extent * all_tile_pairs * 4
    packed_query_rows = schedule.query_tile_size * schedule.query_heads_per_key_value_tile
    return StreamingAttentionBackwardWorkEstimate(
        logical_query_key_tile_pairs=logical_tile_pairs,
        fully_restricted_tile_pairs=fully_restricted,
        query_gradient_contract_invocations=query_contracts,
        full_domain_query_gradient_contract_invocations=full_domain_query_contracts,
        key_value_gradient_contract_invocations=grouped_key_value_contracts,
        scalar_head_key_value_contract_invocations=scalar_key_value_contracts,
        full_domain_scalar_head_key_value_contract_invocations=full_domain_scalar_key_value_contracts,
        packed_query_rows=packed_query_rows,
        peak_score_tile_elements=packed_query_rows * schedule.key_value_tile_size,
        peak_query_tile_elements=packed_query_rows * query_feature.extent,
        key_value_gradient_accumulator_elements=2 * schedule.key_value_tile_size * query_feature.extent,
    )


def execute_streaming_attention_backward(
    program: StreamingAttentionBackwardProgram,
    inputs: dict[str, np.ndarray],
    forward_execution: StreamingAttentionExecution,
    output_cotangent: np.ndarray,
) -> StreamingAttentionBackwardExecution:
    """Execute the reverse stream without score/probability materialization."""
    forward = program.forward
    values = _validated_inputs(forward.source, inputs)
    query_value, key_value = forward.qk.inputs
    value_value = forward.pv.inputs[1]
    query = values[query_value.name]
    key = values[key_value.name]
    value = values[value_value.name]
    if output_cotangent.shape != forward.finalize.output.shape:
        raise ValueError(
            f"output cotangent has shape {output_cotangent.shape}, expected {forward.finalize.output.shape}"
        )
    if forward_execution.output.shape != forward.finalize.output.shape:
        raise ValueError("saved streaming output has the wrong shape")
    if forward_execution.row_max.shape != forward.state.row_max.shape:
        raise ValueError("saved row maximum has the wrong shape")
    if forward_execution.row_sum_exp.shape != forward.state.row_sum_exp.shape:
        raise ValueError("saved row sum-exp has the wrong shape")

    query_axis = next(axis for axis in query_value.axes if axis.label == AttentionScoreAxis.QUERY.value)
    key_axis = next(axis for axis in key_value.axes if axis.label == AttentionScoreAxis.KEY.value)
    row_axes = forward.state.row_max.axes
    score_axes = forward.qk.output.axes
    query_cotangent = np.zeros(query_value.shape, dtype=np.float32)
    key_cotangent = np.zeros(key_value.shape, dtype=np.float32)
    value_cotangent = np.zeros(value_value.shape, dtype=np.float32)

    for query_start in range(0, query_axis.extent, forward.schedule.query_tile_size):
        query_stop = min(query_start + forward.schedule.query_tile_size, query_axis.extent)
        query_slices = {query_axis: slice(query_start, query_stop)}
        query_tile = _slice_array(query, query_value, query_slices).astype(np.float32)
        output_tile = _slice_array(forward_execution.output, forward.finalize.output, query_slices).astype(np.float32)
        output_cotangent_tile = _slice_array(output_cotangent, forward.finalize.output, query_slices).astype(np.float32)
        row_slices = tuple(slice(query_start, query_stop) if axis == query_axis else slice(None) for axis in row_axes)
        row_max = forward_execution.row_max[row_slices]
        row_sum_exp = forward_execution.row_sum_exp[row_slices]
        output_dot = np.sum(output_cotangent_tile * output_tile, axis=-1)
        query_gradient_tile = np.zeros(query_tile.shape, dtype=np.float32)

        for key_start in range(0, key_axis.extent, forward.schedule.key_value_tile_size):
            key_stop = min(key_start + forward.schedule.key_value_tile_size, key_axis.extent)
            tile_slices = {
                query_axis: slice(query_start, query_stop),
                key_axis: slice(key_start, key_stop),
            }
            key_tile = _slice_array(key, key_value, tile_slices).astype(np.float32)
            value_tile = _slice_array(value, value_value, tile_slices).astype(np.float32)
            raw_score = _contract(forward.qk, (query_tile, key_tile))
            score_bindings = {forward.qk.output.name: raw_score}
            for score_input in forward.score_map.inputs[1:]:
                score_bindings[score_input.name] = _slice_array(
                    values[score_input.name],
                    score_input,
                    tile_slices,
                )
            mapped_score = _evaluate_map(forward.score_map, score_bindings)
            probability = np.exp(mapped_score - _align_array(row_max, row_axes, score_axes))
            probability /= _align_array(row_sum_exp, row_axes, score_axes)
            key_head_map = forward.qk.index_maps_for_input(1)[0]
            head_indices = key_head_map.indices()
            key_expanded = np.take(key_tile, head_indices, axis=key_value.axes.index(key_head_map.operand_axis))
            value_expanded = np.take(value_tile, head_indices, axis=value_value.axes.index(key_head_map.operand_axis))
            d_probability = np.einsum("bqhv,bkhv->bhqk", output_cotangent_tile, value_expanded)
            d_probability *= program.output_scale
            d_mapped_score = probability * (d_probability - output_dot[:, :, :, None].transpose(0, 2, 1, 3))
            score_derivative = _evaluate_map(program.score_map_vjp, score_bindings)
            d_raw_score = d_mapped_score * score_derivative
            query_gradient_tile += np.einsum("bhqk,bkhd->bqhd", d_raw_score, key_expanded)
            d_key_expanded = np.einsum("bhqk,bqhd->bkhd", d_raw_score, query_tile)
            d_value_expanded = program.output_scale * np.einsum(
                "bhqk,bqhv->bkhv",
                probability,
                output_cotangent_tile,
            )
            key_axis_index = key_value.axes.index(key_axis)
            key_destination = [slice(None)] * key_cotangent.ndim
            value_destination = [slice(None)] * value_cotangent.ndim
            key_destination[key_axis_index] = slice(key_start, key_stop)
            value_destination[value_value.axes.index(key_axis)] = slice(key_start, key_stop)
            for query_head, key_value_head in enumerate(head_indices):
                key_cotangent[tuple(key_destination)][:, :, key_value_head, :] += d_key_expanded[:, :, query_head, :]
                value_cotangent[tuple(value_destination)][:, :, key_value_head, :] += d_value_expanded[
                    :, :, query_head, :
                ]
        query_destination = [slice(None)] * query_cotangent.ndim
        query_destination[query_value.axes.index(query_axis)] = slice(query_start, query_stop)
        query_cotangent[tuple(query_destination)] = query_gradient_tile
    return StreamingAttentionBackwardExecution(
        query_cotangent=query_cotangent,
        key_cotangent=key_cotangent,
        value_cotangent=value_cotangent,
    )


def _normalized_output_scale(forward: StreamingAttentionProgram) -> float:
    expression = forward.finalize.expression
    scale = 1.0
    if expression.kind is ScalarExpressionKind.MULTIPLY:
        left, right = expression.operands
        if left.constant is not None:
            scale = float(left.constant)
            expression = right
        elif right.constant is not None:
            scale = float(right.constant)
            expression = left
        else:
            raise ValueError("normalized-exp finalization scale must be a scalar constant")
    if expression.kind is not ScalarExpressionKind.DIVIDE:
        raise ValueError("streaming backward requires normalized weighted-value finalization")
    numerator, denominator = expression.operands
    if (
        numerator.input_name != forward.state.weighted_value_accumulator.name
        or denominator.input_name != forward.state.row_sum_exp.name
    ):
        raise ValueError("streaming backward finalization does not reference the derived Fold state")
    if not np.isfinite(scale):
        raise ValueError("streaming backward output scale must be finite")
    return scale
