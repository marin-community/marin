# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experimental attention-pattern selector over lossless generic StableHLO IR.

This module identifies one bounded attention-shaped training graph and then
regenerates a reverse with Shuttle's symbolic reference VJP. It is not an
accepted plugin frontend and does not establish JAX-owned reverse generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

import numpy as np

from tile_lifetime.ir import DType
from tile_lifetime.stablehlo_algebra_import import (
    ImportedContractNode,
    ImportedDomainRestrictionNode,
    ImportedFoldNode,
    ImportedMapNode,
    ImportedStableHLOAlgebra,
)
from tile_lifetime.stablehlo_import import CompareAttributes, ConstantAttributes, ReductionAttributes
from tile_lifetime.streaming_attention import (
    StreamingAttentionProgram,
    StreamingTileSchedule,
    derive_streaming_attention,
)
from tile_lifetime.streaming_attention_backward import (
    StreamingAttentionBackwardMaximumVJP,
    StreamingAttentionBackwardProgram,
    StreamingAttentionBackwardProvenance,
    derive_streaming_attention_backward,
)
from tile_lifetime.tensor_program import (
    AxisIndexMap,
    ContractPrimitive,
    FoldPrimitive,
    FoldReducer,
    MapPrimitive,
    ProgramValue,
    ScalarExpressionKind,
    TensorAxis,
    TensorProgram,
    scalar_binary,
    scalar_constant,
    scalar_input,
    scalar_select,
    scalar_unary,
)

SCALAR_LITERAL = re.compile(r"dense<([^>]+)>")
VIEW_KINDS = frozenset({"broadcast_in_dim", "convert", "reshape", "transpose"})


class ExperimentalStreamingScheduleError(ValueError):
    """A generic algebra graph does not match the experimental schedule."""


@dataclass(frozen=True)
class ExperimentalRecoveredStreamingAttentionTraining:
    """An attention-shaped source match plus regenerated streaming reverse."""

    algebra: ImportedStableHLOAlgebra
    program: StreamingAttentionBackwardProgram
    query: int
    key: int
    value: int
    output_cotangent: int
    forward_output: int
    query_cotangent: int
    key_cotangent: int
    value_cotangent: int
    score_scale: float
    contract_operation_ids: tuple[int, ...]
    normalized_exponential_fold_operation_ids: tuple[int, ...]
    maximum_vjp_tie_fold_operation_id: int
    broadcast_vjp_fold_operation_ids: tuple[int, ...]
    domain_restriction_operation_ids: tuple[int, ...]
    cast_and_view_operation_ids: tuple[int, ...]
    source_operation_ids: tuple[int, ...]


def select_experimental_streaming_attention_training_schedule(
    algebra: ImportedStableHLOAlgebra,
    *,
    schedule: StreamingTileSchedule,
) -> ExperimentalRecoveredStreamingAttentionTraining:
    """Select a bounded attention pattern and regenerate its reverse.

    Selection follows values and generic operation classes without a workload
    key or fixed global operation count. The selected reverse is still rebuilt
    by ``derive_streaming_attention_backward`` instead of imported op by op.
    """
    if len(algebra.inputs) != 4 or len(algebra.outputs) != 4:
        raise ExperimentalStreamingScheduleError("saved-state training selection requires four inputs and four outputs")

    normalized_candidates = []
    for exponential in _maps(algebra, "exponential"):
        try:
            center = _producer(algebra, exponential.inputs[0], ImportedMapNode, "subtract")
            maximum_path, mapped_score = _partition_inputs_by_ancestor_type(
                algebra,
                center.inputs,
                ImportedFoldNode,
            )
            maximum = _origin_through_views(algebra, maximum_path, ImportedFoldNode)
            probability = _single_consumer(algebra, exponential.outputs[0], ImportedMapNode, "divide")
            row_sum = _origin_through_views(algebra, probability.inputs[1], ImportedFoldNode)
            _require_fold(maximum, reducer="maximum", input_id=mapped_score)
            _require_fold(row_sum, reducer="add", input_id=exponential.outputs[0])
            if maximum.attributes.dimensions != row_sum.attributes.dimensions:
                continue
        except (ExperimentalStreamingScheduleError, AttributeError):
            continue
        normalized_candidates.append((exponential, center, mapped_score, maximum, row_sum, probability))
    if len(normalized_candidates) != 1:
        raise ExperimentalStreamingScheduleError(
            f"expected one structurally mergeable normalized-exponential subgraph, found {len(normalized_candidates)}"
        )
    exponential, _center, mapped_score, maximum, row_sum, probability = normalized_candidates[0]

    restriction = _producer(algebra, mapped_score, ImportedDomainRestrictionNode, "select")
    if len(restriction.inputs) != 3:
        raise ExperimentalStreamingScheduleError("domain restriction must have predicate, accepted value, and fill")
    compare = _single_ancestor(algebra, restriction.inputs[0], ImportedMapNode, "compare")
    if compare.attributes != CompareAttributes(direction="LE", compare_type="SIGNED"):
        raise ExperimentalStreamingScheduleError("streaming traversal currently requires an ordered <= predicate")
    if len(_ancestor_operations(algebra, compare.inputs, ImportedMapNode, source_kind="iota")) != 2:
        raise ExperimentalStreamingScheduleError("ordered domain predicate must originate at two iota maps")
    if not np.isneginf(_scalar_constant(algebra, restriction.inputs[2])):
        raise ExperimentalStreamingScheduleError("normalized-exponential domain fill must be negative infinity")

    score_scale = _producer(algebra, restriction.inputs[1], ImportedMapNode, "multiply")
    qk_path, scale_path = _partition_inputs_by_ancestor_type(algebra, score_scale.inputs, ImportedContractNode)
    qk = _single_ancestor(algebra, qk_path, ImportedContractNode, "dot_general")
    scale = _scalar_constant(algebra, scale_path)
    if not np.isfinite(scale):
        raise ExperimentalStreamingScheduleError("score scale must be finite")
    qk_inputs = tuple(_source_input_through_views(algebra, value_id) for value_id in qk.inputs)
    if len(set(qk_inputs)) != 2:
        raise ExperimentalStreamingScheduleError("first Contract operands do not originate at two source inputs")
    query, key = sorted(qk_inputs, key=lambda value_id: algebra.value(value_id).shape[2], reverse=True)
    query_shape = algebra.value(query).shape
    key_shape = algebra.value(key).shape
    if query_shape[2] <= key_shape[2] or query_shape[2] % key_shape[2]:
        raise ExperimentalStreamingScheduleError("selected Contract lacks a regular grouped-axis relation")

    remaining_inputs = tuple(value_id for value_id in algebra.inputs if value_id not in (query, key))
    value_candidates = tuple(
        value_id for value_id in remaining_inputs if algebra.value(value_id).shape[:3] == key_shape[:3]
    )
    cotangent_candidates = tuple(
        value_id for value_id in remaining_inputs if algebra.value(value_id).shape[:3] == query_shape[:3]
    )
    if len(value_candidates) != 1 or len(cotangent_candidates) != 1:
        raise ExperimentalStreamingScheduleError(
            "could not identify weighted value and output cotangent by dependencies"
        )
    value = value_candidates[0]
    output_cotangent = cotangent_candidates[0]

    dependencies = {output: _ancestor_input_ids(algebra, (output,)) for output in algebra.outputs}
    query_outputs = tuple(output for output in algebra.outputs if algebra.value(output).shape == query_shape)
    primal_dependencies = {query, key, value}
    reverse_dependencies = {*primal_dependencies, output_cotangent}
    forward_candidates = tuple(output for output in query_outputs if dependencies[output] == primal_dependencies)
    query_cotangent_candidates = tuple(
        output for output in query_outputs if dependencies[output] == reverse_dependencies
    )
    if len(forward_candidates) != 1 or len(query_cotangent_candidates) != 1:
        raise ExperimentalStreamingScheduleError("could not identify primal and query-cotangent outputs by dependencies")
    forward_output = forward_candidates[0]
    query_cotangent = query_cotangent_candidates[0]
    key_value_outputs = tuple(output for output in algebra.outputs if algebra.value(output).shape == key_shape)
    key_cotangent_candidates = tuple(output for output in key_value_outputs if value in dependencies[output])
    value_cotangent_candidates = tuple(output for output in key_value_outputs if value not in dependencies[output])
    if len(key_cotangent_candidates) != 1 or len(value_cotangent_candidates) != 1:
        raise ExperimentalStreamingScheduleError("could not identify key/value cotangents by dependencies")
    key_cotangent = key_cotangent_candidates[0]
    value_cotangent = value_cotangent_candidates[0]

    forward_pv = _latest_ancestor_contract(algebra, forward_output, exclude=qk.source_operation_id)
    reverse_terminal_contracts = tuple(
        _latest_ancestor_contract(algebra, output, exclude=qk.source_operation_id)
        for output in (value_cotangent, key_cotangent, query_cotangent)
    )
    assigned = {qk.source_operation_id, forward_pv.source_operation_id}
    assigned.update(operation.source_operation_id for operation in reverse_terminal_contracts)
    d_probability_candidates = tuple(
        operation
        for operation in _ancestor_operations(algebra, (query_cotangent, key_cotangent), ImportedContractNode)
        if operation.source_operation_id not in assigned
        and _ancestor_input_ids(algebra, operation.inputs) == {value, output_cotangent}
    )
    if len(d_probability_candidates) != 1:
        raise ExperimentalStreamingScheduleError("could not identify one probability cotangent Contract structurally")
    d_probability = d_probability_candidates[0]

    tie_compares = tuple(
        operation
        for operation in _maps(algebra, "compare")
        if operation.attributes == CompareAttributes(direction="EQ", compare_type="FLOAT")
        and mapped_score in _ancestor_values(algebra, operation.inputs)
        and maximum.outputs[0] in _ancestor_values(algebra, operation.inputs)
    )
    if len(tie_compares) != 1:
        raise ExperimentalStreamingScheduleError("could not identify one source maximum-tie comparison")
    tie_fold = _single_descendant_fold(algebra, tie_compares[0].outputs[0])
    broadcast_folds = tuple(
        operation
        for operation in _ancestor_operations(
            algebra,
            (key_cotangent, value_cotangent),
            ImportedFoldNode,
        )
        if operation.source_operation_id
        not in {
            maximum.source_operation_id,
            row_sum.source_operation_id,
            tie_fold.source_operation_id,
        }
        and len(operation.attributes.dimensions) > 1
    )
    if not broadcast_folds:
        raise ExperimentalStreamingScheduleError("grouped-axis reverse has no source broadcast-adjoint Fold")

    forward = _build_streaming_program_from_source_values(
        algebra,
        query=query,
        key=key,
        value=value,
        score_scale=scale,
        schedule=schedule,
        accumulation_dtype=algebra.value(qk.outputs[0]).dtype,
        output_dtype=algebra.value(forward_output).dtype,
    )
    backward = replace(
        derive_streaming_attention_backward(forward),
        provenance=StreamingAttentionBackwardProvenance.EXPERIMENTAL_REGENERATED_REVERSE_FROM_JAX_VJP_ALGEBRA,
        maximum_vjp=StreamingAttentionBackwardMaximumVJP.JAX_EQUAL_SPLIT,
    )
    source_operation_ids = tuple(operation.source_operation_id for operation in algebra.operations)
    cast_and_view_operation_ids = tuple(
        operation.source_operation_id for operation in algebra.operations if operation.source_kind in VIEW_KINDS
    )
    return ExperimentalRecoveredStreamingAttentionTraining(
        algebra=algebra,
        program=backward,
        query=query,
        key=key,
        value=value,
        output_cotangent=output_cotangent,
        forward_output=forward_output,
        query_cotangent=query_cotangent,
        key_cotangent=key_cotangent,
        value_cotangent=value_cotangent,
        score_scale=scale,
        contract_operation_ids=(
            qk.source_operation_id,
            forward_pv.source_operation_id,
            d_probability.source_operation_id,
            *(operation.source_operation_id for operation in reverse_terminal_contracts),
        ),
        normalized_exponential_fold_operation_ids=(
            maximum.source_operation_id,
            row_sum.source_operation_id,
        ),
        maximum_vjp_tie_fold_operation_id=tie_fold.source_operation_id,
        broadcast_vjp_fold_operation_ids=tuple(operation.source_operation_id for operation in broadcast_folds),
        domain_restriction_operation_ids=(compare.source_operation_id, restriction.source_operation_id),
        cast_and_view_operation_ids=cast_and_view_operation_ids,
        source_operation_ids=source_operation_ids,
    )


def _build_streaming_program_from_source_values(
    algebra: ImportedStableHLOAlgebra,
    *,
    query: int,
    key: int,
    value: int,
    score_scale: float,
    schedule: StreamingTileSchedule,
    accumulation_dtype: DType,
    output_dtype: DType,
) -> StreamingAttentionProgram:
    query_shape = algebra.value(query).shape
    key_shape = algebra.value(key).shape
    value_shape = algebra.value(value).shape
    batch = TensorAxis(0, query_shape[0], "batch")
    query_token = TensorAxis(1, query_shape[1], "query")
    key_token = TensorAxis(2, key_shape[1], "key")
    query_head = TensorAxis(3, query_shape[2], "head")
    key_value_head = TensorAxis(4, key_shape[2], "key_value_head")
    key_feature = TensorAxis(5, query_shape[3], "key_feature")
    value_feature = TensorAxis(6, value_shape[3], "value_feature")
    score_axes = (batch, query_head, query_token, key_token)
    row_axes = score_axes[:-1]
    query_value = ProgramValue("query", (batch, query_token, query_head, key_feature), algebra.value(query).dtype)
    key_value = ProgramValue("key", (batch, key_token, key_value_head, key_feature), algebra.value(key).dtype)
    weighted_value_input = ProgramValue(
        "value",
        (batch, key_token, key_value_head, value_feature),
        algebra.value(value).dtype,
    )
    raw_score = ProgramValue("score.raw", score_axes, accumulation_dtype)
    mapped_score = ProgramValue("score.mapped", score_axes, accumulation_dtype)
    row_max = ProgramValue("score.row_max", row_axes, accumulation_dtype)
    centered_score = ProgramValue("score.centered", score_axes, accumulation_dtype)
    exponentials = ProgramValue("score.exponential", score_axes, accumulation_dtype)
    row_sum = ProgramValue("score.row_sum_exp", row_axes, accumulation_dtype)
    weighted_value = ProgramValue(
        "normalized_weighted_reduction.accumulator",
        (batch, query_token, query_head, value_feature),
        accumulation_dtype,
    )
    output = ProgramValue("normalized_weighted_reduction.output", weighted_value.axes, output_dtype)
    query_position = ProgramValue("query.position", (query_token,), DType.INT32)
    key_position = ProgramValue("key.position", (key_token,), DType.INT32)
    head_map = AxisIndexMap(
        domain_axis=query_head,
        operand_axis=key_value_head,
        divisor=query_shape[2] // key_shape[2],
    )
    score_expression = scalar_select(
        scalar_binary(
            ScalarExpressionKind.LESS_EQUAL,
            scalar_input(key_position.name),
            scalar_input(query_position.name),
        ),
        scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            scalar_input(raw_score.name),
            scalar_constant(score_scale),
        ),
        scalar_constant(float("-inf")),
    )
    operations = (
        ContractPrimitive(
            "first_contract",
            (query_value, key_value),
            raw_score,
            (key_feature,),
            accumulation_dtype,
            ((), (head_map,)),
        ),
        MapPrimitive("score_map", (raw_score, query_position, key_position), mapped_score, score_expression),
        FoldPrimitive(
            "normalized_exp_max", mapped_score, row_max, (key_token,), FoldReducer.MAXIMUM, accumulation_dtype
        ),
        MapPrimitive(
            "center",
            (mapped_score, row_max),
            centered_score,
            scalar_binary(
                ScalarExpressionKind.SUBTRACT,
                scalar_input(mapped_score.name),
                scalar_input(row_max.name),
            ),
        ),
        MapPrimitive(
            "exponential",
            (centered_score,),
            exponentials,
            scalar_unary(ScalarExpressionKind.EXP, scalar_input(centered_score.name)),
        ),
        FoldPrimitive("normalized_exp_sum", exponentials, row_sum, (key_token,), FoldReducer.SUM, accumulation_dtype),
        ContractPrimitive(
            "weighted_contract",
            (exponentials, weighted_value_input),
            weighted_value,
            (key_token,),
            accumulation_dtype,
            ((), (head_map,)),
        ),
        MapPrimitive(
            "normalize",
            (weighted_value, row_sum),
            output,
            scalar_binary(
                ScalarExpressionKind.DIVIDE,
                scalar_input(weighted_value.name),
                scalar_input(row_sum.name),
            ),
        ),
    )
    source = TensorProgram(
        inputs=(query_value, key_value, weighted_value_input, query_position, key_position),
        operations=operations,
        outputs=(output,),
    )
    return derive_streaming_attention(source, schedule=schedule)


def _maps(algebra: ImportedStableHLOAlgebra, source_kind: str) -> tuple[ImportedMapNode, ...]:
    return tuple(
        operation
        for operation in algebra.operations
        if isinstance(operation, ImportedMapNode) and operation.source_kind == source_kind
    )


def _producer(algebra, value_id, expected_type, source_kind):
    operation = algebra.producer(value_id)
    if not isinstance(operation, expected_type) or operation.source_kind != source_kind:
        found = "input" if operation is None else f"{type(operation).__name__}/{operation.source_kind}"
        raise ExperimentalStreamingScheduleError(f"expected {expected_type.__name__}/{source_kind}, found {found}")
    return operation


def _single_consumer(algebra, value_id, expected_type, source_kind):
    matches = tuple(
        operation
        for operation in algebra.consumers(value_id)
        if isinstance(operation, expected_type) and operation.source_kind == source_kind
    )
    if len(matches) != 1:
        raise ExperimentalStreamingScheduleError(f"expected one {source_kind} consumer, found {len(matches)}")
    return matches[0]


def _origin_through_views(algebra, value_id, expected_type):
    current = value_id
    while True:
        operation = algebra.producer(current)
        if isinstance(operation, expected_type):
            return operation
        if (
            not isinstance(operation, ImportedMapNode)
            or operation.source_kind not in VIEW_KINDS
            or len(operation.inputs) != 1
        ):
            raise ExperimentalStreamingScheduleError(f"expected {expected_type.__name__} through source views")
        current = operation.inputs[0]


def _partition_inputs_by_ancestor_type(algebra, inputs, expected_type):
    matches = tuple(bool(_ancestor_operations(algebra, (value_id,), expected_type)) for value_id in inputs)
    if len(matches) != 2 or matches.count(True) != 1:
        raise ExperimentalStreamingScheduleError(f"could not partition operands by {expected_type.__name__} ancestry")
    index = matches.index(True)
    return inputs[index], inputs[1 - index]


def _single_ancestor(algebra, value_id, expected_type, source_kind):
    matches = tuple(
        operation
        for operation in _ancestor_operations(algebra, (value_id,), expected_type)
        if operation.source_kind == source_kind
    )
    if len(matches) != 1:
        raise ExperimentalStreamingScheduleError(f"expected one ancestor {source_kind}, found {len(matches)}")
    return matches[0]


def _ancestor_operations(algebra, roots, expected_type, *, source_kind=None):
    ids = _ancestor_operation_ids(algebra, roots)
    return tuple(
        operation
        for operation in algebra.operations
        if operation.source_operation_id in ids
        and isinstance(operation, expected_type)
        and (source_kind is None or operation.source_kind == source_kind)
    )


def _ancestor_operation_ids(algebra, roots):
    result = set()
    pending = list(roots)
    visited = set()
    while pending:
        value_id = pending.pop()
        if value_id in visited:
            continue
        visited.add(value_id)
        operation = algebra.producer(value_id)
        if operation is not None:
            result.add(operation.source_operation_id)
            pending.extend(operation.inputs)
    return frozenset(result)


def _ancestor_values(algebra, roots):
    result = set()
    pending = list(roots)
    while pending:
        value_id = pending.pop()
        if value_id in result:
            continue
        result.add(value_id)
        operation = algebra.producer(value_id)
        if operation is not None:
            pending.extend(operation.inputs)
    return frozenset(result)


def _ancestor_input_ids(algebra, roots):
    return set(algebra.inputs).intersection(_ancestor_values(algebra, roots))


def _source_input_through_views(algebra, value_id):
    current = value_id
    while current not in algebra.inputs:
        operation = algebra.producer(current)
        if (
            not isinstance(operation, ImportedMapNode)
            or operation.source_kind not in VIEW_KINDS
            or len(operation.inputs) != 1
        ):
            raise ExperimentalStreamingScheduleError(
                "Contract operand does not originate at an input through source views"
            )
        current = operation.inputs[0]
    return current


def _require_fold(operation, *, reducer, input_id):
    attributes = operation.attributes
    if (
        not isinstance(attributes, ReductionAttributes)
        or attributes.reducer != reducer
        or operation.inputs[0] != input_id
    ):
        raise ExperimentalStreamingScheduleError(f"expected source {reducer} Fold over the selected value")


def _scalar_constant(algebra, value_id):
    constants = _ancestor_operations(algebra, (value_id,), ImportedMapNode, source_kind="constant")
    if len(constants) != 1 or not isinstance(constants[0].attributes, ConstantAttributes):
        raise ExperimentalStreamingScheduleError("expected one source scalar constant")
    match = SCALAR_LITERAL.search(constants[0].attributes.literal)
    if match is None:
        raise ExperimentalStreamingScheduleError("source constant is not a supported dense literal")
    literal = match.group(1)
    if literal.startswith("0xFF800000"):
        return float("-inf")
    try:
        return float(literal)
    except ValueError as error:
        raise ExperimentalStreamingScheduleError(f"unsupported source scalar literal {literal!r}") from error


def _latest_ancestor_contract(algebra, value_id, *, exclude):
    operations = tuple(
        operation
        for operation in _ancestor_operations(algebra, (value_id,), ImportedContractNode)
        if operation.source_operation_id != exclude
    )
    if not operations:
        raise ExperimentalStreamingScheduleError("selected output has no downstream Contract")
    return max(operations, key=lambda operation: operation.source_index)


def _single_descendant_fold(algebra, value_id):
    pending = [value_id]
    visited = set()
    folds = []
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        for consumer in algebra.consumers(current):
            if isinstance(consumer, ImportedFoldNode):
                folds.append(consumer)
            elif isinstance(consumer, ImportedMapNode) and consumer.source_kind in VIEW_KINDS:
                pending.extend(consumer.outputs)
    if len(folds) != 1:
        raise ExperimentalStreamingScheduleError(f"expected one descendant Fold, found {len(folds)}")
    return folds[0]
