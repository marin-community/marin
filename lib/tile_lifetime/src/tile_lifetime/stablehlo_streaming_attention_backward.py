# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover generic streaming reverse algebra from a JAX-owned attention VJP."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

import numpy as np

from tile_lifetime.stablehlo_import import (
    CompareAttributes,
    ConstantAttributes,
    ReductionAttributes,
    StableHLOGraph,
    StableHLOOperation,
)
from tile_lifetime.streaming_attention import (
    StreamingTileSchedule,
    apply_causal_score_mask,
    build_attention_tensor_program,
    derive_streaming_attention,
    scaled_score_map,
)
from tile_lifetime.streaming_attention_backward import (
    StreamingAttentionBackwardMaximumVJP,
    StreamingAttentionBackwardProgram,
    StreamingAttentionBackwardProvenance,
    derive_streaming_attention_backward,
)

SCALAR_LITERAL = re.compile(r"dense<([^>]+)>")
VIEW_KINDS = frozenset({"broadcast_in_dim", "convert", "reshape", "transpose"})


class StableHLOStreamingAttentionBackwardError(ValueError):
    """A structured mismatch in a candidate JAX-differentiated tensor graph."""

    def __init__(self, stage: str, reason: str, *, operation_ids: tuple[int, ...] = ()):
        self.stage = stage
        self.reason = reason
        self.operation_ids = operation_ids
        super().__init__(f"{stage}: {reason}")


@dataclass(frozen=True)
class RecoveredStableHLOStreamingAttentionBackward:
    """Visible generic reverse program and its source-HLO role assignment."""

    graph: StableHLOGraph
    program: StreamingAttentionBackwardProgram
    query: int
    key: int
    value: int
    output_cotangent: int
    forward_output: int | None
    query_cotangent: int
    key_cotangent: int
    value_cotangent: int
    score_scale: float
    contract_operation_ids: tuple[int, ...]
    normalized_exponential_fold_operation_ids: tuple[int, ...]
    maximum_vjp_tie_fold_operation_id: int
    broadcast_vjp_fold_operation_ids: tuple[int, ...]
    domain_restriction_operation_ids: tuple[int, ...]
    source_operation_ids: tuple[int, ...]


def recover_stablehlo_streaming_attention_backward(
    graph: StableHLOGraph,
    *,
    schedule: StreamingTileSchedule,
) -> RecoveredStableHLOStreamingAttentionBackward:
    """Recover a generic Contract/Map/Fold reverse from ordinary JAX VJP HLO.

    JAX remains the automatic-differentiation owner. This pass assigns generic
    roles to the generic forward/reverse Contracts and normalized-exponential Folds;
    it never recognizes or dispatches on an attention/model name.
    """
    if len(graph.inputs) != 4 or len(graph.outputs) not in (3, 4):
        raise StableHLOStreamingAttentionBackwardError(
            "signature",
            "expected four primal/cotangent inputs and either three cotangents or forward output plus cotangents",
        )
    if any(len(graph.value(value_id).shape) != 4 for value_id in (*graph.inputs, *graph.outputs)):
        raise StableHLOStreamingAttentionBackwardError("signature", "all inputs and outputs must be rank four")

    exponentials = _operations(graph, "exponential")
    if len(exponentials) != 1:
        raise StableHLOStreamingAttentionBackwardError(
            "normalized_exponential",
            f"expected one exponential Map, found {len(exponentials)}",
        )
    exponential = exponentials[0]
    center = _producer(graph, exponential.inputs[0], "subtract", "normalized_exponential")
    maximum_path_id, masked_score_id = _partition_inputs_by_ancestor_kind(
        graph,
        center,
        kind="reduce",
        stage="normalized_exponential",
    )
    maximum = _origin_operation_through_views(graph, maximum_path_id, "reduce", "normalized_exponential")
    _require_reduction(graph, maximum, reducer="maximum", input_id=masked_score_id, stage="normalized_exponential")

    probability = _single_consumer(graph, exponential.outputs[0], "divide", "normalized_exponential")
    row_sum = _origin_operation_through_views(graph, probability.inputs[1], "reduce", "normalized_exponential")
    _require_reduction(
        graph,
        row_sum,
        reducer="add",
        input_id=exponential.outputs[0],
        stage="normalized_exponential",
    )
    if maximum.attributes.dimensions != row_sum.attributes.dimensions:
        raise StableHLOStreamingAttentionBackwardError(
            "normalized_exponential",
            "maximum and sum Folds use different domains",
            operation_ids=(maximum.id, row_sum.id),
        )

    score_select = _producer(graph, masked_score_id, "select", "domain_restriction")
    if len(score_select.inputs) != 3:
        raise StableHLOStreamingAttentionBackwardError(
            "domain_restriction",
            "score selection requires predicate, true, and false operands",
            operation_ids=(score_select.id,),
        )
    causal_compare = _single_ancestor_operation(graph, score_select.inputs[0], "compare", "domain_restriction")
    if causal_compare.attributes != CompareAttributes(direction="LE", compare_type="SIGNED"):
        raise StableHLOStreamingAttentionBackwardError(
            "domain_restriction",
            "predicate must compare key_position <= query_position",
            operation_ids=(causal_compare.id,),
        )
    if len(_ancestor_operations_of_kind(graph, causal_compare.inputs, "iota")) != 2:
        raise StableHLOStreamingAttentionBackwardError(
            "domain_restriction",
            "causal predicate must originate at two logical position iotas",
            operation_ids=(causal_compare.id,),
        )
    _require_negative_infinity(graph, score_select.inputs[2], stage="domain_restriction")

    score_scale = _producer(graph, score_select.inputs[1], "multiply", "score_map")
    qk_path, scale_path = _partition_inputs_by_ancestor_kind(graph, score_scale, kind="dot_general", stage="score_map")
    qk = _single_ancestor_operation(graph, qk_path, "dot_general", "qk_contract")
    scale = _scalar_constant(graph, scale_path, stage="score_map")
    if not np.isfinite(scale):
        raise StableHLOStreamingAttentionBackwardError("score_map", "score scale must be finite")

    qk_inputs = tuple(_source_input_through_views(graph, value_id, "qk_contract") for value_id in qk.inputs)
    if len(set(qk_inputs)) != 2:
        raise StableHLOStreamingAttentionBackwardError(
            "qk_contract",
            "QK operands do not originate at two distinct function inputs",
            operation_ids=(qk.id,),
        )
    ordered_qk_inputs = sorted(qk_inputs, key=lambda value_id: graph.value(value_id).shape[2], reverse=True)
    query, key = ordered_qk_inputs
    query_shape = graph.value(query).shape
    key_shape = graph.value(key).shape
    if query_shape[0] != key_shape[0] or query_shape[-1] != key_shape[-1]:
        raise StableHLOStreamingAttentionBackwardError("qk_contract", "Q/K batch or feature dimensions differ")
    if query_shape[2] <= key_shape[2] or query_shape[2] % key_shape[2]:
        raise StableHLOStreamingAttentionBackwardError(
            "grouped_head_relation",
            "the first reverse importer requires query heads grouped over fewer K/V heads",
            operation_ids=(qk.id,),
        )

    remaining_inputs = tuple(value_id for value_id in graph.inputs if value_id not in (query, key))
    value_candidates = tuple(
        value_id for value_id in remaining_inputs if graph.value(value_id).shape[:3] == key_shape[:3]
    )
    cotangent_candidates = tuple(
        value_id for value_id in remaining_inputs if graph.value(value_id).shape[:3] == query_shape[:3]
    )
    if len(value_candidates) != 1 or len(cotangent_candidates) != 1:
        raise StableHLOStreamingAttentionBackwardError(
            "signature",
            "could not distinguish value and output-cotangent inputs from the GQA index relation",
        )
    value = value_candidates[0]
    output_cotangent = cotangent_candidates[0]
    value_shape = graph.value(value).shape
    if value_shape[:3] != key_shape[:3]:
        raise StableHLOStreamingAttentionBackwardError("signature", "K/V batch, token, or head axes differ")

    output_dependencies = {output_id: _ancestor_input_ids(graph, (output_id,)) for output_id in graph.outputs}
    query_outputs = tuple(output_id for output_id in graph.outputs if graph.value(output_id).shape == query_shape)
    expected_cotangent_dependencies = {query, key, value, output_cotangent}
    query_cotangent_candidates = tuple(
        output_id for output_id in query_outputs if output_dependencies[output_id] == expected_cotangent_dependencies
    )
    forward_output_candidates = tuple(
        output_id for output_id in query_outputs if output_dependencies[output_id] == {query, key, value}
    )
    if len(query_cotangent_candidates) != 1:
        raise StableHLOStreamingAttentionBackwardError(
            "cotangent_outputs",
            "could not identify one query cotangent by its generic data dependencies",
        )
    if len(graph.outputs) == 4 and len(forward_output_candidates) != 1:
        raise StableHLOStreamingAttentionBackwardError(
            "forward_output",
            "could not identify one forward output independent of the output cotangent",
        )
    if len(graph.outputs) == 3 and forward_output_candidates:
        raise StableHLOStreamingAttentionBackwardError(
            "forward_output",
            "reverse-only boundary unexpectedly returns a forward output",
        )
    query_cotangent = query_cotangent_candidates[0]
    forward_output = forward_output_candidates[0] if forward_output_candidates else None
    key_value_outputs = tuple(output_id for output_id in graph.outputs if graph.value(output_id).shape == key_shape)
    if len(key_value_outputs) != 2:
        raise StableHLOStreamingAttentionBackwardError(
            "cotangent_outputs",
            "expected two K/V-shaped cotangent outputs",
        )
    key_cotangent_candidates = tuple(
        output_id for output_id in key_value_outputs if value in _ancestor_input_ids(graph, (output_id,))
    )
    value_cotangent_candidates = tuple(
        output_id for output_id in key_value_outputs if value not in _ancestor_input_ids(graph, (output_id,))
    )
    if len(key_cotangent_candidates) != 1 or len(value_cotangent_candidates) != 1:
        raise StableHLOStreamingAttentionBackwardError(
            "cotangent_outputs",
            "could not distinguish K and V cotangents by their generic data dependencies",
        )
    key_cotangent = key_cotangent_candidates[0]
    value_cotangent = value_cotangent_candidates[0]
    for output_id, role in ((query_cotangent, "query"), (key_cotangent, "key")):
        if output_dependencies[output_id] != expected_cotangent_dependencies:
            raise StableHLOStreamingAttentionBackwardError(
                "cotangent_outputs",
                f"{role} cotangent omits a primal or output-cotangent dependency",
            )
    if output_dependencies[value_cotangent] != {query, key, output_cotangent}:
        raise StableHLOStreamingAttentionBackwardError(
            "cotangent_outputs",
            "value cotangent has unexpected data dependencies",
        )

    dots = _operations(graph, "dot_general")
    expected_dot_count = 6 if forward_output is not None else 5
    if len(dots) != expected_dot_count:
        raise StableHLOStreamingAttentionBackwardError(
            "contracts",
            f"expected {expected_dot_count} forward/reverse Contracts, found {len(dots)}",
            operation_ids=tuple(operation.id for operation in dots),
        )
    terminal_dots = tuple(
        _latest_ancestor_dot(graph, output_id, exclude=qk.id)
        for output_id in (value_cotangent, key_cotangent, query_cotangent)
    )
    if len({operation.id for operation in terminal_dots}) != 3:
        raise StableHLOStreamingAttentionBackwardError("contracts", "cotangent outputs do not have distinct Contracts")
    forward_pv = _latest_ancestor_dot(graph, forward_output, exclude=qk.id) if forward_output is not None else None
    assigned_dot_ids = {qk.id, *(operation.id for operation in terminal_dots)}
    if forward_pv is not None:
        assigned_dot_ids.add(forward_pv.id)
    remaining_dots = tuple(operation for operation in dots if operation.id not in assigned_dot_ids)
    if len(remaining_dots) != 1 or _ancestor_input_ids(graph, remaining_dots[0].inputs) != {
        value,
        output_cotangent,
    }:
        raise StableHLOStreamingAttentionBackwardError(
            "contracts",
            "the fourth reverse Contract is not the output-cotangent/value contraction",
        )
    d_probability = remaining_dots[0]

    tie_compares = tuple(
        operation
        for operation in _operations(graph, "compare")
        if operation.attributes == CompareAttributes(direction="EQ", compare_type="FLOAT")
        and masked_score_id in _ancestor_values(graph, operation.inputs)
        and maximum.outputs[0] in _ancestor_values(graph, operation.inputs)
    )
    if len(tie_compares) != 1:
        raise StableHLOStreamingAttentionBackwardError(
            "maximum_vjp",
            f"expected one maximum-tie comparison, found {len(tie_compares)}",
        )
    tie_fold = _single_descendant_reduction(graph, tie_compares[0].outputs[0], stage="maximum_vjp")
    _require_reducer(tie_fold, "add", "maximum_vjp")
    broadcast_vjp_folds = tuple(
        operation
        for operation in _operations(graph, "reduce")
        if operation.id not in {maximum.id, row_sum.id, tie_fold.id} and len(operation.attributes.dimensions) > 1
    )
    if len(broadcast_vjp_folds) != 2:
        raise StableHLOStreamingAttentionBackwardError(
            "grouped_head_relation",
            f"expected two broadcast-adjoint Folds, found {len(broadcast_vjp_folds)}",
            operation_ids=tuple(operation.id for operation in broadcast_vjp_folds),
        )

    tensor_program = build_attention_tensor_program(
        batch_size=query_shape[0],
        query_length=query_shape[1],
        key_length=key_shape[1],
        query_heads=query_shape[2],
        key_value_heads=key_shape[2],
        key_dimension=query_shape[3],
        value_dimension=value_shape[3],
        score_map=apply_causal_score_mask(scaled_score_map(scale)),
        input_dtype=graph.value(query).dtype,
        accumulation_dtype=graph.value(qk.outputs[0]).dtype,
    )
    forward = derive_streaming_attention(tensor_program, schedule=schedule)
    backward = replace(
        derive_streaming_attention_backward(forward),
        provenance=StreamingAttentionBackwardProvenance.JAX_VJP_HLO_RECOVERY,
        maximum_vjp=StreamingAttentionBackwardMaximumVJP.JAX_EQUAL_SPLIT,
    )
    source_operation_ids = tuple(sorted(_ancestor_operation_ids(graph, graph.outputs)))
    if source_operation_ids != tuple(operation.id for operation in graph.operations):
        raise StableHLOStreamingAttentionBackwardError(
            "coverage",
            "differentiated graph contains operations outside the recovered outputs",
        )
    return RecoveredStableHLOStreamingAttentionBackward(
        graph=graph,
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
            qk.id,
            *((forward_pv.id,) if forward_pv is not None else ()),
            d_probability.id,
            *(operation.id for operation in terminal_dots),
        ),
        normalized_exponential_fold_operation_ids=(maximum.id, row_sum.id),
        maximum_vjp_tie_fold_operation_id=tie_fold.id,
        broadcast_vjp_fold_operation_ids=tuple(operation.id for operation in broadcast_vjp_folds),
        domain_restriction_operation_ids=(causal_compare.id, score_select.id),
        source_operation_ids=source_operation_ids,
    )


def _operations(graph: StableHLOGraph, kind: str) -> tuple[StableHLOOperation, ...]:
    return tuple(operation for operation in graph.operations if operation.kind == kind)


def _producer(graph: StableHLOGraph, value_id: int, kind: str, stage: str) -> StableHLOOperation:
    operation = graph.producer(value_id)
    if operation is None or operation.kind != kind:
        found = "input" if operation is None else operation.kind
        raise StableHLOStreamingAttentionBackwardError(stage, f"expected {kind}, found {found}")
    return operation


def _single_consumer(graph: StableHLOGraph, value_id: int, kind: str, stage: str) -> StableHLOOperation:
    consumers = tuple(operation for operation in graph.consumers(value_id) if operation.kind == kind)
    if len(consumers) != 1:
        raise StableHLOStreamingAttentionBackwardError(stage, f"expected one {kind} consumer, found {len(consumers)}")
    return consumers[0]


def _partition_inputs_by_ancestor_kind(
    graph: StableHLOGraph,
    operation: StableHLOOperation,
    *,
    kind: str,
    stage: str,
) -> tuple[int, int]:
    matches = tuple(bool(_ancestor_operations_of_kind(graph, (value_id,), kind)) for value_id in operation.inputs)
    if len(matches) != 2 or matches.count(True) != 1:
        raise StableHLOStreamingAttentionBackwardError(
            stage,
            f"could not distinguish {kind}-derived and non-{kind}-derived operands",
            operation_ids=(operation.id,),
        )
    index = matches.index(True)
    return operation.inputs[index], operation.inputs[1 - index]


def _single_ancestor_operation(graph: StableHLOGraph, value_id: int, kind: str, stage: str) -> StableHLOOperation:
    operations = _ancestor_operations_of_kind(graph, (value_id,), kind)
    if len(operations) != 1:
        raise StableHLOStreamingAttentionBackwardError(
            stage,
            f"expected one ancestor {kind}, found {len(operations)}",
            operation_ids=tuple(operation.id for operation in operations),
        )
    return operations[0]


def _origin_operation_through_views(
    graph: StableHLOGraph,
    value_id: int,
    kind: str,
    stage: str,
) -> StableHLOOperation:
    current = value_id
    while True:
        operation = graph.producer(current)
        if operation is None:
            raise StableHLOStreamingAttentionBackwardError(stage, f"expected {kind}, reached a function input")
        if operation.kind == kind:
            return operation
        if operation.kind not in VIEW_KINDS or len(operation.inputs) != 1:
            raise StableHLOStreamingAttentionBackwardError(
                stage,
                f"expected {kind} through views, found {operation.kind}",
                operation_ids=(operation.id,),
            )
        current = operation.inputs[0]


def _ancestor_operations_of_kind(
    graph: StableHLOGraph,
    roots: tuple[int, ...],
    kind: str,
) -> tuple[StableHLOOperation, ...]:
    ids = _ancestor_operation_ids(graph, roots)
    return tuple(operation for operation in graph.operations if operation.id in ids and operation.kind == kind)


def _ancestor_operation_ids(graph: StableHLOGraph, roots: tuple[int, ...]) -> frozenset[int]:
    operation_ids: set[int] = set()
    pending = list(roots)
    visited: set[int] = set()
    while pending:
        value_id = pending.pop()
        if value_id in visited:
            continue
        visited.add(value_id)
        producer = graph.producer(value_id)
        if producer is None:
            continue
        operation_ids.add(producer.id)
        pending.extend(producer.inputs)
    return frozenset(operation_ids)


def _ancestor_values(graph: StableHLOGraph, roots: tuple[int, ...]) -> frozenset[int]:
    values: set[int] = set()
    pending = list(roots)
    while pending:
        value_id = pending.pop()
        if value_id in values:
            continue
        values.add(value_id)
        producer = graph.producer(value_id)
        if producer is not None:
            pending.extend(producer.inputs)
    return frozenset(values)


def _ancestor_input_ids(graph: StableHLOGraph, roots: tuple[int, ...]) -> set[int]:
    return set(graph.inputs).intersection(_ancestor_values(graph, roots))


def _source_input_through_views(graph: StableHLOGraph, value_id: int, stage: str) -> int:
    current = value_id
    while current not in graph.inputs:
        operation = graph.producer(current)
        if operation is None or operation.kind not in VIEW_KINDS or len(operation.inputs) != 1:
            found = "no producer" if operation is None else operation.kind
            raise StableHLOStreamingAttentionBackwardError(stage, f"operand view path stops at {found}")
        current = operation.inputs[0]
    return current


def _require_reduction(
    graph: StableHLOGraph,
    operation: StableHLOOperation,
    *,
    reducer: str,
    input_id: int,
    stage: str,
) -> None:
    _require_reducer(operation, reducer, stage)
    if operation.inputs[0] != input_id:
        raise StableHLOStreamingAttentionBackwardError(
            stage,
            f"{reducer} Fold does not consume the expected value",
            operation_ids=(operation.id,),
        )
    input_rank = len(graph.value(input_id).shape)
    if operation.attributes.dimensions != (input_rank - 1,):
        raise StableHLOStreamingAttentionBackwardError(
            stage,
            f"{reducer} Fold does not reduce the final logical key axis",
            operation_ids=(operation.id,),
        )


def _require_reducer(operation: StableHLOOperation, reducer: str, stage: str) -> None:
    attributes = operation.attributes
    if not isinstance(attributes, ReductionAttributes) or attributes.reducer != reducer:
        raise StableHLOStreamingAttentionBackwardError(
            stage,
            f"expected a {reducer} Fold",
            operation_ids=(operation.id,),
        )


def _require_negative_infinity(graph: StableHLOGraph, value_id: int, *, stage: str) -> None:
    value = _scalar_constant(graph, value_id, stage=stage)
    if not np.isneginf(value):
        raise StableHLOStreamingAttentionBackwardError(stage, "domain restriction fill must be negative infinity")


def _scalar_constant(graph: StableHLOGraph, value_id: int, *, stage: str) -> float:
    constants = _ancestor_operations_of_kind(graph, (value_id,), "constant")
    if len(constants) != 1 or not isinstance(constants[0].attributes, ConstantAttributes):
        raise StableHLOStreamingAttentionBackwardError(stage, "expected one scalar constant operand")
    match = SCALAR_LITERAL.search(constants[0].attributes.literal)
    if match is None:
        raise StableHLOStreamingAttentionBackwardError(stage, "constant does not use a supported dense literal")
    literal = match.group(1)
    if literal.startswith("0xFF800000"):
        return float("-inf")
    try:
        return float(literal)
    except ValueError as error:
        raise StableHLOStreamingAttentionBackwardError(stage, f"unsupported scalar literal {literal!r}") from error


def _latest_ancestor_dot(graph: StableHLOGraph, value_id: int, *, exclude: int) -> StableHLOOperation:
    operations = tuple(
        operation
        for operation in _ancestor_operations_of_kind(graph, (value_id,), "dot_general")
        if operation.id != exclude
    )
    if not operations:
        raise StableHLOStreamingAttentionBackwardError("contracts", "cotangent output lacks a reverse Contract")
    return max(operations, key=lambda operation: operation.id)


def _single_descendant_reduction(graph: StableHLOGraph, value_id: int, *, stage: str) -> StableHLOOperation:
    pending = [value_id]
    visited: set[int] = set()
    reductions: list[StableHLOOperation] = []
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        for consumer in graph.consumers(current):
            if consumer.kind == "reduce":
                reductions.append(consumer)
            elif consumer.kind in VIEW_KINDS:
                pending.extend(consumer.outputs)
    if len(reductions) != 1:
        raise StableHLOStreamingAttentionBackwardError(
            stage,
            f"expected one tie-count Fold, found {len(reductions)}",
            operation_ids=tuple(operation.id for operation in reductions),
        )
    return reductions[0]
