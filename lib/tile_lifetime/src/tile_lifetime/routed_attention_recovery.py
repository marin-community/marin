# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover natural routed selected-attention StableHLO into generic algebra."""

import re
from dataclasses import dataclass

import numpy as np

from shuttle.stablehlo_import import (
    CompareAttributes,
    CompositeAttributes,
    ConstantAttributes,
    DotAttributes,
    GatherAttributes,
    ReductionAttributes,
    StableHLOGraph,
    StableHLOOperation,
)
from tile_lifetime.plan import SemanticErasureReport, SemanticLoweringStep
from tile_lifetime.relation import RelationPlan
from tile_lifetime.routed_attention import (
    IndexDomainRestriction,
    RelationSelectionProgram,
    build_routed_attention_relation,
    execute_relation_selection,
)
from tile_lifetime.routed_attention_plan import (
    RoutedAttentionPlanConfig,
    RoutedStreamingAttentionCompilation,
    compile_routed_streaming_attention_candidates,
)
from tile_lifetime.semantic_erasure import semantic_erasure_errors, tensor_program_scheduling_keys
from tile_lifetime.streaming_attention import (
    StreamingAttentionProgram,
    StreamingTileSchedule,
    apply_causal_score_mask,
    build_attention_tensor_program,
    derive_streaming_attention,
    scaled_score_map,
)
from tile_lifetime.tensor_program import TensorProgram


class RoutedAttentionRecoveryError(ValueError):
    """A stage-specific failure to recover generic routed-attention algebra."""

    def __init__(self, stage: str, reason: str):
        self.stage = stage
        self.reason = reason
        super().__init__(f"{stage}: {reason}")


@dataclass(frozen=True)
class RecoveredRoutedAttentionProgram:
    """Post-recognition program containing only generic selection and tensor algebra."""

    relation_selection: RelationSelectionProgram
    tensor_program: TensorProgram
    domain_restrictions: tuple[IndexDomainRestriction, ...]
    query_input: str
    key_input: str
    value_input: str
    source_semantics: tuple[str, ...]
    source_operation_ids: tuple[int, ...]
    semantic_erasure_report: SemanticErasureReport

    @property
    def generic_operation_kinds(self) -> tuple[str, ...]:
        """Exact generic representation made visible to scheduling."""
        tensor_kinds = tuple(
            type(operation).__name__.removesuffix("Primitive") for operation in self.tensor_program.operations
        )
        return ("Contract", "DomainRestriction", "Relation", "RelationPlan", *tensor_kinds)


@dataclass(frozen=True)
class NaturalRoutedAttentionCompilation:
    """Natural frontend, runtime relation, and scheduled generated body."""

    recovered: RecoveredRoutedAttentionProgram
    relation: RelationPlan
    streaming_program: StreamingAttentionProgram
    scheduled: RoutedStreamingAttentionCompilation


def recover_routed_attention_program(
    graph: StableHLOGraph,
) -> RecoveredRoutedAttentionProgram:
    """Erase natural router/selected-attention syntax into generic Shuttle algebra."""
    input_by_name = {graph.value(value_id).name: value_id for value_id in graph.inputs}
    required = ("query", "key", "value", "query_metadata", "key_value_metadata")
    missing = tuple(name for name in required if name not in input_by_name)
    if missing:
        raise RoutedAttentionRecoveryError("inputs", f"missing named inputs {missing}")

    top_k = _one(graph, "composite", stage="selection")
    if not isinstance(top_k.attributes, CompositeAttributes):
        raise RoutedAttentionRecoveryError("selection", "top-k composite has no imported attributes")
    match = re.fullmatch(r"(\d+) : i64", dict(top_k.attributes.attributes).get("k", ""))
    if top_k.attributes.name != "chlo.top_k" or top_k.attributes.version != 1 or match is None:
        raise RoutedAttentionRecoveryError("selection", "expected the version-1 generic chlo.top_k composite")
    selected_count = int(match.group(1))
    if len(top_k.outputs) != 2:
        raise RoutedAttentionRecoveryError("selection", "top-k must return values and integer indices")
    selected_id = top_k.outputs[1]

    router_dot = graph.producer(_trace_through(graph, top_k.inputs[0], ("select",))[0])
    if router_dot is None or router_dot.kind != "dot_general":
        raise RoutedAttentionRecoveryError("selection", "top-k scores do not originate at a Contract")
    if set(router_dot.inputs) != {input_by_name["query_metadata"], input_by_name["key_value_metadata"]}:
        raise RoutedAttentionRecoveryError("selection", "router Contract does not consume the metadata inputs")
    if not isinstance(router_dot.attributes, DotAttributes) or router_dot.attributes != DotAttributes(
        lhs_batching_dimensions=(),
        rhs_batching_dimensions=(),
        lhs_contracting_dimensions=(1,),
        rhs_contracting_dimensions=(1,),
    ):
        raise RoutedAttentionRecoveryError("selection", "router Contract has an unsupported index relation")
    router_select = graph.producer(top_k.inputs[0])
    assert router_select is not None
    if router_select.kind != "select" or router_select.inputs[1] != router_dot.outputs[0]:
        raise RoutedAttentionRecoveryError("selection", "router domain restriction is not an explicit select")
    router_compare = graph.producer(router_select.inputs[0])
    if router_compare is None or not _is_less_equal(router_compare):
        raise RoutedAttentionRecoveryError("selection", "router domain restriction is not right_index <= left_index")
    _require_negative_infinity(graph, router_select.inputs[2], "selection")

    query_shape = graph.value(input_by_name["query"]).shape
    key_shape = graph.value(input_by_name["key"]).shape
    value_shape = graph.value(input_by_name["value"]).shape
    if len(query_shape) != 4 or len(key_shape) != 4 or len(value_shape) != 4:
        raise RoutedAttentionRecoveryError("inputs", "Q/K/V must have BSHD rank-four shapes")
    if query_shape[0] != 1 or key_shape[0] != 1 or value_shape[0] != 1:
        raise RoutedAttentionRecoveryError("inputs", "the first routed prototype requires batch size one")
    if key_shape[:3] != value_shape[:3] or query_shape[1] != key_shape[1]:
        raise RoutedAttentionRecoveryError("inputs", "Q/K/V sequence and K/V head shapes are inconsistent")
    if query_shape[-1] != key_shape[-1]:
        raise RoutedAttentionRecoveryError("inputs", "Q and K feature dimensions differ")
    if query_shape[2] % key_shape[2]:
        raise RoutedAttentionRecoveryError("inputs", "query heads are not grouped over key/value heads")

    metadata_shape = graph.value(input_by_name["query_metadata"]).shape
    right_metadata_shape = graph.value(input_by_name["key_value_metadata"]).shape
    if len(metadata_shape) != 2 or metadata_shape != right_metadata_shape:
        raise RoutedAttentionRecoveryError("selection", "left/right relation metadata shapes must match")
    block_count, feature_count = metadata_shape
    if query_shape[1] % block_count:
        raise RoutedAttentionRecoveryError("selection", "sequence length is not divisible by relation blocks")
    gathers = tuple(operation for operation in graph.operations if operation.kind == "gather")
    if len(gathers) != 2:
        raise RoutedAttentionRecoveryError("relation", f"expected two selected payload gathers, found {len(gathers)}")
    for gather in gathers:
        if not isinstance(gather.attributes, GatherAttributes):
            raise RoutedAttentionRecoveryError("relation", "selected payload gather lacks an index mapping")
        if selected_id not in _ancestors(graph, gather.inputs[1]):
            raise RoutedAttentionRecoveryError("relation", "selected payload gather is not indexed by top-k")
    key_gather = next(
        (operation for operation in gathers if input_by_name["key"] in _ancestors(graph, operation.inputs[0])),
        None,
    )
    value_gather = next(
        (operation for operation in gathers if input_by_name["value"] in _ancestors(graph, operation.inputs[0])),
        None,
    )
    if key_gather is None or value_gather is None or key_gather is value_gather:
        raise RoutedAttentionRecoveryError("relation", "could not distinguish selected K and V payload gathers")

    qk = _one_dot_with_ancestor(graph, input_by_name["query"], key_gather.outputs[0], stage="qk_contract")
    scale_multiply = _single_consumer(graph, qk.outputs[0], "multiply", "score_map")
    scale = _scalar_constant_operand(graph, scale_multiply, exclude=qk.outputs[0], stage="score_map")
    if not np.isfinite(scale):
        raise RoutedAttentionRecoveryError("score_map", "score scale is not finite")
    score_select = _downstream_single(graph, scale_multiply.outputs[0], "select", through=("broadcast_in_dim",))
    if score_select is None:
        raise RoutedAttentionRecoveryError("domain_restriction", "scores lack an explicit token-domain select")
    # JAX may broadcast the predicate more than once before selection.
    _, compare_base = _peel(graph, score_select.inputs[0], ("broadcast_in_dim",))
    score_compare = graph.producer(compare_base)
    if score_compare is None or not _is_less_equal(score_compare):
        raise RoutedAttentionRecoveryError(
            "domain_restriction",
            "score restriction is not key_position <= query_position",
        )
    if selected_id not in _ancestors(graph, score_compare.outputs[0]):
        raise RoutedAttentionRecoveryError("domain_restriction", "score restriction does not use selected right blocks")
    _require_negative_infinity(graph, score_select.inputs[2], "domain_restriction")

    reductions = tuple(operation for operation in graph.operations if operation.kind == "reduce")
    reducers = tuple(
        operation.attributes.reducer for operation in reductions if isinstance(operation.attributes, ReductionAttributes)
    )
    if reducers.count("maximum") != 1 or reducers.count("add") != 1:
        raise RoutedAttentionRecoveryError("normalized_fold", f"expected max and sum Fold reducers, found {reducers}")
    exponentials = tuple(operation for operation in graph.operations if operation.kind == "exponential")
    divides = tuple(operation for operation in graph.operations if operation.kind == "divide")
    if len(exponentials) != 1 or len(divides) != 1:
        raise RoutedAttentionRecoveryError("normalized_fold", "normalized exponential must contain one exp and divide")

    pv = _one_dot_with_ancestor(graph, value_gather.outputs[0], divides[0].outputs[0], stage="pv_contract")
    if not any(pv.id in _ancestors_operations(graph, output_id) for output_id in graph.outputs):
        raise RoutedAttentionRecoveryError("pv_contract", "PV Contract does not produce a function output")

    relation_restriction = IndexDomainRestriction(
        left_axis="left_block",
        right_axis="right_block",
        predicate="left_greater_equal_right",
    )
    token_restriction = IndexDomainRestriction(
        left_axis="query_position",
        right_axis="key_position",
        predicate="left_greater_equal_right",
    )
    selection = RelationSelectionProgram(
        left_input="query_metadata",
        right_input="key_value_metadata",
        left_count=block_count,
        right_count=block_count,
        feature_count=feature_count,
        selected_count=selected_count,
        restriction=relation_restriction,
    )
    score_map = apply_causal_score_mask(scaled_score_map(scale))
    tensor_program = build_attention_tensor_program(
        batch_size=1,
        query_length=query_shape[1],
        key_length=key_shape[1],
        query_heads=query_shape[2],
        key_value_heads=key_shape[2],
        key_dimension=query_shape[3],
        value_dimension=value_shape[3],
        score_map=score_map,
        input_dtype=graph.value(input_by_name["query"]).dtype,
        accumulation_dtype=graph.value(qk.outputs[0]).dtype,
    )
    source_ids = tuple(sorted(_ancestors_operations(graph, graph.outputs[0])))
    source_semantics = ("top_k", "selected_exact_attention", "normalized_exponential", "causal_predicate")
    scheduling_keys = (
        (
            f"relation_selection:left_rank=2:right_rank=2:selected={selected_count}:accumulate=fp32:"
            f"{selection.selection_semantics.scheduling_key}"
        ),
        "domain_restriction:binary_affine_index_predicate",
        "relation_plan:runtime_binary_edges:dual_orientation",
        *tensor_program_scheduling_keys(tensor_program),
    )
    provisional_report = SemanticErasureReport(
        source_semantics=source_semantics,
        lowering_steps=(
            SemanticLoweringStep("top_k", ("Contract", "DomainRestriction", "Relation")),
            SemanticLoweringStep(
                "selected_exact_attention",
                ("RelationPlan", "Contract", "Map", "Fold", "DomainRestriction"),
            ),
            SemanticLoweringStep("normalized_exponential", ("Map", "Fold")),
            SemanticLoweringStep("causal_predicate", ("DomainRestriction",)),
        ),
        scheduling_keys=scheduling_keys,
    )
    erasure_report = SemanticErasureReport(
        source_semantics=provisional_report.source_semantics,
        lowering_steps=provisional_report.lowering_steps,
        scheduling_keys=provisional_report.scheduling_keys,
        validation_errors=semantic_erasure_errors(provisional_report),
    )
    return RecoveredRoutedAttentionProgram(
        relation_selection=selection,
        tensor_program=tensor_program,
        domain_restrictions=(relation_restriction, token_restriction),
        query_input="query",
        key_input="key",
        value_input="value",
        source_semantics=source_semantics,
        source_operation_ids=source_ids,
        semantic_erasure_report=erasure_report,
    )


def compile_natural_routed_attention(
    recovered: RecoveredRoutedAttentionProgram,
    *,
    runtime_inputs: dict[str, np.ndarray],
    schedule: StreamingTileSchedule,
    config: RoutedAttentionPlanConfig,
    padding_quantum: int = 1,
) -> NaturalRoutedAttentionCompilation:
    """Build runtime RelationPlan and generated streaming candidates from erased semantics."""
    erasure_errors = semantic_erasure_errors(recovered.semantic_erasure_report)
    if erasure_errors:
        raise RoutedAttentionRecoveryError("name_erasure", "; ".join(erasure_errors))
    selection = execute_relation_selection(recovered.relation_selection, runtime_inputs)
    relation = build_routed_attention_relation(
        selection.indices,
        edge_valid=selection.valid,
        kv_rank_by_block=np.zeros(recovered.relation_selection.right_count, dtype=np.int32),
        kv_local_block_by_block=np.arange(recovered.relation_selection.right_count, dtype=np.int32),
        padding_quantum=padding_quantum,
    )
    streaming = derive_streaming_attention(recovered.tensor_program, schedule=schedule)
    scheduled = compile_routed_streaming_attention_candidates(streaming, relation, config)
    return NaturalRoutedAttentionCompilation(
        recovered=recovered,
        relation=relation,
        streaming_program=streaming,
        scheduled=scheduled,
    )


def _one(graph: StableHLOGraph, kind: str, *, stage: str) -> StableHLOOperation:
    matches = tuple(operation for operation in graph.operations if operation.kind == kind)
    if len(matches) != 1:
        raise RoutedAttentionRecoveryError(stage, f"expected one {kind}, found {len(matches)}")
    return matches[0]


def _one_dot_with_ancestor(
    graph: StableHLOGraph,
    first_ancestor: int,
    second_ancestor: int,
    *,
    stage: str,
) -> StableHLOOperation:
    matches = tuple(
        operation
        for operation in graph.operations
        if operation.kind == "dot_general"
        and len(operation.inputs) == 2
        and (
            (
                first_ancestor in _ancestors(graph, operation.inputs[0])
                and second_ancestor in _ancestors(graph, operation.inputs[1])
            )
            or (
                first_ancestor in _ancestors(graph, operation.inputs[1])
                and second_ancestor in _ancestors(graph, operation.inputs[0])
            )
        )
    )
    if len(matches) != 1:
        raise RoutedAttentionRecoveryError(
            stage,
            f"expected one Contract over the required values, found {len(matches)}",
        )
    return matches[0]


def _single_consumer(graph: StableHLOGraph, value_id: int, kind: str, stage: str) -> StableHLOOperation:
    matches = tuple(operation for operation in graph.consumers(value_id) if operation.kind == kind)
    if len(matches) != 1:
        raise RoutedAttentionRecoveryError(stage, f"value does not feed exactly one {kind}")
    return matches[0]


def _scalar_constant_operand(
    graph: StableHLOGraph,
    operation: StableHLOOperation,
    *,
    exclude: int,
    stage: str,
) -> float:
    candidates = tuple(value_id for value_id in operation.inputs if value_id != exclude)
    if len(candidates) != 1:
        raise RoutedAttentionRecoveryError(stage, "map does not contain one scalar operand")
    _, base = _peel(graph, candidates[0], ("broadcast_in_dim",))
    constant = graph.producer(base)
    if constant is None or not isinstance(constant.attributes, ConstantAttributes):
        raise RoutedAttentionRecoveryError(stage, "scalar operand is not a constant")
    match = re.search(r"dense<([^>]+)>", constant.attributes.literal)
    if match is None:
        raise RoutedAttentionRecoveryError(stage, "could not parse scalar constant")
    return float(match.group(1))


def _trace_through(
    graph: StableHLOGraph,
    value_id: int,
    kinds: tuple[str, ...],
) -> tuple[int, tuple[StableHLOOperation, ...]]:
    traversed: list[StableHLOOperation] = []
    current = value_id
    while (producer := graph.producer(current)) is not None and producer.kind in kinds:
        traversed.append(producer)
        # Select's true value is the semantic payload.
        current = producer.inputs[1] if producer.kind == "select" else producer.inputs[0]
    return current, tuple(traversed)


def _peel(
    graph: StableHLOGraph,
    value_id: int,
    kinds: tuple[str, ...],
) -> tuple[tuple[StableHLOOperation, ...], int]:
    current = value_id
    operations: list[StableHLOOperation] = []
    while (producer := graph.producer(current)) is not None and producer.kind in kinds:
        operations.append(producer)
        current = producer.inputs[0]
    return tuple(operations), current


def _downstream_single(
    graph: StableHLOGraph,
    value_id: int,
    kind: str,
    *,
    through: tuple[str, ...],
) -> StableHLOOperation | None:
    pending = [value_id]
    found: list[StableHLOOperation] = []
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        for consumer in graph.consumers(current):
            if consumer.kind == kind:
                found.append(consumer)
            elif consumer.kind in through:
                pending.extend(consumer.outputs)
    return found[0] if len(found) == 1 else None


def _is_less_equal(operation: StableHLOOperation) -> bool:
    return (
        operation.kind == "compare"
        and isinstance(operation.attributes, CompareAttributes)
        and operation.attributes.direction == "LE"
    )


def _require_negative_infinity(graph: StableHLOGraph, value_id: int, stage: str) -> None:
    _, base = _peel(graph, value_id, ("broadcast_in_dim",))
    producer = graph.producer(base)
    if producer is None or not isinstance(producer.attributes, ConstantAttributes):
        raise RoutedAttentionRecoveryError(stage, "masked value is not a constant")
    if "FF800000" not in producer.attributes.literal and "-inf" not in producer.attributes.literal.lower():
        raise RoutedAttentionRecoveryError(stage, "masked value is not negative infinity")


def _ancestors(graph: StableHLOGraph, value_id: int) -> set[int]:
    result = {value_id}
    producer = graph.producer(value_id)
    if producer is None:
        return result
    for input_id in producer.inputs:
        result.update(_ancestors(graph, input_id))
    return result


def _ancestors_operations(graph: StableHLOGraph, value_id: int) -> set[int]:
    result: set[int] = set()
    producer = graph.producer(value_id)
    if producer is None:
        return result
    result.add(producer.id)
    for input_id in producer.inputs:
        result.update(_ancestors_operations(graph, input_id))
    return result
