# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experimental whole-pattern projected-attention recovery."""

import re
from dataclasses import dataclass

import numpy as np

from shuttle.experimental.stablehlo_import import (
    CompareAttributes,
    CompositeAttributes,
    ConstantAttributes,
    DotAttributes,
    ReductionAttributes,
    StableHLOGraph,
    StableHLOOperation,
)
from shuttle.ir import DType
from tile_lifetime.plan import SemanticErasureReport, SemanticLoweringStep
from tile_lifetime.relation import RelationPlan
from tile_lifetime.routed_attention import (
    IndexDomainRestriction,
    ProjectedBlockSelectionProgram,
    build_grouped_routed_attention_relation,
    execute_projected_block_selection,
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


class ProjectedRoutedAttentionRecoveryError(ValueError):
    """A stage-specific projected-relation recovery failure."""

    def __init__(self, stage: str, reason: str):
        self.stage = stage
        self.reason = reason
        super().__init__(f"{stage}: {reason}")


@dataclass(frozen=True)
class RecoveredProjectedRoutedAttentionProgram:
    """Named frontend syntax erased into projected Selection and tensor algebra."""

    relation_selection: ProjectedBlockSelectionProgram
    tensor_program: TensorProgram
    domain_restrictions: tuple[IndexDomainRestriction, ...]
    source_semantics: tuple[str, ...]
    source_operation_ids: tuple[int, ...]
    semantic_erasure_report: SemanticErasureReport

    @property
    def generic_operation_kinds(self) -> tuple[str, ...]:
        """Generic semantic structure visible to scheduling."""
        tensor_kinds = tuple(
            type(operation).__name__.removesuffix("Primitive") for operation in self.tensor_program.operations
        )
        return (
            "Contract",
            "Contract",
            "Map",
            "DomainRestriction",
            "Fold",
            "Selection",
            "Relation",
            "RelationPlan",
            *tensor_kinds,
        )


@dataclass(frozen=True)
class NaturalProjectedRoutedAttentionCompilation:
    """Runtime projected relation and generated orientation candidates."""

    recovered: RecoveredProjectedRoutedAttentionProgram
    selected_right_blocks: np.ndarray
    edge_valid: np.ndarray
    relation: RelationPlan
    streaming_program: StreamingAttentionProgram
    scheduled: RoutedStreamingAttentionCompilation


def recover_projected_routed_attention_program(
    graph: StableHLOGraph,
) -> RecoveredProjectedRoutedAttentionProgram:
    """Recover natural projected routing without retaining a workload identity."""
    input_by_name = {graph.value(value_id).name: value_id for value_id in graph.inputs}
    required = (
        "query_hidden",
        "key_value_hidden",
        "query_weight",
        "key_weight",
        "value_weight",
        "left_index_weight",
        "right_index_weight",
    )
    missing = tuple(name for name in required if name not in input_by_name)
    if missing:
        raise ProjectedRoutedAttentionRecoveryError("inputs", f"missing named inputs {missing}")

    query_hidden_id = input_by_name["query_hidden"]
    key_value_hidden_id = input_by_name["key_value_hidden"]
    query_hidden_shape = graph.value(query_hidden_id).shape
    key_value_hidden_shape = graph.value(key_value_hidden_id).shape
    if len(query_hidden_shape) != 2 or len(key_value_hidden_shape) != 2:
        raise ProjectedRoutedAttentionRecoveryError(
            "inputs", "query and key/value hidden inputs must have [token, feature] shapes"
        )
    source_count, source_feature_count = query_hidden_shape
    right_count, right_source_feature_count = key_value_hidden_shape
    if source_count > right_count:
        raise ProjectedRoutedAttentionRecoveryError(
            "inputs", "bottom-right causal prefill requires query count no larger than key/value count"
        )
    projection_source = {
        "query_weight": query_hidden_id,
        "key_weight": key_value_hidden_id,
        "value_weight": key_value_hidden_id,
        "left_index_weight": query_hidden_id,
        "right_index_weight": key_value_hidden_id,
    }
    projections = {
        name: _projection_dot(graph, projection_source[name], input_by_name[name], stage="projection")
        for name in projection_source
    }
    expected_projection = DotAttributes(
        lhs_batching_dimensions=(),
        rhs_batching_dimensions=(),
        lhs_contracting_dimensions=(1,),
        rhs_contracting_dimensions=(0,),
    )
    if any(operation.attributes != expected_projection for operation in projections.values()):
        raise ProjectedRoutedAttentionRecoveryError("projection", "projection Contract index relation is unsupported")

    left_weight_shape = graph.value(input_by_name["left_index_weight"]).shape
    right_weight_shape = graph.value(input_by_name["right_index_weight"]).shape
    if len(left_weight_shape) != 2 or len(right_weight_shape) != 2:
        raise ProjectedRoutedAttentionRecoveryError("selection", "index projection weights must be matrices")
    relation_feature_count = right_weight_shape[1]
    if left_weight_shape[0] != source_feature_count or right_weight_shape[0] != right_source_feature_count:
        raise ProjectedRoutedAttentionRecoveryError("selection", "index projection input dimensions differ")
    if left_weight_shape[1] % relation_feature_count:
        raise ProjectedRoutedAttentionRecoveryError("selection", "left index features do not form equal groups")
    group_count = left_weight_shape[1] // relation_feature_count

    top_k = _one(graph, "composite", stage="selection")
    if not isinstance(top_k.attributes, CompositeAttributes):
        raise ProjectedRoutedAttentionRecoveryError("selection", "top-k has no imported composite attributes")
    match = re.fullmatch(r"(\d+) : i64", dict(top_k.attributes.attributes).get("k", ""))
    if top_k.attributes.name != "chlo.top_k" or top_k.attributes.version != 1 or match is None:
        raise ProjectedRoutedAttentionRecoveryError("selection", "expected version-1 generic chlo.top_k")
    selected_count = int(match.group(1))
    if len(top_k.outputs) != 2:
        raise ProjectedRoutedAttentionRecoveryError("selection", "top-k must return values and indices")
    selected_id = top_k.outputs[1]

    top_k_input = graph.producer(top_k.inputs[0])
    if top_k_input is None or top_k_input.kind != "select":
        raise ProjectedRoutedAttentionRecoveryError("selection", "forced local selection is not an explicit select")
    block_folds = tuple(
        operation
        for operation in graph.operations
        if operation.kind == "reduce"
        and isinstance(operation.attributes, ReductionAttributes)
        and operation.attributes.reducer == "maximum"
        and operation.id in _ancestor_operations(graph, top_k.inputs[0])
    )
    if len(block_folds) != 1:
        raise ProjectedRoutedAttentionRecoveryError(
            "selection",
            f"expected one block-maximum Fold before top-k, found {len(block_folds)}",
        )
    block_fold = block_folds[0]
    if block_fold.outputs[0] not in _ancestors(graph, top_k_input.inputs[2]):
        raise ProjectedRoutedAttentionRecoveryError("selection", "forced select does not preserve block Fold scores")
    _require_positive_infinity(graph, top_k_input.inputs[1], "selection")
    block_input_shape = graph.value(block_fold.inputs[0]).shape
    if len(block_input_shape) != 4 or block_input_shape[:2] != (source_count, group_count):
        raise ProjectedRoutedAttentionRecoveryError("selection", "block Fold has an unsupported logical shape")
    right_block_count, right_block_size = block_input_shape[2:]
    if right_block_count * right_block_size != right_count:
        raise ProjectedRoutedAttentionRecoveryError("selection", "block Fold does not partition the token domain")

    token_score = _dot_with_projection_ancestors(
        graph,
        projections["left_index_weight"],
        projections["right_index_weight"],
        stage="selection",
    )
    if any(graph.value(value_id).dtype is not DType.BF16 for value_id in token_score.inputs):
        raise ProjectedRoutedAttentionRecoveryError(
            "selection", "index projections must expose a BF16 boundary before the FP32 score Contract"
        )
    index_scale_map = _single_consumer(graph, token_score.outputs[0], "multiply", "selection")
    index_scale = _scalar_constant_operand(graph, index_scale_map, exclude=token_score.outputs[0], stage="selection")
    token_selects = tuple(
        operation
        for operation in graph.operations
        if operation.kind == "select"
        and operation.id in _ancestor_operations(graph, block_fold.inputs[0])
        and index_scale_map.outputs[0] in _ancestors(graph, operation.outputs[0])
    )
    if len(token_selects) != 1:
        raise ProjectedRoutedAttentionRecoveryError("selection", "causal token restriction is not one explicit select")
    token_compare = graph.producer(token_selects[0].inputs[0])
    if token_compare is None:
        raise ProjectedRoutedAttentionRecoveryError("selection", "token restriction has no compare producer")
    _require_bottom_right_compare(
        graph,
        token_compare,
        query_count=source_count,
        key_value_count=right_count,
        stage="selection",
    )
    _require_negative_infinity(graph, token_selects[0].inputs[2], "selection")

    selected_shape = graph.value(selected_id).shape
    selection_validity_compares = tuple(
        operation
        for operation in graph.operations
        if operation.kind == "compare"
        and isinstance(operation.attributes, CompareAttributes)
        and operation.attributes.direction == "LE"
        and graph.value(operation.outputs[0]).shape == selected_shape
        and selected_id in _ancestors(graph, operation.inputs[0])
    )
    if len(selection_validity_compares) != 1:
        raise ProjectedRoutedAttentionRecoveryError(
            "selection",
            "underfilled top-k slots must have one explicit Boolean validity value",
        )
    selection_validity = selection_validity_compares[0].outputs[0]

    gathers = tuple(operation for operation in graph.operations if operation.kind == "gather")
    if len(gathers) != 2 or any(selected_id not in _ancestors(graph, gather.inputs[1]) for gather in gathers):
        raise ProjectedRoutedAttentionRecoveryError("relation", "selected K/V gathers do not originate at top-k")
    if any(selection_validity not in _ancestors(graph, gather.inputs[1]) for gather in gathers):
        raise ProjectedRoutedAttentionRecoveryError(
            "relation",
            "selected K/V gathers do not replace invalid top-k slots before indexing",
        )

    query_projection = projections["query_weight"]
    key_projection = projections["key_weight"]
    value_projection = projections["value_weight"]
    qk = _dot_with_projection_ancestors(graph, query_projection, key_projection, stage="qk_contract")
    attention_scale_map = _single_consumer(graph, qk.outputs[0], "multiply", "score_map")
    attention_scale = _scalar_constant_operand(
        graph,
        attention_scale_map,
        exclude=qk.outputs[0],
        stage="score_map",
    )
    score_selects = tuple(
        operation
        for operation in graph.operations
        if operation.kind == "select" and attention_scale_map.outputs[0] in _ancestors(graph, operation.outputs[0])
    )
    if len(score_selects) < 2:
        raise ProjectedRoutedAttentionRecoveryError(
            "domain_restriction",
            "selected attention lacks separate token and route-validity restrictions",
        )
    if not any(selected_id in _ancestors(graph, operation.inputs[0]) for operation in score_selects):
        raise ProjectedRoutedAttentionRecoveryError("domain_restriction", "score restrictions ignore selected blocks")
    score_compares = tuple(
        candidate
        for operation in score_selects
        for candidate in graph.operations
        if candidate.kind == "compare" and candidate.id in _ancestor_operations(graph, operation.inputs[0])
    )
    position_offset = right_count - source_count
    if not any(
        isinstance(operation.attributes, CompareAttributes)
        and operation.attributes.direction == "LE"
        and selected_id in _ancestors(graph, operation.inputs[0])
        and _contains_iota_extent(graph, operation.inputs[1], source_count)
        and _contains_integer_constant(graph, operation.inputs[1], position_offset)
        for operation in score_compares
    ):
        raise ProjectedRoutedAttentionRecoveryError(
            "domain_restriction", "selected attention does not use bottom-right causal positions"
        )

    reductions = tuple(operation for operation in graph.operations if operation.kind == "reduce")
    softmax_reducers = tuple(
        operation.attributes.reducer
        for operation in reductions
        if operation is not block_fold and isinstance(operation.attributes, ReductionAttributes)
    )
    if softmax_reducers.count("maximum") != 1 or softmax_reducers.count("add") != 1:
        raise ProjectedRoutedAttentionRecoveryError(
            "normalized_fold",
            f"expected max and sum normalized Fold reducers, found {softmax_reducers}",
        )
    exponentials = tuple(operation for operation in graph.operations if operation.kind == "exponential")
    divides = tuple(operation for operation in graph.operations if operation.kind == "divide")
    if len(exponentials) != 1 or len(divides) != 1:
        raise ProjectedRoutedAttentionRecoveryError("normalized_fold", "normalized exponential is incomplete")
    pv = _dot_with_projection_ancestors(graph, value_projection, None, stage="pv_contract", second=divides[0].outputs[0])
    if not any(pv.id in _ancestor_operations(graph, output) for output in graph.outputs):
        raise ProjectedRoutedAttentionRecoveryError("pv_contract", "PV Contract does not produce an output")

    query_weight_shape = graph.value(input_by_name["query_weight"]).shape
    key_weight_shape = graph.value(input_by_name["key_weight"]).shape
    value_weight_shape = graph.value(input_by_name["value_weight"]).shape
    if key_weight_shape != value_weight_shape or key_weight_shape[1] % group_count:
        raise ProjectedRoutedAttentionRecoveryError("inputs", "main K/V projection shapes are inconsistent")
    head_dimension = key_weight_shape[1] // group_count
    if query_weight_shape[1] % head_dimension:
        raise ProjectedRoutedAttentionRecoveryError("inputs", "query projection does not form complete heads")
    query_heads = query_weight_shape[1] // head_dimension
    if query_heads % group_count:
        raise ProjectedRoutedAttentionRecoveryError("inputs", "query heads do not map evenly onto relation groups")

    token_restriction = IndexDomainRestriction(
        left_axis="query_position",
        right_axis="key_position",
        predicate="left_greater_equal_right",
    )
    selection = ProjectedBlockSelectionProgram(
        source_input="query_hidden",
        left_weight_input="left_index_weight",
        right_weight_input="right_index_weight",
        source_count=source_count,
        source_feature_count=source_feature_count,
        group_count=group_count,
        relation_feature_count=relation_feature_count,
        right_block_size=right_block_size,
        selected_count=selected_count,
        score_scale=index_scale,
        token_restriction=token_restriction,
        force_local_block=True,
        projection_output_dtype="bf16",
        right_source_input="key_value_hidden",
        right_source_feature_count=right_source_feature_count,
        right_count=right_count,
        left_position_offset=position_offset,
        right_position_offset=0,
    )
    tensor_program = build_attention_tensor_program(
        batch_size=1,
        query_length=source_count,
        key_length=right_count,
        query_heads=query_heads,
        key_value_heads=group_count,
        key_dimension=head_dimension,
        value_dimension=head_dimension,
        score_map=apply_causal_score_mask(scaled_score_map(attention_scale)),
        input_dtype=graph.value(qk.inputs[0]).dtype,
        accumulation_dtype=graph.value(qk.outputs[0]).dtype,
    )
    source_semantics = (
        "projected_block_selection",
        "selected_weighted_reduction",
        "normalized_exponential",
        "causal_predicate",
    )
    scheduling_keys = (
        "contract:index_left:output_rank=3:output=bf16:accumulate=fp32",
        "contract:index_right:output_rank=2:output=bf16:accumulate=fp32",
        "map:scale",
        "domain_restriction:binary_affine_index_predicate",
        f"fold:maximum:block={right_block_size}:accumulate=fp32",
        (f"selection:stable_top_k:k={selected_count}:force_local=1:" f"{selection.selection_semantics.scheduling_key}"),
        "relation_plan:runtime_binary_edges:grouped_left:dual_orientation",
        *tensor_program_scheduling_keys(tensor_program),
    )
    provisional = SemanticErasureReport(
        source_semantics=source_semantics,
        lowering_steps=(
            SemanticLoweringStep(
                "projected_block_selection",
                ("Contract", "Map", "DomainRestriction", "Fold", "Selection", "Relation", "RelationPlan"),
            ),
            SemanticLoweringStep(
                "selected_weighted_reduction",
                ("Contract", "Map", "Fold", "DomainRestriction"),
            ),
            SemanticLoweringStep("normalized_exponential", ("Map", "Fold")),
            SemanticLoweringStep("causal_predicate", ("DomainRestriction",)),
        ),
        scheduling_keys=scheduling_keys,
    )
    report = SemanticErasureReport(
        source_semantics=provisional.source_semantics,
        lowering_steps=provisional.lowering_steps,
        scheduling_keys=provisional.scheduling_keys,
        validation_errors=semantic_erasure_errors(provisional),
    )
    return RecoveredProjectedRoutedAttentionProgram(
        relation_selection=selection,
        tensor_program=tensor_program,
        domain_restrictions=(token_restriction,),
        source_semantics=source_semantics,
        source_operation_ids=tuple(sorted(_ancestor_operations(graph, graph.outputs[0]))),
        semantic_erasure_report=report,
    )


def compile_natural_projected_routed_attention(
    recovered: RecoveredProjectedRoutedAttentionProgram,
    *,
    runtime_inputs: dict[str, np.ndarray],
    schedule: StreamingTileSchedule,
    config: RoutedAttentionPlanConfig,
    padding_quantum: int = 1,
) -> NaturalProjectedRoutedAttentionCompilation:
    """Execute generic Selection, build RelationPlan, and enumerate schedules."""
    errors = semantic_erasure_errors(recovered.semantic_erasure_report)
    if errors:
        raise ProjectedRoutedAttentionRecoveryError("name_erasure", "; ".join(errors))
    selection = execute_projected_block_selection(recovered.relation_selection, runtime_inputs)
    relation = build_grouped_routed_attention_relation(
        selection.indices,
        edge_valid=selection.valid,
        padding_quantum=padding_quantum,
    )
    streaming = derive_streaming_attention(recovered.tensor_program, schedule=schedule)
    scheduled = compile_routed_streaming_attention_candidates(streaming, relation, config)
    return NaturalProjectedRoutedAttentionCompilation(
        recovered=recovered,
        selected_right_blocks=selection.indices,
        edge_valid=selection.valid,
        relation=relation,
        streaming_program=streaming,
        scheduled=scheduled,
    )


def _one(graph: StableHLOGraph, kind: str, *, stage: str) -> StableHLOOperation:
    matches = tuple(operation for operation in graph.operations if operation.kind == kind)
    if len(matches) != 1:
        raise ProjectedRoutedAttentionRecoveryError(stage, f"expected one {kind}, found {len(matches)}")
    return matches[0]


def _projection_dot(
    graph: StableHLOGraph,
    source_id: int,
    weight_id: int,
    *,
    stage: str,
) -> StableHLOOperation:
    matches = tuple(
        operation
        for operation in graph.operations
        if operation.kind == "dot_general"
        and weight_id in operation.inputs
        and source_id in _ancestors(graph, operation.inputs[0])
    )
    if len(matches) != 1:
        raise ProjectedRoutedAttentionRecoveryError(stage, f"expected one projection for weight value {weight_id}")
    return matches[0]


def _dot_with_projection_ancestors(
    graph: StableHLOGraph,
    first: StableHLOOperation,
    second_projection: StableHLOOperation | None,
    *,
    stage: str,
    second: int | None = None,
) -> StableHLOOperation:
    matches = []
    for operation in graph.operations:
        if operation.kind != "dot_general" or operation in (first, second_projection):
            continue
        lhs_operations = _ancestor_operations(graph, operation.inputs[0])
        rhs_operations = _ancestor_operations(graph, operation.inputs[1])
        first_on_lhs = first.id in lhs_operations
        first_on_rhs = first.id in rhs_operations
        if second_projection is not None:
            second_on_lhs = second_projection.id in lhs_operations
            second_on_rhs = second_projection.id in rhs_operations
        else:
            assert second is not None
            second_on_lhs = second in _ancestors(graph, operation.inputs[0])
            second_on_rhs = second in _ancestors(graph, operation.inputs[1])
        if (first_on_lhs and not second_on_lhs and second_on_rhs and not first_on_rhs) or (
            first_on_rhs and not second_on_rhs and second_on_lhs and not first_on_lhs
        ):
            matches.append(operation)
    if len(matches) != 1:
        raise ProjectedRoutedAttentionRecoveryError(stage, f"expected one matching Contract, found {len(matches)}")
    return matches[0]


def _single_consumer(graph: StableHLOGraph, value_id: int, kind: str, stage: str) -> StableHLOOperation:
    matches = tuple(operation for operation in graph.consumers(value_id) if operation.kind == kind)
    if len(matches) != 1:
        raise ProjectedRoutedAttentionRecoveryError(stage, f"value does not feed exactly one {kind}")
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
        raise ProjectedRoutedAttentionRecoveryError(stage, "Map does not contain one scalar operand")
    base = _peel(graph, candidates[0], ("broadcast_in_dim",))
    constant = graph.producer(base)
    if constant is None or not isinstance(constant.attributes, ConstantAttributes):
        raise ProjectedRoutedAttentionRecoveryError(stage, "Map scalar is not constant")
    match = re.search(r"dense<([^>]+)>", constant.attributes.literal)
    if match is None:
        raise ProjectedRoutedAttentionRecoveryError(stage, "could not parse Map scalar")
    return float(match.group(1))


def _require_bottom_right_compare(
    graph: StableHLOGraph,
    operation: StableHLOOperation,
    *,
    query_count: int,
    key_value_count: int,
    stage: str,
) -> None:
    current = operation
    while current.kind == "broadcast_in_dim":
        producer = graph.producer(current.inputs[0])
        if producer is None:
            raise ProjectedRoutedAttentionRecoveryError(stage, "causal predicate broadcast has no producer")
        current = producer
    if not (
        current.kind == "compare"
        and isinstance(current.attributes, CompareAttributes)
        and current.attributes.direction == "LE"
    ):
        raise ProjectedRoutedAttentionRecoveryError(stage, "token restriction is not key_position <= query_position")
    position_offset = key_value_count - query_count
    if not _contains_iota_extent(graph, current.inputs[0], key_value_count):
        raise ProjectedRoutedAttentionRecoveryError(stage, "causal predicate left side is not the key position domain")
    if not _contains_iota_extent(graph, current.inputs[1], query_count):
        raise ProjectedRoutedAttentionRecoveryError(
            stage, "causal predicate right side is not the query position domain"
        )
    if not _contains_integer_constant(graph, current.inputs[1], position_offset):
        raise ProjectedRoutedAttentionRecoveryError(
            stage,
            f"query positions do not use the required bottom-right offset {position_offset}",
        )


def _contains_iota_extent(graph: StableHLOGraph, value_id: int, extent: int) -> bool:
    return any(
        operation.kind == "iota" and len(operation.outputs) == 1 and graph.value(operation.outputs[0]).shape == (extent,)
        for operation in graph.operations
        if operation.id in _ancestor_operations(graph, value_id)
    )


def _contains_integer_constant(graph: StableHLOGraph, value_id: int, expected: int) -> bool:
    if expected == 0:
        # A canonicalizer may remove addition by zero entirely.
        return True
    for operation in graph.operations:
        if operation.id not in _ancestor_operations(graph, value_id):
            continue
        if not isinstance(operation.attributes, ConstantAttributes):
            continue
        match = re.fullmatch(r"dense<(-?\d+)> : tensor<i\d+>", operation.attributes.literal)
        if match is not None and int(match.group(1)) == expected:
            return True
    return False


def _require_negative_infinity(graph: StableHLOGraph, value_id: int, stage: str) -> None:
    _require_infinity(graph, value_id, stage, positive=False)


def _require_positive_infinity(graph: StableHLOGraph, value_id: int, stage: str) -> None:
    _require_infinity(graph, value_id, stage, positive=True)


def _require_infinity(graph: StableHLOGraph, value_id: int, stage: str, *, positive: bool) -> None:
    base = _peel(graph, value_id, ("broadcast_in_dim",))
    producer = graph.producer(base)
    if producer is None or not isinstance(producer.attributes, ConstantAttributes):
        raise ProjectedRoutedAttentionRecoveryError(stage, "masked value is not constant")
    literal = producer.attributes.literal.lower()
    if positive:
        matched = "7f800000" in literal or ("inf" in literal and "-inf" not in literal)
    else:
        matched = "ff800000" in literal or "-inf" in literal
    if not matched:
        sign = "positive" if positive else "negative"
        raise ProjectedRoutedAttentionRecoveryError(stage, f"masked value is not {sign} infinity")


def _peel(graph: StableHLOGraph, value_id: int, kinds: tuple[str, ...]) -> int:
    current = value_id
    while (producer := graph.producer(current)) is not None and producer.kind in kinds:
        current = producer.inputs[0]
    return current


def _ancestors(graph: StableHLOGraph, value_id: int) -> set[int]:
    result = {value_id}
    producer = graph.producer(value_id)
    if producer is None:
        return result
    for input_id in producer.inputs:
        result.update(_ancestors(graph, input_id))
    return result


def _ancestor_operations(graph: StableHLOGraph, value_id: int) -> set[int]:
    result: set[int] = set()
    producer = graph.producer(value_id)
    if producer is None:
        return result
    result.add(producer.id)
    for input_id in producer.inputs:
        result.update(_ancestor_operations(graph, input_id))
    return result
