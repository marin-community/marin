# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reference-only whole-pattern recovery for a shared-plus-routed MoE graph."""

import re
from dataclasses import dataclass

from shuttle.experimental.stablehlo_import import (
    BroadcastAttributes,
    CompareAttributes,
    CompositeAttributes,
    ConstantAttributes,
    DotAttributes,
    GatherAttributes,
    ReductionAttributes,
    StableHLOGraph,
    StableHLOOperation,
)
from shuttle.ir import DType
from tile_lifetime.ir import TensorGraph

INTEGER_ATTRIBUTE = re.compile(r"^(\d+) : i64$")
SCALAR_LITERAL = re.compile(r"dense<([^>]+)>")


class MoESemanticRecoveryError(ValueError):
    """Structured rejection of a StableHLO MoE fragment."""

    def __init__(self, *, stage: str, reason: str, source_location: str | None = None):
        self.stage = stage
        self.reason = reason
        self.source_location = source_location
        location = f" at {source_location}" if source_location is not None else ""
        super().__init__(f"{stage}{location}: {reason}")


@dataclass(frozen=True)
class RecoveredMoERegion:
    """Recovered MoE semantic graph and source-operation provenance."""

    graph: TensorGraph
    source_operation_ids: tuple[int, ...]


def recover_moe_region(
    stablehlo_graph: StableHLOGraph,
    *,
    gemm_accumulation_dtype: DType,
) -> RecoveredMoERegion:
    """Recover the exact ordinary JAX top-k, shared MLP, routed MLP, and combine graph."""
    if len(stablehlo_graph.inputs) != 8 or len(stablehlo_graph.outputs) != 3:
        raise MoESemanticRecoveryError(
            stage="signature",
            reason="MoE region requires eight inputs and output, expert-index, and route-weight results",
        )
    x_id, router_weight_id, shared_gate_id, shared_up_id, shared_down_id = stablehlo_graph.inputs[:5]
    routed_gate_id, routed_up_id, routed_down_id = stablehlo_graph.inputs[5:]

    try:
        router_dot = _dot_using_weight(stablehlo_graph, router_weight_id)
        _validate_dot(router_dot, lhs_batch=(), rhs_batch=(), lhs_contract=(1,), rhs_contract=(0,))
        if router_dot.inputs[0] != x_id:
            raise ValueError("router projection does not consume the token input")
        router_logits_id = _converted_dot_output(stablehlo_graph, router_dot)
        top_k_op, expert_indices_id, route_weights_id, top_k = _recover_router(
            stablehlo_graph,
            router_logits_id,
        )
    except ValueError as error:
        raise _structured("router", error) from error

    try:
        shared_gate_dot = _dot_using_weight(stablehlo_graph, shared_gate_id)
        shared_up_dot = _dot_using_weight(stablehlo_graph, shared_up_id)
        shared_down_dot = _dot_using_weight(stablehlo_graph, shared_down_id)
        for operation in (shared_gate_dot, shared_up_dot, shared_down_dot):
            _validate_dot(operation, lhs_batch=(), rhs_batch=(), lhs_contract=(1,), rhs_contract=(1,))
        if shared_gate_dot.inputs[0] != x_id or shared_up_dot.inputs[0] != x_id:
            raise ValueError("shared gate and up projections do not consume the token input")
        shared_gate_value = _converted_dot_output(stablehlo_graph, shared_gate_dot)
        shared_up_value = _converted_dot_output(stablehlo_graph, shared_up_dot)
        shared_hidden = _recover_swiglu(stablehlo_graph, shared_gate_value, shared_up_value)
        if shared_down_dot.inputs[0] != shared_hidden:
            raise ValueError("shared down projection does not consume the recovered SwiGLU value")
        shared_output_id = _converted_dot_output(stablehlo_graph, shared_down_dot)
    except ValueError as error:
        raise _structured("shared_expert", error) from error

    try:
        global_experts = stablehlo_graph.value(router_logits_id).shape[1]
        for weight_id in (routed_gate_id, routed_up_id, routed_down_id):
            if stablehlo_graph.value(weight_id).shape[0] != global_experts:
                raise ValueError("ordinary semantic routed weights must contain the router's global expert axis")
        routed_gate_gather = _gather_using_weight(stablehlo_graph, routed_gate_id, expert_indices_id)
        routed_up_gather = _gather_using_weight(stablehlo_graph, routed_up_id, expert_indices_id)
        routed_down_gather = _gather_using_weight(stablehlo_graph, routed_down_id, expert_indices_id)
        routed_gate_dot = _dot_using_value(stablehlo_graph, routed_gate_gather.outputs[0])
        routed_up_dot = _dot_using_value(stablehlo_graph, routed_up_gather.outputs[0])
        routed_down_dot = _dot_using_value(stablehlo_graph, routed_down_gather.outputs[0])
        for operation in (routed_gate_dot, routed_up_dot):
            _validate_dot(operation, lhs_batch=(0,), rhs_batch=(0,), lhs_contract=(1,), rhs_contract=(3,))
            if operation.inputs[0] != x_id:
                raise ValueError("routed gate/up projection does not consume the token input")
        _validate_dot(
            routed_down_dot,
            lhs_batch=(0, 1),
            rhs_batch=(0, 1),
            lhs_contract=(2,),
            rhs_contract=(3,),
        )
        routed_gate_value = _converted_dot_output(stablehlo_graph, routed_gate_dot)
        routed_up_value = _converted_dot_output(stablehlo_graph, routed_up_dot)
        routed_hidden = _recover_swiglu(stablehlo_graph, routed_gate_value, routed_up_value)
        if routed_down_dot.inputs[0] != routed_hidden:
            raise ValueError("routed down projection does not consume the recovered SwiGLU value")
        routed_output_id = _converted_dot_output(stablehlo_graph, routed_down_dot)
    except ValueError as error:
        raise _structured("routed_experts", error) from error

    try:
        combine = _recover_weighted_combine(
            stablehlo_graph,
            shared_output_id=shared_output_id,
            routed_output_id=routed_output_id,
            route_weights_id=route_weights_id,
        )
        if stablehlo_graph.outputs[1:] != (expert_indices_id, route_weights_id):
            raise ValueError("returned router metadata does not match the recovered top-k values")
        reachable = set().union(
            *(_reachable_operation_ids(stablehlo_graph, output_id) for output_id in stablehlo_graph.outputs)
        )
        all_operation_ids = {operation.id for operation in stablehlo_graph.operations}
        if reachable != all_operation_ids:
            raise ValueError(f"MoE region has unmatched operations {sorted(all_operation_ids - reachable)}")
    except ValueError as error:
        raise _structured("combine", error) from error

    semantic_graph = TensorGraph()
    source_values = {value_id: stablehlo_graph.value(value_id) for value_id in stablehlo_graph.inputs}
    x_source = source_values[x_id]
    x = semantic_graph.input(x_source.name, shape=x_source.shape, dtype=x_source.dtype)
    parameters = {
        value_id: semantic_graph.parameter(value.name, shape=value.shape, dtype=value.dtype)
        for value_id, value in source_values.items()
        if value_id != x_id
    }
    router_logits = semantic_graph.linear(
        x,
        parameters[router_weight_id],
        name="router_logits",
        accumulation_dtype=gemm_accumulation_dtype,
        source_location=router_dot.source_location,
    )
    expert_indices, route_weights = semantic_graph.top_k_router(
        router_logits,
        name="routes",
        top_k=top_k,
        normalize_weights=True,
        source_location=top_k_op.source_location,
    )
    shared_output = semantic_graph.shared_expert_mlp(
        x,
        parameters[shared_gate_id],
        parameters[shared_up_id],
        parameters[shared_down_id],
        name="shared_expert",
        accumulation_dtype=gemm_accumulation_dtype,
        source_location=shared_gate_dot.source_location,
    )
    routed_output = semantic_graph.routed_expert_mlp(
        x,
        expert_indices,
        parameters[routed_gate_id],
        parameters[routed_up_id],
        parameters[routed_down_id],
        name="routed_experts",
        accumulation_dtype=gemm_accumulation_dtype,
        source_location=routed_gate_dot.source_location,
    )
    semantic_graph.weighted_expert_combine(
        shared_output,
        routed_output,
        route_weights,
        name="moe_output",
        source_location=combine.source_location,
    )
    return RecoveredMoERegion(
        graph=semantic_graph,
        source_operation_ids=tuple(sorted(all_operation_ids)),
    )


def _recover_router(
    graph: StableHLOGraph,
    router_logits_id: int,
) -> tuple[StableHLOOperation, int, int, int]:
    consumers = graph.consumers(router_logits_id)
    if len(consumers) != 1 or consumers[0].kind != "composite":
        raise ValueError("router logits do not feed exactly one StableHLO top-k composite")
    operation = consumers[0]
    if not isinstance(operation.attributes, CompositeAttributes):
        raise ValueError("top-k composite has no imported attributes")
    attributes = operation.attributes
    if attributes.name != "chlo.top_k" or attributes.version != 1 or len(operation.outputs) != 2:
        raise ValueError("router must use the version-1 chlo.top_k composite with value and index outputs")
    attribute_values = dict(attributes.attributes)
    match = INTEGER_ATTRIBUTE.fullmatch(attribute_values.get("k", ""))
    if match is None or set(attribute_values) != {"k"}:
        raise ValueError("top-k composite must carry one static i64 k attribute")
    top_k = int(match.group(1))
    top_values_id, expert_indices_id = operation.outputs
    if not 0 < top_k <= graph.value(router_logits_id).shape[1]:
        raise ValueError("top-k k must be positive and no larger than the global expert count")
    if graph.value(expert_indices_id).dtype is not DType.INT32:
        raise ValueError("top-k expert indices must be INT32")
    route_weights_id = graph.outputs[2]
    _validate_top_k_softmax(graph, top_values_id, route_weights_id)
    return operation, expert_indices_id, route_weights_id, top_k


def _validate_top_k_softmax(graph: StableHLOGraph, top_values_id: int, route_weights_id: int) -> None:
    divide = _producer(graph, route_weights_id, "divide")
    exponential_id, denominator_id = divide.inputs
    exponential = _producer(graph, exponential_id, "exponential")
    subtract = _producer(graph, exponential.inputs[0], "subtract")
    converted = _producer(graph, subtract.inputs[0], "convert")
    if converted.inputs != (top_values_id,) or graph.value(converted.outputs[0]).dtype is not DType.FP32:
        raise ValueError("top-k values are not explicitly converted to FP32 before normalization")
    maximum_path, maximum_id = _peel_broadcasts(graph, subtract.inputs[1])
    maximum = _producer(graph, maximum_id, "reduce")
    _validate_reduction(graph, maximum, source=converted.outputs[0], reducer="maximum", dimension=1)
    if tuple(operation.attributes.dimensions for operation in maximum_path) != ((0, 1), (0,)):
        raise ValueError("top-k maximum is not broadcast back over [tokens, top_k]")
    sum_path, sum_id = _peel_broadcasts(graph, denominator_id)
    summation = _producer(graph, sum_id, "reduce")
    _validate_reduction(graph, summation, source=exponential_id, reducer="add", dimension=1)
    if tuple(operation.attributes.dimensions for operation in sum_path) != ((0, 1), (0,)):
        raise ValueError("top-k sum is not broadcast back over [tokens, top_k]")


def _recover_swiglu(graph: StableHLOGraph, gate_id: int, up_id: int) -> int:
    matches = tuple(
        operation for operation in graph.consumers(up_id) if operation.kind == "multiply" and up_id in operation.inputs
    )
    if len(matches) != 1:
        raise ValueError("up projection does not feed exactly one SwiGLU multiply")
    multiply = matches[0]
    silu_id = _other_input(multiply, up_id)
    divide = _producer(graph, silu_id, "divide")
    if divide.inputs[0] != gate_id:
        raise ValueError("SwiGLU SiLU numerator is not the gate projection")
    denominator = _producer(graph, divide.inputs[1], "add")
    exponential_inputs = tuple(
        value_id for value_id in denominator.inputs if _producer_kind(graph, value_id) == "exponential"
    )
    if len(exponential_inputs) != 1:
        raise ValueError("SiLU denominator does not contain exactly one exponential")
    exponential = _producer(graph, exponential_inputs[0], "exponential")
    negate = _producer(graph, exponential.inputs[0], "negate")
    if negate.inputs != (gate_id,):
        raise ValueError("SiLU exponential does not consume the negated gate")
    one_id = _other_input(denominator, exponential.outputs[0])
    _, constant_id = _peel_broadcasts(graph, one_id)
    if _constant_number(_producer(graph, constant_id, "constant")) != 1.0:
        raise ValueError("SiLU denominator does not add one")
    return multiply.outputs[0]


def _gather_using_weight(
    graph: StableHLOGraph,
    weight_id: int,
    expert_indices_id: int,
) -> StableHLOOperation:
    matches = tuple(operation for operation in graph.consumers(weight_id) if operation.kind == "gather")
    if len(matches) != 1:
        raise ValueError(f"weight {graph.value(weight_id).name} does not feed exactly one expert gather")
    gather = matches[0]
    if not isinstance(gather.attributes, GatherAttributes):
        raise ValueError("expert gather has no imported dimension metadata")
    weight_shape = graph.value(weight_id).shape
    expected = GatherAttributes(
        offset_dimensions=(2, 3),
        collapsed_slice_dimensions=(0,),
        start_index_map=(0,),
        index_vector_dimension=2,
        slice_sizes=(1, *weight_shape[1:]),
    )
    if gather.attributes != expected:
        raise ValueError(f"expert gather has unsupported dimension mapping {gather.attributes}")
    start_indices = _producer(graph, gather.inputs[1], "broadcast_in_dim")
    if not isinstance(start_indices.attributes, BroadcastAttributes) or start_indices.attributes.dimensions != (0, 1):
        raise ValueError("expert gather does not append one canonical index-vector dimension")
    select = _producer(graph, start_indices.inputs[0], "select")
    if select.inputs[2] != expert_indices_id:
        raise ValueError("expert gather does not select from the recovered top-k indices")
    compare = _producer(graph, select.inputs[0], "compare")
    if (
        not isinstance(compare.attributes, CompareAttributes)
        or compare.attributes.direction != "LT"
        or compare.attributes.compare_type != "SIGNED"
        or compare.inputs[0] != expert_indices_id
        or _broadcast_scalar_integer(graph, compare.inputs[1]) != 0
    ):
        raise ValueError("expert gather index canonicalization is not signed index < 0")
    adjust = _producer(graph, select.inputs[1], "add")
    if expert_indices_id not in adjust.inputs:
        raise ValueError("negative expert indices are not adjusted from the top-k index")
    expert_count_id = _other_input(adjust, expert_indices_id)
    if _broadcast_scalar_integer(graph, expert_count_id) != weight_shape[0]:
        raise ValueError("negative expert indices are not adjusted by the expert count")
    return gather


def _recover_weighted_combine(
    graph: StableHLOGraph,
    *,
    shared_output_id: int,
    routed_output_id: int,
    route_weights_id: int,
) -> StableHLOOperation:
    combine = _producer(graph, graph.outputs[0], "add")
    if shared_output_id not in combine.inputs:
        raise ValueError("final add does not consume the shared expert output")
    routed_sum_id = _other_input(combine, shared_output_id)
    routed_convert = _producer(graph, routed_sum_id, "convert")
    reduction = _producer(graph, routed_convert.inputs[0], "reduce")
    if not isinstance(reduction.attributes, ReductionAttributes) or reduction.attributes != ReductionAttributes(
        dimensions=(1,), reducer="add"
    ):
        raise ValueError("weighted routed outputs are not summed over the top-k axis")
    if _constant_number(_producer(graph, reduction.inputs[1], "constant")) != 0.0:
        raise ValueError("weighted route reduction does not use a zero initializer")
    multiply = _producer(graph, reduction.inputs[0], "multiply")
    routed_converts = tuple(
        _producer(graph, value_id, "convert")
        for value_id in multiply.inputs
        if graph.producer(value_id) is not None and graph.producer(value_id).kind == "convert"
    )
    if len(routed_converts) != 1 or routed_converts[0].inputs != (routed_output_id,):
        raise ValueError("weighted combine does not convert the routed expert output to FP32")
    weight_id = _other_input(multiply, routed_converts[0].outputs[0])
    weight_path, weight_base = _peel_broadcasts(graph, weight_id)
    if weight_base != route_weights_id:
        raise ValueError("weighted combine does not consume the recovered route weights")
    if tuple(operation.attributes.dimensions for operation in weight_path) != ((0, 1, 2), (0, 1)):
        raise ValueError("route weights are not broadcast over only the hidden dimension")
    return combine


def _dot_using_weight(graph: StableHLOGraph, weight_id: int) -> StableHLOOperation:
    matches = tuple(
        operation
        for operation in graph.consumers(weight_id)
        if operation.kind == "dot_general" and operation.inputs[1] == weight_id
    )
    if len(matches) != 1:
        raise ValueError(f"weight {graph.value(weight_id).name} does not feed exactly one dot")
    return matches[0]


def _dot_using_value(graph: StableHLOGraph, value_id: int) -> StableHLOOperation:
    matches = tuple(
        operation
        for operation in graph.consumers(value_id)
        if operation.kind == "dot_general" and operation.inputs[1] == value_id
    )
    if len(matches) != 1:
        raise ValueError("gathered expert weight does not feed exactly one dot")
    return matches[0]


def _validate_dot(
    operation: StableHLOOperation,
    *,
    lhs_batch: tuple[int, ...],
    rhs_batch: tuple[int, ...],
    lhs_contract: tuple[int, ...],
    rhs_contract: tuple[int, ...],
) -> None:
    expected = DotAttributes(lhs_batch, rhs_batch, lhs_contract, rhs_contract)
    if operation.attributes != expected:
        raise ValueError(f"dot at {operation.source_location} has unsupported dimensions {operation.attributes}")


def _converted_dot_output(graph: StableHLOGraph, dot: StableHLOOperation) -> int:
    consumers = graph.consumers(dot.outputs[0])
    if len(consumers) != 1 or consumers[0].kind != "convert":
        raise ValueError("FP32 dot does not feed one explicit BF16 conversion")
    output_id = consumers[0].outputs[0]
    if graph.value(dot.outputs[0]).dtype is not DType.FP32 or graph.value(output_id).dtype is not DType.BF16:
        raise ValueError("expert dot must explicitly convert FP32 accumulation to BF16")
    return output_id


def _validate_reduction(
    graph: StableHLOGraph,
    operation: StableHLOOperation,
    *,
    source: int,
    reducer: str,
    dimension: int,
) -> None:
    if operation.inputs[0] != source or operation.attributes != ReductionAttributes((dimension,), reducer):
        raise ValueError(f"router softmax {reducer} reduction has the wrong source or axis")
    initializer = _producer(graph, operation.inputs[1], "constant")
    if reducer == "add" and _constant_number(initializer) != 0.0:
        raise ValueError("router softmax sum does not use zero initialization")
    if reducer == "maximum" and not _negative_infinity(initializer):
        raise ValueError("router softmax maximum does not use negative-infinity initialization")


def _peel_broadcasts(
    graph: StableHLOGraph,
    value_id: int,
) -> tuple[tuple[StableHLOOperation, ...], int]:
    operations: list[StableHLOOperation] = []
    while True:
        producer = graph.producer(value_id)
        if producer is None or producer.kind != "broadcast_in_dim":
            return tuple(operations), value_id
        if not isinstance(producer.attributes, BroadcastAttributes):
            raise ValueError("broadcast has no imported dimension metadata")
        operations.append(producer)
        value_id = producer.inputs[0]


def _broadcast_scalar_integer(graph: StableHLOGraph, value_id: int) -> int:
    broadcasts, constant_id = _peel_broadcasts(graph, value_id)
    if len(broadcasts) != 1 or broadcasts[0].attributes != BroadcastAttributes(()) or graph.value(constant_id).shape:
        raise ValueError("expected one scalar integer broadcast")
    return int(_constant_number(_producer(graph, constant_id, "constant")))


def _constant_number(operation: StableHLOOperation) -> float:
    if not isinstance(operation.attributes, ConstantAttributes):
        raise ValueError("constant has no imported literal")
    match = SCALAR_LITERAL.search(operation.attributes.literal)
    if match is None:
        raise ValueError("constant is not a scalar literal")
    return float(match.group(1))


def _negative_infinity(operation: StableHLOOperation) -> bool:
    if not isinstance(operation.attributes, ConstantAttributes):
        return False
    literal = operation.attributes.literal.upper()
    return "0XFF800000" in literal or "-INF" in literal


def _producer(graph: StableHLOGraph, value_id: int, expected_kind: str) -> StableHLOOperation:
    operation = graph.producer(value_id)
    if operation is None or operation.kind != expected_kind:
        actual = "input" if operation is None else operation.kind
        raise ValueError(f"expected {expected_kind} producing {graph.value(value_id).name}, found {actual}")
    return operation


def _producer_kind(graph: StableHLOGraph, value_id: int) -> str | None:
    operation = graph.producer(value_id)
    return operation.kind if operation is not None else None


def _other_input(operation: StableHLOOperation, known_input: int) -> int:
    remaining = tuple(value_id for value_id in operation.inputs if value_id != known_input)
    if len(operation.inputs) != 2 or len(remaining) != 1:
        raise ValueError(f"{operation.kind} does not have one input besides the expected value")
    return remaining[0]


def _reachable_operation_ids(graph: StableHLOGraph, value_id: int) -> set[int]:
    operation = graph.producer(value_id)
    if operation is None:
        return set()
    reachable = {operation.id}
    for input_id in operation.inputs:
        reachable.update(_reachable_operation_ids(graph, input_id))
    return reachable


def _structured(stage: str, error: ValueError) -> MoESemanticRecoveryError:
    if isinstance(error, MoESemanticRecoveryError):
        return error
    return MoESemanticRecoveryError(stage=stage, reason=str(error))
