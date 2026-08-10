# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frontend canonicalization from normalized StableHLO.

The named ``TensorGraph`` operations returned here are transitional frontend
objects. Current scheduling must erase them into generic Flow or TensorProgram
algebra first. Named RMS recovery lives in ``reference_semantic_recovery``;
``reference_pipeline`` contains the historical callers that schedule names
directly for comparison artifacts.
"""

import math
import re
from collections.abc import Callable
from dataclasses import dataclass

from shuttle.experimental.stablehlo_import import (
    BroadcastAttributes,
    CompareAttributes,
    ConcatenateAttributes,
    ConstantAttributes,
    DotAttributes,
    IotaAttributes,
    ReductionAttributes,
    SliceAttributes,
    StableHLOGraph,
    StableHLOOperation,
    TransposeAttributes,
)
from shuttle.ir import DType
from tile_lifetime.ir import TensorGraph, TensorValue

SCALAR_LITERAL = re.compile(r"dense<([^>]+)>")


class SemanticRecoveryError(ValueError):
    """Raised when a StableHLO fragment does not have the required semantics."""


@dataclass(frozen=True)
class RecoveredAttentionRegion:
    """Recovered exact causal GQA graph and its source operation mapping."""

    graph: TensorGraph
    source_operation_ids: tuple[int, ...]


@dataclass(frozen=True)
class RecoveredDenseRegion:
    """Recovered connected dense Llama region and source operation mapping."""

    graph: TensorGraph
    source_operation_ids: tuple[int, ...]


def recover_dense_transformer_region(
    stablehlo_graph: StableHLOGraph,
    *,
    gemm_accumulation_dtype: DType,
) -> RecoveredDenseRegion:
    """Recover the bounded JAX dense debug region through the next QKV/RoPE."""
    if len(stablehlo_graph.inputs) != 10 or len(stablehlo_graph.outputs) != 4:
        raise SemanticRecoveryError("dense debug region requires ten inputs and four outputs")
    x_id, qkv_weight_id, output_weight_id, mlp_gamma_id, gate_up_weight_id = stablehlo_graph.inputs[:5]
    down_weight_id, next_gamma_id, next_qkv_weight_id, sine_id, cosine_id = stablehlo_graph.inputs[5:]

    qkv_dot = _dot_using_weight(stablehlo_graph, qkv_weight_id)
    output_dot = _dot_using_weight(stablehlo_graph, output_weight_id)
    gate_up_dot = _dot_using_weight(stablehlo_graph, gate_up_weight_id)
    down_dot = _dot_using_weight(stablehlo_graph, down_weight_id)
    next_qkv_dot = _dot_using_weight(stablehlo_graph, next_qkv_weight_id)
    for operation in (qkv_dot, output_dot, gate_up_dot, down_dot, next_qkv_dot):
        _validate_matrix_dot(operation)

    if qkv_dot.inputs[0] != x_id:
        raise SemanticRecoveryError("initial QKV projection does not consume the residual input")
    query_id, key_id, value_id = _recover_qkv_partition(stablehlo_graph, qkv_dot)
    attention_output_id, rotated_query_id, rotated_key_id = _recover_internal_attention(
        stablehlo_graph,
        output_dot.inputs[0],
        value_id=value_id,
    )
    _validate_rope(stablehlo_graph, query_id, rotated_query_id, sine_id=sine_id, cosine_id=cosine_id)
    _validate_rope(stablehlo_graph, key_id, rotated_key_id, sine_id=sine_id, cosine_id=cosine_id)

    attention_view = _producer(stablehlo_graph, output_dot.inputs[0], expected_kind="reshape")
    if attention_view.inputs[0] != attention_output_id:
        raise SemanticRecoveryError("attention flattening view does not consume the recovered attention output")
    projected_id = _converted_dot_output(stablehlo_graph, output_dot)
    first_residual = _binary_consumer(stablehlo_graph, projected_id, x_id, expected_kind="add")
    mlp_epsilon, mlp_reduction, mlp_normalized_id = _recover_rms_before_dot(
        stablehlo_graph,
        gate_up_dot,
        residual_id=first_residual.outputs[0],
        gamma_id=mlp_gamma_id,
    )
    gate_up_id = _converted_dot_output(stablehlo_graph, gate_up_dot)
    activated_id = _recover_pairwise_swiglu(stablehlo_graph, gate_up_id)
    if down_dot.inputs[0] != activated_id:
        raise SemanticRecoveryError("down projection does not consume the recovered pairwise SwiGLU output")
    down_id = _converted_dot_output(stablehlo_graph, down_dot)
    second_residual = _binary_consumer(
        stablehlo_graph,
        down_id,
        first_residual.outputs[0],
        expected_kind="add",
    )
    if stablehlo_graph.outputs[0] != second_residual.outputs[0]:
        raise SemanticRecoveryError("first returned value is not the second residual stream")
    next_epsilon, next_reduction, next_normalized_id = _recover_rms_before_dot(
        stablehlo_graph,
        next_qkv_dot,
        residual_id=second_residual.outputs[0],
        gamma_id=next_gamma_id,
    )
    next_query_id, next_key_id, next_value_id = _recover_qkv_partition(stablehlo_graph, next_qkv_dot)
    next_rotated_query_id, next_rotated_key_id, returned_value_id = stablehlo_graph.outputs[1:]
    _validate_rope(
        stablehlo_graph,
        next_query_id,
        next_rotated_query_id,
        sine_id=sine_id,
        cosine_id=cosine_id,
    )
    _validate_rope(
        stablehlo_graph,
        next_key_id,
        next_rotated_key_id,
        sine_id=sine_id,
        cosine_id=cosine_id,
    )
    if returned_value_id != next_value_id:
        raise SemanticRecoveryError("final returned V is not the following QKV projection's V partition")

    reachable_ids = set().union(
        *(_reachable_operation_ids(stablehlo_graph, output_id) for output_id in stablehlo_graph.outputs)
    )
    all_ids = {operation.id for operation in stablehlo_graph.operations}
    if reachable_ids != all_ids:
        raise SemanticRecoveryError(f"dense debug region has unmatched operations {sorted(all_ids - reachable_ids)}")

    semantic_graph = TensorGraph()
    source_values = {value_id: stablehlo_graph.value(value_id) for value_id in stablehlo_graph.inputs}
    x_source = source_values[x_id]
    x = semantic_graph.input(x_source.name, shape=x_source.shape, dtype=x_source.dtype)
    parameters = {
        value_id: semantic_graph.parameter(value.name, shape=value.shape, dtype=value.dtype)
        for value_id, value in source_values.items()
        if value_id != x_id
    }
    query_shape = stablehlo_graph.value(query_id).shape
    batch, sequence, query_heads, head_dimension = query_shape
    key_value_heads = stablehlo_graph.value(key_id).shape[2]
    x_bsh = semantic_graph.view(
        x,
        shape=(batch, sequence, x.shape[-1]),
        name="x_bsh",
        source_location=qkv_dot.source_location,
    )
    query, key, value = semantic_graph.qkv_projection(
        x_bsh,
        parameters[qkv_weight_id],
        name="qkv",
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dimension=head_dimension,
        accumulation_dtype=gemm_accumulation_dtype,
        source_location=qkv_dot.source_location,
    )
    query, key = semantic_graph.rope(
        query,
        key,
        parameters[sine_id],
        parameters[cosine_id],
        name="rotated",
        rotary_dimension=head_dimension,
        source_location=stablehlo_graph.producer(rotated_query_id).source_location,
    )
    attention = semantic_graph.scaled_dot_product_attention(
        query,
        key,
        value,
        name="attention",
        scale=head_dimension**-0.5,
        causal=True,
        accumulation_dtype=stablehlo_graph.value(_attention_score_dot(stablehlo_graph).outputs[0]).dtype,
        source_location=_attention_score_dot(stablehlo_graph).source_location,
    )
    attention_flat = semantic_graph.view(
        attention,
        shape=stablehlo_graph.value(output_dot.inputs[0]).shape,
        name="attention_flat",
        source_location=attention_view.source_location,
    )
    projected = semantic_graph.linear(
        attention_flat,
        parameters[output_weight_id],
        name="projected",
        accumulation_dtype=gemm_accumulation_dtype,
        source_location=output_dot.source_location,
    )
    x1 = semantic_graph.residual_add(projected, x, name="x1", source_location=first_residual.source_location)
    mlp_input = semantic_graph.rms_norm(
        x1,
        parameters[mlp_gamma_id],
        name="mlp_input",
        axis=-1,
        epsilon=mlp_epsilon,
        reduction_dtype=stablehlo_graph.value(mlp_reduction.inputs[0]).dtype,
        source_location=mlp_reduction.source_location,
    )
    gate_up = semantic_graph.linear(
        mlp_input,
        parameters[gate_up_weight_id],
        name="gate_up",
        accumulation_dtype=gemm_accumulation_dtype,
        source_location=gate_up_dot.source_location,
    )
    activated = semantic_graph.pairwise_swiglu(
        gate_up,
        name="activated",
        source_location=stablehlo_graph.producer(activated_id).source_location,
    )
    down = semantic_graph.linear(
        activated,
        parameters[down_weight_id],
        name="down",
        accumulation_dtype=gemm_accumulation_dtype,
        source_location=down_dot.source_location,
    )
    x2 = semantic_graph.residual_add(down, x1, name="x2", source_location=second_residual.source_location)
    next_input = semantic_graph.rms_norm(
        x2,
        parameters[next_gamma_id],
        name="next_input",
        axis=-1,
        epsilon=next_epsilon,
        reduction_dtype=stablehlo_graph.value(next_reduction.inputs[0]).dtype,
        source_location=next_reduction.source_location,
    )
    next_input_bsh = semantic_graph.view(
        next_input,
        shape=(batch, sequence, x.shape[-1]),
        name="next_input_bsh",
        source_location=next_qkv_dot.source_location,
    )
    next_query, next_key, _ = semantic_graph.qkv_projection(
        next_input_bsh,
        parameters[next_qkv_weight_id],
        name="next_qkv",
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dimension=head_dimension,
        accumulation_dtype=gemm_accumulation_dtype,
        source_location=next_qkv_dot.source_location,
    )
    semantic_graph.rope(
        next_query,
        next_key,
        parameters[sine_id],
        parameters[cosine_id],
        name="next_rotated",
        rotary_dimension=head_dimension,
        source_location=stablehlo_graph.producer(next_rotated_query_id).source_location,
    )
    if mlp_normalized_id != gate_up_dot.inputs[0] or next_normalized_id != next_qkv_dot.inputs[0]:
        raise SemanticRecoveryError("recovered normalization does not feed its expected projection")
    return RecoveredDenseRegion(graph=semantic_graph, source_operation_ids=tuple(sorted(all_ids)))


def recover_attention_region(
    stablehlo_graph: StableHLOGraph,
    *,
    output_name: str,
    output_index: int = 0,
) -> RecoveredAttentionRegion:
    """Recover exact causal GQA from the initial normalized JAX StableHLO form."""
    output_id = stablehlo_graph.outputs[output_index]
    output_convert = _producer(stablehlo_graph, output_id, expected_kind="convert")
    output_transpose = _producer(stablehlo_graph, output_convert.inputs[0], expected_kind="transpose")
    if not isinstance(output_transpose.attributes, TransposeAttributes) or output_transpose.attributes.permutation != (
        0,
        3,
        1,
        2,
    ):
        raise SemanticRecoveryError("attention output does not transpose PV into [batch, query, head, dimension]")
    pv_dot = _producer(stablehlo_graph, output_transpose.inputs[0], expected_kind="dot_general")
    _validate_pv_dot(pv_dot)

    repeated_value_id, probability_value_id = pv_dot.inputs
    probability_convert = _producer(stablehlo_graph, probability_value_id, expected_kind="convert")
    probability_divide = _producer(stablehlo_graph, probability_convert.inputs[0], expected_kind="divide")
    exponential = _producer(stablehlo_graph, probability_divide.inputs[0], expected_kind="exponential")
    sum_broadcasts, sum_base = _peel(stablehlo_graph, probability_divide.inputs[1], kinds=("broadcast_in_dim",))
    sum_reduction = _producer(stablehlo_graph, sum_base, expected_kind="reduce")
    _validate_softmax_reduction(stablehlo_graph, sum_reduction, reducer="add", input_id=exponential.outputs[0])

    subtract = _producer(stablehlo_graph, exponential.inputs[0], expected_kind="subtract")
    masked_score_id, row_maximum_id = subtract.inputs
    maximum_broadcasts, maximum_base = _peel(stablehlo_graph, row_maximum_id, kinds=("broadcast_in_dim",))
    maximum_guard = _producer(stablehlo_graph, maximum_base, expected_kind="maximum")
    reduced_maximum_id, guard_infinity_id = _partition_inputs(
        stablehlo_graph,
        maximum_guard,
        predicate=lambda value_id: _origin_kind(stablehlo_graph, value_id, through=()) == "reduce",
        description="row-maximum reduction",
    )
    max_reduction = _producer(stablehlo_graph, reduced_maximum_id, expected_kind="reduce")
    _validate_softmax_reduction(stablehlo_graph, max_reduction, reducer="maximum", input_id=masked_score_id)
    _require_negative_infinity(stablehlo_graph, guard_infinity_id)

    mask_select = _producer(stablehlo_graph, masked_score_id, expected_kind="select")
    if len(mask_select.inputs) != 3:
        raise SemanticRecoveryError("causal mask select does not have predicate, true, and false inputs")
    mask_id, scaled_score_id, masked_fill_id = mask_select.inputs
    _require_negative_infinity(stablehlo_graph, masked_fill_id)
    _validate_causal_mask(stablehlo_graph, mask_id, query_axis=2, key_axis=3)

    score_scale = _producer(stablehlo_graph, scaled_score_id, expected_kind="multiply")
    scale_value_id, score_dot_id = _partition_inputs(
        stablehlo_graph,
        score_scale,
        predicate=lambda value_id: _origin_kind(stablehlo_graph, value_id, through=("broadcast_in_dim",)) == "constant",
        description="scalar attention scale",
    )
    scale_path, scale_constant_id = _peel(stablehlo_graph, scale_value_id, kinds=("broadcast_in_dim",))
    scale = _constant_float(_producer(stablehlo_graph, scale_constant_id, expected_kind="constant"))
    qk_dot = _producer(stablehlo_graph, score_dot_id, expected_kind="dot_general")
    _validate_qk_dot(qk_dot)

    query_input_id, repeated_key_id = qk_dot.inputs
    if query_input_id not in stablehlo_graph.inputs:
        raise SemanticRecoveryError("QK query operand is not a function input")
    key_input_id, key_ratio = _recover_repeated_heads(stablehlo_graph, repeated_key_id, role="key")
    value_input_id, value_ratio = _recover_repeated_heads(stablehlo_graph, repeated_value_id, role="value")
    if key_ratio != value_ratio:
        raise SemanticRecoveryError("K and V use different grouped-query head mappings")

    if stablehlo_graph.consumers(qk_dot.outputs[0]) != (score_scale,):
        raise SemanticRecoveryError("QK score tensor has a consumer outside the attention fragment")
    if stablehlo_graph.consumers(probability_divide.outputs[0]) != (probability_convert,):
        raise SemanticRecoveryError("probability tensor has a consumer outside the attention fragment")
    if stablehlo_graph.consumers(probability_convert.outputs[0]) != (pv_dot,):
        raise SemanticRecoveryError("converted probability tensor has a consumer outside the PV contraction")

    semantic_graph = TensorGraph()
    query = _graph_input(semantic_graph, stablehlo_graph, query_input_id)
    key = _graph_input(semantic_graph, stablehlo_graph, key_input_id)
    value = _graph_input(semantic_graph, stablehlo_graph, value_input_id)
    semantic_graph.scaled_dot_product_attention(
        query,
        key,
        value,
        name=output_name,
        scale=scale,
        causal=True,
        accumulation_dtype=stablehlo_graph.value(qk_dot.outputs[0]).dtype,
        source_location=qk_dot.source_location,
    )

    matched_ids = {
        output_convert.id,
        output_transpose.id,
        pv_dot.id,
        probability_convert.id,
        probability_divide.id,
        exponential.id,
        sum_reduction.id,
        subtract.id,
        maximum_guard.id,
        max_reduction.id,
        mask_select.id,
        score_scale.id,
        qk_dot.id,
        *[operation.id for operation in sum_broadcasts],
        *[operation.id for operation in maximum_broadcasts],
        *[operation.id for operation in scale_path],
    }
    matched_ids.update(_reachable_operation_ids(stablehlo_graph, output_id))
    return RecoveredAttentionRegion(graph=semantic_graph, source_operation_ids=tuple(sorted(matched_ids)))


def _dot_using_weight(graph: StableHLOGraph, weight_id: int) -> StableHLOOperation:
    matches = tuple(
        operation
        for operation in graph.consumers(weight_id)
        if operation.kind == "dot_general" and len(operation.inputs) == 2 and operation.inputs[1] == weight_id
    )
    if len(matches) != 1:
        raise SemanticRecoveryError(f"weight {graph.value(weight_id).name} does not feed exactly one matrix dot")
    return matches[0]


def _converted_dot_output(graph: StableHLOGraph, dot: StableHLOOperation) -> int:
    consumers = graph.consumers(dot.outputs[0])
    if len(consumers) != 1 or consumers[0].kind != "convert":
        raise SemanticRecoveryError(f"dot at {dot.source_location} is not followed by one explicit output conversion")
    converted = consumers[0].outputs[0]
    if graph.value(dot.outputs[0]).dtype is not DType.FP32 or graph.value(converted).dtype is not DType.BF16:
        raise SemanticRecoveryError("dense debug GEMM must explicitly convert FP32 accumulation to BF16")
    return converted


def _recover_qkv_partition(graph: StableHLOGraph, dot: StableHLOOperation) -> tuple[int, int, int]:
    packed_id = _converted_dot_output(graph, dot)
    slices = tuple(operation for operation in graph.consumers(packed_id) if operation.kind == "slice")
    if len(slices) != 3 or len(graph.consumers(packed_id)) != 3:
        raise SemanticRecoveryError("combined QKV output is not partitioned by exactly three static slices")
    ordered: list[tuple[int, StableHLOOperation]] = []
    cursor = 0
    packed_shape = graph.value(packed_id).shape
    for operation in slices:
        if not isinstance(operation.attributes, SliceAttributes):
            raise SemanticRecoveryError("QKV partition slice has no static bounds")
        attributes = operation.attributes
        if attributes.start_indices[0] != 0 or attributes.limit_indices[0] != packed_shape[0]:
            raise SemanticRecoveryError("QKV partition does not preserve the token dimension")
        if attributes.strides != (1, 1):
            raise SemanticRecoveryError("QKV partition slices must have unit stride")
        ordered.append((attributes.start_indices[1], operation))
    outputs: list[int] = []
    for start, operation in sorted(ordered):
        attributes = operation.attributes
        assert isinstance(attributes, SliceAttributes)
        if start != cursor:
            raise SemanticRecoveryError("QKV partitions are not contiguous")
        cursor = attributes.limit_indices[1]
        consumers = graph.consumers(operation.outputs[0])
        if len(consumers) != 1 or consumers[0].kind != "reshape":
            raise SemanticRecoveryError("QKV partition is not reshaped into BSHD")
        outputs.append(consumers[0].outputs[0])
    if cursor != packed_shape[1]:
        raise SemanticRecoveryError("QKV partitions do not cover the packed projection width")
    query_shape, key_shape, value_shape = (graph.value(value_id).shape for value_id in outputs)
    if len(query_shape) != 4 or len(key_shape) != 4 or key_shape != value_shape:
        raise SemanticRecoveryError("QKV partitions do not produce one Q and matching K/V BSHD tensors")
    return outputs[0], outputs[1], outputs[2]


def _recover_internal_attention(
    graph: StableHLOGraph,
    flattened_output_id: int,
    *,
    value_id: int,
) -> tuple[int, int, int]:
    flatten = _producer(graph, flattened_output_id, expected_kind="reshape")
    output_convert = _producer(graph, flatten.inputs[0], expected_kind="convert")
    output_transpose = _producer(graph, output_convert.inputs[0], expected_kind="transpose")
    if not isinstance(output_transpose.attributes, TransposeAttributes) or output_transpose.attributes.permutation != (
        0,
        3,
        1,
        2,
    ):
        raise SemanticRecoveryError("attention output does not transpose PV into BSHD")
    pv_dot = _producer(graph, output_transpose.inputs[0], expected_kind="dot_general")
    _validate_pv_dot(pv_dot)
    repeated_value_id, probability_value_id = pv_dot.inputs
    if _recover_repeated_heads_from(graph, repeated_value_id, role="value")[0] != value_id:
        raise SemanticRecoveryError("PV contraction does not consume the QKV projection's V partition")

    probability_convert = _producer(graph, probability_value_id, expected_kind="convert")
    probability_divide = _producer(graph, probability_convert.inputs[0], expected_kind="divide")
    exponential = _producer(graph, probability_divide.inputs[0], expected_kind="exponential")
    _, sum_base = _peel(graph, probability_divide.inputs[1], kinds=("broadcast_in_dim",))
    sum_reduction = _producer(graph, sum_base, expected_kind="reduce")
    _validate_softmax_reduction(graph, sum_reduction, reducer="add", input_id=exponential.outputs[0])
    subtract = _producer(graph, exponential.inputs[0], expected_kind="subtract")
    masked_score_id, row_maximum_id = subtract.inputs
    _, maximum_base = _peel(graph, row_maximum_id, kinds=("broadcast_in_dim",))
    maximum_guard = _producer(graph, maximum_base, expected_kind="maximum")
    reduced_maximum_id, guard_infinity_id = _partition_inputs(
        graph,
        maximum_guard,
        predicate=lambda candidate: _origin_kind(graph, candidate, through=()) == "reduce",
        description="row-maximum reduction",
    )
    max_reduction = _producer(graph, reduced_maximum_id, expected_kind="reduce")
    _validate_softmax_reduction(graph, max_reduction, reducer="maximum", input_id=masked_score_id)
    _require_negative_infinity(graph, guard_infinity_id)
    mask_select = _producer(graph, masked_score_id, expected_kind="select")
    mask_id, scaled_score_id, masked_fill_id = mask_select.inputs
    _require_negative_infinity(graph, masked_fill_id)
    _validate_causal_mask(graph, mask_id, query_axis=2, key_axis=3)
    score_scale = _producer(graph, scaled_score_id, expected_kind="multiply")
    scale_value_id, score_dot_id = _partition_inputs(
        graph,
        score_scale,
        predicate=lambda candidate: _origin_kind(graph, candidate, through=("broadcast_in_dim",)) == "constant",
        description="scalar attention scale",
    )
    _, scale_constant_id = _peel(graph, scale_value_id, kinds=("broadcast_in_dim",))
    scale = _constant_float(_producer(graph, scale_constant_id, expected_kind="constant"))
    qk_dot = _producer(graph, score_dot_id, expected_kind="dot_general")
    _validate_qk_dot(qk_dot)
    query_id, repeated_key_id = qk_dot.inputs
    key_id, _ = _recover_repeated_heads_from(graph, repeated_key_id, role="key")
    if not math.isclose(scale, graph.value(query_id).shape[-1] ** -0.5, rel_tol=1e-6):
        raise SemanticRecoveryError("attention scale is not inverse square root of the head dimension")
    return output_convert.outputs[0], query_id, key_id


def _recover_repeated_heads_from(graph: StableHLOGraph, value_id: int, *, role: str) -> tuple[int, int]:
    final_reshape = _producer(graph, value_id, expected_kind="reshape")
    broadcast = _producer(graph, final_reshape.inputs[0], expected_kind="broadcast_in_dim")
    initial_reshape = _producer(graph, broadcast.inputs[0], expected_kind="reshape")
    input_id = initial_reshape.inputs[0]
    input_shape = graph.value(input_id).shape
    inserted_shape = graph.value(initial_reshape.outputs[0]).shape
    broadcast_shape = graph.value(broadcast.outputs[0]).shape
    output_shape = graph.value(final_reshape.outputs[0]).shape
    if len(input_shape) != 4 or inserted_shape != (*input_shape[:3], 1, input_shape[3]):
        raise SemanticRecoveryError(f"repeated {role} heads do not insert one group axis")
    ratio = broadcast_shape[3]
    if ratio <= 1 or broadcast_shape != (*input_shape[:3], ratio, input_shape[3]):
        raise SemanticRecoveryError(f"repeated {role} heads do not expand only the group axis")
    if output_shape != (input_shape[0], input_shape[1], input_shape[2] * ratio, input_shape[3]):
        raise SemanticRecoveryError(f"repeated {role} heads do not flatten KV and group axes")
    return input_id, ratio


def _validate_rope(
    graph: StableHLOGraph,
    input_id: int,
    output_id: int,
    *,
    sine_id: int,
    cosine_id: int,
) -> None:
    final_reshape = _producer(graph, output_id, expected_kind="reshape")
    concatenate = _producer(graph, final_reshape.inputs[0], expected_kind="concatenate")
    if graph.value(output_id).shape != graph.value(input_id).shape:
        raise SemanticRecoveryError("RoPE changes the Q/K shape")
    if len(concatenate.inputs) != 2:
        raise SemanticRecoveryError("RoPE does not concatenate one rotated even/odd pair")
    if not isinstance(concatenate.attributes, ConcatenateAttributes) or concatenate.attributes.dimension != 4:
        raise SemanticRecoveryError("RoPE does not concatenate adjacent pairs along the innermost pair dimension")
    pair_values = []
    for value_id in concatenate.inputs:
        broadcasts, base = _peel(graph, value_id, kinds=("broadcast_in_dim",))
        if not broadcasts:
            raise SemanticRecoveryError("RoPE pair result is not expanded for adjacent concatenation")
        pair_values.append(base)
    pair_operations = tuple(graph.producer(value_id) for value_id in pair_values)
    if {operation.kind if operation is not None else None for operation in pair_operations} != {"add", "subtract"}:
        raise SemanticRecoveryError("RoPE does not use the exact even subtract / odd add form")
    ancestors = _ancestor_value_ids(graph, output_id)
    if not {input_id, sine_id, cosine_id}.issubset(ancestors):
        raise SemanticRecoveryError("RoPE output does not depend on Q/K and both sine/cosine tables")
    pair_reshape_consumers = tuple(
        operation
        for operation in graph.consumers(input_id)
        if operation.kind == "reshape" and graph.value(operation.outputs[0]).shape[-1] == 2
    )
    if len(pair_reshape_consumers) != 1:
        raise SemanticRecoveryError("RoPE input is not reshaped into adjacent pairs")
    slices = tuple(
        operation for operation in graph.consumers(pair_reshape_consumers[0].outputs[0]) if operation.kind == "slice"
    )
    if len(slices) != 2:
        raise SemanticRecoveryError("RoPE does not extract exactly two adjacent pair components")
    ordered_slices = sorted(
        slices,
        key=lambda operation: (
            operation.attributes.start_indices[-1] if isinstance(operation.attributes, SliceAttributes) else -1
        ),
    )
    components = []
    for operation in ordered_slices:
        consumers = graph.consumers(operation.outputs[0])
        if len(consumers) != 1 or consumers[0].kind != "reshape":
            raise SemanticRecoveryError("RoPE pair component is not squeezed exactly once")
        components.append(consumers[0].outputs[0])
    even_id, odd_id = components
    labels = {even_id: "even", odd_id: "odd", sine_id: "sine", cosine_id: "cosine"}
    signatures: dict[str, set[frozenset[str]]] = {"add": set(), "subtract": set()}
    for operation in pair_operations:
        assert operation is not None
        for multiply_id in operation.inputs:
            multiply = _producer(graph, multiply_id, expected_kind="multiply")
            sources = []
            for operand_id in multiply.inputs:
                if operand_id in labels:
                    sources.append(labels[operand_id])
                    continue
                _, base_id = _peel(graph, operand_id, kinds=("broadcast_in_dim",))
                if base_id in labels:
                    sources.append(labels[base_id])
            signature = frozenset(sources)
            signatures[operation.kind].add(signature)
    if signatures["subtract"] != {frozenset(("even", "cosine")), frozenset(("odd", "sine"))}:
        raise SemanticRecoveryError("RoPE even component is not even*cosine - odd*sine")
    if signatures["add"] != {frozenset(("even", "sine")), frozenset(("odd", "cosine"))}:
        raise SemanticRecoveryError("RoPE odd component is not even*sine + odd*cosine")


def _ancestor_value_ids(graph: StableHLOGraph, value_id: int) -> set[int]:
    ancestors = {value_id}
    producer = graph.producer(value_id)
    if producer is None:
        return ancestors
    for input_id in producer.inputs:
        ancestors.update(_ancestor_value_ids(graph, input_id))
    return ancestors


def _binary_consumer(
    graph: StableHLOGraph,
    left_id: int,
    right_id: int,
    *,
    expected_kind: str,
) -> StableHLOOperation:
    matches = tuple(
        operation
        for operation in graph.consumers(left_id)
        if operation.kind == expected_kind
        and len(operation.inputs) == 2
        and set(operation.inputs) == {left_id, right_id}
    )
    if len(matches) != 1:
        raise SemanticRecoveryError(f"values do not feed exactly one {expected_kind} operation")
    return matches[0]


def _recover_rms_before_dot(
    graph: StableHLOGraph,
    consumer_dot: StableHLOOperation,
    *,
    residual_id: int,
    gamma_id: int,
) -> tuple[float, StableHLOOperation, int]:
    normalized_convert = _producer(graph, consumer_dot.inputs[0], expected_kind="convert")
    scaled_normalized = _producer(graph, normalized_convert.inputs[0], expected_kind="multiply")
    inverse_rms_value, gamma_scaled_value = _partition_inputs(
        graph,
        scaled_normalized,
        predicate=lambda value_id: _origin_kind(graph, value_id, through=("broadcast_in_dim",)) == "rsqrt",
        description="inverse-RMS broadcast",
    )
    _, inverse_rms_base = _peel(graph, inverse_rms_value, kinds=("broadcast_in_dim",))
    inverse_rms = _producer(graph, inverse_rms_base, expected_kind="rsqrt")
    gamma_scaled = _producer(graph, gamma_scaled_value, expected_kind="multiply")
    gamma_value, rms_input_fp32 = _partition_inputs(
        graph,
        gamma_scaled,
        predicate=lambda value_id: gamma_id in _ancestor_value_ids(graph, value_id),
        description="gamma broadcast",
    )
    _, gamma_base = _peel(graph, gamma_value, kinds=("broadcast_in_dim", "convert"))
    if gamma_base != gamma_id:
        raise SemanticRecoveryError("RMS gamma path does not originate at the expected parameter")
    residual_convert = _producer(graph, rms_input_fp32, expected_kind="convert")
    if residual_convert.inputs != (residual_id,):
        raise SemanticRecoveryError("RMSNorm does not consume the expected residual stream")
    epsilon, reduction, _ = _recover_rms_reduction(graph, inverse_rms, rms_input_fp32=rms_input_fp32)
    attributes = _reduction_attributes(reduction)
    if attributes.dimensions != (len(graph.value(residual_id).shape) - 1,):
        raise SemanticRecoveryError("RMSNorm does not reduce the hidden dimension")
    return epsilon, reduction, normalized_convert.outputs[0]


def _recover_pairwise_swiglu(graph: StableHLOGraph, gate_up_id: int) -> int:
    reshapes = tuple(operation for operation in graph.consumers(gate_up_id) if operation.kind == "reshape")
    if len(reshapes) != 1 or graph.value(reshapes[0].outputs[0]).shape[-1] != 2:
        raise SemanticRecoveryError("gate/up projection is not reshaped into adjacent pairs")
    pair_value = reshapes[0].outputs[0]
    slices = tuple(operation for operation in graph.consumers(pair_value) if operation.kind == "slice")
    if len(slices) != 2:
        raise SemanticRecoveryError("pairwise SwiGLU does not extract gate and up")
    ordered = sorted(
        slices,
        key=lambda operation: (
            operation.attributes.start_indices[-1] if isinstance(operation.attributes, SliceAttributes) else -1
        ),
    )
    gate_reshape = graph.consumers(ordered[0].outputs[0])
    up_reshape = graph.consumers(ordered[1].outputs[0])
    if len(gate_reshape) != 1 or len(up_reshape) != 1:
        raise SemanticRecoveryError("pairwise SwiGLU slices are not squeezed exactly once")
    gate_id = gate_reshape[0].outputs[0]
    up_id = up_reshape[0].outputs[0]
    divides = tuple(
        operation
        for operation in graph.consumers(gate_id)
        if operation.kind == "divide" and operation.inputs[0] == gate_id
    )
    if len(divides) != 1:
        raise SemanticRecoveryError("SwiGLU gate is not multiplied by the exact sigmoid quotient")
    denominator = _producer(graph, divides[0].inputs[1], expected_kind="add")
    exponential_inputs = tuple(
        value_id for value_id in denominator.inputs if _origin_kind(graph, value_id, through=()) == "exponential"
    )
    if len(exponential_inputs) != 1:
        raise SemanticRecoveryError("SwiGLU denominator does not contain exp(-gate)")
    exponential = _producer(graph, exponential_inputs[0], expected_kind="exponential")
    negate = _producer(graph, exponential.inputs[0], expected_kind="negate")
    if negate.inputs != (gate_id,):
        raise SemanticRecoveryError("SwiGLU exponential does not consume the negated gate")
    one_inputs = tuple(value_id for value_id in denominator.inputs if value_id not in exponential_inputs)
    if len(one_inputs) != 1:
        raise SemanticRecoveryError("SwiGLU denominator does not have one additive identity input")
    _, one_base = _peel(graph, one_inputs[0], kinds=("broadcast_in_dim",))
    if _constant_float(_producer(graph, one_base, expected_kind="constant")) != 1.0:
        raise SemanticRecoveryError("SwiGLU denominator additive identity is not exactly one")
    activated = _binary_consumer(graph, divides[0].outputs[0], up_id, expected_kind="multiply")
    return activated.outputs[0]


def _attention_score_dot(graph: StableHLOGraph) -> StableHLOOperation:
    matches = tuple(
        operation
        for operation in graph.operations
        if operation.kind == "dot_general"
        and isinstance(operation.attributes, DotAttributes)
        and operation.attributes.lhs_batching_dimensions == (0, 2)
        and operation.attributes.lhs_contracting_dimensions == (3,)
    )
    if len(matches) != 1:
        raise SemanticRecoveryError("dense region does not contain exactly one QK score contraction")
    return matches[0]


def _recover_rms_reduction(
    graph: StableHLOGraph,
    inverse_rms: StableHLOOperation,
    *,
    rms_input_fp32: int,
) -> tuple[float, StableHLOOperation, tuple[int, ...]]:
    epsilon_add = _producer(graph, inverse_rms.inputs[0], expected_kind="add")
    epsilon_value, mean_value = _partition_inputs(
        graph,
        epsilon_add,
        predicate=lambda value_id: _origin_kind(graph, value_id, through=("broadcast_in_dim",)) == "constant",
        description="epsilon constant",
    )
    epsilon_path, epsilon_constant_value = _peel(graph, epsilon_value, kinds=("broadcast_in_dim",))
    epsilon_constant = _producer(graph, epsilon_constant_value, expected_kind="constant")
    epsilon = _constant_float(epsilon_constant)

    mean = _producer(graph, mean_value, expected_kind="divide")
    reduction_value, divisor_value = mean.inputs
    reduction_broadcasts, reduction_base = _peel(graph, reduction_value, kinds=("broadcast_in_dim",))
    reduction = _producer(graph, reduction_base, expected_kind="reduce")
    divisor_path, divisor_base = _peel(graph, divisor_value, kinds=("broadcast_in_dim",))
    divisor_constant = _producer(graph, divisor_base, expected_kind="constant")

    reduction_attributes = _reduction_attributes(reduction)
    hidden_size = graph.value(rms_input_fp32).shape[-1]
    if _constant_float(divisor_constant) != hidden_size:
        raise SemanticRecoveryError("RMS mean divisor does not equal the hidden dimension")
    if reduction_attributes.reducer != "add":
        raise SemanticRecoveryError(f"RMS reduction uses {reduction_attributes.reducer!r}, expected 'add'")

    square = _producer(graph, reduction.inputs[0], expected_kind="multiply")
    if square.inputs != (rms_input_fp32, rms_input_fp32):
        raise SemanticRecoveryError("RMS reduction input is not the square of the normalized activation")
    initial_value = _producer(graph, reduction.inputs[1], expected_kind="constant")
    if _constant_float(initial_value) != 0.0:
        raise SemanticRecoveryError("RMS sum reduction does not use a zero initial value")

    path_ids = (
        square.id,
        initial_value.id,
        reduction.id,
        *tuple(operation.id for operation in reduction_broadcasts),
        divisor_constant.id,
        *tuple(operation.id for operation in divisor_path),
        mean.id,
        epsilon_constant.id,
        *tuple(operation.id for operation in epsilon_path),
        epsilon_add.id,
        inverse_rms.id,
    )
    return epsilon, reduction, path_ids


def _validate_qk_dot(operation: StableHLOOperation) -> None:
    if not isinstance(operation.attributes, DotAttributes):
        raise SemanticRecoveryError("QK contraction has no dimension metadata")
    attributes = operation.attributes
    if (
        attributes.lhs_batching_dimensions != (0, 2)
        or attributes.rhs_batching_dimensions != (0, 2)
        or attributes.lhs_contracting_dimensions != (3,)
        or attributes.rhs_contracting_dimensions != (3,)
    ):
        raise SemanticRecoveryError("QK contraction does not batch over batch/head and contract head dimension")


def _validate_pv_dot(operation: StableHLOOperation) -> None:
    if not isinstance(operation.attributes, DotAttributes):
        raise SemanticRecoveryError("PV contraction has no dimension metadata")
    attributes = operation.attributes
    if (
        attributes.lhs_batching_dimensions != (0, 2)
        or attributes.rhs_batching_dimensions != (0, 1)
        or attributes.lhs_contracting_dimensions != (1,)
        or attributes.rhs_contracting_dimensions != (3,)
    ):
        raise SemanticRecoveryError("PV contraction does not batch over batch/head and contract key sequence")


def _validate_softmax_reduction(
    graph: StableHLOGraph,
    operation: StableHLOOperation,
    *,
    reducer: str,
    input_id: int,
) -> None:
    attributes = _reduction_attributes(operation)
    if attributes.dimensions != (3,) or attributes.reducer != reducer:
        raise SemanticRecoveryError(f"softmax {reducer} reduction must reduce key-sequence axis 3, found {attributes}")
    if operation.inputs[0] != input_id:
        raise SemanticRecoveryError(f"softmax {reducer} reduction does not consume the expected value")
    initial_value = _producer(graph, operation.inputs[1], expected_kind="constant")
    if reducer == "add" and _constant_float(initial_value) != 0.0:
        raise SemanticRecoveryError("softmax sum reduction does not use a zero initial value")
    if reducer == "maximum" and not _is_negative_infinity_constant(initial_value):
        raise SemanticRecoveryError("softmax maximum reduction does not use negative infinity")


def _recover_repeated_heads(graph: StableHLOGraph, value_id: int, *, role: str) -> tuple[int, int]:
    final_reshape = _producer(graph, value_id, expected_kind="reshape")
    broadcast = _producer(graph, final_reshape.inputs[0], expected_kind="broadcast_in_dim")
    initial_reshape = _producer(graph, broadcast.inputs[0], expected_kind="reshape")
    input_id = initial_reshape.inputs[0]
    if input_id not in graph.inputs:
        raise SemanticRecoveryError(f"repeated {role} heads do not originate at a function input")

    input_shape = graph.value(input_id).shape
    inserted_shape = graph.value(initial_reshape.outputs[0]).shape
    broadcast_shape = graph.value(broadcast.outputs[0]).shape
    output_shape = graph.value(final_reshape.outputs[0]).shape
    if len(input_shape) != 4 or inserted_shape != (*input_shape[:3], 1, input_shape[3]):
        raise SemanticRecoveryError(f"{role} head replication does not insert one group axis after KV heads")
    if not isinstance(broadcast.attributes, BroadcastAttributes) or broadcast.attributes.dimensions != (0, 1, 2, 3, 4):
        raise SemanticRecoveryError(f"{role} head replication uses a non-canonical broadcast")
    ratio = broadcast_shape[3]
    if ratio <= 1 or broadcast_shape != (*input_shape[:3], ratio, input_shape[3]):
        raise SemanticRecoveryError(f"{role} head replication does not expand only the inserted group axis")
    expected_output = (input_shape[0], input_shape[1], input_shape[2] * ratio, input_shape[3])
    if output_shape != expected_output:
        raise SemanticRecoveryError(f"{role} head replication does not flatten adjacent KV and group axes")
    return input_id, ratio


def _validate_causal_mask(graph: StableHLOGraph, value_id: int, *, query_axis: int, key_axis: int) -> None:
    predicate_broadcasts, compare_value_id = _peel(graph, value_id, kinds=("broadcast_in_dim",))
    compare = _producer(graph, compare_value_id, expected_kind="compare")
    if not isinstance(compare.attributes, CompareAttributes):
        raise SemanticRecoveryError("causal-mask comparison has no comparison metadata")
    if compare.attributes.direction != "LE" or compare.attributes.compare_type != "SIGNED":
        raise SemanticRecoveryError("causal mask must compare signed key_position <= query_position")
    if _broadcasted_iota_axis(graph, compare.inputs[0]) != key_axis:
        raise SemanticRecoveryError("causal mask's left iota does not vary along the key axis")
    if _broadcasted_iota_axis(graph, compare.inputs[1]) != query_axis:
        raise SemanticRecoveryError("causal mask's right iota does not vary along the query axis")
    if not predicate_broadcasts:
        raise SemanticRecoveryError("causal mask is not broadcast over query heads")


def _broadcasted_iota_axis(graph: StableHLOGraph, value_id: int) -> int:
    broadcasts, iota_value_id = _peel(graph, value_id, kinds=("broadcast_in_dim",))
    iota = _producer(graph, iota_value_id, expected_kind="iota")
    if not isinstance(iota.attributes, IotaAttributes):
        raise SemanticRecoveryError("causal mask iota has no dimension metadata")
    axis = iota.attributes.dimension
    for broadcast in reversed(broadcasts):
        if not isinstance(broadcast.attributes, BroadcastAttributes):
            raise SemanticRecoveryError("causal mask broadcast has no dimension metadata")
        axis = broadcast.attributes.dimensions[axis]
    return axis


def _require_negative_infinity(graph: StableHLOGraph, value_id: int) -> None:
    _, constant_value_id = _peel(graph, value_id, kinds=("broadcast_in_dim",))
    constant = _producer(graph, constant_value_id, expected_kind="constant")
    if not _is_negative_infinity_constant(constant):
        raise SemanticRecoveryError("causal/softmax sentinel is not exact negative infinity")


def _is_negative_infinity_constant(operation: StableHLOOperation) -> bool:
    if not isinstance(operation.attributes, ConstantAttributes):
        return False
    literal = operation.attributes.literal.upper()
    return "0XFF800000" in literal or "-INF" in literal


def _reachable_operation_ids(graph: StableHLOGraph, value_id: int) -> set[int]:
    producer = graph.producer(value_id)
    if producer is None:
        return set()
    reachable = {producer.id}
    for input_id in producer.inputs:
        reachable.update(_reachable_operation_ids(graph, input_id))
    return reachable


def _producer(graph: StableHLOGraph, value_id: int, *, expected_kind: str) -> StableHLOOperation:
    operation = graph.producer(value_id)
    if operation is None:
        raise SemanticRecoveryError(f"value {graph.value(value_id).name} has no producer; expected {expected_kind}")
    if operation.kind != expected_kind:
        raise SemanticRecoveryError(f"expected {expected_kind} at {operation.source_location}, found {operation.kind}")
    return operation


def _partition_inputs(
    graph: StableHLOGraph,
    operation: StableHLOOperation,
    *,
    predicate: Callable[[int], bool],
    description: str,
) -> tuple[int, int]:
    matching = tuple(value_id for value_id in operation.inputs if predicate(value_id))
    remaining = tuple(value_id for value_id in operation.inputs if value_id not in matching)
    if len(matching) != 1 or len(remaining) != 1:
        raise SemanticRecoveryError(
            f"operation {operation.kind} at {operation.source_location} does not have one {description} input"
        )
    return matching[0], remaining[0]


def _peel(graph: StableHLOGraph, value_id: int, *, kinds: tuple[str, ...]) -> tuple[tuple[StableHLOOperation, ...], int]:
    operations: list[StableHLOOperation] = []
    while True:
        producer = graph.producer(value_id)
        if producer is None or producer.kind not in kinds:
            return tuple(operations), value_id
        if len(producer.inputs) != 1:
            raise SemanticRecoveryError(f"{producer.kind} at {producer.source_location} is not unary")
        operations.append(producer)
        value_id = producer.inputs[0]


def _origin_kind(graph: StableHLOGraph, value_id: int, *, through: tuple[str, ...]) -> str | None:
    _, base = _peel(graph, value_id, kinds=through)
    producer = graph.producer(base)
    return producer.kind if producer is not None else None


def _originates_at_rank_one_input(graph: StableHLOGraph, value_id: int) -> bool:
    _, base = _peel(graph, value_id, kinds=("broadcast_in_dim", "convert"))
    return base in graph.inputs and len(graph.value(base).shape) == 1


def _validate_matrix_dot(operation: StableHLOOperation) -> None:
    if not isinstance(operation.attributes, DotAttributes):
        raise SemanticRecoveryError(f"dot at {operation.source_location} has no dimension metadata")
    attributes = operation.attributes
    if (
        attributes.lhs_batching_dimensions
        or attributes.rhs_batching_dimensions
        or attributes.lhs_contracting_dimensions != (1,)
        or attributes.rhs_contracting_dimensions != (0,)
    ):
        raise SemanticRecoveryError(f"dot at {operation.source_location} is not a rank-two right multiplication")


def _reduction_attributes(operation: StableHLOOperation) -> ReductionAttributes:
    if not isinstance(operation.attributes, ReductionAttributes):
        raise SemanticRecoveryError(f"reduction at {operation.source_location} has no reduction metadata")
    return operation.attributes


def _constant_float(operation: StableHLOOperation) -> float:
    if not isinstance(operation.attributes, ConstantAttributes):
        raise SemanticRecoveryError(f"constant at {operation.source_location} has no literal")
    match = SCALAR_LITERAL.search(operation.attributes.literal)
    if match is None:
        raise SemanticRecoveryError(f"constant at {operation.source_location} is not a scalar literal")
    return float(match.group(1))


def _graph_input(graph: TensorGraph, source: StableHLOGraph, value_id: int) -> TensorValue:
    value = source.value(value_id)
    return graph.input(value.name, shape=value.shape, dtype=value.dtype)


def _graph_parameter(graph: TensorGraph, source: StableHLOGraph, value_id: int) -> TensorValue:
    value = source.value(value_id)
    return graph.parameter(value.name, shape=value.shape, dtype=value.dtype)
