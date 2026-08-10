# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Historical named StableHLO recovery used by reference planners."""

from dataclasses import dataclass

from shuttle.ir import DType
from shuttle.stablehlo_import import StableHLOGraph
from tile_lifetime.ir import TensorGraph
from tile_lifetime.semantic_recovery import (
    SemanticRecoveryError,
    _graph_input,
    _graph_parameter,
    _origin_kind,
    _originates_at_rank_one_input,
    _partition_inputs,
    _peel,
    _producer,
    _recover_rms_reduction,
    _reduction_attributes,
    _validate_matrix_dot,
)


@dataclass(frozen=True)
class RecoveredReferenceRMSRegion:
    """Named RMS TensorGraph and its source-operation mapping."""

    graph: TensorGraph
    source_operation_ids: tuple[int, ...]


def recover_reference_rms_region(
    stablehlo_graph: StableHLOGraph,
    *,
    gemm_accumulation_dtype: DType,
    output_name: str,
    output_index: int = 0,
) -> RecoveredReferenceRMSRegion:
    """Recover a named Linear/Residual/RMS/Linear reference graph."""
    output_dot = _producer(stablehlo_graph, stablehlo_graph.outputs[output_index], expected_kind="dot_general")
    _validate_matrix_dot(output_dot)

    normalized_convert = _producer(stablehlo_graph, output_dot.inputs[0], expected_kind="convert")
    scaled_normalized = _producer(stablehlo_graph, normalized_convert.inputs[0], expected_kind="multiply")
    inverse_rms_value, gamma_scaled_value = _partition_inputs(
        stablehlo_graph,
        scaled_normalized,
        predicate=lambda value_id: _origin_kind(stablehlo_graph, value_id, through=("broadcast_in_dim",)) == "rsqrt",
        description="inverse-RMS broadcast",
    )

    inverse_rms_broadcasts, inverse_rms_base = _peel(
        stablehlo_graph,
        inverse_rms_value,
        kinds=("broadcast_in_dim",),
    )
    inverse_rms = _producer(stablehlo_graph, inverse_rms_base, expected_kind="rsqrt")
    gamma_scaled = _producer(stablehlo_graph, gamma_scaled_value, expected_kind="multiply")

    gamma_value, rms_input_fp32 = _partition_inputs(
        stablehlo_graph,
        gamma_scaled,
        predicate=lambda value_id: _originates_at_rank_one_input(stablehlo_graph, value_id),
        description="gamma broadcast",
    )
    gamma_path, gamma_input_id = _peel(stablehlo_graph, gamma_value, kinds=("broadcast_in_dim", "convert"))
    if gamma_input_id not in stablehlo_graph.inputs:
        raise SemanticRecoveryError("gamma broadcast does not originate at a function input")

    rms_input_convert = _producer(stablehlo_graph, rms_input_fp32, expected_kind="convert")
    residual_add = _producer(stablehlo_graph, rms_input_convert.inputs[0], expected_kind="add")
    projection_value, residual_input_id = _partition_inputs(
        stablehlo_graph,
        residual_add,
        predicate=lambda value_id: _origin_kind(stablehlo_graph, value_id, through=()) == "dot_general",
        description="projection GEMM",
    )
    projection_dot = _producer(stablehlo_graph, projection_value, expected_kind="dot_general")
    _validate_matrix_dot(projection_dot)
    if residual_input_id not in stablehlo_graph.inputs:
        raise SemanticRecoveryError("residual addition's non-GEMM input is not a function input")

    epsilon, reduction, rms_path_ids = _recover_rms_reduction(
        stablehlo_graph,
        inverse_rms,
        rms_input_fp32=rms_input_fp32,
    )
    reduction_attributes = _reduction_attributes(reduction)
    residual_shape = stablehlo_graph.value(residual_add.outputs[0]).shape
    if reduction_attributes.dimensions != (len(residual_shape) - 1,):
        raise SemanticRecoveryError(
            f"RMS reduction dimensions {reduction_attributes.dimensions} do not cover the hidden dimension"
        )

    x_input_id, weight_0_input_id = projection_dot.inputs
    weight_1_input_id = output_dot.inputs[1]
    for value_id, role in (
        (x_input_id, "projection input"),
        (weight_0_input_id, "first weight"),
        (weight_1_input_id, "second weight"),
    ):
        if value_id not in stablehlo_graph.inputs:
            raise SemanticRecoveryError(f"{role} is not a function input")

    semantic_graph = TensorGraph()
    x = _graph_input(semantic_graph, stablehlo_graph, x_input_id)
    residual = _graph_input(semantic_graph, stablehlo_graph, residual_input_id)
    weight_0 = _graph_parameter(semantic_graph, stablehlo_graph, weight_0_input_id)
    gamma = _graph_parameter(semantic_graph, stablehlo_graph, gamma_input_id)
    weight_1 = _graph_parameter(semantic_graph, stablehlo_graph, weight_1_input_id)

    projected = semantic_graph.linear(
        x,
        weight_0,
        name="projected",
        accumulation_dtype=gemm_accumulation_dtype,
        source_location=projection_dot.source_location,
    )
    residual_sum = semantic_graph.residual_add(
        projected,
        residual,
        name="residual_sum",
        source_location=residual_add.source_location,
    )
    normalized = semantic_graph.rms_norm(
        residual_sum,
        gamma,
        name="normalized",
        axis=reduction_attributes.dimensions[0],
        epsilon=epsilon,
        reduction_dtype=stablehlo_graph.value(rms_input_fp32).dtype,
        source_location=inverse_rms.source_location,
    )
    semantic_graph.linear(
        normalized,
        weight_1,
        name=output_name,
        accumulation_dtype=gemm_accumulation_dtype,
        source_location=output_dot.source_location,
    )

    source_ids = (
        projection_dot.id,
        residual_add.id,
        rms_input_convert.id,
        *rms_path_ids,
        *tuple(operation.id for operation in gamma_path),
        gamma_scaled.id,
        *tuple(operation.id for operation in inverse_rms_broadcasts),
        scaled_normalized.id,
        normalized_convert.id,
        output_dot.id,
    )
    return RecoveredReferenceRMSRegion(
        graph=semantic_graph,
        source_operation_ids=tuple(dict.fromkeys(source_ids)),
    )
