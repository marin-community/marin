# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from tile_lifetime import (
    DType,
    GemmSkeleton,
    MaterializationDisposition,
    NumericalPolicy,
    ReductionSkeleton,
    StreamingAttentionSkeleton,
    TransformSkeleton,
    compile_erased_dense_transformer_region,
    compile_gemm_program,
    erase_dense_transformer_semantics,
    pairwise_product_expression,
    pairwise_silu_product_expression,
    validate_plan_semantic_erasure,
)
from tile_lifetime.compiler import RowScalePlacement
from tile_lifetime.dense_flow import FlowMap, FlowMapIteration
from tile_lifetime.dense_region import compile_dense_transformer_region
from tile_lifetime.gemm_program import GENERIC_H100_GEMM_BACKEND
from tile_lifetime.ir import TensorGraph
from tile_lifetime.quack_gemm_codegen import generate_quack_gemm
from tile_lifetime.tensor_program import deserialize_scalar_expression

BATCH = 1
SEQUENCE = 128
TOKENS = BATCH * SEQUENCE
HIDDEN = 512
INTERMEDIATE = 1408
QUERY_HEADS = 8
KEY_VALUE_HEADS = 2
HEAD_DIMENSION = 64
QKV_WIDTH = (QUERY_HEADS + 2 * KEY_VALUE_HEADS) * HEAD_DIMENSION


def _dense_llama_region() -> TensorGraph:
    graph = TensorGraph()
    x = graph.input("x", shape=(TOKENS, HIDDEN), dtype=DType.BF16)
    qkv_weight = graph.parameter("qkv_weight", shape=(HIDDEN, QKV_WIDTH), dtype=DType.BF16)
    output_weight = graph.parameter("output_weight", shape=(HIDDEN, HIDDEN), dtype=DType.BF16)
    mlp_gamma = graph.parameter("mlp_gamma", shape=(HIDDEN,), dtype=DType.BF16)
    gate_up_weight = graph.parameter("gate_up_weight", shape=(HIDDEN, 2 * INTERMEDIATE), dtype=DType.BF16)
    down_weight = graph.parameter("down_weight", shape=(INTERMEDIATE, HIDDEN), dtype=DType.BF16)
    next_gamma = graph.parameter("next_gamma", shape=(HIDDEN,), dtype=DType.BF16)
    next_qkv_weight = graph.parameter("next_qkv_weight", shape=(HIDDEN, QKV_WIDTH), dtype=DType.BF16)
    sine = graph.parameter("rope_sine", shape=(SEQUENCE, HEAD_DIMENSION // 2), dtype=DType.BF16)
    cosine = graph.parameter("rope_cosine", shape=(SEQUENCE, HEAD_DIMENSION // 2), dtype=DType.BF16)

    x_bsh = graph.view(x, shape=(BATCH, SEQUENCE, HIDDEN), name="x_bsh")
    query, key, value = graph.qkv_projection(
        x_bsh,
        qkv_weight,
        name="qkv",
        query_heads=QUERY_HEADS,
        key_value_heads=KEY_VALUE_HEADS,
        head_dimension=HEAD_DIMENSION,
        accumulation_dtype=DType.FP32,
    )
    query, key = graph.rope(
        query,
        key,
        sine,
        cosine,
        name="rotated",
        rotary_dimension=HEAD_DIMENSION,
    )
    attention = graph.scaled_dot_product_attention(
        query,
        key,
        value,
        name="attention",
        scale=HEAD_DIMENSION**-0.5,
        causal=True,
        accumulation_dtype=DType.FP32,
    )
    attention_flat = graph.view(attention, shape=(TOKENS, HIDDEN), name="attention_flat")
    projected = graph.linear(attention_flat, output_weight, name="projected", accumulation_dtype=DType.FP32)
    x1 = graph.residual_add(projected, x, name="x1")
    mlp_input = graph.rms_norm(
        x1,
        mlp_gamma,
        name="mlp_input",
        axis=-1,
        epsilon=1e-6,
        reduction_dtype=DType.FP32,
    )
    gate_up = graph.linear(
        mlp_input,
        gate_up_weight,
        name="gate_up",
        accumulation_dtype=DType.FP32,
    )
    activated = graph.pairwise_swiglu(gate_up, name="activated")
    down = graph.linear(activated, down_weight, name="down", accumulation_dtype=DType.FP32)
    x2 = graph.residual_add(down, x1, name="x2")
    next_input = graph.rms_norm(
        x2,
        next_gamma,
        name="next_input",
        axis=-1,
        epsilon=1e-6,
        reduction_dtype=DType.FP32,
    )
    next_input_bsh = graph.view(next_input, shape=(BATCH, SEQUENCE, HIDDEN), name="next_input_bsh")
    next_query, next_key, _ = graph.qkv_projection(
        next_input_bsh,
        next_qkv_weight,
        name="next_qkv",
        query_heads=QUERY_HEADS,
        key_value_heads=KEY_VALUE_HEADS,
        head_dimension=HEAD_DIMENSION,
        accumulation_dtype=DType.FP32,
    )
    graph.rope(
        next_query,
        next_key,
        sine,
        cosine,
        name="next_rotated",
        rotary_dimension=HEAD_DIMENSION,
    )
    return graph


def test_dense_transformer_region_composes_expected_skeletons() -> None:
    plan = compile_dense_transformer_region(
        _dense_llama_region(),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )

    assert [type(skeleton) for skeleton in plan.skeletons] == [
        GemmSkeleton,
        StreamingAttentionSkeleton,
        GemmSkeleton,
        ReductionSkeleton,
        GemmSkeleton,
        GemmSkeleton,
        ReductionSkeleton,
        GemmSkeleton,
    ]
    assert [skeleton.name for skeleton in plan.skeletons] == [
        "contract_partition_pairwise_linear_maps",
        "streaming_normalized_weighted_fold",
        "contract_maps_and_fold_partials",
        "combine_fold_partials",
        "contract_consumer_prologue_row_scale_pairwise_map",
        "contract_maps_and_fold_partials",
        "combine_fold_partials",
        "contract_partition_pairwise_linear_maps",
    ]
    assert not any(isinstance(skeleton, TransformSkeleton) for skeleton in plan.skeletons)
    assert plan.semantic_erasure_report is not None
    assert plan.semantic_erasure_report.is_clean
    assert len(plan.semantic_erasure_report.scheduling_keys) == 36
    validate_plan_semantic_erasure(plan)


def test_dense_transformer_region_attaches_all_memory_bound_operations() -> None:
    plan = compile_dense_transformer_region(
        _dense_llama_region(),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )

    output_projection = plan.skeletons[2]
    gate_up = plan.skeletons[4]
    down_projection = plan.skeletons[5]
    next_qkv = plan.skeletons[7]
    assert isinstance(output_projection, GemmSkeleton)
    assert isinstance(gate_up, GemmSkeleton)
    assert isinstance(down_projection, GemmSkeleton)
    assert isinstance(next_qkv, GemmSkeleton)
    assert [attachment.operation for attachment in output_projection.epilogue] == [
        "add",
        "multiply",
        "partial_sum_square",
    ]
    assert [attachment.operation for attachment in gate_up.prologue] == ["scale_row"]
    assert [attachment.operation for attachment in gate_up.epilogue] == ["pairwise_map"]
    assert [attachment.operation for attachment in down_projection.epilogue] == [
        "add",
        "multiply",
        "partial_sum_square",
    ]
    assert [attachment.operation for attachment in next_qkv.prologue] == ["scale_row", "view"]
    assert [attachment.operation for attachment in next_qkv.epilogue] == [
        "partition",
        "pairwise_linear_map",
        "pairwise_linear_map",
    ]
    assert gate_up.backend == GENERIC_H100_GEMM_BACKEND
    assert gate_up.physical_tile_shape == (128, 256, 64)
    assert gate_up.cluster_shape == (1, 1, 1)
    assert next_qkv.backend == GENERIC_H100_GEMM_BACKEND
    assert next_qkv.physical_tile_shape == (128, 256, 64)
    assert next_qkv.cluster_shape == (1, 1, 1)


def test_dense_transformer_region_can_delay_rms_scale_in_consumer_epilogues() -> None:
    plan = compile_dense_transformer_region(
        _dense_llama_region(),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        rms_scale_placement=RowScalePlacement.CONSUMER_EPILOGUE,
    )

    gate_up = plan.skeletons[4]
    output_projection = plan.skeletons[2]
    down_projection = plan.skeletons[5]
    next_qkv = plan.skeletons[7]
    assert isinstance(output_projection, GemmSkeleton)
    assert isinstance(gate_up, GemmSkeleton)
    assert isinstance(down_projection, GemmSkeleton)
    assert isinstance(next_qkv, GemmSkeleton)
    assert output_projection.physical_tile_shape == (128, 256, 64)
    assert output_projection.cluster_shape == (1, 1, 1)
    assert down_projection.physical_tile_shape == (128, 256, 64)
    assert down_projection.cluster_shape == (1, 1, 1)
    assert gate_up.backend == GENERIC_H100_GEMM_BACKEND
    assert gate_up.prologue == ()
    assert [attachment.operation for attachment in gate_up.epilogue] == ["scale_row", "pairwise_map"]
    assert next_qkv.backend == GENERIC_H100_GEMM_BACKEND
    assert [attachment.operation for attachment in next_qkv.prologue] == ["view"]
    assert [attachment.operation for attachment in next_qkv.epilogue] == [
        "scale_row",
        "partition",
        "pairwise_linear_map",
        "pairwise_linear_map",
    ]
    assert plan.materialization("mlp_input").disposition is MaterializationDisposition.EPILOGUE_ONLY
    assert plan.materialization("next_input").disposition is MaterializationDisposition.EPILOGUE_ONLY


def test_full_dense_pairwise_semantic_mutation_uses_the_same_generator() -> None:
    erased = erase_dense_transformer_semantics(_dense_llama_region())
    pairwise = next(
        operation
        for operation in erased.operations
        if isinstance(operation, FlowMap)
        and operation.iteration is FlowMapIteration.ADJACENT_PAIR
        and operation.expressions == (pairwise_silu_product_expression(),)
    )
    mutated_map = replace(pairwise, expressions=(pairwise_product_expression(),))
    mutated = erased.with_operations(
        tuple(mutated_map if operation is pairwise else operation for operation in erased.operations)
    )

    plan = compile_erased_dense_transformer_region(
        mutated,
        row_scale_placement=RowScalePlacement.CONSUMER_PROLOGUE,
    )

    skeleton = plan.skeletons[4]
    assert isinstance(skeleton, GemmSkeleton)
    assert skeleton.epilogue[-1].operation == "pairwise_map"
    expression = deserialize_scalar_expression(dict(skeleton.epilogue[-1].attributes)["expression_ast"])
    assert expression == pairwise_product_expression()
    source = generate_quack_gemm(compile_gemm_program(skeleton)).source
    assert "left_0 * right_0" in source
    assert "swiglu" not in source
    validate_plan_semantic_erasure(plan)


def test_full_dense_recovery_uses_dataflow_instead_of_operation_slots() -> None:
    erased = erase_dense_transformer_semantics(_dense_llama_region())
    reordered = erased.with_operations(tuple(reversed(erased.operations)))

    plan = compile_erased_dense_transformer_region(
        reordered,
        row_scale_placement=RowScalePlacement.CONSUMER_PROLOGUE,
    )

    assert [type(skeleton) for skeleton in plan.skeletons] == [
        GemmSkeleton,
        StreamingAttentionSkeleton,
        GemmSkeleton,
        ReductionSkeleton,
        GemmSkeleton,
        GemmSkeleton,
        ReductionSkeleton,
        GemmSkeleton,
    ]
    validate_plan_semantic_erasure(plan)


def test_dense_transformer_region_materialization_boundaries_are_explicit() -> None:
    plan = compile_dense_transformer_region(
        _dense_llama_region(),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )

    assert plan.materialization("x_bsh").disposition is MaterializationDisposition.ALIAS
    assert plan.materialization("attention_flat").disposition is MaterializationDisposition.ALIAS
    assert plan.materialization("qkv.query").disposition is MaterializationDisposition.EPILOGUE_ONLY
    assert plan.materialization("qkv.key").disposition is MaterializationDisposition.EPILOGUE_ONLY
    assert plan.materialization("gate_up").disposition is MaterializationDisposition.EPILOGUE_ONLY
    assert plan.materialization("mlp_input").disposition is MaterializationDisposition.PROLOGUE_ONLY
    assert plan.materialization("next_input").disposition is MaterializationDisposition.PROLOGUE_ONLY
    assert plan.materialization("x1").disposition is MaterializationDisposition.MATERIALIZE
    assert plan.materialization("x2").disposition is MaterializationDisposition.MATERIALIZE
    assert plan.sequence_squared_materializations == ()
    assert all(rewrite.applied for rewrite in plan.rewrites)
