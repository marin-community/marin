# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from tile_lifetime import (
    DType,
    GemmSkeleton,
    MaterializationDisposition,
    NumericalPolicy,
    ReductionSkeleton,
    StreamingAttentionSkeleton,
    TransformSkeleton,
    compile_erased_dense_program,
    compile_gemm_program,
    erase_dense_semantics,
    execute_tensor_program,
    validate_erased_tensor_program,
    validate_plan_semantic_erasure,
)
from tile_lifetime.attention import compile_reference_attention_region
from tile_lifetime.compiler import RowScalePlacement, compile_reference_region
from tile_lifetime.gemm_program import GENERIC_H100_GEMM_BACKEND
from tile_lifetime.ir import TensorGraph
from tile_lifetime.plan import NumericalEquivalence
from tile_lifetime.semantic_erasure import SemanticErasureError
from tile_lifetime.tensor_program import MapPrimitive, ScalarExpressionKind, scalar_binary, scalar_constant
from tile_lifetime.tile_program import TilePrimitive, TileProgramStage


def _rms_region(
    *,
    axis: int = -1,
    add_normalized_consumer: bool = False,
    output_features: int = 24,
) -> TensorGraph:
    tokens = 8
    hidden = 16
    graph = TensorGraph()
    x = graph.input("x", shape=(tokens, hidden), dtype=DType.BF16)
    residual = graph.input("residual", shape=(tokens, hidden), dtype=DType.BF16)
    weight_0 = graph.parameter("weight_0", shape=(hidden, hidden), dtype=DType.BF16)
    gamma_size = hidden if axis in {-1, 1} else tokens
    gamma = graph.parameter("gamma", shape=(gamma_size,), dtype=DType.BF16)
    weight_1 = graph.parameter("weight_1", shape=(hidden, output_features), dtype=DType.BF16)

    projected = graph.linear(x, weight_0, name="projected", accumulation_dtype=DType.FP32)
    residual_sum = graph.residual_add(projected, residual, name="residual_sum")
    normalized = graph.rms_norm(
        residual_sum,
        gamma,
        name="normalized",
        axis=axis,
        epsilon=1e-6,
        reduction_dtype=DType.FP32,
    )
    graph.linear(normalized, weight_1, name="output", accumulation_dtype=DType.FP32)
    if add_normalized_consumer:
        other = graph.input("other", shape=normalized.shape, dtype=normalized.dtype)
        graph.residual_add(normalized, other, name="observed_normalized")
    return graph


def _layer_norm_region(*, output_features: int = 24, epsilon: float = 1e-5) -> TensorGraph:
    tokens = 8
    hidden = 16
    graph = TensorGraph()
    x = graph.input("x", shape=(tokens, hidden), dtype=DType.BF16)
    residual = graph.input("residual", shape=(tokens, hidden), dtype=DType.BF16)
    weight_0 = graph.parameter("weight_0", shape=(hidden, hidden), dtype=DType.BF16)
    gamma = graph.parameter("gamma", shape=(hidden,), dtype=DType.BF16)
    beta = graph.parameter("beta", shape=(hidden,), dtype=DType.BF16)
    weight_1 = graph.parameter("weight_1", shape=(hidden, output_features), dtype=DType.BF16)

    projected = graph.linear(x, weight_0, name="projected", accumulation_dtype=DType.FP32)
    residual_sum = graph.residual_add(projected, residual, name="residual_sum")
    normalized = graph.layer_norm(
        residual_sum,
        gamma,
        beta,
        name="normalized",
        axis=-1,
        epsilon=epsilon,
        reduction_dtype=DType.FP32,
    )
    graph.linear(normalized, weight_1, name="output", accumulation_dtype=DType.FP32)
    return graph


def test_centered_affine_normalization_uses_consumer_preparation_under_reassociation_policy() -> None:
    plan = compile_reference_region(
        _layer_norm_region(),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        rms_scale_placement=RowScalePlacement.CONSUMER_PROLOGUE,
    )

    assert [type(skeleton) for skeleton in plan.skeletons] == [
        GemmSkeleton,
        ReductionSkeleton,
        ReductionSkeleton,
        GemmSkeleton,
    ]
    producer, mean_reduction, inverse_reduction, consumer = plan.skeletons
    assert isinstance(producer, GemmSkeleton)
    assert [attachment.operation for attachment in producer.epilogue] == [
        "add",
        "partial_sum",
        "partial_sum_square",
        "store_tile",
    ]
    assert isinstance(mean_reduction, ReductionSkeleton)
    assert isinstance(inverse_reduction, ReductionSkeleton)
    assert inverse_reduction.auxiliary_inputs == (mean_reduction.output,)
    assert isinstance(consumer, GemmSkeleton)
    assert [attachment.operation for attachment in consumer.prologue] == [
        "subtract",
        "scale_row",
        "multiply",
        "add",
    ]
    assert plan.materialization("normalized").disposition is MaterializationDisposition.PROLOGUE_ONLY
    assert plan.rewrites[0].name == "place_centered_affine_in_consumer_contract_preparation"
    assert "not source-order LayerNorm statistics" in plan.rewrites[0].numerical_effect
    assert "LayerNormOp" in plan.semantic_erasure_report.source_semantics
    assert all("layer" not in key.lower() for key in plan.semantic_erasure_report.scheduling_keys)
    producer_program = compile_gemm_program(producer)
    assert producer_program.tile_program.primitives_at(TileProgramStage.FINALIZATION) == (
        TilePrimitive.ADD,
        TilePrimitive.PARTIAL_SUM,
        TilePrimitive.PARTIAL_SUM_SQUARE,
        TilePrimitive.STORE,
        TilePrimitive.STORE,
        TilePrimitive.STORE,
    )
    consumer_program = compile_gemm_program(consumer)
    assert consumer_program.tile_program.primitives_at(TileProgramStage.PREPARATION) == (
        TilePrimitive.SUBTRACT,
        TilePrimitive.SCALE_ROW,
        TilePrimitive.MULTIPLY,
        TilePrimitive.ADD,
        TilePrimitive.CONVERT,
    )
    preparation = consumer_program.tile_program.operations_at(TileProgramStage.PREPARATION)
    assert tuple(dict(operation.attributes).get("input.1_delivery") for operation in preparation[:-1]) == (
        "row",
        None,
        "feature",
        "feature",
    )
    validate_plan_semantic_erasure(plan)


def test_centered_affine_normalization_delayed_output_has_column_corrections() -> None:
    plan = compile_reference_region(
        _layer_norm_region(),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        rms_scale_placement=RowScalePlacement.CONSUMER_EPILOGUE,
    )

    assert [type(skeleton) for skeleton in plan.skeletons] == [
        GemmSkeleton,
        ReductionSkeleton,
        ReductionSkeleton,
        GemmSkeleton,
        GemmSkeleton,
        GemmSkeleton,
    ]
    gamma_projection, beta_projection, consumer = plan.skeletons[-3:]
    assert isinstance(gamma_projection, GemmSkeleton)
    assert gamma_projection.shape == (1, 24, 16)
    assert isinstance(beta_projection, GemmSkeleton)
    assert beta_projection.shape == (1, 24, 16)
    assert isinstance(consumer, GemmSkeleton)
    assert [attachment.operation for attachment in consumer.epilogue] == [
        "scale_row",
        "multiply",
        "multiply",
        "subtract",
        "add",
    ]
    assert plan.rewrites[0].name == "move_centered_affine_through_right_contract"
    assert "gamma W" in plan.rewrites[0].numerical_effect
    for skeleton in (gamma_projection, beta_projection, consumer):
        assert isinstance(skeleton, GemmSkeleton)
        compile_gemm_program(skeleton)
    validate_plan_semantic_erasure(plan)


def test_changed_centered_variance_map_uses_same_generic_placement() -> None:
    first = compile_reference_region(
        _layer_norm_region(epsilon=1e-5),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        rms_scale_placement=RowScalePlacement.CONSUMER_PROLOGUE,
    )
    changed = compile_reference_region(
        _layer_norm_region(epsilon=3e-5),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        rms_scale_placement=RowScalePlacement.CONSUMER_PROLOGUE,
    )

    assert [type(skeleton) for skeleton in first.skeletons] == [type(skeleton) for skeleton in changed.skeletons]
    first_inverse = first.skeletons[2]
    changed_inverse = changed.skeletons[2]
    assert isinstance(first_inverse, ReductionSkeleton)
    assert isinstance(changed_inverse, ReductionSkeleton)
    assert first_inverse.operator != changed_inverse.operator
    assert changed_inverse.operator.endswith("+ 3e-05)")


def test_erased_centered_affine_program_matches_materialized_reference() -> None:
    erased = erase_dense_semantics(_layer_norm_region())
    rng = np.random.default_rng(23)
    inputs = {value.name: rng.normal(size=value.shape).astype(np.float32) for value in erased.program.inputs}

    actual = execute_tensor_program(erased.program, inputs)["output"]
    residual_sum = inputs["x"] @ inputs["weight_0"] + inputs["residual"]
    mean = np.mean(residual_sum, axis=-1, keepdims=True)
    variance = np.mean(np.square(residual_sum), axis=-1, keepdims=True) - np.square(mean)
    normalized = (residual_sum - mean) / np.sqrt(variance + 1e-5)
    expected = (normalized * inputs["gamma"] + inputs["beta"]) @ inputs["weight_1"]

    difference = np.abs(actual - expected)
    assert float(np.max(difference)) < 3e-5
    assert float(np.mean(difference)) < 3e-6


def test_delayed_centered_affine_identity_is_real_exact_but_changes_bf16_order() -> None:
    rng = np.random.default_rng(29)
    source = rng.normal(size=(8, 16)).astype(np.float64)
    gamma = rng.normal(size=(16,)).astype(np.float64)
    beta = rng.normal(size=(16,)).astype(np.float64)
    weight = rng.normal(size=(16, 24)).astype(np.float64)
    mean = np.mean(source, axis=-1, keepdims=True)
    inverse_scale = np.reciprocal(np.sqrt(np.mean(np.square(source), axis=-1, keepdims=True) - np.square(mean) + 1e-5))

    materialized = (((source - mean) * inverse_scale) * gamma + beta) @ weight
    delayed = inverse_scale * ((source * gamma) @ weight) - (mean * inverse_scale) * (gamma @ weight) + beta @ weight
    np.testing.assert_allclose(materialized, delayed, rtol=2e-13, atol=2e-13)

    source_bf16 = jnp.asarray(source, dtype=jnp.bfloat16)
    gamma_bf16 = jnp.asarray(gamma, dtype=jnp.bfloat16)
    beta_bf16 = jnp.asarray(beta, dtype=jnp.bfloat16)
    weight_bf16 = jnp.asarray(weight, dtype=jnp.bfloat16)
    mean_fp32 = jnp.asarray(mean, dtype=jnp.float32)
    inverse_fp32 = jnp.asarray(inverse_scale, dtype=jnp.float32)
    source_ordered = (
        jnp.asarray(
            (source_bf16.astype(jnp.float32) - mean_fp32) * inverse_fp32 * gamma_bf16 + beta_bf16,
            dtype=jnp.bfloat16,
        )
        @ weight_bf16
    )
    delayed_bf16 = (
        inverse_fp32 * (jnp.asarray(source_bf16 * gamma_bf16, dtype=jnp.bfloat16) @ weight_bf16)
        - (mean_fp32 * inverse_fp32) * (gamma_bf16 @ weight_bf16)
        + beta_bf16 @ weight_bf16
    )
    assert float(jnp.max(jnp.abs(source_ordered.astype(jnp.float32) - delayed_bf16))) > 0.0


def test_centered_affine_bitwise_policy_materializes() -> None:
    plan = compile_reference_region(_layer_norm_region(), numerical_policy=NumericalPolicy.BITWISE_EXACT)

    assert not plan.rewrites[0].applied
    assert plan.materialization("normalized").disposition is MaterializationDisposition.MATERIALIZE


def test_sum_and_sum_squares_variance_is_not_source_order_two_pass_variance() -> None:
    centered_offsets = np.asarray([-0.5, -0.25, 0.0, 0.25, 0.5] * 4, dtype=np.float32)
    values = np.float32(10_000.0) + centered_offsets
    mean = np.mean(values, dtype=np.float32)

    moment_variance = np.mean(values * values, dtype=np.float32) - mean * mean
    source_order_two_pass_variance = np.mean((values - mean) * (values - mean), dtype=np.float32)

    assert moment_variance == 0.0
    assert source_order_two_pass_variance == 0.125


def test_compile_region_legal_rms_region_delays_scale() -> None:
    plan = compile_reference_region(_rms_region(), numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER)

    assert [type(skeleton) for skeleton in plan.skeletons] == [GemmSkeleton, ReductionSkeleton, GemmSkeleton]
    assert [attachment.operation for attachment in plan.skeletons[0].epilogue] == [
        "add",
        "multiply",
        "partial_sum_square",
    ]
    assert [attachment.operation for attachment in plan.skeletons[2].epilogue] == ["scale_row"]
    assert plan.skeletons[2].backend == GENERIC_H100_GEMM_BACKEND
    assert plan.materialization("normalized").disposition is MaterializationDisposition.EPILOGUE_ONLY
    assert plan.rewrites[0].applied
    assert plan.sequence_squared_materializations == ()
    assert plan.rewrites[0].numerical_equivalence is NumericalEquivalence.ALGEBRAICALLY_EXACT
    generated = compile_gemm_program(plan.skeletons[0])
    assert generated.tile_program.primitives_at(TileProgramStage.FINALIZATION) == (
        TilePrimitive.ADD,
        TilePrimitive.MULTIPLY,
        TilePrimitive.PARTIAL_SUM_SQUARE,
        TilePrimitive.CONVERT,
        TilePrimitive.STORE,
        TilePrimitive.STORE,
    )


def test_compile_region_can_scale_rms_in_consumer_prologue() -> None:
    plan = compile_reference_region(
        _rms_region(),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        rms_scale_placement=RowScalePlacement.CONSUMER_PROLOGUE,
    )

    consumer = plan.skeletons[2]
    assert isinstance(consumer, GemmSkeleton)
    assert consumer.backend == GENERIC_H100_GEMM_BACKEND
    assert consumer.physical_tile_shape == (128, 256, 64)
    assert consumer.cluster_shape == (1, 1, 1)
    assert consumer.pingpong is False
    assert [attachment.operation for attachment in consumer.prologue] == ["scale_row"]
    assert consumer.epilogue == ()
    assert plan.materialization("normalized").disposition is MaterializationDisposition.PROLOGUE_ONLY
    assert plan.rewrites[0].name == "place_row_scalar_in_consumer_contract_preparation"
    assert "BF16 conversion before WGMMA" in plan.rewrites[0].numerical_effect


def test_consumer_prologue_uses_measured_wide_projection_cluster() -> None:
    plan = compile_reference_region(
        _rms_region(output_features=28_672),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        rms_scale_placement=RowScalePlacement.CONSUMER_PROLOGUE,
    )

    consumer = plan.skeletons[2]
    assert isinstance(consumer, GemmSkeleton)
    assert consumer.physical_tile_shape == (128, 256, 64)
    assert consumer.cluster_shape == (1, 2, 1)


def test_compile_region_observed_normalized_activation_uses_materialized_fallback() -> None:
    plan = compile_reference_region(
        _rms_region(add_normalized_consumer=True), numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER
    )

    assert isinstance(plan.skeletons[1], TransformSkeleton)
    assert plan.materialization("normalized").disposition is MaterializationDisposition.MATERIALIZE
    assert not plan.rewrites[0].applied
    assert any("2 consumers" in reason for reason in plan.rewrites[0].rejection_reasons)


def test_compile_region_non_hidden_rms_axis_uses_materialized_fallback() -> None:
    plan = compile_reference_region(_rms_region(axis=0), numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER)

    assert not plan.rewrites[0].applied
    assert any("reduction axis" in reason for reason in plan.rewrites[0].rejection_reasons)


def test_compile_region_bitwise_policy_uses_materialized_fallback() -> None:
    plan = compile_reference_region(_rms_region(), numerical_policy=NumericalPolicy.BITWISE_EXACT)

    assert not plan.rewrites[0].applied
    assert plan.rewrites[0].numerical_equivalence is NumericalEquivalence.BITWISE_EXACT
    assert any("bitwise-exact" in reason for reason in plan.rewrites[0].rejection_reasons)


def test_dense_frontend_names_erase_before_candidate_selection() -> None:
    plan = compile_reference_region(_rms_region(), numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER)

    report = plan.semantic_erasure_report
    assert report is not None
    assert "RMSNormOp" in report.source_semantics
    assert report.is_clean
    assert all("rms" not in key.lower() for key in report.scheduling_keys)
    assert {primitive for step in report.lowering_steps for primitive in step.generic_primitives} == {
        "Contract",
        "Fold",
        "Map",
    }
    validate_plan_semantic_erasure(plan)


def test_erased_dense_program_matches_independent_materialized_reference() -> None:
    erased = erase_dense_semantics(_rms_region())
    rng = np.random.default_rng(19)
    inputs = {value.name: rng.normal(size=value.shape).astype(np.float32) for value in erased.program.inputs}

    actual = execute_tensor_program(erased.program, inputs)["output"]
    summed = inputs["x"] @ inputs["weight_0"] + inputs["residual"]
    inverse = np.reciprocal(np.sqrt(np.mean(np.square(summed), axis=-1, keepdims=True) + 1e-6))
    expected = (summed * inputs["gamma"] * inverse) @ inputs["weight_1"]

    difference = np.abs(actual - expected)
    assert float(np.max(difference)) < 2e-5
    assert float(np.mean(difference)) < 2e-6


def test_changed_row_scalar_expression_uses_same_generic_placement() -> None:
    erased = erase_dense_semantics(_rms_region())
    row_finalize = next(
        operation
        for operation in erased.program.operations
        if isinstance(operation, MapPrimitive) and operation.expression.kind is ScalarExpressionKind.RSQRT
    )
    changed_finalize = replace(
        row_finalize,
        expression=scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            row_finalize.expression,
            scalar_constant(0.5),
        ),
    )
    changed_operations = tuple(
        changed_finalize if operation is row_finalize else operation for operation in erased.program.operations
    )
    changed = erased.with_program(replace(erased.program, operations=changed_operations))

    plan = compile_erased_dense_program(
        changed,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        scale_placement=RowScalePlacement.CONSUMER_PROLOGUE,
    )

    assert [type(skeleton) for skeleton in plan.skeletons] == [GemmSkeleton, ReductionSkeleton, GemmSkeleton]
    reduction = plan.skeletons[1]
    assert isinstance(reduction, ReductionSkeleton)
    assert reduction.operator.endswith("* 0.5")
    assert [attachment.operation for attachment in plan.skeletons[2].prologue] == ["scale_row"]
    validate_plan_semantic_erasure(plan)


def test_semantic_erasure_validator_rejects_named_scheduling_key() -> None:
    erased = erase_dense_semantics(_rms_region())
    contaminated = replace(
        erased,
        report=replace(erased.report, scheduling_keys=("rmsnorm_before_gemm",)),
    )

    with pytest.raises(SemanticErasureError, match="retains named semantics"):
        validate_erased_tensor_program(contaminated)


def test_delayed_rms_scale_is_real_exact_but_changes_bf16_rounding() -> None:
    rng = np.random.default_rng(7)
    residual_sum = rng.normal(size=(8, 16)).astype(np.float32)
    gamma = rng.normal(size=(16,)).astype(np.float32)
    weight = rng.normal(size=(16, 24)).astype(np.float32)
    inverse_rms = 1.0 / np.sqrt(np.mean(np.square(residual_sum), axis=-1, keepdims=True) + 1e-6)

    source_fp64 = ((residual_sum.astype(np.float64) * gamma) * inverse_rms) @ weight
    delayed_fp64 = ((residual_sum.astype(np.float64) * gamma) @ weight) * inverse_rms
    np.testing.assert_allclose(source_fp64, delayed_fp64, rtol=1e-12, atol=1e-12)

    source_bf16 = jnp.asarray(residual_sum * gamma * inverse_rms, dtype=jnp.bfloat16) @ jnp.asarray(
        weight, dtype=jnp.bfloat16
    )
    delayed_bf16 = (
        jnp.asarray(residual_sum * gamma, dtype=jnp.bfloat16) @ jnp.asarray(weight, dtype=jnp.bfloat16)
    ) * jnp.asarray(inverse_rms, dtype=jnp.float32)
    maximum_difference = float(jnp.max(jnp.abs(source_bf16.astype(jnp.float32) - delayed_bf16)))

    assert 0.0 < maximum_difference < 0.1


def test_consumer_prologue_scale_tracks_bf16_source_order_more_closely() -> None:
    rng = np.random.default_rng(7)
    residual_sum = rng.normal(size=(8, 16)).astype(np.float32)
    gamma = rng.normal(size=(16,)).astype(np.float32)
    weight = jnp.asarray(rng.normal(size=(16, 24)), dtype=jnp.bfloat16)
    inverse_rms = 1.0 / np.sqrt(np.mean(np.square(residual_sum), axis=-1, keepdims=True) + 1e-6)

    source = jnp.asarray(residual_sum * gamma * inverse_rms, dtype=jnp.bfloat16) @ weight
    stored_gamma_scaled = jnp.asarray(residual_sum * gamma, dtype=jnp.bfloat16)
    epilogue_scaled = (stored_gamma_scaled @ weight) * jnp.asarray(inverse_rms, dtype=jnp.float32)
    prologue_scaled = (
        jnp.asarray(
            stored_gamma_scaled.astype(jnp.float32) * inverse_rms,
            dtype=jnp.bfloat16,
        )
        @ weight
    )

    epilogue_error = jnp.mean(jnp.abs(epilogue_scaled.astype(jnp.float32) - source.astype(jnp.float32)))
    prologue_error = jnp.mean(jnp.abs(prologue_scaled.astype(jnp.float32) - source.astype(jnp.float32)))
    assert float(prologue_error) < float(epilogue_error)


def _attention_region() -> TensorGraph:
    graph = TensorGraph()
    query = graph.input("query", shape=(1, 128, 8, 64), dtype=DType.BF16)
    key = graph.input("key", shape=(1, 128, 2, 64), dtype=DType.BF16)
    value = graph.input("value", shape=(1, 128, 2, 64), dtype=DType.BF16)
    graph.scaled_dot_product_attention(
        query,
        key,
        value,
        name="attention_output",
        scale=0.125,
        causal=True,
        accumulation_dtype=DType.FP32,
    )
    return graph


def test_reference_attention_region_streams_exact_causal_gqa() -> None:
    plan = compile_reference_attention_region(
        _attention_region(),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )

    assert len(plan.skeletons) == 1
    skeleton = plan.skeletons[0]
    assert isinstance(skeleton, StreamingAttentionSkeleton)
    assert skeleton.causal
    assert skeleton.query_heads // skeleton.key_value_heads == 4
    assert skeleton.backend == "official_flashattention_3_hopper"
    assert skeleton.pipeline_stages == 2
    assert skeleton.producer_threads == 32
    assert skeleton.consumer_threads == 384
    assert skeleton.pack_gqa
    assert skeleton.mma_pv_is_rs
    assert skeleton.intra_warpgroup_overlap
    assert skeleton.persistent_scheduler
    assert skeleton.online_state == (
        "attention_output.online.max",
        "attention_output.online.sum",
        "attention_output.online.output",
    )
    assert plan.materialization("attention_output.scores").disposition is (
        MaterializationDisposition.INTERNAL_ATTENTION_STATE
    )
    assert plan.materialization("attention_output.probabilities").disposition is (
        MaterializationDisposition.INTERNAL_ATTENTION_STATE
    )
    assert plan.rewrites[0].applied


def test_reference_attention_region_bitwise_policy_keeps_quadratic_intermediates() -> None:
    plan = compile_reference_attention_region(_attention_region(), numerical_policy=NumericalPolicy.BITWISE_EXACT)

    assert isinstance(plan.skeletons[0], TransformSkeleton)
    assert plan.materialization("attention_output.scores").disposition is MaterializationDisposition.MATERIALIZE
    assert plan.materialization("attention_output.probabilities").disposition is MaterializationDisposition.MATERIALIZE
    assert not plan.rewrites[0].applied
