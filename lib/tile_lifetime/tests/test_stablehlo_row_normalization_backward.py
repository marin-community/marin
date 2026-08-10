# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tile_lifetime import (
    RowStatisticKind,
    compile_stablehlo_row_normalization_backward,
)
from tile_lifetime.cuda_axis_fold_codegen import (
    AxisFoldPipelineSchedule,
    AxisFoldTiledReductionStrategy,
    evaluate_axis_fold_pipeline,
    evaluate_axis_fold_program,
    generate_cuda_axis_fold,
)
from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.stablehlo_import import import_stablehlo
from tile_lifetime.stablehlo_row_normalization_backward import (
    StableHLORowNormalizationBackwardError,
    compile_stablehlo_row_normalization_backward_ffi,
)
from tile_lifetime.xla_axis_fold_pipeline_ffi import (
    audit_axis_fold_pipeline_hlo_replacement,
    plan_axis_fold_pipeline_hlo_replacement,
    replace_axis_fold_pipeline_hlo_with_custom_call,
)
from tile_lifetime.xla_hlo_recovery import parse_hlo_module_text


def _natural_jax_vjp(*, centered: bool, rows: int = 4, hidden: int = 8, epsilon: float = 1e-5):
    def normalization(x, feature_scale):
        local = x.astype(jnp.float32)
        if centered:
            local -= jnp.mean(local, axis=-1, keepdims=True)
        inverse = jax.lax.rsqrt(jnp.mean(jnp.square(local), axis=-1, keepdims=True) + epsilon)
        return (local * inverse * feature_scale.astype(jnp.float32)).astype(jnp.bfloat16)

    def reverse(x, feature_scale, cotangent):
        _, pullback = jax.vjp(normalization, x, feature_scale)
        return pullback(cotangent)

    arguments = (
        jnp.zeros((rows, hidden), dtype=jnp.bfloat16),
        jnp.ones((hidden,), dtype=jnp.bfloat16),
        jnp.zeros((rows, hidden), dtype=jnp.bfloat16),
    )
    exported = jax.export.export(jax.jit(reverse))(*arguments)
    graph = import_stablehlo(
        exported.mlir_module_serialized,
        input_names=("arbitrary_matrix_a", "arbitrary_vector", "arbitrary_matrix_b"),
    )
    return reverse, graph


@pytest.mark.parametrize(
    ("centered", "statistic_kind", "input_reduction_count"),
    [
        (False, RowStatisticKind.UNCENTERED_SECOND_MOMENT, 1),
        (True, RowStatisticKind.CENTERED_SECOND_MOMENT, 2),
    ],
)
def test_jax_vjp_recovers_generic_row_folds_and_matches_bf16_outputs(
    centered: bool,
    statistic_kind: RowStatisticKind,
    input_reduction_count: int,
) -> None:
    reverse, graph = _natural_jax_vjp(centered=centered)
    compilation = compile_stablehlo_row_normalization_backward(graph, threads=8)
    recovered = compilation.recovered
    programs = compilation.programs

    assert recovered.statistic_kind is statistic_kind
    assert recovered.rows == 4
    assert recovered.hidden == 8
    assert len(programs.input_cotangent.reductions) == input_reduction_count

    rng = np.random.default_rng(19)
    x = rng.normal(size=(4, 8)).astype(np.float32)
    feature_scale = rng.normal(size=(8,)).astype(np.float32)
    cotangent = rng.normal(size=(4, 8)).astype(np.float32)
    x_bf16 = jnp.asarray(x, dtype=jnp.bfloat16)
    scale_bf16 = jnp.asarray(feature_scale, dtype=jnp.bfloat16)
    cotangent_bf16 = jnp.asarray(cotangent, dtype=jnp.bfloat16)
    expected_input, expected_scale = reverse(x_bf16, scale_bf16, cotangent_bf16)

    source = np.asarray(x_bf16, dtype=np.float32)
    local = source - np.mean(source, axis=1, keepdims=True) if centered else source
    inverse = np.reciprocal(np.sqrt(np.mean(local * local, axis=1) + 1e-5))
    standardized = local * inverse[:, None]
    projected = np.asarray(cotangent_bf16, dtype=np.float32)
    actual_input = evaluate_axis_fold_program(
        programs.input_cotangent,
        {
            "projected": projected,
            "feature_scale": np.asarray(scale_bf16, dtype=np.float32),
            "standardized": standardized,
            "inverse_scale": inverse,
        },
    )
    actual_scale = evaluate_axis_fold_program(
        programs.feature_scale_cotangent,
        {"projected": projected, "standardized": standardized},
    )
    actual_input_bf16 = np.asarray(jnp.asarray(actual_input, dtype=jnp.bfloat16), dtype=np.float32)
    actual_scale_bf16 = np.asarray(jnp.asarray(actual_scale, dtype=jnp.bfloat16), dtype=np.float32)
    expected_input_array = np.asarray(expected_input, dtype=np.float32)
    expected_scale_array = np.asarray(expected_scale, dtype=np.float32)
    input_error = np.abs(actual_input_bf16 - expected_input_array)
    scale_error = np.abs(actual_scale_bf16 - expected_scale_array)

    assert float(input_error.max()) <= 0.015625
    assert float(input_error.mean()) <= 0.001
    assert float(scale_error.max()) <= 0.03125
    assert float(scale_error.mean()) <= 0.004
    generated = generate_cuda_axis_fold(programs.input_cotangent).source.lower()
    assert "rms" not in generated
    assert "layernorm" not in generated


def test_natural_uncentered_vjp_becomes_whole_entry_generated_ffi_pipeline() -> None:
    reverse, graph = _natural_jax_vjp(centered=False)

    compilation = compile_stablehlo_row_normalization_backward_ffi(
        graph,
        target_name="shuttle.row_statistic_backward_v1",
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        threads=8,
    )

    assert compilation.recovered.epsilon == pytest.approx(1e-5)
    assert compilation.input_bindings == (
        ("primal", graph.inputs[0]),
        ("feature_scale", graph.inputs[1]),
        ("output_cotangent", graph.inputs[2]),
    )
    assert tuple(value_id for _, value_id in compilation.output_bindings) == graph.outputs
    assert compilation.replaced_operation_ids == tuple(operation.id for operation in graph.operations)
    assert [stage.output_name for stage in compilation.pipeline.stages] == [
        "inverse_scale",
        "input_cotangent",
        "feature_scale_cotangent",
    ]
    assert [stage.expose_output for stage in compilation.pipeline.stages] == [False, True, True]
    assert [(value.name, value.shape) for value in compilation.generated.inputs] == [
        ("primal", (4, 8)),
        ("feature_scale", (8,)),
        ("output_cotangent", (4, 8)),
    ]
    assert [(value.name, value.shape) for value in compilation.generated.outputs] == [
        ("input_cotangent", (4, 8)),
        ("feature_scale_cotangent", (8,)),
    ]
    assert "inverse_scale_storage = scratch.Allocate" in compilation.generated.source
    assert "__bfloat162float" in compilation.generated.source
    assert "__float2bfloat16_rn" in compilation.generated.source
    assert "rmsnorm" not in compilation.generated.source.lower()
    assert "layernorm" not in compilation.generated.source.lower()

    rng = np.random.default_rng(41)
    x = jnp.asarray(rng.normal(size=(4, 8)), dtype=jnp.bfloat16)
    feature_scale = jnp.asarray(rng.normal(size=(8,)), dtype=jnp.bfloat16)
    cotangent = jnp.asarray(rng.normal(size=(4, 8)), dtype=jnp.bfloat16)
    expected_input, expected_scale = reverse(x, feature_scale, cotangent)
    actual_input, actual_scale = evaluate_axis_fold_pipeline(
        compilation.pipeline,
        {
            "primal": np.asarray(x, dtype=np.float32),
            "feature_scale": np.asarray(feature_scale, dtype=np.float32),
            "output_cotangent": np.asarray(cotangent, dtype=np.float32),
        },
    )
    actual_input = np.asarray(jnp.asarray(actual_input, dtype=jnp.bfloat16), dtype=np.float32)
    actual_scale = np.asarray(jnp.asarray(actual_scale, dtype=jnp.bfloat16), dtype=np.float32)

    np.testing.assert_allclose(actual_input, np.asarray(expected_input, dtype=np.float32), atol=0.016, rtol=0.01)
    np.testing.assert_allclose(actual_scale, np.asarray(expected_scale, dtype=np.float32), atol=0.032, rtol=0.01)


def test_natural_uncentered_vjp_can_select_generic_same_domain_fold_coalescing() -> None:
    _, graph = _natural_jax_vjp(centered=False)

    compilation = compile_stablehlo_row_normalization_backward_ffi(
        graph,
        target_name="shuttle.row_statistic_backward_coalesced_v1",
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        threads=8,
        pipeline_schedule=AxisFoldPipelineSchedule.COALESCE_COMPATIBLE_ROW_STAGES,
    )

    assert compilation.generated.pipeline_schedule is AxisFoldPipelineSchedule.COALESCE_COMPATIBLE_ROW_STAGES
    assert "ShuttleAxisFoldKernel0And1" in compilation.generated.source
    assert "ShuttleAxisFoldKernel2" in compilation.generated.source
    assert "inverse_scale_storage = scratch.Allocate" in compilation.generated.source


def test_natural_uncentered_vjp_can_select_warp_finalized_feature_fold() -> None:
    _, graph = _natural_jax_vjp(centered=False, rows=64, hidden=64)

    barrier_tree = compile_stablehlo_row_normalization_backward_ffi(
        graph,
        target_name="shuttle.row_statistic_backward_barrier_tree_v1",
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        threads=256,
        feature_groups_per_block=32,
    )
    warp_finalize = compile_stablehlo_row_normalization_backward_ffi(
        graph,
        target_name="shuttle.row_statistic_backward_warp_finalize_v1",
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        threads=256,
        feature_groups_per_block=32,
        feature_tiled_reduction_strategy=AxisFoldTiledReductionStrategy.WARP_FINALIZE,
    )

    barrier_feature = barrier_tree.pipeline.stages[-1].program
    warp_feature = warp_finalize.pipeline.stages[-1].program
    assert warp_feature.tiled_reduction_strategy is AxisFoldTiledReductionStrategy.WARP_FINALIZE
    assert warp_feature.semantic_fingerprint == barrier_feature.semantic_fingerprint
    assert warp_finalize.generated.semantic_fingerprints == barrier_tree.generated.semantic_fingerprints
    assert warp_finalize.generated.source_sha256 != barrier_tree.generated.source_sha256
    warp_kernel = warp_finalize.generated.source.split("ShuttleAxisFoldKernel2", maxsplit=1)[1]
    assert warp_kernel.count("__syncthreads();") == 1
    assert "for (int stride" not in warp_kernel


def test_natural_scalar_mutation_changes_generated_pipeline_without_physical_switch() -> None:
    _, original_graph = _natural_jax_vjp(centered=False, epsilon=1e-5)
    _, mutated_graph = _natural_jax_vjp(centered=False, epsilon=2e-4)

    original = compile_stablehlo_row_normalization_backward_ffi(
        original_graph,
        target_name="shuttle.row_statistic_backward_mutation_v1",
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        threads=8,
    )
    mutated = compile_stablehlo_row_normalization_backward_ffi(
        mutated_graph,
        target_name="shuttle.row_statistic_backward_mutation_v1",
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        threads=8,
    )

    assert original.recovered.epsilon == pytest.approx(1e-5)
    assert mutated.recovered.epsilon == pytest.approx(2e-4)
    assert original.generated.semantic_fingerprints != mutated.generated.semantic_fingerprints
    assert original.generated.source_sha256 != mutated.generated.source_sha256
    assert original.generated.inputs == mutated.generated.inputs
    assert original.generated.outputs == mutated.generated.outputs


def test_bounded_ffi_path_fails_closed_for_source_order_and_executes_centered_statistics() -> None:
    centered_reverse, centered_graph = _natural_jax_vjp(centered=True)
    _, uncentered_graph = _natural_jax_vjp(centered=False)

    with pytest.raises(StableHLORowNormalizationBackwardError, match="ALLOW_ROUNDING_REORDER"):
        compile_stablehlo_row_normalization_backward_ffi(
            uncentered_graph,
            target_name="shuttle.row_statistic_source_order_v1",
            numerical_policy=NumericalPolicy.BITWISE_EXACT,
            threads=8,
        )
    centered = compile_stablehlo_row_normalization_backward_ffi(
        centered_graph,
        target_name="shuttle.row_statistic_centered_v1",
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        threads=8,
        feature_groups_per_block=8,
        feature_outputs_per_group=2,
    )
    assert centered.recovered.statistic_kind is RowStatisticKind.CENTERED_SECOND_MOMENT
    assert tuple(stage.output_name for stage in centered.pipeline.stages) == (
        "row_mean",
        "inverse_scale",
        "input_cotangent",
        "feature_scale_cotangent",
    )
    assert centered.generated.semantic_fingerprints == tuple(
        stage.program.semantic_fingerprint for stage in centered.pipeline.stages
    )
    assert "ShuttleAxisFoldKernel3" in centered.generated.source
    assert centered.pipeline.stages[-1].program.outputs_per_group == 2
    assert "kProgram3OutputsPerGroup = 2" in centered.generated.source
    assert "layernorm" not in centered.generated.source.lower()

    rng = np.random.default_rng(31)
    x = jnp.asarray(rng.normal(size=(4, 8)), dtype=jnp.bfloat16)
    feature_scale = jnp.asarray(rng.normal(size=(8,)), dtype=jnp.bfloat16)
    cotangent = jnp.asarray(rng.normal(size=(4, 8)), dtype=jnp.bfloat16)
    expected_input, expected_scale = centered_reverse(x, feature_scale, cotangent)
    actual_input, actual_scale = evaluate_axis_fold_pipeline(
        centered.pipeline,
        {
            "primal": np.asarray(x, dtype=np.float32),
            "feature_scale": np.asarray(feature_scale, dtype=np.float32),
            "output_cotangent": np.asarray(cotangent, dtype=np.float32),
        },
    )
    actual_input_bf16 = np.asarray(jnp.asarray(actual_input, dtype=jnp.bfloat16), dtype=np.float32)
    actual_scale_bf16 = np.asarray(jnp.asarray(actual_scale, dtype=jnp.bfloat16), dtype=np.float32)
    np.testing.assert_allclose(actual_input_bf16, np.asarray(expected_input, dtype=np.float32), rtol=0, atol=0.015625)
    np.testing.assert_allclose(actual_scale_bf16, np.asarray(expected_scale, dtype=np.float32), rtol=0, atol=0.03125)

    with pytest.raises(StableHLORowNormalizationBackwardError, match="separate-stage"):
        compile_stablehlo_row_normalization_backward_ffi(
            centered_graph,
            target_name="shuttle.row_statistic_centered_coalesced_v1",
            numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
            threads=8,
            pipeline_schedule=AxisFoldPipelineSchedule.COALESCE_COMPATIBLE_ROW_STAGES,
        )


@pytest.mark.parametrize("centered", [False, True])
def test_generated_pipeline_replaces_and_audits_natural_whole_entry_hlo(centered: bool) -> None:
    reverse, graph = _natural_jax_vjp(centered=centered)
    target = f"shuttle.row_statistic_whole_entry_{'centered' if centered else 'uncentered'}_v1"
    arguments = (
        jnp.zeros((4, 8), dtype=jnp.bfloat16),
        jnp.ones((8,), dtype=jnp.bfloat16),
        jnp.zeros((4, 8), dtype=jnp.bfloat16),
    )
    hlo = jax.jit(reverse).lower(*arguments).compiler_ir("hlo").as_hlo_text()
    compilation = compile_stablehlo_row_normalization_backward_ffi(
        graph,
        target_name=target,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        threads=8,
    )

    plan = plan_axis_fold_pipeline_hlo_replacement(
        hlo,
        compilation.generated,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    transformed = replace_axis_fold_pipeline_hlo_with_custom_call(
        hlo,
        plan,
        target=compilation.generated.target_name,
    )
    audit = audit_axis_fold_pipeline_hlo_replacement(
        hlo,
        transformed,
        plan,
        target=compilation.generated.target_name,
    )
    transformed_entry = parse_hlo_module_text(transformed).computation(plan.entry)

    assert tuple(value.buffer_name for value in plan.inputs) == ("primal", "feature_scale", "output_cotangent")
    assert len(plan.replaced_instructions) > 20
    assert audit.dead_internal_instructions == plan.replaced_instructions
    assert audit.copy_count[0] == audit.copy_count[1]
    assert audit.transpose_count[0] == audit.transpose_count[1]
    assert all("shuttle.axis_fold.pipeline.output" in name for name in transformed_entry.root.operands)
    assert transformed.count(f'custom_call_target="{target}"') == 1

    call_operands = ", ".join(f"%{value.instruction}" for value in plan.inputs)
    swapped_operands = ", ".join(f"%{value.instruction}" for value in (plan.inputs[1], plan.inputs[0], *plan.inputs[2:]))
    wrong_operands = transformed.replace(
        f"custom-call({call_operands})",
        f"custom-call({swapped_operands})",
        1,
    )
    with pytest.raises(ValueError, match="operands changed"):
        audit_axis_fold_pipeline_hlo_replacement(
            hlo,
            wrong_operands,
            plan,
            target=compilation.generated.target_name,
        )

    wrong_api = transformed.replace("api_version=API_VERSION_TYPED_FFI", "api_version=API_VERSION_STATUS_RETURNING", 1)
    with pytest.raises(ValueError, match="typed-FFI API"):
        audit_axis_fold_pipeline_hlo_replacement(
            hlo,
            wrong_api,
            plan,
            target=compilation.generated.target_name,
        )
