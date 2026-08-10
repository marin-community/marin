# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
import pytest

from shuttle.ir import DType
from tile_lifetime.cuda_axis_fold_codegen import (
    AxisFoldDirection,
    AxisFoldInput,
    AxisFoldInputLayout,
    AxisFoldOutputKind,
    AxisFoldPipeline,
    AxisFoldPipelineSchedule,
    AxisFoldPipelineStage,
    AxisFoldProgram,
    AxisFoldReassociation,
    AxisFoldReduction,
    AxisFoldTiledReductionStrategy,
    evaluate_axis_fold_pipeline,
    evaluate_axis_fold_program,
    generate_cuda_axis_fold,
    generate_cuda_axis_fold_ffi,
    generate_cuda_axis_fold_pipeline_ffi,
)
from tile_lifetime.tensor_program import ScalarExpressionKind, scalar_binary, scalar_constant, scalar_input


def _scaled_column_sum_program(*, threads: int = 64) -> AxisFoldProgram:
    value = scalar_input("value")
    scale = scalar_input("scale")
    return AxisFoldProgram(
        rows=5,
        columns=3,
        inputs=(
            AxisFoldInput("value", DType.FP32, AxisFoldInputLayout.ELEMENT),
            AxisFoldInput("scale", DType.FP32, AxisFoldInputLayout.COLUMN),
        ),
        reductions=(AxisFoldReduction("total", value),),
        reduction_axis=AxisFoldDirection.ROWS,
        output_kind=AxisFoldOutputKind.REDUCED,
        output_expression=scalar_binary(ScalarExpressionKind.MULTIPLY, scalar_input("total"), scale),
        output_dtype=DType.FP32,
        threads=threads,
    )


def test_axis_fold_evaluator_matches_independent_column_reduction() -> None:
    program = _scaled_column_sum_program()
    values = np.arange(15, dtype=np.float32).reshape(5, 3) - 3.0
    scale = np.asarray([0.5, -2.0, 3.0], dtype=np.float32)

    actual = evaluate_axis_fold_program(program, {"value": values, "scale": scale})

    np.testing.assert_array_equal(actual, np.sum(values, axis=0, dtype=np.float32) * scale)


def test_axis_fold_semantics_are_independent_of_schedule_and_mutate_through_ast() -> None:
    program = _scaled_column_sum_program()
    another_schedule = replace(program, threads=128)
    tiled_schedule = replace(program, threads=128, groups_per_block=16)
    two_output_schedule = replace(tiled_schedule, outputs_per_group=2)
    mutation = replace(
        program,
        output_expression=scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            program.output_expression,
            scalar_constant(0.5),
        ),
    )
    values = np.arange(15, dtype=np.float32).reshape(5, 3)
    scale = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

    assert another_schedule.semantic_fingerprint == program.semantic_fingerprint
    assert tiled_schedule.semantic_fingerprint == program.semantic_fingerprint
    assert two_output_schedule.semantic_fingerprint == program.semantic_fingerprint
    assert generate_cuda_axis_fold(another_schedule).source != generate_cuda_axis_fold(program).source
    assert "constexpr int kGroupsPerBlock = 16" in generate_cuda_axis_fold(tiled_schedule).source
    assert "stride * kGroupsPerBlock" in generate_cuda_axis_fold(tiled_schedule).source
    generated_two_output = generate_cuda_axis_fold(two_output_schedule)
    assert "constexpr int kOutputsPerGroup = 2" in generated_two_output.source
    assert "output_lane * kGroupsPerBlock + group_lane" in generated_two_output.source
    assert mutation.semantic_fingerprint != program.semantic_fingerprint
    np.testing.assert_array_equal(
        evaluate_axis_fold_program(mutation, {"value": values, "scale": scale}),
        evaluate_axis_fold_program(program, {"value": values, "scale": scale}) * 0.5,
    )


def test_tiled_row_fold_can_finalize_all_feature_groups_with_one_warp() -> None:
    barrier_tree = replace(
        _scaled_column_sum_program(threads=256),
        rows=2048,
        columns=4096,
        groups_per_block=32,
    )
    warp_finalize = replace(
        barrier_tree,
        tiled_reduction_strategy=AxisFoldTiledReductionStrategy.WARP_FINALIZE,
    )
    values = np.arange(35, dtype=np.float32).reshape(7, 5)
    scale = np.linspace(0.5, 1.5, 5, dtype=np.float32)

    assert warp_finalize.semantic_fingerprint == barrier_tree.semantic_fingerprint
    np.testing.assert_array_equal(
        evaluate_axis_fold_program(
            replace(warp_finalize, rows=7, columns=5),
            {"value": values, "scale": scale},
        ),
        evaluate_axis_fold_program(
            replace(barrier_tree, rows=7, columns=5),
            {"value": values, "scale": scale},
        ),
    )

    torch_source = generate_cuda_axis_fold(warp_finalize).source
    ffi_source = generate_cuda_axis_fold_ffi((warp_finalize,), target_name="shuttle.warp_finalize_v1").source
    for source in (torch_source, ffi_source):
        assert source.count("__syncthreads();") == 1
        assert "for (int stride" not in source
        assert "partial *" in source
        assert "if (threadIdx.x <" in source
    assert torch_source != generate_cuda_axis_fold(barrier_tree).source


@pytest.mark.parametrize(
    ("groups", "outputs", "threads"),
    [(16, 1, 256), (32, 2, 256), (32, 1, 2048)],
)
def test_warp_finalized_tiled_row_fold_rejects_unsupported_physical_shapes(
    groups: int,
    outputs: int,
    threads: int,
) -> None:
    with pytest.raises(ValueError, match="warp-finalized"):
        replace(
            _scaled_column_sum_program(),
            threads=threads,
            groups_per_block=groups,
            outputs_per_group=outputs,
            tiled_reduction_strategy=AxisFoldTiledReductionStrategy.WARP_FINALIZE,
        )


def test_tiled_row_fold_multiple_outputs_preserve_tail_column_semantics() -> None:
    program = replace(
        _scaled_column_sum_program(threads=64),
        rows=7,
        columns=35,
        groups_per_block=16,
        outputs_per_group=2,
    )
    values = np.arange(program.rows * program.columns, dtype=np.float32).reshape(program.rows, program.columns)
    scale = np.linspace(0.5, 1.5, program.columns, dtype=np.float32)

    actual = evaluate_axis_fold_program(program, {"value": values, "scale": scale})
    generated = generate_cuda_axis_fold_ffi((program,), target_name="shuttle.axis_fold_two_output_v1")

    np.testing.assert_array_equal(actual, np.sum(values, axis=0, dtype=np.float32) * scale)
    assert "kProgram0OutputsPerGroup = 2" in generated.source
    assert "kProgram0GroupsPerBlock * kProgram0OutputsPerGroup" in generated.source
    assert "if (group < kProgram0Columns)" in generated.source


def test_axis_fold_codegen_exposes_generated_scalar_body_without_workload_kernel() -> None:
    generated = generate_cuda_axis_fold(_scaled_column_sum_program())

    assert generated.source_sha256
    assert "__fadd_rn" in generated.source
    assert "__fmul_rn" in generated.source
    assert "shuttle_axis_fold_kernel" in generated.source
    assert "rms" not in generated.source.lower()
    assert "layernorm" not in generated.source.lower()
    assert "atomic" not in generated.source.lower()


def test_source_ordered_fold_uses_one_worker() -> None:
    program = _scaled_column_sum_program(threads=1)

    source_ordered = replace(program, reassociation=AxisFoldReassociation.SOURCE_ORDERED)

    assert source_ordered.semantic_fingerprint != program.semantic_fingerprint
    assert "constexpr int kThreads = 1" in generate_cuda_axis_fold(source_ordered).source


def test_axis_fold_ffi_composes_element_and_reduced_programs_without_torch() -> None:
    value = scalar_input("value")
    row_scale = scalar_input("row_scale")
    element_program = AxisFoldProgram(
        rows=5,
        columns=3,
        inputs=(
            AxisFoldInput("value", DType.FP32, AxisFoldInputLayout.ELEMENT),
            AxisFoldInput("row_scale", DType.FP32, AxisFoldInputLayout.ROW),
        ),
        reductions=(AxisFoldReduction("row_total", value),),
        reduction_axis=AxisFoldDirection.COLUMNS,
        output_kind=AxisFoldOutputKind.ELEMENT,
        output_expression=scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            row_scale,
            scalar_binary(
                ScalarExpressionKind.SUBTRACT,
                value,
                scalar_binary(
                    ScalarExpressionKind.DIVIDE,
                    scalar_input("row_total"),
                    scalar_constant(3.0),
                ),
            ),
        ),
        output_dtype=DType.FP32,
        threads=64,
    )
    reduced_program = replace(_scaled_column_sum_program(threads=64), groups_per_block=16)

    generated = generate_cuda_axis_fold_ffi(
        (element_program, reduced_program),
        target_name="shuttle.axis_fold_test_v1",
    )

    assert generated.handler_symbol == "shuttle_axis_fold_test_v1"
    assert [(value.name, value.dtype, value.rank) for value in generated.inputs] == [
        ("value", DType.FP32, 2),
        ("row_scale", DType.FP32, 1),
        ("scale", DType.FP32, 1),
    ]
    assert [(value.dtype, value.rank) for value in generated.outputs] == [
        (DType.FP32, 2),
        (DType.FP32, 1),
    ]
    assert generated.semantic_fingerprints == (
        element_program.semantic_fingerprint,
        reduced_program.semantic_fingerprint,
    )
    assert "XLA_FFI_DEFINE_HANDLER_SYMBOL" in generated.source
    assert "ShuttleAxisFoldKernel0" in generated.source
    assert "ShuttleAxisFoldKernel1" in generated.source
    assert "kProgram1GroupsPerBlock = 16" in generated.source
    assert "torch" not in generated.source.lower()


def test_axis_fold_ffi_semantic_mutation_regenerates_physical_source() -> None:
    program = _scaled_column_sum_program()
    mutated = replace(
        program,
        output_expression=scalar_binary(
            ScalarExpressionKind.ADD,
            program.output_expression,
            scalar_constant(1.0),
        ),
    )

    original = generate_cuda_axis_fold_ffi((program,), target_name="shuttle.axis_fold_mutation_v1")
    changed = generate_cuda_axis_fold_ffi((mutated,), target_name="shuttle.axis_fold_mutation_v1")

    assert original.semantic_fingerprints != changed.semantic_fingerprints
    assert original.source_sha256 != changed.source_sha256


def test_axis_fold_pipeline_uses_internal_scratch_between_generated_stages() -> None:
    value = scalar_input("value")
    row_total = scalar_input("row_total")
    first = AxisFoldProgram(
        rows=5,
        columns=3,
        inputs=(AxisFoldInput("value", DType.FP32, AxisFoldInputLayout.ELEMENT),),
        reductions=(AxisFoldReduction("total", value),),
        reduction_axis=AxisFoldDirection.COLUMNS,
        output_kind=AxisFoldOutputKind.REDUCED,
        output_expression=scalar_input("total"),
        output_dtype=DType.FP32,
        threads=64,
    )
    second = AxisFoldProgram(
        rows=5,
        columns=3,
        inputs=(
            AxisFoldInput("value", DType.FP32, AxisFoldInputLayout.ELEMENT),
            AxisFoldInput("row_total", DType.FP32, AxisFoldInputLayout.ROW),
        ),
        reductions=(AxisFoldReduction("ignored", scalar_constant(0.0)),),
        reduction_axis=AxisFoldDirection.COLUMNS,
        output_kind=AxisFoldOutputKind.ELEMENT,
        output_expression=scalar_binary(ScalarExpressionKind.ADD, value, row_total),
        output_dtype=DType.FP32,
        threads=64,
    )
    pipeline = AxisFoldPipeline(
        (
            AxisFoldPipelineStage("row_total", first, expose_output=False),
            AxisFoldPipelineStage("result", second, expose_output=True),
        )
    )

    generated = generate_cuda_axis_fold_pipeline_ffi(pipeline, target_name="shuttle.axis_fold_pipeline_v1")
    values = np.arange(15, dtype=np.float32).reshape(5, 3)
    (actual,) = evaluate_axis_fold_pipeline(pipeline, {"value": values})

    assert [(item.name, item.shape) for item in generated.inputs] == [("value", (5, 3))]
    assert [(item.name, item.shape) for item in generated.outputs] == [("result", (5, 3))]
    assert "row_total_storage = scratch.Allocate" in generated.source
    assert "ShuttleAxisFoldKernel0" in generated.source
    assert "ShuttleAxisFoldKernel1" in generated.source
    np.testing.assert_array_equal(actual, values + np.sum(values, axis=1, keepdims=True))


def test_axis_fold_pipeline_scalar_mutation_regenerates_same_physical_family() -> None:
    program = _scaled_column_sum_program()
    pipeline = AxisFoldPipeline((AxisFoldPipelineStage("result", program, expose_output=True),))
    mutated_program = replace(
        program,
        output_expression=scalar_binary(
            ScalarExpressionKind.ADD,
            program.output_expression,
            scalar_constant(7.0),
        ),
    )
    mutated = replace(
        pipeline,
        stages=(AxisFoldPipelineStage("result", mutated_program, expose_output=True),),
    )

    original_source = generate_cuda_axis_fold_pipeline_ffi(
        pipeline,
        target_name="shuttle.axis_fold_pipeline_mutation_v1",
    )
    changed_source = generate_cuda_axis_fold_pipeline_ffi(
        mutated,
        target_name="shuttle.axis_fold_pipeline_mutation_v1",
    )

    assert original_source.semantic_fingerprints != changed_source.semantic_fingerprints
    assert original_source.source_sha256 != changed_source.source_sha256
    assert "ShuttleAxisFoldKernel0" in changed_source.source


def test_axis_fold_pipeline_coalesces_compatible_row_stages_without_changing_semantics() -> None:
    value = scalar_input("value")
    row_total = scalar_input("row_total")
    first = AxisFoldProgram(
        rows=5,
        columns=3,
        inputs=(AxisFoldInput("value", DType.FP32, AxisFoldInputLayout.ELEMENT),),
        reductions=(AxisFoldReduction("total", value),),
        reduction_axis=AxisFoldDirection.COLUMNS,
        output_kind=AxisFoldOutputKind.REDUCED,
        output_expression=scalar_input("total"),
        output_dtype=DType.FP32,
        threads=64,
    )
    second = AxisFoldProgram(
        rows=5,
        columns=3,
        inputs=(
            AxisFoldInput("value", DType.FP32, AxisFoldInputLayout.ELEMENT),
            AxisFoldInput("row_total", DType.FP32, AxisFoldInputLayout.ROW),
        ),
        reductions=(AxisFoldReduction("local_total", value),),
        reduction_axis=AxisFoldDirection.COLUMNS,
        output_kind=AxisFoldOutputKind.ELEMENT,
        output_expression=scalar_binary(ScalarExpressionKind.ADD, row_total, scalar_input("local_total")),
        output_dtype=DType.FP32,
        threads=64,
    )
    pipeline = AxisFoldPipeline(
        (
            AxisFoldPipelineStage("row_total", first, expose_output=False),
            AxisFoldPipelineStage("result", second, expose_output=True),
        )
    )

    separate = generate_cuda_axis_fold_pipeline_ffi(
        pipeline,
        target_name="shuttle.axis_fold_coalescing_v1",
    )
    coalesced = generate_cuda_axis_fold_pipeline_ffi(
        pipeline,
        target_name="shuttle.axis_fold_coalescing_v1",
        schedule=AxisFoldPipelineSchedule.COALESCE_COMPATIBLE_ROW_STAGES,
    )
    values = np.arange(15, dtype=np.float32).reshape(5, 3)
    (actual,) = evaluate_axis_fold_pipeline(pipeline, {"value": values})

    assert coalesced.semantic_fingerprints == separate.semantic_fingerprints
    assert coalesced.source_sha256 != separate.source_sha256
    assert coalesced.pipeline_schedule is AxisFoldPipelineSchedule.COALESCE_COMPATIBLE_ROW_STAGES
    assert "ShuttleAxisFoldKernel0And1" in coalesced.source
    assert "ShuttleAxisFoldKernel0<<<" not in coalesced.source
    assert "ShuttleAxisFoldKernel1<<<" not in coalesced.source
    assert "row_total_storage = scratch.Allocate" in coalesced.source
    expected = np.repeat(2.0 * np.sum(values, axis=1, keepdims=True), values.shape[1], axis=1)
    np.testing.assert_array_equal(actual, expected)


def test_axis_fold_pipeline_rejects_coalescing_when_second_reduction_uses_first_output() -> None:
    value = scalar_input("value")
    row_total = scalar_input("row_total")
    first = AxisFoldProgram(
        rows=5,
        columns=3,
        inputs=(AxisFoldInput("value", DType.FP32, AxisFoldInputLayout.ELEMENT),),
        reductions=(AxisFoldReduction("total", value),),
        reduction_axis=AxisFoldDirection.COLUMNS,
        output_kind=AxisFoldOutputKind.REDUCED,
        output_expression=scalar_input("total"),
        output_dtype=DType.FP32,
        threads=64,
    )
    dependent_contribution = scalar_binary(ScalarExpressionKind.MULTIPLY, value, row_total)
    second = AxisFoldProgram(
        rows=5,
        columns=3,
        inputs=(
            AxisFoldInput("value", DType.FP32, AxisFoldInputLayout.ELEMENT),
            AxisFoldInput("row_total", DType.FP32, AxisFoldInputLayout.ROW),
        ),
        reductions=(AxisFoldReduction("dependent", dependent_contribution),),
        reduction_axis=AxisFoldDirection.COLUMNS,
        output_kind=AxisFoldOutputKind.ELEMENT,
        output_expression=scalar_binary(ScalarExpressionKind.ADD, row_total, scalar_input("dependent")),
        output_dtype=DType.FP32,
        threads=64,
    )
    pipeline = AxisFoldPipeline(
        (
            AxisFoldPipelineStage("row_total", first, expose_output=False),
            AxisFoldPipelineStage("result", second, expose_output=True),
        )
    )

    with pytest.raises(ValueError, match="no compatible adjacent row stages"):
        generate_cuda_axis_fold_pipeline_ffi(
            pipeline,
            target_name="shuttle.axis_fold_illegal_coalescing_v1",
            schedule=AxisFoldPipelineSchedule.COALESCE_COMPATIBLE_ROW_STAGES,
        )
