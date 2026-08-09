# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np

from tile_lifetime.cuda_axis_fold_codegen import (
    AxisFoldDirection,
    AxisFoldInput,
    AxisFoldInputLayout,
    AxisFoldOutputKind,
    AxisFoldProgram,
    AxisFoldReassociation,
    AxisFoldReduction,
    evaluate_axis_fold_program,
    generate_cuda_axis_fold,
)
from tile_lifetime.ir import DType
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
    assert generate_cuda_axis_fold(another_schedule).source != generate_cuda_axis_fold(program).source
    assert mutation.semantic_fingerprint != program.semantic_fingerprint
    np.testing.assert_array_equal(
        evaluate_axis_fold_program(mutation, {"value": values, "scale": scale}),
        evaluate_axis_fold_program(program, {"value": values, "scale": scale}) * 0.5,
    )


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
