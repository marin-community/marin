# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
import pytest

from shuttle.ir import DType
from tile_lifetime.tensor_program import ScalarExpressionKind, TensorAxis, scalar_binary, scalar_input
from tile_lifetime.tiled_fold_finalize import (
    FoldFeatureLayout,
    FoldPartialAddressing,
    FoldPhysicalAxis,
    TiledFoldAxes,
    TiledFoldFinalizeProgram,
    TiledFoldFinalizeSchedule,
    TiledFoldInputLayout,
    deterministic_weighted_sum_fold_program,
    evaluate_tiled_fold_finalize,
    normalized_exponential_fold_program,
)


def _axes(*, partials: int = 4, rows: int = 3, features: int = 8) -> TiledFoldAxes:
    return TiledFoldAxes(
        partial=TensorAxis(100, partials, "partial"),
        row=TensorAxis(101, rows, "row"),
        feature=TensorAxis(102, features, "feature"),
    )


def _schedule(
    *,
    axes: TiledFoldAxes,
    partial_lanes: int,
    addressing: FoldPartialAddressing,
    feature_tile: int = 64,
    shared_buffers: int = 2,
) -> TiledFoldFinalizeSchedule:
    if addressing is FoldPartialAddressing.DENSE:
        input_layout = TiledFoldInputLayout(
            addressing=addressing,
            value_axis_order=(FoldPhysicalAxis.PARTIAL, FoldPhysicalAxis.ROW, FoldPhysicalAxis.FEATURE),
            scalar_axis_order=(FoldPhysicalAxis.PARTIAL, FoldPhysicalAxis.ROW),
            feature_layout=FoldFeatureLayout.STG128_LANE_PERMUTED,
        )
    else:
        input_layout = TiledFoldInputLayout(
            addressing=addressing,
            value_axis_order=(FoldPhysicalAxis.SOURCE, FoldPhysicalAxis.FEATURE),
            scalar_axis_order=(FoldPhysicalAxis.ROW, FoldPhysicalAxis.PARTIAL),
            index_axis_order=(FoldPhysicalAxis.ROW, FoldPhysicalAxis.PARTIAL),
        )
    return TiledFoldFinalizeSchedule(
        axes=axes,
        partial_addressing=addressing,
        row_tile=8,
        feature_tile=feature_tile,
        vector_bytes=16,
        shared_stages=4,
        threads=256,
        partial_lanes=partial_lanes,
        shared_buffers=shared_buffers,
        input_layout=input_layout,
    )


def test_normalized_exponential_fold_matches_materialized_reference_with_invalid_partials() -> None:
    rng = np.random.default_rng(11)
    axes = _axes()
    values = rng.normal(size=(4, 3, 8)).astype(np.float32)
    log_normalizers = rng.normal(size=(4, 3)).astype(np.float32)
    valid = np.array(
        [
            [True, True, True],
            [False, True, True],
            [False, False, True],
            [False, False, True],
        ],
        dtype=np.bool_,
    )
    # Invalid payload must be invisible, including an unwritten underfilled
    # top-k slot containing nonsensical data that would overflow exp.
    values[1:, 0] = np.nan
    log_normalizers[1:, 0] = np.float32(1e30)
    program = normalized_exponential_fold_program(
        _schedule(axes=axes, partial_lanes=32, addressing=FoldPartialAddressing.DENSE),
        partial_value_dtype=DType.FP32,
        output_dtype=DType.FP32,
    )

    actual = evaluate_tiled_fold_finalize(program, values, log_normalizers, partial_valid=valid)
    masked_lse = np.where(valid, log_normalizers, -np.inf)
    common = np.max(masked_lse, axis=0)
    weights = np.where(valid, np.exp(masked_lse - common), 0.0).astype(np.float32)
    safe_values = np.where(valid[..., None], values, 0.0)
    expected_denominator = np.asarray(np.sum(weights, axis=0, dtype=np.float32))
    expected = np.sum(weights[..., None] * safe_values, axis=0, dtype=np.float32) / expected_denominator[:, None]

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_weighted_sum_fold_matches_independent_vectorized_reference_in_source_order() -> None:
    axes = _axes(partials=3, rows=5, features=6)
    # Dyadic values make the independent vectorized sum exact while still
    # exercising validity and per-row weights.
    values = np.arange(90, dtype=np.float32).reshape(3, 5, 6) / np.float32(8)
    weights = np.array(
        [[0.25, 0.5, 1.0, 0.125, 0.75], [0.5, 0.25, 0.5, 0.25, 0.125], [0.25] * 5],
        dtype=np.float32,
    )
    valid = np.array(
        [
            [True, True, True, True, True],
            [True, False, True, False, True],
            [False, False, True, True, True],
        ],
        dtype=np.bool_,
    )
    program = deterministic_weighted_sum_fold_program(
        _schedule(axes=axes, partial_lanes=1, addressing=FoldPartialAddressing.INDEXED),
        partial_value_dtype=DType.FP32,
        output_dtype=DType.FP32,
    )

    actual = evaluate_tiled_fold_finalize(program, values, weights, partial_valid=valid)
    expected = np.sum(
        np.where(valid[..., None], values * weights[..., None], 0.0),
        axis=0,
        dtype=np.float32,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, evaluate_tiled_fold_finalize(program, values, weights, partial_valid=valid))


def test_same_tiled_skeleton_schedule_accepts_two_generic_fold_programs() -> None:
    axes = _axes(partials=2, rows=2, features=4)
    schedule = _schedule(axes=axes, partial_lanes=1, addressing=FoldPartialAddressing.DENSE)
    normalized = normalized_exponential_fold_program(
        schedule,
        partial_value_dtype=DType.BF16,
        output_dtype=DType.BF16,
    )
    weighted = deterministic_weighted_sum_fold_program(
        schedule,
        partial_value_dtype=DType.BF16,
        output_dtype=DType.BF16,
    )
    values = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
            [[5.0, 4.0, 3.0, 2.0], [6.0, 5.0, 4.0, 3.0]],
        ],
        dtype=np.float32,
    )
    valid = np.ones((2, 2), dtype=np.bool_)

    normalized_output = evaluate_tiled_fold_finalize(
        normalized,
        values,
        np.array([[0.0, 0.5], [1.0, -0.5]], dtype=np.float32),
        partial_valid=valid,
    )
    weighted_output = evaluate_tiled_fold_finalize(
        weighted,
        values,
        np.array([[0.25, 0.75], [0.75, 0.25]], dtype=np.float32),
        partial_valid=valid,
    )

    assert normalized.schedule is weighted.schedule
    assert normalized_output.shape == weighted_output.shape == (2, 4)
    assert not np.array_equal(normalized_output, weighted_output)


def test_128_feature_ping_pong_schedule_preserves_fold_semantics() -> None:
    rng = np.random.default_rng(17)
    axes = _axes(partials=8, rows=3, features=128)
    schedule = _schedule(
        axes=axes,
        partial_lanes=32,
        addressing=FoldPartialAddressing.DENSE,
        feature_tile=128,
        shared_buffers=2,
    )
    program = normalized_exponential_fold_program(
        schedule,
        partial_value_dtype=DType.FP32,
        output_dtype=DType.FP32,
    )
    values = rng.normal(size=(8, 3, 128)).astype(np.float32)
    log_normalizers = rng.normal(size=(8, 3)).astype(np.float32)
    valid = np.ones((8, 3), dtype=np.bool_)
    valid[5:, 0] = False

    actual = evaluate_tiled_fold_finalize(program, values, log_normalizers, partial_valid=valid)
    masked = np.where(valid, log_normalizers, -np.inf)
    common = np.max(masked, axis=0)
    weights = np.where(valid, np.exp(masked - common), 0.0).astype(np.float32)
    expected = (
        np.sum(weights[..., None] * values, axis=0, dtype=np.float32)
        / np.sum(
            weights,
            axis=0,
            dtype=np.float32,
        )[:, None]
    )

    assert schedule.feature_tile == axes.feature.extent
    assert schedule.shared_buffers == 2
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    ("feature_tile", "shared_buffers"),
    ((64, 1), (64, 2), (128, 2)),
)
def test_fold_candidate_schedules_share_normalized_exponential_semantics(
    feature_tile: int,
    shared_buffers: int,
) -> None:
    rng = np.random.default_rng(23)
    axes = _axes(partials=8, rows=3, features=128)
    program = normalized_exponential_fold_program(
        _schedule(
            axes=axes,
            partial_lanes=32,
            addressing=FoldPartialAddressing.DENSE,
            feature_tile=feature_tile,
            shared_buffers=shared_buffers,
        ),
        partial_value_dtype=DType.FP32,
        output_dtype=DType.FP32,
    )
    values = rng.normal(size=(8, 3, 128)).astype(np.float32)
    scalar = rng.normal(size=(8, 3)).astype(np.float32)
    valid = np.ones((8, 3), dtype=np.bool_)
    valid[6:, 1] = False

    actual = evaluate_tiled_fold_finalize(program, values, scalar, partial_valid=valid)
    masked = np.where(valid, scalar, -np.inf)
    common = np.max(masked, axis=0)
    weights = np.where(valid, np.exp(masked - common), 0.0).astype(np.float32)
    expected = (
        np.sum(weights[..., None] * values, axis=0, dtype=np.float32)
        / np.sum(
            weights,
            axis=0,
            dtype=np.float32,
        )[:, None]
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_fold_candidate_set_changes_only_physical_schedule() -> None:
    axes = _axes(partials=16, rows=8, features=128)
    programs = tuple(
        normalized_exponential_fold_program(
            _schedule(
                axes=axes,
                partial_lanes=32,
                addressing=FoldPartialAddressing.DENSE,
                feature_tile=feature_tile,
                shared_buffers=shared_buffers,
            ),
            partial_value_dtype=DType.BF16,
            output_dtype=DType.BF16,
        )
        for feature_tile, shared_buffers in ((64, 1), (64, 2), (128, 2))
    )

    assert programs[0].semantics == programs[1].semantics == programs[2].semantics
    assert len({program.schedule for program in programs}) == 3


@pytest.mark.parametrize(
    ("shared_buffers", "message"),
    ((0, "schedule parameters must be positive"), (3, "shared-buffer count must be a power of two")),
)
def test_schedule_rejects_invalid_shared_buffer_count(shared_buffers: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _schedule(
            axes=_axes(partials=8, rows=3, features=128),
            partial_lanes=32,
            addressing=FoldPartialAddressing.DENSE,
            feature_tile=128,
            shared_buffers=shared_buffers,
        )


def test_weight_ast_mutation_changes_semantics_without_changing_the_skeleton() -> None:
    axes = _axes(partials=2, rows=2, features=2)
    schedule = _schedule(axes=axes, partial_lanes=1, addressing=FoldPartialAddressing.DENSE)
    program = deterministic_weighted_sum_fold_program(
        schedule,
        partial_value_dtype=DType.FP32,
        output_dtype=DType.FP32,
    )
    squared_weight = scalar_binary(
        ScalarExpressionKind.MULTIPLY,
        program.semantics.weight_expression,
        scalar_input("partial_scalar"),
    )
    mutated = TiledFoldFinalizeProgram(
        replace(program.semantics, weight_expression=squared_weight),
        schedule,
    )
    values = np.ones((2, 2, 2), dtype=np.float32)
    weights = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=np.float32)
    valid = np.ones((2, 2), dtype=np.bool_)

    original = evaluate_tiled_fold_finalize(program, values, weights, partial_valid=valid)
    changed = evaluate_tiled_fold_finalize(mutated, values, weights, partial_valid=valid)

    assert mutated.schedule is program.schedule
    assert not np.array_equal(original, changed)


def test_source_ordered_fold_rejects_parallel_partial_reassociation() -> None:
    axes = _axes()
    with pytest.raises(ValueError, match="requires one logical partial lane"):
        deterministic_weighted_sum_fold_program(
            _schedule(axes=axes, partial_lanes=32, addressing=FoldPartialAddressing.DENSE),
            partial_value_dtype=DType.FP32,
            output_dtype=DType.FP32,
        )


def test_normalized_exponential_fold_rejects_empty_rows() -> None:
    axes = _axes(partials=2, rows=2, features=4)
    program = normalized_exponential_fold_program(
        _schedule(axes=axes, partial_lanes=32, addressing=FoldPartialAddressing.DENSE),
        partial_value_dtype=DType.FP32,
        output_dtype=DType.FP32,
    )
    values = np.ones((2, 2, 4), dtype=np.float32)
    lse = np.zeros((2, 2), dtype=np.float32)
    valid = np.array([[True, False], [True, False]], dtype=np.bool_)

    with pytest.raises(ValueError, match=r"no valid finite partials: \[1\]"):
        evaluate_tiled_fold_finalize(program, values, lse, partial_valid=valid)
