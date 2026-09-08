# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_olmix_swarm_single_phase_dsp_20260901 as benchmark,
)


def synthetic_pool() -> benchmark.Pool:
    weights = np.asarray(
        [
            [0.70, 0.20, 0.10],
            [0.55, 0.30, 0.15],
            [0.30, 0.50, 0.20],
            [0.15, 0.25, 0.60],
            [0.20, 0.65, 0.15],
            [0.45, 0.10, 0.45],
        ]
    )
    inventory = np.asarray([1.0, 2.0, 4.0])
    exposures = weights / inventory[None, :]
    outcomes = np.column_stack(
        [
            1.2 - 0.15 * (1.0 - np.exp(-exposures @ np.asarray([1.0, 0.5, 0.25]))),
            0.9 - 0.08 * (1.0 - np.exp(-exposures @ np.asarray([0.4, 0.8, 0.2]))),
        ]
    )
    return benchmark.Pool(
        name="synthetic",
        runs=tuple(f"r{index}" for index in range(len(weights))),
        buckets=("b0", "b1", "b2"),
        tasks=("t0", "t1"),
        weights=weights,
        exposures=exposures,
        outcomes=outcomes,
        input_hashes={},
    )


def test_shared_task_head_uses_one_bucket_response_across_tasks() -> None:
    pool = synthetic_pool()
    rows = np.arange(len(pool.runs))
    shape = benchmark.Shape(rate=1.0, ridge=0.3, floor_margin=0.08)

    head = benchmark.fit_variant(pool, rows, "dsp_shared_task_log_link", shape)

    assert isinstance(head, benchmark.Head)
    np.testing.assert_allclose(head.coefficients[:, 0], head.coefficients[:, 1])
    predictions = benchmark.predict_variant(pool, rows, "dsp_shared_task_log_link", shape, head)
    assert predictions.shape == pool.outcomes.shape
    assert np.isfinite(predictions).all()


def test_exact_olmix_macro_variant_returns_scalar_predictions() -> None:
    pool = synthetic_pool()
    rows = np.arange(len(pool.runs))
    macro = pool.outcomes.mean(axis=1, keepdims=True)
    shape = benchmark.Shape(0.0, 0.0, 0.0)

    head = benchmark.fit_variant(pool, rows, "olmix_exact_macro", shape, outcomes=macro)
    predictions = benchmark.predict_variant(pool, rows, "olmix_exact_macro", shape, head)

    assert isinstance(head, benchmark.OlmixLoglinearFit)
    assert predictions.shape == (len(rows), 1)
    assert np.isfinite(predictions).all()


def test_inventory_permutation_breaks_exposure_identity_only() -> None:
    pool = synthetic_pool()
    permuted = benchmark.permuted_exposures(pool)

    assert permuted.shape == pool.exposures.shape
    assert not np.allclose(permuted, pool.exposures)
    observed_rates = permuted.sum(axis=0) / pool.weights.sum(axis=0)
    np.testing.assert_allclose(np.sort(observed_rates), np.sort([1.0, 0.5, 0.25]))


def test_linear_inventory_control_changes_only_exposure_identity() -> None:
    pool = synthetic_pool()
    rows = np.arange(len(pool.runs))
    shape = benchmark.Shape(rate=0.0, ridge=0.3, floor_margin=0.08)

    observed, observed_positive = benchmark.features(pool, rows, "linear_epoch_log_link", shape)
    permuted, permuted_positive = benchmark.features(pool, rows, "linear_epoch_log_link_permuted_inventory", shape)

    np.testing.assert_allclose(observed, np.log1p(pool.exposures))
    np.testing.assert_allclose(permuted, np.log1p(benchmark.permuted_exposures(pool)))
    assert observed.shape == permuted.shape
    assert observed_positive is permuted_positive is False


def test_ridge_grid_brackets_previous_boundary_choices() -> None:
    assert min(benchmark.RIDGE_GRID) < 0.03
    assert max(benchmark.RIDGE_GRID) > 3.0
