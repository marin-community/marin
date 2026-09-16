# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_olmix_swarm_single_phase_dsp_20260901 as incumbent,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_olmix_swarm_taskwise_olmix_vs_dsp_20260901 as benchmark,
)


def synthetic_pool() -> incumbent.Pool:
    weights = np.asarray(
        [
            [0.80, 0.15, 0.05],
            [0.65, 0.25, 0.10],
            [0.45, 0.40, 0.15],
            [0.25, 0.55, 0.20],
            [0.15, 0.30, 0.55],
            [0.10, 0.15, 0.75],
        ]
    )
    exposures = weights / np.asarray([0.5, 1.5, 3.0])[None, :]
    outcomes = np.column_stack(
        [
            0.7 + np.exp(weights @ np.asarray([-0.8, -0.2, 0.1])),
            0.5 + np.exp(np.log1p(exposures) @ np.asarray([-0.2, -0.7, 0.05])),
        ]
    )
    return incumbent.Pool(
        name="synthetic",
        runs=tuple(f"r{index}" for index in range(len(weights))),
        buckets=("b0", "b1", "b2"),
        tasks=("t0", "t1"),
        weights=weights,
        exposures=exposures,
        outcomes=outcomes,
        input_hashes={},
    )


def test_taskwise_olmix_preserves_atomic_predictions_before_macro_average(tmp_path) -> None:
    pool = synthetic_pool()
    split = benchmark.Fold(
        pool=pool.name,
        repeat=0,
        fold=0,
        train=np.asarray([0, 1, 2, 3]),
        test=np.asarray([4, 5]),
    )
    task_predictions = []
    for task in range(len(pool.tasks)):
        path = benchmark.shard_path(
            tmp_path,
            pool=pool.name,
            model="olmix_taskwise_raw",
            repeat=0,
            fold=0,
            task=task,
        )
        benchmark.fit_olmix_task_shard(pool, split, "olmix_taskwise_raw", task, 6, path)
        with np.load(path) as payload:
            task_predictions.append(payload["prediction"])

    compiled = benchmark.load_taskwise_fold(pool, split, "olmix_taskwise_raw", tmp_path)

    assert compiled is not None
    np.testing.assert_allclose(compiled, np.column_stack(task_predictions).mean(axis=1))
    assert not np.allclose(task_predictions[0], task_predictions[1])


def test_log_epoch_transform_uses_materialized_exposure() -> None:
    pool = synthetic_pool()

    np.testing.assert_allclose(
        benchmark.model_inputs(pool, "olmix_taskwise_log_epoch"),
        np.log1p(pool.exposures),
    )
    assert not np.allclose(
        benchmark.model_inputs(pool, "olmix_taskwise_log_epoch"),
        benchmark.model_inputs(pool, "olmix_taskwise_raw"),
    )


def test_coarse_canonical_fit_is_deterministic() -> None:
    pool = synthetic_pool()
    rows = np.arange(len(pool.runs))
    folds = (
        (np.asarray([2, 3, 4, 5]), np.asarray([0, 1])),
        (np.asarray([0, 1, 4, 5]), np.asarray([2, 3])),
        (np.asarray([0, 1, 2, 3]), np.asarray([4, 5])),
    )
    response = pool.outcomes.mean(axis=1)

    first = benchmark.coarse_canonical_fit(pool.exposures[rows], response[rows], folds, seed=7, starts=8)
    second = benchmark.coarse_canonical_fit(pool.exposures[rows], response[rows], folds, seed=7, starts=8)

    np.testing.assert_allclose(first[0], second[0])
    assert first[1] == second[1]
    assert first[2] == second[2]


def test_coarse_canonical_prediction_is_finite() -> None:
    pool = synthetic_pool()
    split = benchmark.Fold(
        pool=pool.name,
        repeat=0,
        fold=0,
        train=np.asarray([0, 1, 2, 3, 4]),
        test=np.asarray([5]),
    )

    prediction, start_id, objective = benchmark.canonical_prediction(
        pool,
        split,
        model="dsp_canonical_macro_coarse",
        maxiter=0,
        restarts=0,
        coarse_starts=4,
    )

    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()
    assert 0 <= start_id < 4
    assert np.isfinite(objective)
