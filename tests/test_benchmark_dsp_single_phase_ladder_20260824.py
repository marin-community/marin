# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_dsp_single_phase_ladder_20260824 as dsp,
)


def reference_profiled_cv_objective(
    exposure: np.ndarray,
    response: np.ndarray,
    vector: np.ndarray,
    rung: dsp.Rung,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    pairs: tuple[tuple[int, int], ...],
) -> float:
    design = dsp.rung_design(exposure, vector, rung, exposure.shape[1])
    total = 0.0
    for train, validation in folds:
        intercept, coefficients = dsp.solve_head(design[train], response[train], pairs)
        residual = intercept + design[validation] @ coefficients - response[validation]
        total += float(residual @ residual)
    return total


@pytest.mark.parametrize("rung_name", [rung.name for rung in dsp.LADDER])
def test_profiled_cv_gradient_matches_central_difference(rung_name: str) -> None:
    generator = np.random.default_rng(7)
    bucket_count = 3
    exposure = np.exp(generator.uniform(np.log(0.05), np.log(200.0), size=(120, bucket_count)))
    rung = next(rung for rung in dsp.LADDER if rung.name == rung_name)
    if rung.per_domain:
        harm = np.asarray([0.4, 0.8, 1.2]) if rung.penalty == "canonical" else np.log(np.asarray([0.7, 1.2, 1.8]))
        vector = np.concatenate([np.log(np.asarray([0.2, 0.35, 0.5])), harm])
    else:
        harm = 0.8 if rung.penalty == "canonical" else float(np.log(1.2))
        vector = np.asarray([np.log(0.3), harm])

    design = dsp.rung_design(exposure, vector, rung, bucket_count)
    coefficients = np.asarray([0.30, 0.22, 0.18, 0.08, 0.06, 0.05])
    response = 1.4 + design @ coefficients + generator.normal(scale=1e-3, size=len(exposure))
    rows = np.arange(len(response))
    folds = tuple((rows[rows % 3 != fold], rows[rows % 3 == fold]) for fold in range(3))
    pairs = ((0, 1),) if rung.tie_pairs else ()

    value, gradient = dsp.profiled_cv_objective_and_gradient(
        exposure,
        response,
        vector,
        rung,
        folds,
        pairs,
    )
    reference_value = reference_profiled_cv_objective(exposure, response, vector, rung, folds, pairs)
    step = 1e-5
    finite_difference = np.empty_like(vector)
    for index in range(len(vector)):
        offset = np.zeros_like(vector)
        offset[index] = step
        upper = reference_profiled_cv_objective(exposure, response, vector + offset, rung, folds, pairs)
        lower = reference_profiled_cv_objective(exposure, response, vector - offset, rung, folds, pairs)
        finite_difference[index] = (upper - lower) / (2.0 * step)

    assert value == pytest.approx(reference_value, rel=0.0, abs=1e-14)
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-5, atol=1e-8)


def test_parallel_restarts_match_serial_fit() -> None:
    generator = np.random.default_rng(19)
    bucket_count = 3
    exposure = np.exp(generator.uniform(np.log(0.05), np.log(80.0), size=(90, bucket_count)))
    rung = next(rung for rung in dsp.LADDER if rung.name == "canonical")
    vector = np.concatenate(
        [
            np.log(np.asarray([0.15, 0.31, 0.61])),
            np.asarray([0.2, 0.8, 1.4]),
        ]
    )
    design = dsp.rung_design(exposure, vector, rung, bucket_count)
    coefficients = np.asarray([0.31, 0.23, 0.15, 0.07, 0.05, 0.03])
    response = 1.2 + design @ coefficients + generator.normal(scale=1e-4, size=len(exposure))
    rows = np.arange(len(response))
    folds = tuple((rows[rows % 3 != fold], rows[rows % 3 == fold]) for fold in range(3))

    serial = dsp.fit_rung(exposure, response, rung, folds, (), seed=4, maxiter=8, restarts=3, workers=1)
    parallel = dsp.fit_rung(exposure, response, rung, folds, (), seed=4, maxiter=8, restarts=3, workers=3)

    for serial_value, parallel_value in zip(serial, parallel, strict=True):
        np.testing.assert_array_equal(serial_value, parallel_value)
