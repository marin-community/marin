# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_olmix_swarm_aggregate_v_20260901 as aggregate_v,
)


def test_aggregate_v_design_splits_linear_under_and_over_exposure() -> None:
    exposure = np.asarray([[0.0, np.expm1(3.0)], [np.expm1(3.0), 0.0]])
    center = np.asarray([1.0, 1.0])

    design = aggregate_v.aggregate_v_design(exposure, center)

    np.testing.assert_allclose(design, [[1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 2.0, 0.0]])


def test_aggregate_v_recovers_synthetic_linear_hinges() -> None:
    log_exposure = np.linspace(0.05, 1.95, 39)
    exposure = np.expm1(log_exposure)[:, None]
    target = 0.9 + 0.7 * np.maximum(1.0 - log_exposure, 0.0)
    target += 0.2 * np.maximum(log_exposure - 1.0, 0.0)
    shape = aggregate_v.AggregateVShape(center_shift=0.0, ridge=0.0)

    head = aggregate_v.fit_aggregate_v(exposure, target, shape)
    prediction = aggregate_v.predict_aggregate_v(head, exposure)

    assert np.all(head.coefficients >= 0.0)
    assert abs(log_exposure[int(np.argmin(prediction))] - 1.0) <= 0.05
    np.testing.assert_allclose(prediction, target, atol=1e-10)


def test_empirical_centers_use_positive_exposure_and_shared_shift() -> None:
    exposure = np.asarray([[0.0, 1.0], [np.expm1(2.0), 3.0], [np.expm1(4.0), 7.0]])

    center = aggregate_v.empirical_centers(exposure, center_shift=0.5)

    np.testing.assert_allclose(center, [3.5, np.log1p(3.0) + 0.5])
