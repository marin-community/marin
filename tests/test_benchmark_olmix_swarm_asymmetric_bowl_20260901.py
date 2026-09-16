# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_olmix_swarm_asymmetric_bowl_20260901 as bowl,
)


def test_asymmetric_bowl_design_splits_under_and_over_exposure() -> None:
    exposure = np.asarray([[0.0, np.e**2 - 1.0], [np.e**2 - 1.0, 0.0]])
    center = np.asarray([1.0, 1.0])

    design = bowl.asymmetric_bowl_design(exposure, center)

    np.testing.assert_allclose(design, [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])


def test_nonnegative_bowl_recovers_synthetic_minimum() -> None:
    log_exposure = np.linspace(0.05, 1.95, 39)
    exposure = np.expm1(log_exposure)[:, None]
    target = 0.9 + 0.7 * np.minimum(log_exposure - 1.0, 0.0) ** 2
    target += 0.2 * np.maximum(log_exposure - 1.0, 0.0) ** 2
    shape = bowl.BowlShape(shared_shift=0.0, ridge=0.0)

    head = bowl.fit_bowl(exposure, target, shape)
    prediction = bowl.predict_bowl(head, exposure)

    assert np.all(head.coefficients >= 0.0)
    assert abs(log_exposure[int(np.argmin(prediction))] - 1.0) <= 0.05
    np.testing.assert_allclose(prediction, target, atol=1e-10)


def test_empirical_centers_apply_one_shared_shift() -> None:
    exposure = np.asarray([[0.0, 1.0], [np.expm1(2.0), 3.0], [np.expm1(4.0), 7.0]])

    center = bowl.empirical_centers(exposure, shared_shift=0.5)

    np.testing.assert_allclose(center, [3.5, np.log1p(3.0) + 0.5])
