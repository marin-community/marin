# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    family_churn_hazard_rpl_20260730 as candidate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    retained_power_law_model_20260728 as rpl,
)


def geometry(families: np.ndarray | None = None) -> rpl.Geometry:
    return rpl.Geometry(
        c0=np.asarray([2.0, 3.0, 5.0, 7.0]),
        c1=np.asarray([0.5, 0.75, 1.25, 1.75]),
        phase_0_fraction=0.8,
        family_index=families,
    )


def policies() -> np.ndarray:
    return np.asarray(
        [
            [[0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]],
            [[0.5, 0.2, 0.2, 0.1], [0.1, 0.4, 0.3, 0.2]],
            [[0.4, 0.2, 0.3, 0.1], [0.2, 0.1, 0.525, 0.175]],
        ],
        dtype=float,
    )


def test_zero_hazard_is_exactly_rpl() -> None:
    pooled = geometry(np.asarray([0, 0, 1, 1]))
    for shape in rpl.shape_grid():
        np.testing.assert_array_equal(
            candidate.design_matrix(policies(), pooled, shape, 0.0),
            rpl.design_matrix(policies(), pooled, shape),
        )


def test_singleton_families_are_exactly_rpl() -> None:
    singleton = geometry()
    for hazard in (0.0, 0.25, 0.5, 1.0, 2.0):
        for shape in rpl.shape_grid():
            np.testing.assert_array_equal(
                candidate.design_matrix(policies(), singleton, shape, hazard),
                rpl.design_matrix(policies(), singleton, shape),
            )


def test_tied_policy_has_zero_churn_and_exact_design() -> None:
    pooled = geometry(np.asarray([0, 0, 1, 1]))
    tied = policies()[[0]]
    np.testing.assert_array_equal(
        candidate.family_churn(tied, pooled),
        np.zeros((1, 2)),
    )
    for hazard in (0.25, 0.5, 1.0, 2.0):
        for shape in rpl.shape_grid():
            np.testing.assert_array_equal(
                candidate.design_matrix(tied, pooled, shape, hazard),
                rpl.design_matrix(tied, pooled, shape),
            )


def test_churn_is_nonnegative_and_phase_swap_invariant() -> None:
    pooled = geometry(np.asarray([0, 0, 1, 1]))
    forward = candidate.family_churn(policies(), pooled)
    reverse = candidate.family_churn(policies()[:, ::-1, :], pooled)
    assert np.all(forward >= 0.0)
    np.testing.assert_allclose(forward, reverse, rtol=0.0, atol=1e-15)


def test_coherent_family_mass_shift_has_zero_churn() -> None:
    pooled = geometry(np.asarray([0, 0, 1, 1]))
    # Each family's conditional composition is fixed; only total family mass moves.
    coherent = policies()[[2]]
    np.testing.assert_allclose(
        candidate.family_churn(coherent, pooled),
        np.zeros((1, 2)),
        rtol=0.0,
        atol=1e-15,
    )


def test_within_family_reallocation_has_positive_churn() -> None:
    pooled = geometry(np.asarray([0, 0, 1, 1]))
    churn = candidate.family_churn(policies()[[1]], pooled)
    assert np.all(churn > 0.0)


def test_hazard_reduces_retained_early_state() -> None:
    pooled = geometry(np.asarray([0, 0, 1, 1]))
    shape = rpl.Shape(1.0, 0.1, 1.5, 0.0, 5.0, 2.0, True)
    baseline = candidate.retained_share(
        policies(),
        pooled,
        shape.retention,
        shape.late_multiplier,
        0.0,
    )
    churned = candidate.retained_share(
        policies(),
        pooled,
        shape.retention,
        shape.late_multiplier,
        1.0,
    )
    assert np.all(churned <= baseline + 1e-15)
    assert np.any(churned < baseline - 1e-12)


def test_negative_hazard_is_rejected() -> None:
    pooled = geometry(np.asarray([0, 0, 1, 1]))
    with pytest.raises(ValueError, match="nonnegative"):
        candidate.retained_share(policies(), pooled, 5.0, 2.0, -0.1)
