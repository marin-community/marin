# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    family_pooled_retained_power_law_20260730 as candidate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import retained_power_law_model_20260728 as rpl


def policies() -> np.ndarray:
    return np.asarray(
        [
            [[0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]],
            [[0.5, 0.2, 0.2, 0.1], [0.1, 0.4, 0.3, 0.2]],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        ],
        dtype=float,
    )


def pooled_geometry() -> rpl.Geometry:
    return rpl.Geometry(
        c0=np.asarray([2.0, 3.0, 5.0, 7.0]),
        c1=np.asarray([0.5, 0.75, 1.25, 1.75]),
        phase_0_fraction=0.8,
        family_index=np.asarray([0, 0, 1, 1]),
    )


def singleton_geometry() -> rpl.Geometry:
    return rpl.Geometry(
        c0=np.asarray([2.0, 3.0, 5.0, 7.0]),
        c1=np.asarray([0.5, 0.75, 1.25, 1.75]),
        phase_0_fraction=0.8,
    )


def test_singleton_families_equal_rpl_for_every_shape() -> None:
    geometry = singleton_geometry()
    for shape in rpl.shape_grid():
        np.testing.assert_array_equal(
            candidate.design_matrix(policies(), geometry, shape),
            rpl.design_matrix(policies(), geometry, shape),
        )


def test_pooled_candidate_preserves_columns_and_penalties() -> None:
    geometry = pooled_geometry()
    for shape in rpl.shape_grid():
        base = rpl.design_matrix(policies(), geometry, shape)
        pooled = candidate.design_matrix(policies(), geometry, shape)
        assert pooled.shape == base.shape
        np.testing.assert_array_equal(
            candidate.penalty_multipliers(geometry, shape),
            rpl.penalty_multipliers(geometry, shape),
        )


def test_mean_and_sum_family_forms_match_after_column_normalization() -> None:
    geometry = pooled_geometry()
    shape = rpl.Shape(0.5, 0.1, 2.0, 0.0, 2.5, 2.0, True)
    retained = rpl.retained_share(policies(), geometry, shape.retention, shape.late_multiplier)
    sum_form = candidate.pooled_family_benefit(retained, geometry, shape)
    mean_form = candidate.mean_family_benefit(retained, geometry, shape)
    sum_normalized = sum_form / np.abs(sum_form).max(axis=0)
    mean_normalized = mean_form / np.abs(mean_form).max(axis=0)
    np.testing.assert_allclose(sum_normalized, mean_normalized, rtol=1e-12, atol=1e-12)


def test_ordering_columns_are_zero_at_tied_policies() -> None:
    geometry = pooled_geometry()
    tied = policies()[[0]]
    for shape in rpl.shape_grid():
        if not shape.ordering_channel:
            continue
        np.testing.assert_allclose(rpl.marginal_phase_block(tied, geometry, shape), 0.0, atol=1e-12)


def test_corner_policies_are_finite() -> None:
    geometry = pooled_geometry()
    corner = policies()[[2]]
    for shape in rpl.shape_grid():
        assert np.all(np.isfinite(candidate.design_matrix(corner, geometry, shape)))
