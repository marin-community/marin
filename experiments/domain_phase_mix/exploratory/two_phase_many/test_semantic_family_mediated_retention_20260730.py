# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    retained_power_law_model_20260728 as rpl,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    semantic_family_mediated_retention_20260730 as candidate,
)


def policies() -> np.ndarray:
    return np.asarray(
        [
            [[0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]],
            [[0.5, 0.2, 0.2, 0.1], [0.1, 0.4, 0.3, 0.2]],
            [[0.7, 0.1, 0.1, 0.1], [0.2, 0.2, 0.3, 0.3]],
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


def mixed_geometry() -> rpl.Geometry:
    return rpl.Geometry(
        c0=np.asarray([2.0, 3.0, 5.0, 7.0]),
        c1=np.asarray([0.5, 0.75, 1.25, 1.75]),
        phase_0_fraction=0.8,
        family_index=np.asarray([0, 0, 1, 2]),
    )


def test_q_zero_is_exactly_rpl() -> None:
    geometry = pooled_geometry()
    for shape in rpl.shape_grid():
        np.testing.assert_array_equal(
            candidate.design_matrix(policies(), geometry, shape, 0.0),
            rpl.design_matrix(policies(), geometry, shape),
        )


def test_singleton_families_are_exactly_rpl_for_every_q() -> None:
    geometry = singleton_geometry()
    for q in (0.0, 0.25, 0.5, 0.75, 1.0):
        for shape in rpl.shape_grid():
            np.testing.assert_array_equal(
                candidate.design_matrix(policies(), geometry, shape, q),
                rpl.design_matrix(policies(), geometry, shape),
            )


def test_tied_policies_are_unchanged_for_every_q() -> None:
    geometry = pooled_geometry()
    tied = policies()[[0]]
    for q in (0.0, 0.25, 0.5, 0.75, 1.0):
        for shape in rpl.shape_grid():
            np.testing.assert_array_equal(
                candidate.design_matrix(tied, geometry, shape, q),
                rpl.design_matrix(tied, geometry, shape),
            )


def test_singleton_members_are_unchanged_in_mixed_geometry() -> None:
    geometry = mixed_geometry()
    _, contrast = candidate.aggregate_and_contrast(policies(), geometry)
    mediated = candidate.mediated_contrast(policies(), geometry, 1.0)
    np.testing.assert_array_equal(mediated[:, 2:], contrast[:, 2:])


def test_family_total_contrast_is_conserved() -> None:
    geometry = pooled_geometry()
    _, original = candidate.aggregate_and_contrast(policies(), geometry)
    mediated = candidate.mediated_contrast(policies(), geometry, 0.8)
    for family in np.unique(geometry.families):
        members = geometry.families == family
        np.testing.assert_allclose(
            mediated[:, members].sum(axis=1),
            original[:, members].sum(axis=1),
            rtol=0.0,
            atol=1e-15,
        )


def test_bucket_state_depends_on_other_family_members() -> None:
    geometry = pooled_geometry()
    base = policies()[[1]].copy()
    shifted = base.copy()
    epsilon = 1e-6
    shifted[0, 1, 1] += epsilon
    shifted[0, 1, 2] -= epsilon

    base_state = candidate.mediated_contrast(base, geometry, 1.0)
    shifted_state = candidate.mediated_contrast(shifted, geometry, 1.0)
    assert shifted_state[0, 0] != pytest.approx(base_state[0, 0])


def test_zero_mass_family_is_finite() -> None:
    geometry = pooled_geometry()
    absent = np.asarray([[[0.0, 0.0, 0.6, 0.4], [0.0, 0.0, 0.3, 0.7]]])
    assert np.all(np.isfinite(candidate.mediated_contrast(absent, geometry, 1.0)))


def test_q_outside_unit_interval_is_rejected() -> None:
    geometry = pooled_geometry()
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        candidate.mediated_contrast(policies(), geometry, -0.1)
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        candidate.mediated_contrast(policies(), geometry, 1.1)
