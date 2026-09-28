# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    centered_hierarchical_rpl_20260730 as candidate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import retained_power_law_model_20260728 as rpl


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


def test_centering_operator_has_one_null_direction_per_family() -> None:
    geometry = pooled_geometry()
    operator = candidate.family_centering_operator(geometry)
    assert np.linalg.matrix_rank(operator, tol=1e-12) == 2
    for family in np.unique(geometry.families):
        constant = (geometry.families == family).astype(float)
        np.testing.assert_allclose(operator @ constant, 0.0, atol=1e-12)


def test_direct_and_hierarchical_blocks_have_same_response_span() -> None:
    geometry = pooled_geometry()
    shape = rpl.Shape(0.5, 0.1, 2.0, 0.0, 2.5, 2.0, True)
    benefit, damage = candidate.response_blocks(policies(), geometry, shape)
    for direct in (benefit, damage):
        hierarchical = rpl._hierarchical_block(direct, geometry)
        direct_projection = hierarchical @ np.linalg.lstsq(hierarchical, direct, rcond=None)[0]
        hierarchical_projection = direct @ np.linalg.lstsq(direct, hierarchical, rcond=None)[0]
        np.testing.assert_allclose(direct_projection, direct, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(hierarchical_projection, hierarchical, rtol=1e-10, atol=1e-10)


def test_nonnegative_response_cones_are_equivalent() -> None:
    geometry = pooled_geometry()
    shape = rpl.Shape(0.5, 0.1, 2.0, 0.0, 2.5, 2.0, True)
    direct, _damage = candidate.response_blocks(policies(), geometry, shape)
    hierarchical = rpl._hierarchical_block(direct, geometry)
    family = np.asarray([0.2, 0.3])
    excess = np.asarray([0.1, 0.4, 0.2, 0.5])
    hierarchical_coefficients = np.concatenate([family, excess])
    direct_coefficients = family[geometry.families] + excess
    np.testing.assert_allclose(
        hierarchical @ hierarchical_coefficients,
        direct @ direct_coefficients,
        rtol=1e-12,
        atol=1e-12,
    )
    direct_only = np.asarray([0.2, 0.4, 0.6, 0.8])
    hierarchy_with_zero_floor = np.concatenate([np.zeros(2), direct_only])
    np.testing.assert_allclose(
        direct @ direct_only,
        hierarchical @ hierarchy_with_zero_floor,
        rtol=1e-12,
        atol=1e-12,
    )


def test_singleton_families_reduce_numerically_to_rpl() -> None:
    geometry = singleton_geometry()
    target = np.asarray([1.01, 0.99, 1.03, 1.08])
    for shape in rpl.shape_grid():
        base = rpl.design_matrix(policies(), geometry, shape)
        centered = candidate.design_matrix(policies(), geometry, shape)
        np.testing.assert_array_equal(centered, base)
        np.testing.assert_array_equal(
            candidate.penalty_operator(geometry, shape),
            np.diag(rpl.penalty_multipliers(geometry, shape)),
        )
        base_intercept, base_coefficients = rpl.solve_head(
            base,
            target,
            0.01,
            rpl.penalty_multipliers(geometry, shape),
        )
        centered_intercept, centered_coefficients = candidate.solve_head(
            centered,
            target,
            0.01,
            candidate.penalty_operator(geometry, shape),
            geometry,
        )
        np.testing.assert_allclose(centered_intercept, base_intercept, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(centered_coefficients, base_coefficients, rtol=0.0, atol=1e-12)


def test_response_penalty_uses_physical_amplitudes_under_unequal_scales() -> None:
    geometry = pooled_geometry()
    shape = rpl.Shape(0.5, 0.1, 2.0, 0.0, 2.5, 2.0, False)
    operator = candidate.penalty_operator(geometry, shape)
    scale = np.asarray([0.1, 3.0, 20.0, 0.4, 7.0, 0.2, 50.0, 2.0, 11.0, 13.0])
    physical = np.asarray([0.5, 1.5, 0.2, 2.2, 3.0, 0.1, 1.7, 0.8, 0.4, 0.6])
    normalized = physical * scale

    transformed = candidate.penalty_in_normalized_coordinates(operator, scale, geometry)
    np.testing.assert_allclose(transformed @ normalized, operator @ physical, rtol=1e-12, atol=1e-12)

    centering = candidate.family_centering_operator(geometry)
    expected = np.concatenate(
        [
            centering @ physical[:4],
            centering @ physical[4:8],
            np.zeros(2),
        ]
    )
    np.testing.assert_allclose(transformed @ normalized, expected, rtol=1e-12, atol=1e-12)


def test_mixed_penalty_keeps_ordering_columns_in_normalized_coordinates() -> None:
    geometry = pooled_geometry()
    shape = rpl.Shape(0.5, 0.1, 2.0, 0.0, 2.5, 2.0, True)
    operator = candidate.penalty_operator(geometry, shape)
    scale = np.geomspace(0.1, 100.0, operator.shape[1])
    physical = np.linspace(0.2, 2.2, operator.shape[1])
    normalized = physical * scale

    transformed = candidate.penalty_in_normalized_coordinates(operator, scale, geometry)
    domains = len(geometry.c0)
    families = len(np.unique(geometry.families))
    phase_start = 2 * domains + 2
    phase_stop = phase_start + 4 * families
    expected = np.zeros(operator.shape[0])
    centering = candidate.family_centering_operator(geometry)
    expected[:domains] = centering @ physical[:domains]
    expected[domains : 2 * domains] = centering @ physical[domains : 2 * domains]
    expected[phase_start:phase_stop] = normalized[phase_start:phase_stop]
    np.testing.assert_allclose(transformed @ normalized, expected, rtol=1e-12, atol=1e-12)


def test_shape_ties_use_canonical_grid_order() -> None:
    shapes = rpl.shape_grid()[:3]
    scores = [
        (0.25, shapes[2], 1.0),
        (0.25, shapes[1], 0.01),
        (0.25, shapes[0], 0.0001),
    ]
    assert candidate._best_shape_and_ridge(scores, shapes) == (shapes[0], 0.0001)
    assert rpl._best_shape_and_ridge(scores, shapes) == (shapes[0], 0.0001)


def test_tied_policies_have_zero_explicit_phase_columns() -> None:
    geometry = pooled_geometry()
    tied = policies()[[0]]
    for shape in rpl.shape_grid():
        if shape.ordering_channel:
            np.testing.assert_allclose(rpl.marginal_phase_block(tied, geometry, shape), 0.0, atol=1e-12)
        np.testing.assert_allclose(rpl.concentration_gap(tied, geometry), 0.0, atol=1e-12)


def test_corner_policies_are_finite() -> None:
    geometry = pooled_geometry()
    corner = policies()[[-1]]
    for shape in rpl.shape_grid():
        assert np.all(np.isfinite(candidate.design_matrix(corner, geometry, shape)))
