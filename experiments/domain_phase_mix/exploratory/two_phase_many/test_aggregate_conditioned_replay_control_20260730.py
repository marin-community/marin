# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    aggregate_conditioned_replay_control_20260730 as model,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_aggregate_conditioned_replay_control_20260730 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import retained_power_law_model_20260728 as retained


def geometry() -> model.Geometry:
    return model.Geometry(
        c0=np.asarray([0.8, 8.0]),
        c1=np.asarray([0.2, 2.0]),
        phase_0_fraction=0.8,
    )


def weights(phase0: list[float], phase1: list[float]) -> np.ndarray:
    return np.asarray([[phase0, phase1]], dtype=float)


def test_replay_pressure_is_one_at_proportional_and_larger_elsewhere() -> None:
    geom = geometry()
    proportional = model.proportional_mixture(geom)
    policies = np.concatenate(
        [
            weights(proportional.tolist(), proportional.tolist()),
            weights([0.5, 0.5], [0.5, 0.5]),
        ],
        axis=0,
    )
    pressure = model.normalized_replay_pressure(policies, geom)
    np.testing.assert_allclose(pressure[0], 1.0)
    assert pressure[1] > pressure[0]


def test_even_phase_costs_are_zero_when_tied_and_nonnegative_when_untied() -> None:
    geom = geometry()
    policies = np.concatenate(
        [
            weights([0.7, 0.3], [0.7, 0.3]),
            weights([0.8, 0.2], [0.3, 0.7]),
        ],
        axis=0,
    )
    information = model.phase_information_cost(policies, geom)
    replay = model.replay_jensen_cost(policies, geom)
    np.testing.assert_allclose(information[0], 0.0)
    np.testing.assert_allclose(replay[0], 0.0)
    assert information[1] > 0.0
    assert replay[1] >= 0.0


def test_aggregate_curvature_matches_finite_difference() -> None:
    geom = geometry()
    shape = model.AggregateShape(
        benefit_exponent=0.5,
        benefit_offset=0.1,
        damage_exponent=2.0,
        damage_threshold=0.0,
    )
    aggregate = np.asarray([0.5, 0.5])
    contrast = np.asarray([-0.2, 0.2])
    phase1_fraction = 1.0 - geom.phase_0_fraction
    untied = weights(
        (aggregate - phase1_fraction * contrast).tolist(),
        (aggregate + geom.phase_0_fraction * contrast).tolist(),
    )
    analytic = model.second_directional_design_matrix(untied, geom, shape)[0]

    epsilon = 1e-4
    plus = aggregate + epsilon * contrast
    minus = aggregate - epsilon * contrast
    center_design = model.aggregate_design_matrix(weights(aggregate.tolist(), aggregate.tolist()), geom, shape)[0]
    plus_design = model.aggregate_design_matrix(weights(plus.tolist(), plus.tolist()), geom, shape)[0]
    minus_design = model.aggregate_design_matrix(weights(minus.tolist(), minus.tolist()), geom, shape)[0]
    finite_difference = (plus_design - 2.0 * center_design + minus_design) / epsilon**2
    np.testing.assert_allclose(analytic, finite_difference, rtol=1e-5, atol=1e-5)
    assert np.all(analytic >= 0.0)


def test_aggregate_curvature_phase_design_is_zero_when_tied() -> None:
    geom = geometry()
    shape = model.AggregateShape(
        benefit_exponent=1.0,
        benefit_offset=0.1,
        damage_exponent=2.0,
        damage_threshold=0.0,
    )
    aggregate = model.AggregateFitted(
        shape=shape,
        ridge=0.0,
        intercept=0.0,
        coefficients=np.ones(4),
        geometry=geom,
    )
    phase = model.PhaseConfig(
        "curvature",
        replay_exponent=0.0,
        use_phase_information=False,
        use_replay_jensen=False,
        use_aggregate_curvature=True,
    )
    design = model.phase_design_matrix(weights([0.5, 0.5], [0.5, 0.5]), aggregate, phase)
    np.testing.assert_allclose(design, 0.0)


def test_control_energy_is_squared_directional_response_and_zero_when_tied() -> None:
    geom = geometry()
    shape = model.AggregateShape(
        benefit_exponent=1.0,
        benefit_offset=0.1,
        damage_exponent=2.0,
        damage_threshold=0.0,
    )
    aggregate = model.AggregateFitted(
        shape=shape,
        ridge=0.0,
        intercept=0.0,
        coefficients=np.asarray([1.0, 2.0, 0.5, 0.25]),
        geometry=geom,
    )
    phase = model.PhaseConfig(
        "control_energy",
        replay_exponent=0.0,
        use_phase_information=False,
        use_replay_jensen=False,
        use_control_energy=True,
    )
    policies = np.concatenate(
        [
            weights([0.5, 0.5], [0.5, 0.5]),
            weights([0.8, 0.2], [0.3, 0.7]),
        ],
        axis=0,
    )
    design = model.phase_design_matrix(policies, aggregate, phase)
    np.testing.assert_allclose(design[:, 1], design[:, 0] ** 2)
    np.testing.assert_allclose(design[0], 0.0)
    assert design[1, 1] > 0.0


def test_late_reactivation_state_reduces_to_aggregate_when_tied() -> None:
    geom = geometry()
    tied = weights([0.7, 0.3], [0.7, 0.3])
    np.testing.assert_allclose(model.late_reactivation_state(tied, geom), [[0.7, 0.3]])


def test_late_reactivation_bregman_prices_late_omission() -> None:
    geom = geometry()
    shape = model.AggregateShape(
        benefit_exponent=0.5,
        benefit_offset=0.1,
        damage_exponent=2.0,
        damage_threshold=0.0,
    )
    tied = weights([0.7, 0.3], [0.7, 0.3])
    aggregate = np.asarray([0.7, 0.3])
    contrast = np.asarray([0.375, -0.375])
    omitted_late = weights(
        (aggregate - geom.phase_1_fraction * contrast).tolist(),
        (aggregate + geom.phase_0_fraction * contrast).tolist(),
    )
    retained = model.late_reactivation_state(omitted_late, geom)
    np.testing.assert_allclose(retained[0, 1], 0.0)
    tied_cost = model.reactivation_bregman_design_matrix(tied, geom, shape)
    omitted_cost = model.reactivation_bregman_design_matrix(omitted_late, geom, shape)
    np.testing.assert_allclose(tied_cost, 0.0)
    assert np.any(omitted_cost > 0.0)


def test_stationary_tied_gradient_makes_fiber_prediction_nonimproving() -> None:
    geom = geometry()
    shape = model.AggregateShape(benefit_exponent=1.0, benefit_offset=0.1, damage_exponent=2.0, damage_threshold=0.0)
    # One domain per family gives two benefit and two damage coefficients. The
    # symmetric coefficients make the aggregate directional derivative equal
    # in both simplex coordinates at the tied midpoint.
    aggregate = model.AggregateFitted(
        shape=shape,
        ridge=0.0,
        intercept=0.0,
        coefficients=np.asarray([1.0, 1.0, 0.0, 0.0]),
        geometry=geom,
    )
    phase = model.PhaseConfig("test", replay_exponent=1.0)
    fitted = model.Fitted(
        aggregate=aggregate,
        phase=phase,
        phase_coefficients=np.asarray([1.0, 0.5, 0.5]),
    )
    tied = weights([0.5, 0.5], [0.5, 0.5])
    untied = weights([0.6, 0.4], [0.1, 0.9])
    assert fitted.predict(untied)[0] >= fitted.predict(tied)[0]


def test_predicted_phase_gain_always_includes_tied_policy() -> None:
    def predict(policies: np.ndarray) -> np.ndarray:
        return (policies[:, 0, 1] - policies[:, 1, 1]) ** 2

    gain = benchmark.predicted_phase_gain(
        predict,
        aggregate=0.4,
        resolution=10,
        phase_0_fraction=0.8,
    )
    assert gain == {"phase_gain": 0.0, "best_contrast": 0.0}


def test_retained_shape_scores_are_identical_across_processes() -> None:
    geom = retained.Geometry(
        c0=np.asarray([0.8, 8.0]),
        c1=np.asarray([0.2, 2.0]),
        phase_0_fraction=0.8,
    )
    policies = np.asarray(
        [
            [[0.2, 0.8], [0.2, 0.8]],
            [[0.4, 0.6], [0.4, 0.6]],
            [[0.6, 0.4], [0.1, 0.9]],
            [[0.8, 0.2], [0.3, 0.7]],
        ]
    )
    target = np.asarray([1.1, 1.0, 0.95, 1.05])
    folds = (
        (np.asarray([0, 2]), np.asarray([1, 3])),
        (np.asarray([1, 3]), np.asarray([0, 2])),
    )
    shapes = retained.shape_grid()[:2]
    serial = retained._shape_batch_scores(policies, target, geom, folds, shapes)
    with ProcessPoolExecutor(max_workers=2) as executor:
        parallel = executor.submit(
            retained._shape_batch_scores,
            policies,
            target,
            geom,
            folds,
            shapes,
        ).result()
    assert serial == parallel
