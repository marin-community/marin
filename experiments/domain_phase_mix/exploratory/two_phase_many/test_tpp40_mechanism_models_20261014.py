# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math

import numpy as np
import pytest
from scipy.integrate import quad
from threadpoolctl import threadpool_limits

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_mechanism_models_20261014 as mechanism
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_temporal_models_20261014 as temporal

SHAPE = {"rate": 0.5, "power": 0.7, "threshold": 2.0}


@pytest.mark.parametrize("model", mechanism.MODEL_NAMES)
def test_mechanism_exact_tied_and_null_predictions_equal_native(model: str) -> None:
    rng = np.random.default_rng(20261015)
    total = np.exp(rng.uniform(-8, 8, (23, 4)))
    alpha = 21856 / 27336
    head = temporal.mariner.Head(-0.2, np.arange(1, 9) * 0.01, 0.6)
    for parameters in mechanism.parameter_grid(model):
        early, late = alpha * total, (1 - alpha) * total
        native = temporal.mariner.design_matrix(early + late, SHAPE)
        matrix, offset = mechanism.design_and_offset(early, late, alpha, SHAPE, model, parameters)
        np.testing.assert_array_equal(matrix, native)
        np.testing.assert_array_equal(temporal.predict_head(head, matrix, offset), head.predict(native))
    early = total * rng.uniform(0, 1, total.shape)
    late = total - early
    matrix, offset = mechanism.design_and_offset(early, late, alpha, SHAPE, model, mechanism.parameter_grid(model)[0])
    native = temporal.mariner.design_matrix(early + late, SHAPE)
    np.testing.assert_array_equal(matrix, native)
    np.testing.assert_array_equal(temporal.predict_head(head, matrix, offset), head.predict(native))


def test_bounded_readouts_match_scalar_probability_and_hazard_references() -> None:
    # power=1 makes the native benefit and both departures hand-computable.
    shape = {"rate": 1.0, "power": 1.0, "threshold": 2.0}
    early, late = np.array([[0.2]]), np.array([[1.8]])
    alpha = 0.8
    native_benefit = 1 - math.exp(-2)
    delta = math.exp(-0.2) - math.exp(-1.6)
    expected_odds = math.exp(math.log(native_benefit / (1 - native_benefit)) + 2 * delta)
    expected_benefit = expected_odds / (1 + expected_odds)
    odds, _ = mechanism.design_and_offset(early, late, alpha, shape, "bounded_benefit_odds", {"kappa": 2.0})
    np.testing.assert_allclose(-odds[0, 0], expected_benefit, rtol=3e-15, atol=3e-15)
    hazard, _ = mechanism.design_and_offset(early, late, alpha, shape, "hazard_recency", {"rho": 0.5})
    np.testing.assert_allclose(-hazard[0, 0], 1 - math.exp(-2.7), rtol=3e-15, atol=3e-15)


@pytest.mark.parametrize("model", mechanism.MODEL_NAMES)
def test_mechanism_benefit_bounds_and_harm_origin_at_extreme_exposures(model: str) -> None:
    total = np.array([[0.0, 1e-20, 1e-8, 1.0, 10.0, 1e6], [0.0, 1e-20, 1e-8, 1.0, 10.0, 1e6]])
    early = total * np.array([[0.0], [1.0]])
    late = total - early
    harm_origin = float(temporal.mariner.harm(np.array(0.0), SHAPE["threshold"]))
    for parameters in mechanism.parameter_grid(model):
        matrix, _ = mechanism.design_and_offset(early, late, 0.8, SHAPE, model, parameters)
        assert np.isfinite(matrix).all()
        assert np.all((-matrix[:, :6] >= 0) & (-matrix[:, :6] <= 1))
        assert np.all(matrix[:, 6:] >= harm_origin)
        np.testing.assert_array_equal(matrix[:, 0], np.zeros(2))
        np.testing.assert_array_equal(matrix[:, 6], np.full(2, harm_origin))


def test_rate_damage_matches_continuous_cumulative_harm_integral() -> None:
    alpha, e0, e1, nu = 0.8, 6.0, 4.0, 0.5
    total = e0 + e1
    threshold = SHAPE["threshold"]

    def damage_rate(time: float, phase_rate: float, start: float, start_time: float) -> float:
        exposure = start + phase_rate * (time - start_time)
        z = math.log1p(exposure) - threshold
        softplus = math.log1p(math.exp(z))
        slope = 2 * softplus / ((1 + math.exp(-z)) * (1 + exposure))
        return (phase_rate / total) ** nu * slope * phase_rate

    origin = math.log1p(math.exp(-threshold)) ** 2
    first = quad(damage_rate, 0, alpha, args=(e0 / alpha, 0, 0), epsabs=1e-12, epsrel=1e-12)[0]
    second = quad(damage_rate, alpha, 1, args=(e1 / (1 - alpha), e0, alpha), epsabs=1e-12, epsrel=1e-12)[0]
    matrix, _ = mechanism.design_and_offset(np.array([[e0]]), np.array([[e1]]), alpha, SHAPE, "rate_damage", {"nu": nu})
    np.testing.assert_allclose(matrix[0, 1], origin + first + second, rtol=3e-14, atol=3e-14)
    # A phase reset substitutes h(e1) for the late cumulative increment.
    reset = origin + (e0 / (alpha * total)) ** nu * (temporal.mariner.harm(np.array(e0), threshold) - origin)
    reset += (e1 / ((1 - alpha) * total)) ** nu * (temporal.mariner.harm(np.array(e1), threshold) - origin)
    assert abs(matrix[0, 1] - reset) > 0.1


def test_rate_damage_single_phase_limits_retain_full_cumulative_harm() -> None:
    alpha, nu, exposure = 0.8, 2.0, 7.0
    matrix, _ = mechanism.design_and_offset(
        np.array([[exposure], [0.0]]),
        np.array([[0.0], [exposure]]),
        alpha,
        SHAPE,
        "rate_damage",
        {"nu": nu},
    )
    h0 = float(temporal.mariner.harm(np.array(0.0), SHAPE["threshold"]))
    increment = float(temporal.mariner.harm(np.array(exposure), SHAPE["threshold"])) - h0
    expected = h0 + np.array([alpha**-nu, (1 - alpha) ** -nu]) * increment
    np.testing.assert_allclose(matrix[:, 1], expected, rtol=3e-15, atol=3e-15)


def test_contrast_magnitude_ignores_empty_buckets_and_is_symmetric_about_tied_fraction() -> None:
    early = np.array([[0.0, 0.6, 1.2], [0.0, 1.0, 1.2]])
    late = np.array([[0.0, 0.4, 0.3], [0.0, 0.0, 0.3]])
    matrix, offset = mechanism.design_and_offset(early, late, 0.8, SHAPE, "contrast_magnitude", {"theta": -0.3})
    # One of three buckets departs by +/-0.2, one is tied, and one is empty.
    np.testing.assert_allclose(offset, [-0.025, -0.025], rtol=3e-15, atol=3e-15)
    np.testing.assert_array_equal(matrix, temporal.mariner.design_matrix(early + late, SHAPE))


def test_conditional_fit_null_scores_and_serialized_predictions_match_native() -> None:
    rng = np.random.default_rng(1907)
    alpha = 0.8
    early = alpha * np.array([2.0, 12.0]) * rng.dirichlet(np.ones(2), 30)
    late = (1 - alpha) * np.array([2.0, 12.0]) * rng.dirichlet(np.ones(2), 30)
    native = temporal.mariner.design_matrix(early + late, SHAPE)
    observed = 0.6 + np.exp(-0.1 + native @ np.array([0.5, 0.4, 0.02, 0.01]))
    observed += rng.normal(0, 0.001, len(observed))
    labels = np.arange(len(observed)) % 3
    ridge, floor_kappa = 0.001, 2.5
    spec = temporal.mariner.FloorSpec(float("nan"), 0, floor_kappa)
    head = temporal.mariner.fit_head(native, observed, ridge, spec)
    errors = []
    for train, held in temporal.inner_splits(labels):
        inner_head = temporal.mariner.fit_head(native[train], observed[train], ridge, spec)
        errors.extend((inner_head.predict(native[held]) - observed[held]) ** 2)
    native_score = math.sqrt(float(np.mean(errors)))
    aggregate = temporal.TemporalFit(
        "aggregate_mariner",
        alpha,
        np.array([True, False]),
        SHAPE,
        ridge,
        floor_kappa,
        False,
        native_score,
        native_score,
        {},
        head,
        (),
    )
    for model in mechanism.MODEL_NAMES:
        fitted = mechanism.fit_extension(early, late, observed, labels, aggregate, model)
        np.testing.assert_allclose(
            fitted.selection_profile[0]["conditional_inner_rmse"], native_score, rtol=3e-14, atol=3e-14
        )
        replay = mechanism.MechanismFit.from_json(json.loads(json.dumps(fitted.to_json())))
        np.testing.assert_array_equal(replay.predict(early, late), fitted.predict(early, late))
        np.testing.assert_array_equal(
            fitted.predict(alpha * (early + late), (1 - alpha) * (early + late)),
            fitted.head.predict(
                temporal.mariner.design_matrix(alpha * (early + late) + (1 - alpha) * (early + late), fitted.shape)
            ),
        )


def test_ridge_gamma_refinement_reproduces_independent_native_floor_search() -> None:
    rng = np.random.default_rng(41)
    alpha = 0.8
    early = rng.uniform(0.0, 3.0, (30, 2))
    late = rng.uniform(0.0, 1.0, (30, 2))
    matrix = temporal.mariner.design_matrix(early + late, SHAPE)
    observed = 0.6 + np.exp(-0.7 + matrix @ np.array([0.7, 0.5, 0.1, 0.2]))
    observed += rng.normal(0, 0.0003, len(observed))
    labels = np.arange(len(observed)) % 3
    original = temporal.TemporalFit(
        "aggregate_mariner",
        alpha,
        np.array([True, False]),
        SHAPE,
        0.1,
        2.5,
        False,
        1.0,
        1.0,
        {},
        temporal.mariner.fit_head(matrix, observed, 0.1, temporal.mariner.FloorSpec(float("nan"), 0, 2.5)),
        (),
    )
    refined = mechanism.refine_ridge_gamma(early, late, observed, labels, original)
    independent = []
    with threadpool_limits(limits=1):
        for ridge in (0.1, 0.0, 0.001, 0.01, 1.0):
            kappa, flat, score = temporal.mariner.search_kappa(
                matrix, observed, ridge, temporal.mariner.Anchor(float("nan"), 0), temporal.inner_splits(labels)
            )
            independent.append((score, ridge, kappa, flat))
        selected = min(enumerate(independent), key=lambda item: (item[1][0], item[0]))[1]
        expected = temporal.mariner.fit_head(
            matrix, observed, selected[1], temporal.mariner.FloorSpec(float("nan"), 0, selected[2])
        )
    np.testing.assert_array_equal(refined.predict(early, late), expected.predict(matrix))
    for actual, reference in zip(refined.selection_profile, independent, strict=True):
        assert actual["conditional_inner_rmse"] == reference[0]
        assert actual["kappa"] == reference[2]
    assert refined.aggregate_inner_rmse == selected[0]
    assert refined.shape == original.shape
