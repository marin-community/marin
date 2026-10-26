# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_joint_estimation_20261014 as frozen
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_tangent_models_20261015 as models

SHAPE = {"rate": 0.25, "power": 0.7, "threshold": 3.0}
ALPHA = 0.8
COEFFICIENTS = np.array([0.5, 0.8, 0.6, 0.03, 0.02, 0.01])


def geometry(seed: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    total = np.exp(rng.uniform(-2, 3, (rows, 3)))
    fraction = rng.uniform(0.15, 0.95, total.shape)
    return fraction * total, (1 - fraction) * total


def independent_truth(early: np.ndarray, late: np.ndarray, model: str, strength: float) -> np.ndarray:
    total = early + late
    benefit = 1 - np.exp(-((0.25 * total) ** 0.7))
    if model == "tangent_benefit_odds":
        at = ALPHA * total
        derivative = 0.7 * 0.25**0.7 * at ** (-0.3) * np.exp(-((0.25 * at) ** 0.7))
        departure = (at - early) * derivative
    else:
        departure = np.exp(-((0.25 * early) ** 0.7)) - np.exp(-((0.25 * ALPHA * total) ** 0.7))
    odds = benefit / (1 - benefit) * np.exp(strength * departure)
    benefit = odds / (1 + odds)
    harm = np.logaddexp(0, np.log1p(total) - 3) ** 2
    return 0.6 + np.exp(1 - benefit @ COEFFICIENTS[:3] + harm @ COEFFICIENTS[3:])


@pytest.mark.parametrize("power", (0.3, 0.7, 1.0))
def test_original_frozen_head_parity_and_exact_native_restrictions(power: float) -> None:
    shape = {**SHAPE, "power": power}
    early, late = geometry(927, 20)
    early = np.vstack([early, [0, 1e-250, 1e250]])
    late = np.vstack([late, [0, 1e-250, 1e250]])
    for strength in (0.0, 0.5, 4.0, 16.0, 128.0):
        old = frozen.JointFit(
            "bounded_benefit_odds",
            ALPHA,
            shape,
            {"kappa": strength},
            0.0,
            None,
            0.6,
            0.0,
            frozen.mariner.Head(1.0, COEFFICIENTS, 0.6),
            (),
            {},
        )
        replay = models.JointFit.from_json(old.to_json())
        np.testing.assert_allclose(replay.predict(early, late), old.predict(early, late), atol=2e-12, rtol=2e-12)
        for model in ("bounded_benefit_odds", "tangent_benefit_odds"):
            total = early + late
            tied0, tied1 = ALPHA * total, (1 - ALPHA) * total
            tied_matrix, _ = models.design_and_offset(tied0, tied1, ALPHA, shape, model, {"kappa": strength})
            np.testing.assert_array_equal(tied_matrix, frozen.mariner.design_matrix(tied0 + tied1, shape))
            null_matrix, _ = models.design_and_offset(early, late, ALPHA, shape, model, {"kappa": 0.0})
            np.testing.assert_array_equal(null_matrix, frozen.mariner.design_matrix(total, shape))


def test_tangent_law_has_finite_endpoints_and_removes_concavity_bonus() -> None:
    total = np.tile(np.array([0.0, 1e-250, 1e-10, 0.2, 2.0, 20.0, 1e250]), (3, 1))
    early = np.array([0.0, 0.5, 1.0])[:, None] * total
    late = total - early
    width = total.shape[1]
    tangent, _ = models.design_and_offset(early, late, ALPHA, SHAPE, "tangent_benefit_odds", {"kappa": 128.0})
    original, _ = models.design_and_offset(early, late, ALPHA, SHAPE, "bounded_benefit_odds", {"kappa": 128.0})
    assert np.isfinite(tangent).all()
    assert np.isfinite(original).all()
    np.testing.assert_array_equal(tangent[:, 0], np.zeros(3))
    np.testing.assert_array_equal(tangent[:, width - 1], -np.ones(3))
    assert np.max(original[:, :width] - tangent[:, :width]) < 2e-14
    assert np.max(tangent[:, :width] - original[:, :width]) > 0.01


@pytest.mark.parametrize("model", ("bounded_benefit_odds", "tangent_benefit_odds"))
def test_known_floor_recovers_law_and_replays_per_strength_predictions(model: str) -> None:
    early, late = geometry(503, 90)
    observed = independent_truth(early, late, model, 32.0)
    labels = np.arange(len(early)) % 3
    fit = models.fit_known_floor(
        early,
        late,
        observed,
        labels,
        ALPHA,
        model,
        floor=0.6,
        shapes=(SHAPE, {**SHAPE, "rate": 0.5}),
        test_grid=models.TestGrid(models.parameter_grid(model), (0.01, 0.0), (2.5,)),
    )
    query0, query1 = geometry(701, 24)
    expected = independent_truth(query0, query1, model, 32.0)
    np.testing.assert_allclose(fit.predict(query0, query1), expected, rtol=2e-12, atol=2e-12)
    restored = models.JointFit.from_json(json.loads(json.dumps(fit.to_json(), allow_nan=False)))
    np.testing.assert_array_equal(restored.predict(query0, query1), fit.predict(query0, query1))
    profiles = models.strength_profiles(fit, early, late, observed, labels, query0, query1)
    generating = next(profile for profile in profiles if profile["strength"] == 32.0)
    np.testing.assert_allclose(generating["inner_predictions"], observed, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(generating["query_predictions"], expected, rtol=2e-12, atol=2e-12)
    for profile in profiles:
        candidate = models.JointFit.from_json(profile["model"])
        np.testing.assert_array_equal(candidate.predict(query0, query1), profile["query_predictions"])


def test_isolated_joint_search_matches_frozen_schedule_without_changing_native_fit() -> None:
    early, late = geometry(206, 45)
    observed = independent_truth(early, late, "bounded_benefit_odds", 4.0)
    labels = np.arange(len(early)) % 3
    grid = models.TestGrid(({"kappa": 0.0}, {"kappa": 4.0}), (0.01,), (1.0, 2.5, 6.0))
    native_before = frozen.fit_joint(
        early, late, observed, labels, ALPHA, "bounded_benefit_odds", shapes=(SHAPE,), test_grid=grid
    )
    expanded = models.fit_joint(
        early, late, observed, labels, ALPHA, "bounded_benefit_odds", shapes=(SHAPE,), test_grid=grid
    )
    native_after = frozen.fit_joint(
        early, late, observed, labels, ALPHA, "bounded_benefit_odds", shapes=(SHAPE,), test_grid=grid
    )
    np.testing.assert_array_equal(native_before.predict(early, late), native_after.predict(early, late))
    np.testing.assert_allclose(expanded.predict(early, late), native_before.predict(early, late), atol=2e-12, rtol=2e-12)
    assert abs(expanded.inner_rmse - native_before.inner_rmse) < 1e-13


@pytest.mark.parametrize("model", ("bounded_benefit_odds", "tangent_benefit_odds"))
def test_expanded_search_retains_the_complete_off_grid_aggregate_candidate(model: str) -> None:
    base0, base1 = geometry(412, 30)
    early, late = np.tile(base0, (3, 1)), np.tile(base1, (3, 1))
    labels = np.repeat(np.arange(3), len(base0))
    observed = independent_truth(early, late, model, 0.0)
    null = models.fit_known_floor(
        early,
        late,
        observed,
        labels,
        ALPHA,
        "aggregate_mariner",
        floor=0.6,
        shapes=(SHAPE,),
        test_grid=models.TestGrid(({},), (0.0,), (2.5,)),
    )
    gamma = float((np.median(observed) - 0.6) / (np.median(observed) - np.min(observed)))
    null = models.JointFit(replace(null.selected, kappa=gamma, known_floor=None))
    fit = models.fit_joint(
        early,
        late,
        observed,
        labels,
        ALPHA,
        model,
        shapes=(SHAPE,),
        matched_null_fit=null,
        test_grid=models.TestGrid(({"kappa": 0.0}, {"kappa": 128.0}), (0.0,), (1.0, 2.5, 6.0)),
    )
    np.testing.assert_allclose(fit.predict(early, late), observed, rtol=2e-12, atol=2e-12)
    assert fit.inner_rmse < 1e-12
    assert fit.parameters["kappa"] == 0.0
    query0, query1 = geometry(22, 17)
    total = query0 + query1
    np.testing.assert_array_equal(
        fit.predict(query0, query1), fit.head.predict(frozen.mariner.design_matrix(total, SHAPE))
    )


def process_replay(fit: models.JointFit, early: np.ndarray, late: np.ndarray) -> tuple[models.JointFit, np.ndarray]:
    return models.JointFit.from_json(fit.to_json()), fit.predict(early, late)


def test_fitted_record_roundtrips_through_spawned_process() -> None:
    early, late = geometry(19, 7)
    fit = models.JointFit(
        frozen.JointFit(
            "tangent_benefit_odds",
            ALPHA,
            SHAPE,
            {"kappa": 128.0},
            0.01,
            2.5,
            None,
            0.0,
            frozen.mariner.Head(1.0, COEFFICIENTS, 0.6),
            (),
            {},
        )
    )
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as executor:
        restored, predicted = executor.submit(process_replay, fit, early, late).result(timeout=30)
    np.testing.assert_array_equal(predicted, fit.predict(early, late))
    np.testing.assert_array_equal(restored.predict(early, late), predicted)
