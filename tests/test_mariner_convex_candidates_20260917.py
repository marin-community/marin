# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    mariner_convex_candidates_20260917 as candidates,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    mariner_harm_screen_models_20260917 as previous,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    screen_mariner_convex_rms_20260917 as rms,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as base,
)

SPECS = candidates.candidate_specs()
CONVEX = tuple(spec for spec in SPECS if spec.guaranteed_convex)


def features_at(weights: np.ndarray) -> base.Features:
    return base.features_from_panel(
        weights,
        np.asarray([8.0, 64.0, 256.0]),
        ("dolma3_cc/art_high", "dolma3_cc/art_low", "dolma3_arxiv"),
        early_fraction=None,
        label="convex_fixture",
    )


@pytest.mark.parametrize("spec", CONVEX, ids=lambda spec: spec.model_id)
def test_convex_mixture_logits_and_exponential_predictions_satisfy_jensen(spec):
    rng = np.random.default_rng(71)
    first = rng.dirichlet(np.ones(3), 200)
    second = rng.dirichlet(np.ones(3), 200)
    # Include boundary mixtures so the check does not rely only on smooth interiors.
    first[:3] = np.eye(3)
    second[:3] = np.eye(3)[::-1]
    fraction = rng.uniform(size=(200, 1))
    middle = fraction * first + (1.0 - fraction) * second
    for shape in ({"rate": 0.1, "power": 0.3, "threshold": 1.0}, {"rate": 2.0, "power": 1.0, "threshold": 4.0}):
        matrices = [spec.design_function(features_at(w), shape, spec.params).values for w in (first, second, middle)]
        coefficients = rng.uniform(0.0001, 0.02, matrices[0].shape[1])
        logits = [matrix @ coefficients for matrix in matrices]
        line = fraction[:, 0] * logits[0] + (1.0 - fraction[:, 0]) * logits[1]
        assert np.all(logits[2] <= line + 1e-12)
        prediction_line = fraction[:, 0] * np.exp(logits[0]) + (1.0 - fraction[:, 0]) * np.exp(logits[1])
        assert np.all(np.exp(logits[2]) <= prediction_line + 1e-12)


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.model_id)
def test_epoch_derivatives_match_finite_differences_for_every_basis(spec):
    features = features_at(np.full((7, 3), 1.0 / 3.0))
    exposures = np.broadcast_to(np.asarray([0.013, 0.7, 2.31, 7.27, 23.1, 47.3, 123.7])[:, None], (7, 3)).copy()
    features = dataclasses.replace(features, exposures=exposures)
    shape = {"rate": 0.5, "power": 0.7, "threshold": 2.0}
    derivative, buckets = candidates.design_epoch_derivative(spec, features, shape)
    step = 1e-6
    for bucket in range(3):
        perturbation = np.zeros_like(exposures)
        perturbation[:, bucket] = step
        left = spec.design_function(
            dataclasses.replace(features, exposures=exposures - perturbation), shape, spec.params
        )
        right = spec.design_function(
            dataclasses.replace(features, exposures=exposures + perturbation), shape, spec.params
        )
        reference = np.zeros_like(derivative)
        reference[:, buckets == bucket] = derivative[:, buckets == bucket]
        np.testing.assert_allclose((right.values - left.values) / (2.0 * step), reference, rtol=3e-6, atol=1e-8)


def test_huber_hinge_has_matching_slopes_at_both_joins():
    step = 1e-5
    onset, width = 3.0, 8.0
    for join, expected in ((onset, 0.0), (onset + width, 1.0 / candidates.RAW_SCALE)):
        positions = np.asarray([join - step, join, join + step])
        values = candidates.huber_hinge(positions, onset, width)
        np.testing.assert_allclose(np.diff(values) / step, expected, atol=2e-8)


def test_raw_smooth_hinge_changes_onset_without_changing_tail_slope():
    features = features_at(np.zeros((1, 3)))
    features = dataclasses.replace(features, exposures=np.full((1, 3), 1000.0))
    spec = next(spec for spec in SPECS if spec.model_id == "cv17_smooth_d8")
    for threshold in (1.0, 3.0, 6.0):
        derivative, _ = candidates.design_epoch_derivative(
            spec, features, {"rate": 0.1, "power": 0.7, "threshold": threshold}
        )
        np.testing.assert_allclose(derivative[:, 3:], 1.0 / candidates.RAW_SCALE, rtol=1e-12)


@pytest.mark.parametrize("new_id,old_id", [("cv17_mariner", "hs17_mariner"), ("cv17_old_convex", "hs17_convex_linear")])
def test_controls_preserve_existing_fits_and_predictions(new_id, old_id):
    features = features_at(np.random.default_rng(42).dirichlet(np.ones(3), 24))
    spec = next(spec for spec in SPECS if spec.model_id == new_id)
    previous_spec = next(spec for spec in previous.candidate_specs() if spec.model_id == old_id)
    # Exercise actual fit/floor selection on a small shared grid.
    models = []
    for model in (candidates.build_model(spec, features), previous.build_model(previous_spec, features)):
        assert isinstance(model, base.FittedFloorModel)
        assert isinstance(model.base, base.GridModel)
        models.append(
            dataclasses.replace(
                model, base=dataclasses.replace(model.base, shapes=model.base.shapes[:2], ridge_grid=(0.0, 0.1))
            )
        )
    rows = np.arange(features.rows)
    inner = tuple((rows[rows % 3 != fold], rows[rows % 3 == fold]) for fold in range(3))
    response = 1.0 + np.exp(-0.2 * np.log1p(features.exposures).sum(axis=1) + 0.02 * features.exposures[:, 1])
    fits = [model.fit(features, response, rows, inner, 0) for model in models]
    np.testing.assert_allclose(fits[0].cv_table, fits[1].cv_table, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        models[0].predict(fits[0], features, rows), models[1].predict(fits[1], features, rows), rtol=1e-10, atol=1e-12
    )


def test_fractional_weibull_boundary_derivative_is_not_falsely_finite():
    derivative = candidates.negative_benefit_derivative(np.asarray([0.0, 1.0]), 0.5, 0.7)
    assert derivative[0] == -np.inf
    assert np.isfinite(derivative[1])


def test_rms_sensitivity_fit_is_invariant_to_positive_column_units():
    features = features_at(np.random.default_rng(23).dirichlet(np.ones(3), 24))
    spec = next(spec for spec in SPECS if spec.model_id == "cv17_smooth_d1")
    model = rms.build_rms(spec, features)
    assert isinstance(model.base, base.GridModel)
    grid = dataclasses.replace(model.base, shapes=model.base.shapes[:2], ridge_grid=(0.1,))

    def rescaled_builder(query, shape):
        original = grid.builder(query, shape)
        factors = np.asarray([0.1, 1.0, 10.0, 0.01, 1.0, 100.0])
        return base.Design(original.values * factors, original.ridge, original.names)

    first = dataclasses.replace(model, base=grid)
    second = dataclasses.replace(
        model, base=dataclasses.replace(grid, builder=rescaled_builder, model_id="rms_unit_rescaling_check")
    )
    rows = np.arange(features.rows)
    inner = tuple((rows[rows % 3 != fold], rows[rows % 3 == fold]) for fold in range(3))
    response = 1.0 + np.exp(-0.2 * np.log1p(features.exposures).sum(axis=1) + 0.02 * features.exposures[:, 1])
    fits = [candidate.fit(features, response, rows, inner, 0) for candidate in (first, second)]
    np.testing.assert_allclose(fits[0].cv_table, fits[1].cv_table, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(
        first.predict(fits[0], features, rows), second.predict(fits[1], features, rows), rtol=1e-8, atol=1e-10
    )
