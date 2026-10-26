# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import math

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    mariner_harm_screen_models_20260917 as screen,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_registry_20260902 as registry,
)


def _features(weights: np.ndarray | None = None) -> base.Features:
    if weights is None:
        weights = np.random.default_rng(71).dirichlet(np.ones(3), size=30)
    return base.features_from_panel(
        weights,
        np.asarray([5.0, 15.0, 40.0]),
        ("dolma3_cc/art_high", "dolma3_cc/art_low", "dolma3_arxiv"),
        early_fraction=None,
        label="harm_screen_fixture",
    )


def _spec(model_id: str) -> screen.CandidateSpec:
    return next(spec for spec in screen.candidate_specs() if spec.model_id == model_id)


def _folds(rows: int) -> tuple[np.ndarray, base.InnerFolds]:
    train = np.arange(rows)
    return train, tuple((train[train % 3 != fold], train[train % 3 == fold]) for fold in range(3))


def test_centered_mariner_reproduces_canonical_cv_and_predictions():
    features = _features()
    canonical = registry.ENTRY_BY_ID[registry.FROZEN_ID].build(features)
    candidate = screen.build_model(_spec("hs17_mariner"), features)
    train, inner = _folds(features.rows)
    benefit = -np.expm1(-((0.5 * features.exposures) ** 0.7))
    harm = np.log1p((1.0 + features.exposures) / math.exp(2.0)) ** 2
    response = 0.8 + np.exp(0.1 - benefit @ [0.2, 0.3, 0.4] + harm @ [0.05, 0.08, 0.12])
    fits = [model.fit(features, response, train, inner, 0) for model in (canonical, candidate)]

    np.testing.assert_allclose(fits[0].cv_table, fits[1].cv_table, rtol=1e-8, atol=1e-11)
    np.testing.assert_allclose(
        canonical.predict(fits[0], features, train),
        candidate.predict(fits[1], features, train),
        rtol=1e-8,
        atol=1e-10,
    )
    assert math.isclose(fits[0].diagnostics["inner_cv_rmse"], fits[1].diagnostics["inner_cv_rmse"], abs_tol=1e-10)


@pytest.mark.parametrize("power", [1.0, 1.5, 2.0])
@pytest.mark.parametrize("threshold", [1.0, 3.0])
def test_log_harm_derivatives_agree_with_finite_differences(power: float, threshold: float):
    exposure = np.asarray([0.2, 2.0, 10.0, 60.0])
    step = 0.002
    minus = screen.centered_log_harm(exposure - step, threshold, power)
    value = screen.centered_log_harm(exposure, threshold, power)
    plus = screen.centered_log_harm(exposure + step, threshold, power)
    denominator = math.exp(threshold) + 1.0 + exposure
    softplus = np.log(denominator) - threshold
    derivative = power * softplus ** (power - 1.0) / denominator
    curvature = power * softplus ** (power - 2.0) * (power - 1.0 - softplus) / denominator**2

    np.testing.assert_allclose((plus - minus) / (2.0 * step), derivative, rtol=2e-6, atol=1e-9)
    np.testing.assert_allclose((plus - 2.0 * value + minus) / step**2, curvature, rtol=2e-4, atol=2e-8)


def test_centering_after_the_power_preserves_the_linear_onset_term():
    exposure = np.asarray([0.0, 0.001, 1.0, 10.0])
    threshold = 1.0
    before_square = np.log1p(exposure / (1.0 + math.exp(threshold)))
    origin = math.log1p(math.exp(-threshold))
    centered = screen.centered_log_harm(exposure, threshold)

    np.testing.assert_allclose(centered, before_square**2 + 2.0 * origin * before_square, atol=1e-14)
    assert centered[1] > 100 * before_square[1] ** 2


@pytest.mark.parametrize("tag", ["rat", "exp"])
@pytest.mark.parametrize("cap", [1.0, 4.0])
def test_bounded_harm_has_zero_origin_and_the_stated_limit(tag: str, cap: float):
    spec = _spec(f"hs17_bounded_{tag}_c{cap:g}")
    features = _features(np.zeros((4, 3)))
    exposure = np.broadcast_to(np.asarray([0.0, 1e-4, 10.0, 1e300])[:, None], (4, 3)).copy()
    features = dataclasses.replace(features, exposures=exposure)
    design = spec.design_function(features, {"rate": 0.5, "power": 0.7, "threshold": 2.0}, spec.variant_params)
    harm = design.values[:, features.buckets :]

    np.testing.assert_array_equal(harm[0], np.zeros(3))
    assert np.all(np.diff(harm, axis=0) > 0)
    assert np.all(harm <= cap)
    np.testing.assert_allclose(harm[-1], cap, rtol=1e-5)
    original = screen.centered_log_harm(exposure[1], 2.0)
    np.testing.assert_allclose(harm[1], original, rtol=1e-4)


@pytest.mark.parametrize("rate,power,ratio", [(0.25, 0.7, 2.0), (4.0, 0.3, 0.5)])
def test_coupled_harm_inflection_matches_the_benefit_clock_without_clipping(rate: float, power: float, ratio: float):
    inflection = ratio * math.log(2.0) ** (1.0 / power) / rate
    threshold = screen.coupled_threshold(rate, power, ratio)
    offset = 0.1 * inflection
    step = min(0.001 * max(inflection, 1.0), offset / 10.0)
    positions = np.asarray([inflection - offset, inflection, inflection + offset])
    curvature = (
        screen.centered_log_harm(positions + step, threshold)
        - 2.0 * screen.centered_log_harm(positions, threshold)
        + screen.centered_log_harm(positions - step, threshold)
    ) / step**2

    assert curvature[0] > 0.0
    assert curvature[2] < 0.0
    assert abs(curvature[1]) < 2e-6
    assert math.isclose((math.e - 1.0) * math.exp(threshold) - 1.0, inflection, rel_tol=1e-12)


@pytest.mark.parametrize("harm_power", [2.0, 3.0])
def test_tied_benefit_harm_has_the_predicted_interior_minimum(harm_power: float):
    rate, power = 0.5, 0.7
    optimum_benefit = (1.0 / harm_power) ** (1.0 / (harm_power - 1.0))
    optimum = (-math.log1p(-optimum_benefit)) ** (1.0 / power) / rate
    exposure = optimum * np.asarray([0.8, 1.0, 1.2])
    features = _features(np.zeros((3, 3)))
    features = dataclasses.replace(features, exposures=np.broadcast_to(exposure[:, None], (3, 3)).copy())
    spec = _spec(f"hs17_tied_b{harm_power:g}")
    design = spec.design_function(features, {"rate": rate, "power": power}, spec.variant_params)
    # One bucket, alpha=beta=1: eta=-b+b^q. A positive exponential link preserves this minimum.
    prediction = np.exp(design.values[:, 0] + design.values[:, features.buckets])
    assert prediction[1] < prediction[0]
    assert prediction[1] < prediction[2]


def test_pair_interactions_exclude_self_products_and_respect_bucket_permutations():
    values = np.asarray([[3.0, 0.0, 0.0], [2.0, 3.0, 0.0], [1.0, 2.0, 4.0]])
    expected = np.asarray([0.0, 2.0, 14.0 / 3.0])[:, None]
    np.testing.assert_allclose(screen.pair_interaction(values), expected)
    np.testing.assert_array_equal(screen.pair_interaction(values), screen.pair_interaction(values[:, ::-1]))


def test_model_cache_description_contains_the_candidate_basis_and_source_digest():
    features = _features()
    first = screen.build_model(_spec("hs17_bounded_rat_c1"), features)
    second = screen.build_model(_spec("hs17_bounded_rat_c4"), features)
    first_description = json.dumps(base.describe_model(first), sort_keys=True)
    second_description = json.dumps(base.describe_model(second), sort_keys=True)

    assert screen.MODULE_SOURCE_HASH in first_description
    assert first_description != second_description


def test_identity_control_predicts_its_additive_loss_and_no_harm_drops_repetition_columns():
    features = _features()
    identity = screen.build_model(_spec("hs17_identity"), features)
    train, inner = _folds(features.rows)
    response = 1.0 + 0.01 * features.exposures[:, 0]
    fitted = identity.fit(features, response, train, inner, 0)
    design = identity.design(features, fitted.shape)
    expected = fitted.head.intercept + design.values @ fitted.head.coefficients

    np.testing.assert_allclose(identity.predict(fitted, features, train), expected)
    assert math.isnan(fitted.head.floor)
    no_harm = screen.build_model(_spec("hs17_no_harm"), features)
    no_harm_design = no_harm.base.design(features, no_harm.base.shapes[0])
    assert no_harm_design.values.shape == (features.rows, features.buckets)
    assert np.all(no_harm_design.values <= 0.0)
