# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import mariner_convex_round2_models_20260917 as models
from experiments.domain_phase_mix.exploratory.two_phase_many import single_phase_observatory_models_20260902 as base

SPECS = models.candidate_specs()


def features(weights):
    return base.features_from_panel(
        weights,
        np.asarray([8.0, 64.0, 256.0]),
        ("dolma3_cc/art_high", "dolma3_cc/art_low", "dolma3_arxiv"),
        early_fraction=None,
        label="convex_round2_test",
    )


@pytest.mark.parametrize("spec", [s for s in SPECS if s.model_id != "cv2_mariner"], ids=lambda s: s.model_id)
def test_convex_columns_and_outer_links_satisfy_jensen(spec):
    rng = np.random.default_rng(92)
    left = rng.dirichlet(np.ones(3), 100)
    right = rng.dirichlet(np.ones(3), 100)
    fraction = rng.uniform(size=(100, 1))
    left[:3] = np.eye(3)
    right[:3] = np.eye(3)[::-1]
    points = (left, right, fraction * left + (1 - fraction) * right)
    shape = {"rate": 0.5, "power": 0.7, "threshold": 2.0}
    matrices = [models.design(spec, features(point), shape).values for point in points]
    coefficients = rng.uniform(0.001, 0.01, matrices[0].shape[1])
    logits = [matrix @ coefficients for matrix in matrices]
    assert np.all(logits[2] <= fraction[:, 0] * logits[0] + (1 - fraction[:, 0]) * logits[1] + 1e-12)
    if spec.link == "raw":
        predictions = [np.maximum(x, 0) for x in logits]
    elif spec.link == "softplus":
        predictions = [0.3 + 0.8 * np.logaddexp(0, x) for x in logits]
    else:
        predictions = [0.3 + np.exp(x) for x in logits]
    assert np.all(predictions[2] <= fraction[:, 0] * predictions[0] + (1 - fraction[:, 0]) * predictions[1] + 1e-12)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.model_id)
def test_all_basis_jacobians_include_cross_bucket_terms(spec):
    point = features(np.asarray([[0.12, 0.37, 0.51], [0.27, 0.16, 0.57]]))
    shape = {"rate": 0.1, "power": 0.7, "threshold": 2.0}
    jacobian = models.design_epoch_jacobian(spec, point, shape)
    for bucket in range(3):
        delta = np.zeros_like(point.exposures)
        delta[:, bucket] = 1e-5
        left = models.design(spec, dataclasses.replace(point, exposures=point.exposures - delta), shape).values
        right = models.design(spec, dataclasses.replace(point, exposures=point.exposures + delta), shape).values
        np.testing.assert_allclose((right - left) / 2e-5, jacobian[:, :, bucket], rtol=1e-5, atol=1e-8)


def test_inverse_softplus_round_trip_across_small_and_large_outputs():
    values = np.asarray([1e-12, 1e-7, 0.1, 1.0, 30.0, 1000.0, 1e100])
    actual = np.logaddexp(0, models.inverse_softplus(values))
    np.testing.assert_allclose(actual, values, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("model_id", ["cv2_raw_additive", "cv2_softplus"])
def test_custom_links_replay_cv_and_ignore_outer_test_outcomes(model_id):
    spec = next(s for s in SPECS if s.model_id == model_id)
    point = features(np.random.default_rng(48).dirichlet(np.ones(3), 24))
    model = models.build_model(spec, point)
    assert isinstance(model, models.LinkedConvexModel)
    model = dataclasses.replace(
        model, base=dataclasses.replace(model.base, shapes=model.base.shapes[:2], ridge_grid=(0.001, 0.1))
    )
    train = np.arange(18)
    inner = tuple((train[train % 3 != i], train[train % 3 == i]) for i in range(3))
    response = 0.5 + np.exp(-0.2 * np.log1p(point.exposures).sum(axis=1) + 0.004 * point.exposures[:, 1])
    fitted = model.fit(point, response, train, inner, 0)
    changed = response.copy()
    changed[18:] += 100
    repeated = model.fit(point, changed, train, inner, 0)
    np.testing.assert_array_equal(fitted.cv_table, repeated.cv_table)
    np.testing.assert_allclose(
        model.predict(fitted, point, np.arange(24)), model.predict(repeated, point, np.arange(24)), rtol=1e-12
    )
    matrix = model.base.design(point, fitted.shape)
    score = models.linked_cv(
        matrix,
        response,
        fitted.ridge,
        model.base.head_for(fitted.shape),
        model.link,
        float(fitted.diagnostics["kappa"]),
        inner,
    )
    np.testing.assert_allclose(score, fitted.diagnostics["inner_cv_rmse"], rtol=1e-12)
    assert np.all(model.predict(fitted, point, np.arange(24)) >= 0)


@pytest.mark.parametrize("model_id", ["cv2_raw_additive", "cv2_softplus", "cv2_family_imbalance"])
def test_prediction_weight_gradient_includes_outer_link(model_id):
    spec = next(s for s in SPECS if s.model_id == model_id)
    point = features(np.asarray([[0.2, 0.3, 0.5]]))
    shape = {"rate": 0.1, "power": 0.7, "threshold": 2.0}
    matrix = models.design(spec, point, shape)
    head = base.FittedHead(0.3, np.full(matrix.values.shape[1], 0.03), 0.2, matrix.values.shape[1])
    if spec.link in ("raw", "softplus"):
        head = models.LinkedHead(head, 0.2, 0.8, 1.5, spec.link)
    fit = base.Fitted(shape, 0.1, head, {})
    prediction, gradient = models.prediction_and_weight_gradient(spec, fit, point)
    assert np.isfinite(prediction).all()
    for bucket in range(3):
        delta = np.zeros_like(point.weights)
        delta[:, bucket] = 1e-6
        left = features(point.weights - delta)
        right = features(point.weights + delta)
        low = models.prediction_and_weight_gradient(spec, fit, left)[0]
        high = models.prediction_and_weight_gradient(spec, fit, right)[0]
        np.testing.assert_allclose((high - low) / 2e-6, gradient[:, bucket], rtol=1e-5, atol=1e-8)
