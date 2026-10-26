# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    mariner_harm_screen_raw_loss_20260917 as raw,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as base,
)


def _features(exposure: np.ndarray) -> base.Features:
    exposure = np.asarray(exposure).reshape(-1, 1)
    return base.features_from_panel(
        exposure / 20.0, np.asarray([20.0]), ("synthetic",), early_fraction=None, label="raw-loss-test"
    )


def _true_loss(exposure: np.ndarray) -> np.ndarray:
    benefit = 1.0 - np.exp(-((0.2 * exposure) ** 0.8))
    harm = np.logaddexp(0.0, np.log1p(exposure) - 1.3) ** 2 - np.logaddexp(0.0, -1.3) ** 2
    return 1.0 + np.exp(-0.2 - 1.1 * benefit) + 0.025 * harm


def test_raw_loss_fit_recovers_unobserved_interior_minimum():
    exposure = np.linspace(0.0, 20.0, 25)
    response = _true_loss(exposure)
    # At kappa=1.5 this anchor makes the known generating floor exactly 1.
    anchor = 3.0 * float(np.min(response)) - 2.0
    probe = raw.RawLossProbe(raw.RawHarm.SQUARED_LOG, base.HeadSpec(floor_anchor=anchor), (0.0,))
    shape = {"rate": 0.2, "power": 0.8, "threshold": 1.3}
    fitted = probe.fit_at_ridge(_features(exposure), response, np.arange(len(exposure)), shape, 0.0, 7)
    dense = np.linspace(0.0, 20.0, 2001)
    predicted = probe.predict(fitted, _features(dense), np.arange(len(dense)))
    actual = _true_loss(dense)

    assert 0.5 < dense[np.argmin(actual)] < 19.5
    assert abs(dense[np.argmin(predicted)] - dense[np.argmin(actual)]) <= 0.02
    assert np.max(np.abs(predicted - actual)) < 2e-5
    assert fitted.diagnostics["solver_converged_count"] == 2
    assert fitted.diagnostics["intercept_bound_hits"] == 0


@pytest.mark.parametrize("harm", tuple(raw.RawHarm))
@pytest.mark.parametrize("placement", tuple(raw.HarmPlacement))
def test_raw_loss_selected_cv_reconstructs_actual_form_and_training_only_floors(
    harm: raw.RawHarm, placement: raw.HarmPlacement
):
    exposure = np.linspace(0.2, 20.0, 30)
    response = _true_loss(exposure) + 0.008 * np.sin(1.7 * exposure)
    features = _features(exposure)
    train = np.arange(27)
    inner = tuple((train[train % 3 != fold], train[train % 3 == fold]) for fold in range(3))
    probe = raw.RawLossProbe(harm, base.HeadSpec(), (0.0, 1e-4, 0.01), placement=placement)
    shape = {"rate": 0.2, "power": 0.8, "threshold": 1.3}
    fitted = probe.fit(features, response, train, inner, shape, 17)

    squared_error = 0.0
    count = 0
    floors = []
    for fold in fitted.head.inner_fits:
        x = exposure[fold.validation]
        benefit = 1.0 - np.exp(-((0.2 * x) ** 0.8))
        log_softplus = np.logaddexp(0.0, np.log1p(x) - 1.3)
        origin = np.logaddexp(0.0, -1.3)
        if harm is raw.RawHarm.UNSQUARED_LOG:
            damage = log_softplus - origin
        elif harm is raw.RawHarm.CONVEX_LINEAR:
            damage = np.logaddexp(0.0, (x - math.expm1(1.3)) / math.exp(1.3))
            damage -= np.logaddexp(0.0, -math.expm1(1.3) / math.exp(1.3))
        else:
            damage = log_softplus**2 - origin**2
            if harm is raw.RawHarm.BOUNDED_RATIONAL_C1:
                damage = damage / (1.0 + damage)
        head = fold.head
        expected_floor = 1.5 * float(np.min(response[fold.train])) - 0.5 * float(np.median(response[fold.train]))
        assert head.floor == pytest.approx(expected_floor, abs=1e-14)
        floors.append(head.floor)
        # Independent scalar reference catches scoring the wrong family or link.
        if placement is raw.HarmPlacement.RAW_LOSS:
            prediction = head.floor + np.exp(head.intercept - head.alpha[0] * benefit) + head.beta[0] * damage
        else:
            prediction = head.floor + np.exp(head.intercept - head.alpha[0] * benefit + head.beta[0] * damage)
        np.testing.assert_allclose(fold.prediction, prediction, rtol=1e-12, atol=1e-12)
        squared_error += float(np.sum((prediction - response[fold.validation]) ** 2))
        count += len(fold.validation)
    assert np.ptp(floors) > 1e-5
    independent_rmse = math.sqrt(squared_error / count)
    assert fitted.diagnostics["inner_rmse"] == pytest.approx(independent_rmse, abs=1e-12)
    assert fitted.diagnostics["inner_reconstructed_rmse"] == pytest.approx(independent_rmse, abs=1e-12)
    assert fitted.cv_table is not None
    assert fitted.ridge == probe.ridge_grid[int(np.argmin(fitted.cv_table[0]))]

    changed_test_response = response.copy()
    changed_test_response[27:] = -100.0
    refitted = probe.fit(features, changed_test_response, train, inner, shape, 17)
    np.testing.assert_allclose(
        probe.predict(fitted, features, np.arange(30)),
        probe.predict(refitted, features, np.arange(30)),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("placement", tuple(raw.HarmPlacement))
def test_raw_loss_analytic_jacobian_matches_numerical_residual_gradient(placement: raw.HarmPlacement):
    benefit = np.asarray([[0.1, 0.5], [0.8, 0.2], [0.9, 0.7]])
    harm = np.asarray([[0.0, 0.6], [0.4, 0.1], [1.3, 0.9]])
    target = np.asarray([1.1, 0.8, 1.2])
    parameters = np.asarray([-0.2, 0.7, 0.1, 0.04, 0.06])
    _, jacobian = raw.residual_and_jacobian(parameters, benefit, harm, target, 0.2, 0.03, placement)

    def reference(value: np.ndarray) -> np.ndarray:
        data = []
        for brow, hrow, y in zip(benefit, harm, target, strict=True):
            signal = value[0] - sum(b * a for b, a in zip(brow, value[1:3], strict=True))
            damage = sum(h * beta for h, beta in zip(hrow, value[3:], strict=True))
            prediction = (
                math.exp(signal + damage) if placement is raw.HarmPlacement.LOG_DEFICIT else math.exp(signal) + damage
            )
            data.append((0.2 + prediction - y) / math.sqrt(3))
        return np.asarray([*data, *(math.sqrt(0.03) * value[1:])])

    numerical = np.column_stack(
        [
            (reference(parameters + 1e-6 * step) - reference(parameters - 1e-6 * step)) / 2e-6
            for step in np.eye(len(parameters))
        ]
    )
    np.testing.assert_allclose(jacobian, numerical, atol=2e-10, rtol=2e-9)


@pytest.mark.parametrize("placement", tuple(raw.HarmPlacement))
def test_raw_loss_ridge_strength_is_invariant_to_training_row_replication(placement: raw.HarmPlacement):
    exposure = np.linspace(0.0, 20.0, 18)
    response = _true_loss(exposure)
    probe = raw.RawLossProbe(raw.RawHarm.SQUARED_LOG, base.HeadSpec(), (0.002,), placement=placement)
    shape = {"rate": 0.2, "power": 0.8, "threshold": 1.3}
    features = _features(exposure)
    fitted = probe.fit_at_ridge(features, response, np.arange(18), shape, 0.002, 7)
    repeated = probe.fit_at_ridge(
        _features(np.repeat(exposure, 2)), np.repeat(response, 2), np.arange(36), shape, 0.002, 7
    )
    np.testing.assert_allclose(
        probe.predict(fitted, features, np.arange(18)),
        probe.predict(repeated, features, np.arange(18)),
        rtol=1e-8,
        atol=1e-8,
    )


def test_raw_loss_failed_starts_do_not_receive_valid_cv_scores():
    exposure = np.linspace(0.0, 20.0, 18)
    response = _true_loss(exposure)
    probe = raw.RawLossProbe(raw.RawHarm.SQUARED_LOG, base.HeadSpec(), (0.0,), max_nfev=1)
    train = np.arange(18)
    inner = tuple((train[train % 3 != fold], train[train % 3 == fold]) for fold in range(3))
    with pytest.raises(raw.RawLossFitError) as caught:
        probe.fit(_features(exposure), response, train, inner, {"rate": 0.2, "power": 0.8, "threshold": 1.3}, 7)

    assert len(caught.value.runs) == 6
    assert all(not run.converged and run.nfev == 1 for run in caught.value.runs)


def test_raw_loss_per_bucket_geometry_matches_separate_bucket_contributions():
    features = base.features_from_panel(
        np.asarray([[0.1, 0.9], [0.4, 0.6], [0.7, 0.3]]),
        np.asarray([8.0, 20.0]),
        ("a", "b"),
        early_fraction=None,
        label="two-buckets",
    )
    shape = {"rate:0": 0.2, "rate:1": 0.5, "power:0": 0.8, "power:1": 0.5, "threshold:0": 1.3, "threshold:1": 2.0}
    benefit, harm = raw.basis(features, shape, raw.RawHarm.SQUARED_LOG)
    for index in range(2):
        exposure = features.exposures[:, index]
        expected_benefit = 1.0 - np.exp(-((shape[f"rate:{index}"] * exposure) ** shape[f"power:{index}"]))
        expected_harm = np.logaddexp(0.0, np.log1p(exposure) - shape[f"threshold:{index}"]) ** 2
        expected_harm -= np.logaddexp(0.0, -shape[f"threshold:{index}"]) ** 2
        np.testing.assert_allclose(benefit[:, index], expected_benefit, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(harm[:, index], expected_harm, rtol=1e-13, atol=1e-13)
