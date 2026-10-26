# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_single_phase_observatory_20260902 as bench,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    mariner_convex_fitting_round2_20260917 as fitting,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as base,
)


def test_selection_score_can_prefer_better_decision_over_lower_rmse():
    observed = [np.array([0.0, 1.0, 2.0])]
    wrong_minimum = [np.array([0.6, 0.5, 2.0])]
    correct_minimum = [np.array([0.5, 1.5, 2.5])]
    wrong = fitting.prediction_score(observed, wrong_minimum, fitting.Procedure.SELECTION)
    correct = fitting.prediction_score(observed, correct_minimum, fitting.Procedure.SELECTION)
    assert wrong.rmse < correct.rmse
    assert wrong.regret == 1.0
    assert correct.regret == 0.0
    assert correct.objective < wrong.objective


def test_bounded_rms_preserves_penalty_on_small_supported_columns():
    x = np.linspace(0.0, 0.01, 20)
    y = 0.4 + 3 * x
    design = base.Design(x[:, None], np.ones(1), ("small",))
    ridge = 0.1
    spec = base.HeadSpec()
    head = fitting.fit_head(design, y, ridge, spec, fitting.Procedure.BOUNDED_RMS)
    centered = x - x.mean()
    expected_slope = float(centered @ (y - y.mean()) / (centered @ centered + ridge * 0.25))
    assert head.coefficients[0] == pytest.approx(expected_slope, abs=1e-12)
    assert head.intercept == pytest.approx(y.mean() - expected_slope * x.mean(), abs=1e-12)
    assert expected_slope < 0.03


@pytest.mark.parametrize(
    "procedure", (fitting.Procedure.BOUNDED_RMS, fitting.Procedure.SELECTION, fitting.Procedure.JOINT)
)
def test_fit_and_inner_score_ignore_outer_test_outcomes(procedure):
    panel = bench.load_panel("60m_39bucket")
    spec = fitting.CandidateSpec(f"test_cf2_{procedure}", "cv17_smooth_d1", procedure)
    model = fitting.build_model(spec, panel.features)
    grid = dataclasses.replace(model.base, shapes=model.base.shapes[:2], ridge_grid=(0.01, 0.1))
    model = dataclasses.replace(model, reference=dataclasses.replace(model.reference, base=grid, search_evaluations=6))
    train = np.arange(24)
    inner = ((np.arange(12), np.arange(12, 24)), (np.arange(12, 24), np.arange(12)))
    response = panel.group("uncheatable").outcomes[:, 0].copy()
    fitted = model.fit(panel.features, response, train, inner, 0)
    changed = response.copy()
    changed[24:] = -1000.0
    replay = model.fit(panel.features, changed, train, inner, 0)
    np.testing.assert_array_equal(fitted.head.coefficients, replay.head.coefficients)
    assert fitted.shape == replay.shape
    assert fitted.ridge == replay.ridge
    assert fitted.head.floor == replay.head.floor
    score = fitting.cross_validate(
        model.base.design(panel.features, fitted.shape),
        response,
        fitted.ridge,
        model.head_spec(fitted.shape, float(fitted.diagnostics["kappa"])),
        inner,
        procedure,
    )
    assert score.rmse == pytest.approx(fitted.diagnostics["inner_cv_rmse"], abs=1e-12)
    if procedure is fitting.Procedure.SELECTION:
        assert score.objective == pytest.approx(fitted.diagnostics["inner_cv_objective"], abs=1e-12)
