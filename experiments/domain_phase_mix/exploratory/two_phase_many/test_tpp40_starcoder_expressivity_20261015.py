# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Numerical contracts for the all-training estimation diagnostic."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize._numdiff import approx_derivative

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    run_tpp40_starcoder_expressivity_20261015 as diagnostic,
)


@pytest.fixture
def panel() -> diagnostic.Panel:
    first, second = np.meshgrid([0, 0.08, 0.3, 0.6, 1.0], [0, 0.03, 0.2, 0.7, 1.0])
    phase0 = np.column_stack([1 - first.ravel(), first.ravel()])
    phase1 = np.column_stack([1 - second.ravel(), second.ravel()])
    alpha = 0.8
    scales = np.array([1.0, 18.0])
    return diagnostic.Panel(
        alpha * phase0 * scales, (1 - alpha) * phase1 * scales, phase0, phase1, alpha, pd.DataFrame(), pd.DataFrame()
    )


def state(family: str) -> diagnostic.State:
    return diagnostic.State(
        family,
        0.8,
        {"rate": 0.3, "power": 0.7, "threshold": 2.3},
        2.0 if family != "aggregate_mariner" else 0.0,
        0.7,
        -0.3,
        np.array([0.5, 0.7, 0.1, 0.2]),
    )


@pytest.mark.parametrize("family", diagnostic.FAMILIES)
def test_state_replays_frozen_function_and_tied_restriction(panel: diagnostic.Panel, family: str) -> None:
    value = state(family)
    matrix, offset = diagnostic.models.design_and_offset(
        panel.early, panel.late, panel.alpha, value.shape, family, {"kappa": value.strength}
    )
    native_head = diagnostic.models.mariner.Head(value.intercept, value.coefficients, value.floor)
    expected = diagnostic.models.frozen.temporal.predict_head(native_head, matrix, offset)
    np.testing.assert_array_equal(value.predict(panel.early, panel.late), expected)
    restored = diagnostic.State.from_json(value.to_json())
    np.testing.assert_array_equal(restored.predict(panel.early, panel.late), expected)
    total = panel.early + panel.late
    tied_early, tied_late = panel.alpha * total, (1 - panel.alpha) * total
    native = replace(value, family="aggregate_mariner", strength=0.0)
    np.testing.assert_array_equal(value.predict(tied_early, tied_late), native.predict(tied_early, tied_late))


def test_diagnostics_distinguishes_native_and_transformed_saturation(panel: diagnostic.Panel) -> None:
    value = replace(state("bounded_benefit_odds"), strength=128.0)
    record = diagnostic.state_diagnostics(value, panel, value.predict(panel.early, panel.late))
    assert record["native_benefit_endpoints_one"] == 0
    assert record["transformed_benefit_endpoints_one"] > 0


def test_head_jacobian_matches_finite_difference_including_native_clipping(panel: diagnostic.Panel) -> None:
    value = state("tangent_benefit_odds")
    matrix = value.design(panel.early, panel.late)
    for intercept in (-50.0, -0.3, 50.0):
        parameters = np.r_[intercept, value.coefficients]
        predicted, jacobian = diagnostic.head_response_jacobian(parameters, matrix, value.floor)
        expected = diagnostic.models.mariner.Head(intercept, value.coefficients, value.floor).predict(matrix)
        np.testing.assert_array_equal(predicted, expected)
        numerical = approx_derivative(lambda x: diagnostic.head_response_jacobian(x, matrix, value.floor)[0], parameters)
        np.testing.assert_allclose(jacobian, numerical, atol=2e-10, rtol=2e-7)


def test_conditional_head_recovers_synthetic_surface(panel: diagnostic.Panel) -> None:
    truth = state("bounded_benefit_odds")
    observed = truth.predict(panel.early, panel.late)
    start = replace(truth, intercept=0.1, coefficients=np.array([0.2, 0.3, 0.05, 0.07]))
    config = {"method": "trf", "x_scale": "jac", "max_nfev": 500, "ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12}
    record = diagnostic.solve_head(start, panel, observed, config)
    assert record["solver"]["feasible"]
    assert record["endpoint"]["rmse"] < 1e-7
    assert record["endpoint"]["model"]["floor"] == truth.floor
    assert record["endpoint"]["model"]["shape"] == truth.shape
    assert record["endpoint"]["model"]["strength"] == truth.strength


@pytest.mark.parametrize("original_gamma, projected_gamma", [(-2.0, 1.0), (2.7, 2.7), (9.0, 6.0)])
def test_joint_floor_projection_preserves_other_parameters(
    panel: diagnostic.Panel, original_gamma: float, projected_gamma: float
) -> None:
    value = state("tangent_benefit_odds")
    observed = value.predict(panel.early, panel.late)
    median, gap = diagnostic.floor_contract(observed)
    old = replace(value, floor=median - original_gamma * gap)
    start, receipt = diagnostic.joint_start(old, observed, {"gamma": [1.0, 6.0]})
    projected = diagnostic.joint_state(old, start, observed)
    assert receipt["gamma_after_projection"] == pytest.approx(projected_gamma)
    assert projected.floor == pytest.approx(median - projected_gamma * gap)
    assert projected.intercept == old.intercept
    np.testing.assert_array_equal(projected.coefficients, old.coefficients)
    assert projected.shape == old.shape
    assert projected.strength == old.strength


def test_unconverged_joint_start_is_retained_and_null_replays(panel: diagnostic.Panel) -> None:
    null = state("aggregate_mariner")
    observed = null.predict(panel.early, panel.late)
    temporal = replace(null, family="tangent_benefit_odds", strength=0.0)
    np.testing.assert_array_equal(temporal.predict(panel.early, panel.late), observed)
    bounds = {
        "gamma": [1.0, 6.0],
        "rate": [0.05, 4.0],
        "power": [0.3, 1.0],
        "threshold": [1.0, 6.0],
        "kappa": [0.0, 128.0],
    }
    config = {"method": "trf", "x_scale": "jac", "max_nfev": 1, "ftol": 1e-10, "xtol": 1e-10, "gtol": 1e-10}
    record = diagnostic.solve_joint(temporal, panel, observed, config, bounds)
    assert not record["solver"]["success"]
    retained = diagnostic.state_diagnostics(temporal, panel, observed)
    selected = diagnostic.best_record([("truncated_optimizer", record), ("retained_aggregate", {"start": retained})])
    assert selected["rmse"] == 0.0
    assert selected["selected_start"] == "retained_aggregate"


def test_durable_record_detects_changed_output(tmp_path: Path) -> None:
    path = tmp_path / "cell.json"
    identity = {"path": "synthetic_identity", "sha256": "unit-test"}
    record = {"prediction": [0.8, 0.9], "role": "all_training_diagnostic"}
    diagnostic.save_record(path, record, identity)
    assert diagnostic.cached_record(path, identity) == record
    copied = tmp_path / "other_cell.json"
    copied.write_text(path.read_text())
    copied.with_suffix(".complete.json").write_text(path.with_suffix(".complete.json").read_text())
    with pytest.raises(ValueError, match="Invalid completed diagnostic record"):
        diagnostic.cached_record(copied, identity)
    path.write_text('{"prediction":[0.1,0.2]}')
    with pytest.raises(ValueError, match="Invalid completed diagnostic record"):
        diagnostic.cached_record(path, identity)
