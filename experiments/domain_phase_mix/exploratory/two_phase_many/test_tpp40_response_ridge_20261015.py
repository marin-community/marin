# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Ridge objective/Jacobian, frozen zero adoption and held-label isolation."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_response_ridge_20261015 as runner
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_response_ridge_20261015 as model


def toy():
    t = np.linspace(0.05, 2, 25)
    matrix = np.column_stack((-t, t**2))
    observed = 0.6 + np.exp(matrix @ np.array([0.7, 0.08])) + 0.001 * np.sin(7 * t)
    return matrix, observed, [{"amplitudes": [0.35 + 0.1 * i, 0.04 + 0.01 * i]} for i in range(5)]


@pytest.mark.parametrize("floor", [0.6, None])
def test_penalized_residual_jacobian_matches_objective_and_finite_differences(floor):
    matrix, observed, _ = toy()
    amplitudes, ridge = np.array([0.73, 0.09]), 3.0
    point = model.evaluate(matrix, observed, amplitudes, floor, ridge=ridge)
    residual, jacobian, diagnostics = model.residual_jacobian(matrix, observed, point, floor)
    assert diagnostics["jacobian_regular"]
    np.testing.assert_allclose(residual @ residual, point["scaled_objective"], atol=1e-10)
    np.testing.assert_allclose(2 * jacobian.T @ residual, point["gradient"], atol=2e-8)
    step = 1e-6
    for j in range(2):
        direction = np.eye(2)[j] * step
        above = model.evaluate(matrix, observed, amplitudes + direction, floor, ridge=ridge)
        below = model.evaluate(matrix, observed, amplitudes - direction, floor, ridge=ridge)
        finite = (above["scaled_objective"] - below["scaled_objective"]) / (2 * step)
        np.testing.assert_allclose(finite, point["gradient"][j], atol=2e-5, rtol=1e-7)
    original = model.prior.evaluate(matrix, observed, amplitudes, floor)
    assert point["floor"] == original["floor"]
    assert point["scale"] == original["scale"]
    assert point["data_first_order_tolerance_met"] == original["first_order_tolerance_met"]


def test_zero_penalty_is_exact_native_objective_and_does_not_mutate_frozen_solver():
    matrix, observed, starts = toy()
    raw = model.prior.evaluate(matrix, observed, np.array([0.7, 0.08]), 0.6)
    point = model.evaluate(matrix, observed, np.array([0.7, 0.08]), 0.6, ridge=0)
    assert point["scaled_objective"] == raw["scaled_objective"]
    assert point["gradient"] == raw["gradient"]
    source = Path(model.frozen.__file__)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    old_evaluator = model.frozen.prior.evaluate
    fitted = model.fit_response(matrix, observed, 0.6, starts, ridge=100, evaluation_cap=5000)
    assert model.frozen.prior.evaluate is old_evaluator
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    finite = [row for row in fitted["trials"] if row["status"] == "finite"]
    assert fitted["selected"]["scaled_objective"] == min(row["scaled_objective"] for row in finite)
    assert sum(row["actual_evaluation_calls"] for row in fitted["solves"]) <= 25000
    assert len(fitted["solves"]) == 5
    assert all("data_first_order_tolerance_met" in row for row in finite)


@pytest.mark.parametrize("floor", [0.6, None])
def test_penalty_changes_clean_fit_without_supplying_true_floor_to_free_arm(floor):
    matrix, _, starts = toy()
    observed = 0.6 + np.exp(matrix @ np.array([0.7, 0.08]))
    free = model.fit_response(matrix, observed, floor, starts, ridge=0, evaluation_cap=5000)
    penalized = model.fit_response(matrix, observed, floor, starts, ridge=100, evaluation_cap=5000)
    assert np.linalg.norm(penalized["selected"]["amplitudes"]) < np.linalg.norm(free["selected"]["amplitudes"])
    assert penalized["selected"]["raw_mean_squared_bpb"] > free["selected"]["raw_mean_squared_bpb"]
    assert penalized["known_floor"] == floor
    assert penalized["selected"]["penalty"] > 0


def test_zero_adoption_is_byte_exact_and_foreign_resume_rejected(tmp_path):
    matrix, observed, starts = toy()
    old = tmp_path / "old"
    old.mkdir()
    frozen_fit = model.frozen.fit_response(matrix, observed, 0.6, starts, evaluation_cap=5000)
    runner.prior.write(old / "fit.json", frozen_fit)
    saved = {
        "matrix": matrix.tolist(),
        "observed": observed.tolist(),
        "starts": starts,
        "train_indices": list(range(len(observed))),
    }
    target = tmp_path / "new"
    runner.save_fit(target, saved, runner.ARMS[0], 0, {"invented": 1}, {"hash": "a"}, old)
    assert (target / "fit.json").read_bytes() == (old / "fit.json").read_bytes()
    with pytest.raises(ValueError, match="identity"):
        runner.save_fit(target, saved, runner.ARMS[0], 0, {"invented": 1}, {"hash": "changed"}, old)


def test_held_truth_poisoning_cannot_change_fit_or_select_ridge(tmp_path):
    matrix, observed, starts = toy()
    saved = {
        "matrix": matrix.tolist(),
        "observed": observed.tolist(),
        "starts": starts,
        "train_indices": list(range(len(observed))),
    }
    query_matrix = matrix[[1, 5, 1, 5, 10]]
    query = pd.DataFrame(
        {
            "role": ["asymmetric", "asymmetric", "matched_tied", "matched_tied", "actual_tied"],
            "source_index": [1, 5, 1, 5, 10],
            "row_id": ["a", "b", "a", "b", "c"],
        }
    )
    for poison in (0.0, 100.0):
        target = tmp_path / str(poison)
        fit = runner.save_fit(target, saved, runner.ARMS[1], 0.01, {"invented": 1}, {"hash": "a"}, tmp_path / "unused")
        prediction = model.prior.predict(query_matrix, fit["selected"])
        source = tmp_path / f"truth{poison}.csv"
        query.assign(predicted=prediction, truth=prediction + poison).to_csv(source, index=False)
        runner.score_saved_fit(target, fit, saved, 0.01, query_matrix, query, source)
    assert (tmp_path / "0.0/fit.json").read_bytes() == (tmp_path / "100.0/fit.json").read_bytes()
    clean = runner.prior.read(tmp_path / "0.0/scores.json")
    bad = runner.prior.read(tmp_path / "100.0/scores.json")
    assert clean["readout"] == bad["readout"]
    assert clean["metrics"]["query_rmse"] < 1e-12
    assert bad["metrics"]["query_rmse"] > 99


def test_nonregular_and_failed_jacobians_keep_complete_penalized_residual(monkeypatch):
    matrix = np.ones((9, 2))
    observed = np.linspace(0.9, 1.1, 9)
    point = model.evaluate(matrix, observed, np.array([0.2, 0.3]), None, ridge=4.0)
    residual, jacobian, diagnostics = model.residual_jacobian(matrix, observed, point, None)
    assert jacobian is None
    assert not diagnostics["jacobian_regular"]
    assert len(residual) == 11
    np.testing.assert_allclose(residual @ residual, point["scaled_objective"], atol=1e-10)

    def unavailable(*args):
        raise np.linalg.LinAlgError("invented failed decomposition")

    monkeypatch.setattr(model.frozen, "residual_jacobian", unavailable)
    fallback, jacobian, diagnostics = model.residual_jacobian(matrix, observed, point, None)
    assert jacobian is None
    assert diagnostics["jacobian_limitations"] == ["nonfinite_jacobian"]
    np.testing.assert_array_equal(fallback, residual)


def test_total_objective_overflow_rejected_even_with_finite_terms(monkeypatch):
    matrix, observed, _ = toy()
    old = model.prior.evaluate

    def enormous_data(*args):
        return old(*args) | {"scaled_objective": 1e308}

    monkeypatch.setattr(model.prior, "evaluate", enormous_data)
    with pytest.raises(ArithmeticError, match="Nonfinite penalized"):
        model.evaluate(matrix, observed, np.array([1.0, 1.0]), 0.6, ridge=5e307)


def test_zero_penalty_ignores_unneeded_overflowing_amplitude_norm():
    matrix = np.linspace(1.0, 2.0, 11)[:, None] * 1e-155
    amplitudes = np.array([1e155])
    observed = 0.6 + np.exp(matrix @ amplitudes)
    original = model.prior.evaluate(matrix, observed, amplitudes, 0.6)
    point = model.evaluate(matrix, observed, amplitudes, 0.6, ridge=0.0)
    assert point["penalty"] == 0.0
    assert point["scaled_objective"] == original["scaled_objective"]
    assert point["gradient"] == original["gradient"]
