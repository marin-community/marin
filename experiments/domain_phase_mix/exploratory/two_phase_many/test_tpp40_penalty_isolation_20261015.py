# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Behavior checks for conditional penalty isolation using toy data only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import lsq_linear

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_penalty_isolation_20261015 as runner


def toy_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1903)
    matrix = rng.normal(size=(54, 4)) * np.array([0.3, 2.0, 0.7, 0.08])
    response = 0.6 + np.exp(-1.0 + matrix @ np.array([0.2, 0.04, 0.12, 0.8]))
    return matrix, response, np.arange(len(matrix)) % 3


@pytest.mark.parametrize("ridge", [0.0, 0.01, 1.0])
def test_scaled_head_matches_direct_bounded_objective(ridge: float) -> None:
    matrix, response, _ = toy_problem()
    scales = np.array([0.7, 2.0, 0.25, 4.0])
    actual = runner.fixed_floor_head(matrix, response, ridge, 0.6, scales)
    # Independent direct system retains the free intercept instead of centering
    # and QR-reducing the production system.
    design = np.column_stack((np.ones(len(matrix)), matrix))
    rows = np.vstack((design, np.column_stack((np.zeros(4), np.diag(np.sqrt(ridge) * scales)))))
    rhs = np.r_[np.log(response - 0.6), np.zeros(4)]
    reference = lsq_linear(rows, rhs, bounds=(np.r_[-np.inf, np.zeros(4)], np.full(5, np.inf)), tol=1e-13)
    np.testing.assert_allclose(np.r_[actual.intercept, actual.coefficients], reference.x, atol=1e-9, rtol=1e-9)
    native = runner.fixed_floor_head(matrix, response, ridge, 0.6, np.ones(4))
    intercept, coefficients = runner.mariner.nonnegative_solve(matrix, np.log(response - 0.6), ridge)
    np.testing.assert_array_equal(native.coefficients, coefficients)
    assert native.intercept == intercept


def test_reference_penalty_is_invariant_to_column_units() -> None:
    matrix, response, _ = toy_problem()
    scales = np.array([0.7, 2.0, 0.25, 4.0])
    units = np.array([2.0, 0.4, 3.0, 0.2])
    original = runner.fixed_floor_head(matrix, response, 0.1, 0.6, scales)
    converted = runner.fixed_floor_head(matrix * units, response, 0.1, 0.6, scales * units)
    np.testing.assert_allclose(original.predict(matrix), converted.predict(matrix * units), atol=1e-12, rtol=1e-12)
    raw = runner.fixed_floor_head(matrix, response, 0.1, 0.6, np.ones(4))
    raw_converted = runner.fixed_floor_head(matrix * units, response, 0.1, 0.6, np.ones(4))
    assert np.max(abs(raw.predict(matrix) - raw_converted.predict(matrix * units))) > 1e-4


def test_inner_heads_ignore_held_labels_and_selected_head_replays() -> None:
    matrix, response, labels = toy_problem()
    query = matrix[:7] + 0.03
    scales = np.array([0.7, 2.0, 0.25, 4.0])
    fitted = runner.fit_conditional(matrix, response, labels, query, 0.6, scales)
    poisoned = response.copy()
    poisoned[labels == 0] += 0.5
    other = runner.fit_conditional(matrix, poisoned, labels, query, 0.6, scales)
    for left, right in zip(fitted["profile"], other["profile"], strict=True):
        for a, b in zip(left["inner_heads"], right["inner_heads"], strict=True):
            if np.all(labels[a["held_positions"]] == 0):
                assert a["head"] == b["head"]
                np.testing.assert_array_equal(
                    np.asarray(left["inner_predictions"])[a["held_positions"]],
                    np.asarray(right["inner_predictions"])[a["held_positions"]],
                )
    restored = runner.read_head(json.loads(json.dumps(fitted))["head"])
    np.testing.assert_array_equal(restored.predict(query), fitted["query_predictions"])
    truth = 0.6 + np.exp(-1.0 + query @ np.array([0.2, 0.04, 0.12, 0.8]))
    np.testing.assert_allclose(restored.predict(query), truth, atol=1e-10, rtol=1e-10)
    assert fitted["selected_ridge"] == 0.0


def test_resume_rejects_changed_saved_predictions(tmp_path: Path) -> None:
    identity = {"sources": {"toy": "identity"}}
    spec = {"penalty": "raw"}
    output = tmp_path / "prediction.json"
    output.write_text("[1.0]\n")
    receipt = {
        "identity": identity,
        "spec": spec,
        "dependencies": {},
        "outputs": {output.name: runner.benchmark.sha256(output)},
    }
    (tmp_path / "complete.json").write_text(json.dumps(receipt))
    assert runner.completed(tmp_path, spec, identity, {})
    output.write_text("[2.0]\n")
    with pytest.raises(ValueError):
        runner.completed(tmp_path, spec, identity, {})


def test_paired_summary_keeps_null_calibration_and_excludes_null_gain_ratios() -> None:
    records = []
    for world in ("null", "benefit", "rate", "combined"):
        for seed in range(3):
            for arm in runner.PENALTIES:
                raw = arm == "raw"
                calibration = (4.0 if world == "null" else 1.0) * (1.0 if raw else 0.5)
                records.append(
                    {
                        "truth_world": world,
                        "noise_sd": 0.003,
                        "noise_seed": seed,
                        "penalty": arm,
                        "clean_asymmetric_rmse": calibration,
                        "clean_actual_tied_rmse": calibration,
                        "clean_matched_tied_rmse": calibration,
                        "clean_gain_rmse": 0.0 if world == "null" else (2.0 if raw else 1.0),
                        "max_abs_predicted_gain": 0.0,
                    }
                )
    result = runner.paired_summary(records)
    calibration = result["pooled"]["clean_actual_tied_rmse"]
    gain = result["pooled"]["clean_gain_rmse"]
    assert calibration["cells_per_arm"] == 12
    assert calibration["mse_bpb_squared"]["raw"] == 4.75
    assert gain["cells_per_arm"] == 9
    assert gain["mse_bpb_squared"]["raw"] == 4.0
    assert result["directional_continuation_signal"]
    assert result["null_false_gain"] == {"raw": 0.0, "ng2_007_reference": 0.0}
