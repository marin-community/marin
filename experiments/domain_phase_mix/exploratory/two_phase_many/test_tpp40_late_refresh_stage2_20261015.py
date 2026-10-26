# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_late_refresh_stage2_20261015 as runner


def test_gate_applicability_null_absolute_error_and_physical_boundary() -> None:
    baseline = {"clean_actual_tied_rmse": 0.02}
    scores = {"max_abs_predicted_gain": 2e-8, "clean_gain_rmse": 0.0009, "clean_actual_tied_rmse": 0.022}
    null = runner.qualification_checks(runner.WORLDS[0], "late_refresh_hazard", {"lambda": 0.5}, scores, baseline)
    assert not null["null_gain_pass"]
    assert null["gain_recovery_pass"] and null["tied_calibration_pass"]
    boundary = runner.qualification_checks(runner.WORLDS[1], "late_refresh_hazard", {"lambda": 1.0}, scores, baseline)
    assert boundary["null_gain_pass"] and not boundary["kinetic_boundary_pass"]
    missed = {**scores, "clean_gain_rmse": 0.002, "clean_actual_tied_rmse": 0.024}
    positive = runner.qualification_checks(runner.WORLDS[1], "late_refresh_hazard", {"lambda": 0.5}, missed, baseline)
    assert not positive["gain_recovery_pass"] and not positive["tied_calibration_pass"]
    control = runner.qualification_checks(runner.WORLDS[1], "aggregate_mariner", {}, missed, baseline)
    assert not control["correct_family"]
    assert control["gain_recovery_pass"] and control["tied_calibration_pass"]
    # Poor raw performance remains visible but cannot create a retroactive noiseless gate.
    raw_failure = {**scores, "regret_vs_truth_numerical_incumbent": 10.0, "selected_solver_success": False}
    assert (
        runner.qualification_checks(runner.WORLDS[0], "late_refresh_hazard", {"lambda": 0.5}, raw_failure, baseline)
        == null
    )


@pytest.mark.parametrize("strength", (0.0, 0.5))
def test_raw_truth_binding_matches_independent_unclipped_equation(strength: float) -> None:
    early = np.array([[0.0, 8.0], [2.0, 1.0], [4.0, 0.0]])
    late = np.array([[2.0, 0.0], [0.5, 0.25], [1.0, 2.0]])
    alpha, rate, power, threshold = 0.8, 0.25, 0.7, 3.0
    coefficients = np.array([0.5, 0.8, 0.03, 0.02])
    specification = {
        "shape": {"rate": rate, "power": power, "threshold": threshold},
        "head": {"intercept": 35.0, "coefficients": coefficients.tolist(), "floor": 0.6},
        "parameters": {"lambda": strength},
    }
    total = early + late
    hazard = (rate * total) ** power
    ratio = -np.expm1(-((rate * late) ** power)) / -np.expm1(-((rate * (1 - alpha) * total) ** power))
    benefit = -np.expm1(-hazard * ((1 - strength) + strength * ratio))
    harm = np.logaddexp(0.0, np.log1p(total) - threshold) ** 2
    feature_sum = -benefit @ coefficients[:2] + harm @ coefficients[2:]
    expected = 0.6 + np.exp(35.0 + feature_sum)
    truth = runner.SyntheticTruth(specification, alpha)
    np.testing.assert_allclose(truth.predict(early, late), expected, atol=0.0, rtol=4e-15)
    assert np.all(truth.predict(early, late) > np.exp(30))


def test_scientific_prepare_rejects_oracle_approval_and_changed_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "OUTPUT", tmp_path)
    monkeypatch.setattr(runner, "registration", lambda: {})
    current = {"sources": {"toy": "first"}, "environment": {}}
    monkeypatch.setattr(runner, "source_identity", lambda: current)
    (tmp_path / "registration.json").write_text("{}\n")
    (tmp_path / "execution_identity.json").write_text(json.dumps(current))
    approval = {
        "scope": "late_refresh_known_shape_floor_oracle",
        "registration_sha256": runner.benchmark.sha256(tmp_path / "registration.json"),
        "execution_identity_sha256": runner.benchmark.sha256(tmp_path / "execution_identity.json"),
    }
    (tmp_path / "execution_approval.json").write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="Separate exact stage-two"):
        runner.prepare()
    approval["scope"] = "late_refresh_stage2_noiseless"
    (tmp_path / "execution_approval.json").write_text(json.dumps(approval))
    assert runner.execution_identity() == current
    current["sources"]["toy"] = "changed"
    with pytest.raises(ValueError, match="Frozen stage-two"):
        runner.prepare()


def test_reused_completion_checks_output_and_aggregate_dependencies(tmp_path: Path) -> None:
    result, dependency = tmp_path / "model.json", tmp_path / "aggregate_complete.json"
    result.write_text("{}\n")
    dependency.write_text('{"complete":true}\n')
    spec, identity = {"family": "late_refresh_hazard"}, {"source": "toy"}
    receipt = {
        "spec": spec,
        "identity": identity,
        "outputs": {"model.json": runner.benchmark.sha256(result)},
        "dependencies": {str(dependency): runner.benchmark.sha256(dependency)},
    }
    (tmp_path / "complete.json").write_text(json.dumps(receipt))
    assert runner._RUNNER.completed(tmp_path, spec, identity)
    dependency.write_text('{"complete":false}\n')
    with pytest.raises(ValueError):
        runner._RUNNER.completed(tmp_path, spec, identity)
    dependency.write_text('{"complete":true}\n')
    result.write_text('{"head":"changed"}\n')
    with pytest.raises(ValueError):
        runner._RUNNER.completed(tmp_path, spec, identity)
