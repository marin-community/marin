# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_effective_harm_stage3_20261015 as stage3


def test_paired_training_noise_keeps_queries_clean_for_both_worlds() -> None:
    clean, other = np.linspace(0.8, 1.2, 280), np.linspace(1.1, 1.5, 280)
    train = np.arange(0, 280, 2)
    for seed in stage3.SEEDS:
        observed, noise = stage3.noisy_training_labels(clean, train, seed)
        alternate, paired = stage3.noisy_training_labels(other, train, seed)
        np.testing.assert_array_equal(noise, paired)
        np.testing.assert_array_equal(noise, np.random.default_rng(seed).normal(0, 0.003, 280))
        np.testing.assert_array_equal(observed[1::2], clean[1::2])
        np.testing.assert_array_equal(alternate[1::2], other[1::2])
        np.testing.assert_allclose(observed[train] - clean[train], noise[train], atol=2e-16, rtol=0)
        np.testing.assert_allclose(alternate[train] - other[train], noise[train], atol=2e-16, rtol=0)


def test_private_noisy_truth_fit_and_raw_paths_use_effective_harm_including_sink() -> None:
    early = np.array([[8.0, 3.0], [0.0, 0.0], [8.0, 4.0], [2.0, 10.0]])
    late = np.array([[0.0, 0.0], [3.0, 2.0], [2.0, 1.0], [1.0, 0.5]])
    alpha, shape = 0.8, {"rate": 0.25, "power": 0.7, "threshold": 3.0}
    head = {"intercept": 0.1, "coefficients": [0.5, 0.8, 0.4, 0.6], "floor": 0.6}
    total = early + late
    native_hazard = (shape["rate"] * total) ** shape["power"]
    ratio = -np.expm1(-((shape["rate"] * late) ** shape["power"])) / -np.expm1(
        -((shape["rate"] * (1 - alpha) * total) ** shape["power"])
    )
    for strength in (0.0, 0.5, 1.0):
        hazard = native_hazard * ((1 - strength) + strength * ratio)
        exposure = hazard ** (1 / shape["power"]) / shape["rate"]
        matrix = np.c_[np.expm1(-hazard), np.logaddexp(0, np.log1p(exposure) - shape["threshold"]) ** 2]
        expected = head["floor"] + np.exp(head["intercept"] + matrix @ np.asarray(head["coefficients"]))
        specification = {"shape": shape, "head": head, "parameters": {"lambda": strength}}
        oracle_matrix = stage3._RUNNER.stage2.oracle.design(early, late, alpha, shape, strength)
        np.testing.assert_allclose(oracle_matrix, matrix, atol=1e-15, rtol=1e-14)
        raw_truth = stage3._RUNNER._FLOW.stage2.SyntheticTruth(specification, alpha)
        np.testing.assert_allclose(raw_truth.predict(early, late), expected, atol=1e-15, rtol=1e-14)
        fitter = stage3._RUNNER._FITTER
        fitted_matrix, _ = fitter.models.design_and_offset(
            early, late, alpha, shape, stage3.FAMILIES[1], {"lambda": strength}
        )
        np.testing.assert_allclose(fitted_matrix, matrix, atol=1e-15, rtol=1e-14)
        if strength == 1.0:
            np.testing.assert_array_equal(oracle_matrix[0, :2], np.zeros(2))
            np.testing.assert_allclose(oracle_matrix[0, 2:], np.logaddexp(0, -3.0) ** 2, atol=0, rtol=0)
    assert stage3._RUNNER._FLOW._BACKEND is stage3._RUNNER._FITTER._RUNNER
    assert stage3.physical._FITTER.models.MODEL_NAMES == ("aggregate_mariner", "late_refresh_hazard")
    assert stage3.stage2.OUTPUT != stage3.OUTPUT
    assert stage3._RUNNER.stage2.OUTPUT == stage3.stage2.OUTPUT


def test_effective_family_gate_keeps_signed_regret_and_physical_upper_endpoint() -> None:
    native, temporal = stage3.FAMILIES
    scores = {
        family: {
            "clean_actual_tied_rmse": 0.01,
            "max_positive_predicted_gain": 0.0,
            "clean_gain_rmse": 0.001,
            "clean_gain_rms": 0.04,
        }
        for family in stage3.FAMILIES
    }
    raw = {
        family: {
            "predicted_gain_same_aggregate": 0.0,
            "phase_gain_overprediction": 0.0,
            "regret_vs_truth_numerical_incumbent": -0.004,
        }
        for family in stage3.FAMILIES
    }
    fits = {native: SimpleNamespace(parameters={}), temporal: SimpleNamespace(parameters={"lambda": 1.0})}
    raw[temporal]["regret_vs_truth_numerical_incumbent"] = -0.002
    checks = stage3.qualification_checks({"name": "toy", "truth_world": "null"}, scores, raw, fits, stage3.GATES)
    keyed = {row["gate"]: row for row in checks if row["family"] == temporal}
    assert not keyed["kinetic_boundary"]["passed"]
    assert not keyed["raw_regret_vs_aggregate"]["passed"]
    assert keyed["raw_regret_vs_aggregate"]["value"] == -0.002
    assert keyed["raw_regret_vs_aggregate"]["limit"] == -0.003
    fits[temporal].parameters["lambda"] = 0.5
    raw[temporal]["regret_vs_truth_numerical_incumbent"] = -0.004
    scores[native]["clean_gain_rmse"] = 0.04
    scores[native]["clean_actual_tied_rmse"] = 0.38
    raw[native]["phase_gain_overprediction"] = 10.0
    positive = stage3.qualification_checks(
        {"name": "positive", "truth_world": "positive"}, scores, raw, fits, stage3.GATES
    )
    assert all(row["passed"] and row["family"] == temporal for row in positive)


def test_noisy_prepare_requires_separate_identity_bound_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(stage3, "OUTPUT", tmp_path)
    monkeypatch.setattr(stage3, "registration", lambda: {})
    current = {"sources": {"toy": "first"}, "environment": {}}
    monkeypatch.setattr(stage3, "source_identity", lambda: current)
    (tmp_path / "registration.json").write_text("{}\n")
    (tmp_path / "execution_identity.json").write_text(json.dumps(current))
    approval = {
        "scope": "effective_harm_stage2_noiseless",
        "registration_sha256": stage3.benchmark.sha256(tmp_path / "registration.json"),
        "execution_identity_sha256": stage3.benchmark.sha256(tmp_path / "execution_identity.json"),
    }
    (tmp_path / "execution_approval.json").write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="Separate exact effective-harm stage-three"):
        stage3.prepare()
    approval["scope"] = "effective_harm_stage3_paired_noise_and_raw_qualification"
    (tmp_path / "execution_approval.json").write_text(json.dumps(approval))
    assert stage3.execution_identity() == current
    current["sources"]["toy"] = "changed"
    with pytest.raises(ValueError, match="Frozen effective-harm stage-three"):
        stage3.prepare()
