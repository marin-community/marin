# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Toy noise pairing, raw-law parity and qualification-boundary behavior."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_mean_coordinate_stage3_20261015 as stage3


def test_both_family_worlds_share_noise_and_keep_queries_clean() -> None:
    train = np.arange(172)
    clean = np.linspace(0.8, 1.2, 280)
    for seed in stage3.SEEDS:
        expected = np.random.default_rng(seed).normal(0.0, 0.003, 280)
        for index, spec in enumerate(item for item in stage3.context_specs() if item["noise_seed"] == seed):
            truth = clean + index * 0.05
            observed, noise = stage3.noisy_training_labels(truth, train, spec["noise_seed"])
            np.testing.assert_array_equal(noise, expected)
            np.testing.assert_array_equal(observed[172:], truth[172:])
            np.testing.assert_allclose(observed[train] - truth[train], expected[train], atol=2e-16, rtol=0)


@pytest.mark.parametrize("family", stage3.FAMILIES)
def test_noisy_raw_truth_and_fitted_readout_preserve_matching_mean_and_physical_harm(family: str) -> None:
    spec = next(
        item
        for item in stage3.context_specs()
        if item["matching_family"] == family and item["truth_world"] == "positive"
    )
    runner = stage3.context_runner(spec)
    early = np.array([[8.0, 3.0], [0.0, 0.0], [8.0, 4.0], [2.0, 10.0]])
    late = np.array([[0.0, 0.0], [3.0, 2.0], [2.0, 1.0], [1.0, 0.5]])
    alpha, shape = 0.8, {"rate": 0.25, "power": 0.7, "threshold": 3.0}
    head = {"intercept": 35.0, "coefficients": [0.5, 0.8, 0.4, 0.6], "floor": 0.6}
    total = early + late
    hazard = (0.25 * total) ** 0.7
    native = -np.expm1(-hazard)
    late_hazard = (0.25 * late / (1 - alpha)) ** 0.7
    harm = np.logaddexp(0.0, np.log1p(total) - 3.0) ** 2
    for strength in (0.0, 0.5, 1.0):
        if family == "hazard_coordinate_mean":
            benefit = -np.expm1(-((1 - strength) * hazard + strength * late_hazard))
        else:
            benefit = (1 - strength) * native - strength * np.expm1(-late_hazard)
        matrix = np.c_[-benefit, harm]
        expected = 0.6 + np.exp(35.0 + matrix @ np.asarray(head["coefficients"]))
        truth = runner._FLOW.stage2.SyntheticTruth(
            {"shape": shape, "head": head, "parameters": {"lambda": strength}, "world": spec}, alpha
        )
        np.testing.assert_allclose(truth.predict(early, late), expected, atol=0.0, rtol=8e-15)
        assert np.min(truth.predict(early, late)) > np.exp(30)
        actual, _ = runner._FITTER.models.design_and_offset(early, late, alpha, shape, family, {"lambda": strength})
        np.testing.assert_allclose(actual, matrix, atol=1e-15, rtol=1e-14)
        if strength == 1:
            np.testing.assert_array_equal(actual[0, :2], np.zeros(2))
            np.testing.assert_array_equal(actual[0, 2:], harm[0])


@pytest.mark.parametrize("family", stage3.FAMILIES)
def test_signed_regret_and_mean_boundary_gates_remain_unchanged(family: str) -> None:
    spec = next(item for item in stage3.context_specs() if item["matching_family"] == family)
    runner = stage3.context_runner(spec)
    native = "aggregate_mariner"
    scores = {
        model: {
            "clean_actual_tied_rmse": 0.01,
            "max_positive_predicted_gain": 0.0,
            "clean_gain_rmse": 0.001,
            "clean_gain_rms": 0.04,
        }
        for model in (native, family)
    }
    raw = {
        model: {
            "predicted_gain_same_aggregate": 0.0,
            "phase_gain_overprediction": 0.0,
            "regret_vs_truth_numerical_incumbent": -0.004,
        }
        for model in (native, family)
    }
    fits = {native: SimpleNamespace(parameters={}), family: SimpleNamespace(parameters={"lambda": 1.0})}
    raw[family]["regret_vs_truth_numerical_incumbent"] = -0.002
    checks = runner.qualification_checks(spec, scores, raw, fits, stage3.GATES)
    keyed = {item["gate"]: item for item in checks if item["family"] == family}
    assert not keyed["kinetic_boundary"]["passed"]
    assert not keyed["raw_regret_vs_aggregate"]["passed"]
    assert keyed["raw_regret_vs_aggregate"]["value"] == -0.002
    assert keyed["raw_regret_vs_aggregate"]["limit"] == -0.003
    fits[family].parameters["lambda"] = 0.5
    raw[family]["regret_vs_truth_numerical_incumbent"] = -0.004
    raw[family]["predicted_gain_same_aggregate"] = 0.004
    assert not next(
        item
        for item in runner.qualification_checks(spec, scores, raw, fits, stage3.GATES)
        if item["family"] == family and item["gate"] == "false_positive_gain"
    )["passed"]
    positive = {**spec, "truth_world": "positive"}
    raw[native]["phase_gain_overprediction"] = 10.0
    scores[native]["clean_gain_rmse"] = 0.04
    scores[native]["clean_actual_tied_rmse"] = 0.38
    assert all(
        item["passed"] and item["family"] == family
        for item in runner.qualification_checks(positive, scores, raw, fits, stage3.GATES)
    )


def test_draft_with_pending_stage2_seal_cannot_freeze_or_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(stage3, "OUTPUT", tmp_path)
    (tmp_path / "registration.json").write_text(
        json.dumps(
            {
                "candidate_ids": stage3.stage2.CANDIDATES,
                "stage": 3,
                "status": "draft",
                "stage2_result_manifest_sha256": None,
                "stage2_audits": {},
            }
        )
    )
    for command in (stage3.freeze, stage3.prepare):
        with pytest.raises(ValueError):
            command()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["registration.json"]
