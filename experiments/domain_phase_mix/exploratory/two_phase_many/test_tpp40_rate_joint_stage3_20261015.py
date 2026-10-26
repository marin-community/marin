# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_rate_joint_stage3_20261015 as stage3


def test_paired_noise_changes_only_training_labels_and_preserves_inputs() -> None:
    first = np.linspace(0.8, 1.2, 280)
    second = first**2
    saved_first, saved_second = first.copy(), second.copy()
    train = np.arange(0, 280, 2)
    observed_first, noise_first = stage3.noisy_training_labels(first, train, 20261015)
    observed_second, noise_second = stage3.noisy_training_labels(second, train, 20261015)
    np.testing.assert_array_equal(noise_first, noise_second)
    np.testing.assert_array_equal(first, saved_first)
    np.testing.assert_array_equal(second, saved_second)
    np.testing.assert_array_equal(observed_first[1::2], first[1::2])
    np.testing.assert_array_equal(observed_second[1::2], second[1::2])
    np.testing.assert_allclose(observed_first[train] - first[train], noise_first[train], atol=2e-16, rtol=0)
    np.testing.assert_allclose(observed_second[train] - second[train], noise_first[train], atol=2e-16, rtol=0)
    assert np.any(noise_first[1::2] != 0)


def gate_inputs() -> tuple[dict, dict, dict, dict]:
    scores = {
        family: {
            "clean_actual_tied_rmse": 0.001,
            "max_positive_predicted_gain": 0.0,
            "clean_gain_rmse": 0.002,
            "clean_gain_rms": 0.01,
        }
        for family in stage3.FAMILIES
    }
    raw = {
        family: {
            "predicted_gain_same_aggregate": 0.0,
            "phase_gain_overprediction": 0.0,
            "regret_vs_truth_numerical_incumbent": 0.0,
        }
        for family in stage3.FAMILIES
    }
    fits = {family: SimpleNamespace(parameters={"kappa": 0, "nu": 0}) for family in stage3.FAMILIES}
    gates = {
        "null_max_positive_raw_or_menu_sameaggregate_gain_bpb": 0.003,
        "raw_gain_absolute_error_max_bpb": 0.003,
        "raw_signed_regret_vs_best_control_extra_max_bpb": 0.001,
    }
    return scores, raw, fits, gates


def test_null_qualification_retains_signed_regret_and_absolute_gain_error() -> None:
    scores, raw, fits, gates = gate_inputs()
    raw[stage3.FAMILIES[1]]["regret_vs_truth_numerical_incumbent"] = -0.004
    raw[stage3.FAMILIES[3]]["regret_vs_truth_numerical_incumbent"] = -0.002
    raw[stage3.FAMILIES[2]]["phase_gain_overprediction"] = -0.004
    checks = stage3.qualification_checks({"name": "null_seed", "truth_world": "null"}, scores, raw, fits, gates)
    regret = next(item for item in checks if item["gate"] == "raw_regret_vs_best_control")
    gain = next(item for item in checks if item["family"] == stage3.FAMILIES[2] and item["gate"] == "raw_gain_error")
    assert regret["value"] == -0.002 and regret["limit"] == -0.003
    assert not regret["passed"]
    assert gain["value"] == 0.004 and not gain["passed"]
    assert len([item for item in checks if item["gate"] == "false_positive_gain"]) == 4


def test_positive_qualification_requires_strict_gain_improvement_but_not_control_recovery() -> None:
    scores, raw, fits, gates = gate_inputs()
    scores[stage3.FAMILIES[1]]["clean_actual_tied_rmse"] = 9.0
    raw[stage3.FAMILIES[2]]["phase_gain_overprediction"] = 9.0
    fits[stage3.FAMILIES[1]].parameters["kappa"] = 128
    spec = {"name": "positive_seed", "truth_world": "combined"}
    tied = stage3.qualification_checks(spec, scores, raw, fits, gates)
    assert not all(item["passed"] for item in tied)
    scores[stage3.FAMILIES[3]]["clean_gain_rmse"] = 0.001
    improved = stage3.qualification_checks(spec, scores, raw, fits, gates)
    assert all(item["passed"] for item in improved)
    assert all(item["family"] == stage3.FAMILIES[3] for item in improved)
