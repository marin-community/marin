# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_late_refresh_stage3_20261015 as stage3


def gate_inputs() -> tuple[dict, dict, dict]:
    scores = {
        family: {
            "clean_actual_tied_rmse": 0.001,
            "max_positive_predicted_gain": 0.0,
            "clean_gain_rmse": 0.01,
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
    fits = {
        stage3.FAMILIES[0]: SimpleNamespace(parameters={}),
        stage3.FAMILIES[1]: SimpleNamespace(parameters={"lambda": 0.5}),
    }
    return scores, raw, fits


def test_paired_noise_applies_only_to_training_without_changing_clean_truth() -> None:
    clean = np.linspace(0.8, 1.2, 280)
    other = clean**2
    original = clean.copy()
    train = np.arange(0, 280, 2)
    observed, noise = stage3.noisy_training_labels(clean, train, 20261015)
    other_observed, other_noise = stage3.noisy_training_labels(other, train, 20261015)
    np.testing.assert_array_equal(clean, original)
    np.testing.assert_array_equal(noise, other_noise)
    np.testing.assert_array_equal(observed[1::2], clean[1::2])
    np.testing.assert_array_equal(other_observed[1::2], other[1::2])
    np.testing.assert_allclose(observed[train] - clean[train], noise[train], atol=2e-16, rtol=0)
    np.testing.assert_allclose(other_observed[train] - other[train], noise[train], atol=2e-16, rtol=0)
    assert np.any(noise[1::2] != 0)


def test_null_gates_preserve_negative_regret_and_absolute_raw_gain_error() -> None:
    native, temporal = stage3.FAMILIES
    scores, raw, fits = gate_inputs()
    raw[native]["regret_vs_truth_numerical_incumbent"] = -0.004
    raw[temporal]["regret_vs_truth_numerical_incumbent"] = -0.002
    raw[temporal]["phase_gain_overprediction"] = -0.004
    raw[temporal]["predicted_gain_same_aggregate"] = 0.004
    checks = stage3.qualification_checks({"name": "null_seed", "truth_world": "null"}, scores, raw, fits, stage3.GATES)
    by_gate = {row["gate"]: row for row in checks if row["family"] == temporal}
    assert by_gate["raw_regret_vs_aggregate"]["value"] == -0.002
    assert by_gate["raw_regret_vs_aggregate"]["limit"] == -0.003
    assert not by_gate["raw_regret_vs_aggregate"]["passed"]
    assert by_gate["raw_gain_error"]["value"] == 0.004 and not by_gate["raw_gain_error"]["passed"]
    assert not by_gate["false_positive_gain"]["passed"]
    raw[temporal]["predicted_gain_same_aggregate"] = 0.0
    scores[temporal]["max_positive_predicted_gain"] = 0.004
    checks = stage3.qualification_checks({"name": "null_seed", "truth_world": "null"}, scores, raw, fits, stage3.GATES)
    assert not next(row for row in checks if row["family"] == temporal and row["gate"] == "false_positive_gain")[
        "passed"
    ]


def test_positive_control_misspecification_is_report_only_and_gain_improvement_is_strict() -> None:
    native, temporal = stage3.FAMILIES
    scores, raw, fits = gate_inputs()
    scores[native]["clean_actual_tied_rmse"] = 0.388
    raw[native]["phase_gain_overprediction"] = 9.0
    spec = {"name": "positive_seed", "truth_world": "positive"}
    tied = stage3.qualification_checks(spec, scores, raw, fits, stage3.GATES)
    assert not all(row["passed"] for row in tied)
    scores[temporal]["clean_gain_rmse"] = 0.009
    improved = stage3.qualification_checks(spec, scores, raw, fits, stage3.GATES)
    assert all(row["passed"] for row in improved)
    assert all(row["family"] == temporal for row in improved)
    fits[temporal].parameters["lambda"] = 1.0
    bounded = stage3.qualification_checks(spec, scores, raw, fits, stage3.GATES)
    assert not next(row for row in bounded if row["gate"] == "kinetic_boundary")["passed"]
    fits[temporal].parameters["lambda"] = 0.0
    assert next(
        row
        for row in stage3.qualification_checks(spec, scores, raw, fits, stage3.GATES)
        if row["gate"] == "kinetic_boundary"
    )["passed"]


def test_descriptive_comparisons_keep_absolute_errors_and_distinct_reference_deltas() -> None:
    current = {"tied_error": 0.04, "signed_regret": -0.002}
    noiseless = {"tied_error": 0.001, "signed_regret": -0.004}
    matched_null = {"tied_error": 0.006, "signed_regret": 0.001}
    values = stage3.comparison_values(current, noiseless, matched_null, ("tied_error", "signed_regret"))
    assert values["signed_regret"] == -0.002
    assert values["delta_vs_noiseless_signed_regret"] == 0.002
    assert values["delta_vs_paired_null_signed_regret"] == -0.003
    np.testing.assert_allclose(values["delta_vs_noiseless_tied_error"], 0.039, atol=1e-15, rtol=0)
    np.testing.assert_allclose(values["delta_vs_paired_null_tied_error"], 0.034, atol=1e-15, rtol=0)
    assert values["noiseless_same_family_tied_error"] == 0.001
    assert values["paired_null_same_family_tied_error"] == 0.006
