# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Invented-data tests for shared controls and leakage-safe smoke diagnostics."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    run_tpp40_acquisition_retention_stage2_20261015 as runner,
)


def native_fit() -> runner.models.JointFit:
    config = runner.oracle.feature_config(np.array([1.0, 26.0]), "equal_group")
    selected = runner.models.frozen.JointFit(
        "aggregate_mariner",
        0.8,
        {"rate": 0.25, "power": 0.7, "threshold": 3.0},
        {},
        0.01,
        2.5,
        None,
        0.1,
        runner.models.mariner.Head(0.5, np.array([1.0, 2.0, 0.1, 0.1]), 0.6),
        ({"stage": "test", "inner_rmse": 0.1},),
        {"selected_from_failed_refinement": False, "refinement_results": [{"success": True}]},
    )
    return runner.models.JointFit(selected, config)


def test_native_clock_clone_preserves_whole_fit_and_predictions() -> None:
    original = native_fit()
    pooled = runner.oracle.feature_config(np.array([1.0, 26.0]), "global_pool")
    rebound = runner.native_for_clock(original, pooled)
    assert rebound.selected is original.selected
    assert original.config.clock == "equal_group" and rebound.config.clock == "global_pool"
    early, late = np.array([[0.3, 2.0], [0.7, 1.0]]), np.array([[0.2, 0.5], [0.1, 3.0]])
    np.testing.assert_array_equal(rebound.predict(early, late), original.predict(early, late))
    restored = runner.models.JointFit.from_json(rebound.to_json())
    assert restored.selected.to_json() == original.selected.to_json()
    with pytest.raises(ValueError, match="exposure and numerical"):
        runner.native_for_clock(original, runner.oracle.feature_config(np.array([2.0, 26.0]), "global_pool"))


def test_menu_order_uses_predictions_and_stable_identity_not_truth() -> None:
    frame = pd.DataFrame(
        {
            "role": ["matched_tied", "asymmetric", "asymmetric", "actual_tied", "matched_tied"],
            "row_id": ["b", "b", "a", "z", "a"],
        }
    )
    predicted = np.array([1.0, 1.0, 1.0, 2.0, 3.0])
    truth = np.array([1.1, 1.2, 1.05, 0.5, 2.5])
    score = runner.menu_scores(predicted, truth, frame)
    assert score["selected_row_id"] == "a" and score["selected_role"] == "asymmetric"
    assert score["frontier_indices"] == [2]
    assert score["selected_optimism"] == pytest.approx(0.05)
    assert score["regret_at_1"] == pytest.approx(0.55)
    assert score["regret_at_3"] == pytest.approx(0.55)
    assert score["regret_at_5"] == 0
    revised = runner.menu_scores(predicted, truth[::-1], frame)
    assert revised["frontier_indices"] == score["frontier_indices"]
    assert revised["selected_row_id"] == score["selected_row_id"]
    assert score["calibration_bias"] == -score["mean_optimism"]


def test_selected_inner_floors_and_heads_exclude_held_labels() -> None:
    fit = native_fit()
    rng = np.random.default_rng(3412)
    first, last = rng.uniform(0.01, 0.99, (2, 24))
    early = 0.8 * np.c_[first, 1 - first] * np.array([1.0, 26.0])
    late = 0.2 * np.c_[last, 1 - last] * np.array([1.0, 26.0])
    values, labels = fit.predict(early, late), np.arange(24) % 3
    base, heads = runner.selected_inner(fit, early, late, values, labels, None)
    shifted = values.copy()
    shifted[labels == 0] += 0.5
    changed, other = runner.selected_inner(fit, early, late, shifted, labels, None)
    assert heads[0] == other[0]
    np.testing.assert_array_equal(base[labels == 0], changed[labels == 0])
    assert heads[1]["head"]["floor"] != other[1]["head"]["floor"]
    assert sorted(i for record in heads for i in record["held"]) == list(range(24))


def test_null_gate_rejects_invented_phase_gain_despite_good_average_fit() -> None:
    fit = native_fit()
    frame = pd.DataFrame(
        {"role": ["asymmetric"] * 2 + ["matched_tied"] * 2 + ["actual_tied"], "row_id": ["a", "b", "a", "b", "tie"]}
    )
    truth = np.array([1.0, 1.1, 1.0, 1.1, 1.05])
    predicted = truth + np.array([-0.002, 0, 0.002, 0, 0])
    score = runner.diagnostic_scores(fit, predicted, truth, frame) | {
        "query_clipping_count": 0,
        "training_deficit_clip_count": 0,
        "inner_training_deficit_clip_count": 0,
        "inner_prediction_clip_count": 0,
    }
    checks = runner.gate_checks(runner.oracle.WORLDS[0], "equal_group", score, score)
    failed = {item["name"] for item in checks if not item["passed"]}
    assert "exact_null_absolute_gain" in failed and "null_maximum_positive_gain" in failed
    assert "asymmetric_rmse" not in failed and "actual_tied_rmse" not in failed
    assert score["gain_spearman"] is None and score["gain_sign_accuracy"] is None


def test_good_response_recovery_cannot_qualify_an_unresolved_numerical_path() -> None:
    fit = native_fit()
    frame = pd.DataFrame({"role": ["asymmetric", "matched_tied", "actual_tied"], "row_id": ["a", "a", "tie"]})
    truth = np.array([1.0, 1.0, 1.1])
    clean = runner.diagnostic_scores(fit, truth, truth, frame) | {
        "query_clipping_count": 0,
        "training_deficit_clip_count": 0,
        "inner_training_deficit_clip_count": 0,
        "inner_prediction_clip_count": 0,
    }
    assert all(item["passed"] for item in runner.gate_checks(runner.oracle.WORLDS[0], "equal_group", clean, clean))
    for failure in (
        "query_clipping_count",
        "training_deficit_clip_count",
        "inner_training_deficit_clip_count",
        "inner_prediction_clip_count",
        "selected_from_failed_refinement",
        "failed_refinements",
    ):
        invalid = clean | {failure: 1}
        checks = runner.gate_checks(runner.oracle.WORLDS[0], "equal_group", invalid, clean)
        assert {item["name"] for item in checks if not item["passed"]} == {f"unresolved_{failure}"}


def test_support_distance_is_in_phase_weights_not_bucket_epochs() -> None:
    early = np.array([[0.2, 15.6]])
    late = np.array([[0.15, 1.3]])
    query0, query1 = early + np.array([0.08, -2.08]), late.copy()
    distance = runner.support_distance(early, late, query0, query1, np.array([1.0, 26.0]), 0.8)
    np.testing.assert_allclose(distance, [np.sqrt(0.02)], atol=1e-15, rtol=0)


def test_draft_never_touches_prior_truth_or_creates_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "OUTPUT", tmp_path)
    (tmp_path / "registration.json").write_text(json.dumps({"status": "draft", "candidate_id": "TP40-034"}))
    for command in (runner.freeze, runner.prepare, runner.run):
        with pytest.raises(ValueError, match="Register"):
            command()
    assert not (tmp_path / "features.sqlite").exists()
    assert not (tmp_path / "prepared").exists()
