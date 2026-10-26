# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Invented-policy checks for the diagnostic extension's noise and truth paths."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    run_tpp40_acquisition_retention_oracle_extension_20261015 as runner,
)


@pytest.mark.parametrize("world", runner.oracle.WORLDS, ids=lambda world: world["name"])
def test_raw_truth_matches_frozen_law_and_tighter_replay_on_invented_policies(world: dict) -> None:
    alpha, scales = 0.8, np.array([1.0, 26.0])
    early = alpha * np.array([[0.8, 0.2], [0.25, 0.75], [0, 1.0]]) * scales
    late = (1 - alpha) * np.array([[0.3, 0.7], [0.25, 0.75], [1.0, 0]]) * scales
    predictor = runner.TruthPredictor(world, alpha, scales, runner.CONTRACT["search_numerics"], None)
    replay = replace(predictor, numerics=runner.CONTRACT["replay_numerics"])
    expected, _ = runner.oracle.truth_values(early, late, alpha, scales, world, None)
    actual = np.array([predictor(a, b) for a, b in zip(early, late, strict=True)])
    strict = np.array([replay(a, b) for a, b in zip(early, late, strict=True)])
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(strict, expected, rtol=0, atol=1e-11)
    if world["kind"] != "temporal":
        total = early + late
        tied = np.array([predictor(alpha * e, (1 - alpha) * e) for e in total])
        np.testing.assert_allclose(tied, expected, rtol=0, atol=1e-15)


def test_noise_applies_only_to_fit_rows_and_profile_scores_keep_clean_queries() -> None:
    rng = np.random.default_rng(3419)
    scales, alpha = np.array([1.0, 26.0]), 0.8
    first, last = rng.uniform(0.02, 0.98, (2, 30))
    early, late = alpha * np.c_[first, 1 - first] * scales, (1 - alpha) * np.c_[last, 1 - last] * scales
    world = runner.oracle.WORLDS[0]
    clean, _ = runner.oracle.truth_values(early, late, alpha, scales, world, None)
    untouched = clean.copy()
    train, held = np.arange(24), np.arange(24, 30)
    noise = rng.normal(0, 0.003, 30)
    altered = noise.copy()
    altered[held] = 10.0
    observed = runner.noisy_training(clean, noise, train)
    np.testing.assert_array_equal(observed, runner.noisy_training(clean, altered, train))
    total = early[held] + late[held]
    query0 = np.vstack((early[held], alpha * total, alpha * total[:1]))
    query1 = np.vstack((late[held], (1 - alpha) * total, (1 - alpha) * total[:1]))
    truth, _ = runner.oracle.truth_values(query0, query1, alpha, scales, world, None)
    fit = runner.oracle.oracle_fit(
        early[train],
        late[train],
        observed,
        train % 3,
        query0,
        query1,
        alpha,
        runner.oracle.feature_config(scales, "equal_group"),
        (0.0, 0.1),
        0.6,
    )
    selected = fit["profiles"][fit["selected_index"]]
    frame = pd.DataFrame(
        {
            "role": ["asymmetric"] * 6 + ["matched_tied"] * 6 + ["actual_tied"],
            "row_id": [*map(str, held), *map(str, held), "tie"],
        }
    )
    result = runner.scores(np.asarray(selected["query_predictions"]), truth, frame)
    assert result["query_rmse"] > 0
    assert result["actual_tied_maximum_absolute_error"] == pytest.approx(
        abs(selected["query_predictions"][-1] - truth[-1])
    )
    assert fit["conditional_nnls_solves"] == 8
    diagnostics = runner.profile_diagnostics(fit)
    assert diagnostics["runner_up_inner_rmse_gap"] >= 0
    assert diagnostics["inner_rmse_improvement_over_zero"] >= 0
    np.testing.assert_array_equal(clean, untouched)


def test_duplicate_parity_preserves_original_receipt_and_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "PRIOR", tmp_path)
    spec = {"outer_fold": 0, "noise": "clean", "world": {"name": "native_null"}}
    folder = tmp_path / "cells/native_null/aggregate_mariner"
    folder.mkdir(parents=True)
    fit = {"selected_mu": 0, "profiles": [{"query_predictions": [1.0, 2.0]}]}
    (folder / "fit.json").write_text(json.dumps(fit | {"spec": "original"}))
    (folder / "complete.json").write_text('{"preserved":true}\n')
    receipt = (folder / "complete.json").read_bytes()
    result = runner.duplicate_parity(spec, "aggregate_mariner", fit)
    assert result["duplicate_numerical_equal"]
    changed = fit | {"selected_mu": 0.1}
    with pytest.raises(ValueError, match="differs"):
        runner.duplicate_parity(spec, "aggregate_mariner", changed)
    assert (folder / "complete.json").read_bytes() == receipt


def test_menu_order_does_not_use_truth_and_retains_false_gain() -> None:
    frame = pd.DataFrame(
        {
            "role": ["asymmetric", "asymmetric", "matched_tied", "matched_tied", "actual_tied"],
            "row_id": ["b", "a", "b", "a", "tie"],
        }
    )
    predicted = np.array([1.0, 1.0, 1.02, 1.02, 1.1])
    truth = np.array([1.01, 1.03, 1.01, 1.03, 1.05])
    result = runner.scores(predicted, truth, frame)
    assert result["selected_row_id"] == "a"
    assert result["maximum_positive_predicted_gain"] == pytest.approx(0.02)
    assert result["selected_optimism"] == pytest.approx(0.03)
    assert result["gain_sign_count"] == 0 and result["gain_spearman"] is None
    reversed_result = runner.scores(predicted, truth[::-1], frame)
    assert result["frontier_indices"] == reversed_result["frontier_indices"]


def test_draft_cannot_execute_either_diagnostic_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "OUTPUT", tmp_path)
    (tmp_path / "registration.json").write_text(json.dumps({"candidate_id": "TP40-034", "status": "draft"}))
    for command in (runner.freeze, runner.run_oracle, runner.raw_truth):
        with pytest.raises(ValueError, match="Register"):
            command()
    assert not (tmp_path / "features.sqlite").exists()
    assert not (tmp_path / "raw_features.sqlite").exists()
