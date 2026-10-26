# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Small invented oracle/provenance tests; no frozen covariates or outcomes."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    run_tpp40_acquisition_retention_oracle_20261015 as runner,
)


def toy_policies(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    first, last = rng.uniform(0.01, 0.99, (2, count))
    scales = np.array([1.0, 26.0])
    return 0.8 * np.c_[first, 1 - first] * scales, 0.2 * np.c_[last, 1 - last] * scales


@pytest.mark.parametrize("clock", runner.CLOCKS)
def test_matching_oracle_recovery_and_full_profile_replay(clock: str, tmp_path: Path) -> None:
    early, late = toy_policies(30, 340)
    first, last = toy_policies(8, 341)
    total = first + last
    query0, query1 = np.vstack((first, 0.8 * total)), np.vstack((last, 0.2 * total))
    config = runner.feature_config(np.array([1.0, 26.0]), clock)
    cache = runner.models.BenefitCache(tmp_path / "features.sqlite")
    world = {"kind": "temporal", "clock": clock, "mu": 3.0}
    values, _ = runner.truth_values(early, late, 0.8, np.array(config.scales), world, cache)
    clean, _ = runner.truth_values(query0, query1, 0.8, np.array(config.scales), world, cache)
    fit = runner.oracle_fit(
        early, late, values, np.arange(30) % 3, query0, query1, 0.8, config, (0.0, 1.0, 3.0), 0.6, cache
    )
    assert fit["selected_mu"] == 3
    assert fit["conditional_nnls_solves"] == 12
    np.testing.assert_allclose(fit["profiles"][fit["selected_index"]]["query_predictions"], clean, atol=2e-11, rtol=0)
    for profile in fit["profiles"]:
        head = profile["head"]
        model = runner.models.mariner.Head(head["intercept"], np.array(head["coefficients"]), head["floor"])
        expected = runner.predict(model, runner.matrix(query0, query1, 0.8, config, profile["mu"], cache))
        np.testing.assert_array_equal(profile["query_predictions"], expected)
        assert profile["inner_rmse"] == pytest.approx(
            np.sqrt(np.mean((np.array(profile["inner_predictions"]) - values) ** 2)), abs=1e-15
        )
        assert sorted(i for part in profile["inner_heads"] for i in part["held"]) == list(range(30))
        assert all(set(part["train"]).isdisjoint(part["held"]) for part in profile["inner_heads"])


def test_inner_head_excludes_held_labels_and_selection_ignores_query_geometry(tmp_path: Path) -> None:
    early, late = toy_policies(24, 342)
    config = runner.feature_config(np.array([1.0, 26.0]), "equal_group")
    cache = runner.models.BenefitCache(tmp_path / "features.sqlite")
    world = {"kind": "temporal", "clock": "equal_group", "mu": 3.0}
    values, _ = runner.truth_values(early, late, 0.8, np.array(config.scales), world, cache)
    labels = np.arange(24) % 3
    original = runner.oracle_fit(early, late, values, labels, early[:3], late[:3], 0.8, config, (0.0, 3.0), 0.6, cache)
    altered = values.copy()
    altered[labels == 0] += 0.02
    changed = runner.oracle_fit(early, late, altered, labels, early[:3], late[:3], 0.8, config, (0.0, 3.0), 0.6, cache)
    other_queries = runner.oracle_fit(
        early, late, values, labels, early[:3] * 0.5, late[:3] * 2, 0.8, config, (0.0, 3.0), 0.6, cache
    )
    assert original["selected_index"] == other_queries["selected_index"]
    for base, shifted, queried in zip(original["profiles"], changed["profiles"], other_queries["profiles"], strict=True):
        assert base["inner_heads"][0]["head"] == shifted["inner_heads"][0]["head"]
        assert base["inner_heads"] == queried["inner_heads"]
        assert base["inner_predictions"] == queried["inner_predictions"]
        assert base["head"] == queried["head"]


def test_misspecified_null_has_zero_phase_effect_without_changing_head() -> None:
    early, late = toy_policies(15, 343)
    total = early + late
    world = next(world for world in runner.WORLDS if world["kind"] == "heterogeneous")
    first, record = runner.truth_values(early, late, 0.8, np.array([1.0, 26.0]), world, None)
    tied, _ = runner.truth_values(0.8 * total, 0.2 * total, 0.8, np.array([1.0, 26.0]), world, None)
    np.testing.assert_allclose(first, tied, rtol=0, atol=2e-15)
    np.testing.assert_allclose(record["per_bucket_rates"], 0.25 * np.exp([0.25, -0.25]), rtol=0, atol=1e-16)
    assert record["head"] == runner.HEAD


def test_floor_preflight_keeps_infeasibility_and_uses_only_training_rows() -> None:
    rows = pd.DataFrame({"row_id": [f"row{i}" for i in range(13)], "tied": [False] * 12 + [True]})
    parts = []
    for outer in range(5):
        train = [i for i in range(12) if i != outer]
        parts.extend(
            {"outer_fold": outer, "role": "train", "row_id": f"row{i}", "inner_fold": j % 3} for j, i in enumerate(train)
        )
        parts.extend(
            [
                {"outer_fold": outer, "role": "test", "row_id": f"row{outer}", "inner_fold": -1},
                {"outer_fold": outer, "role": "tied_test", "row_id": "row12", "inner_fold": -1},
            ]
        )
    folds = pd.DataFrame(parts)
    values = np.linspace(1.0, 1.05, 13)
    noise = {"clean": np.zeros(13), "noisy": np.linspace(-0.001, 0.001, 13)}
    checks = runner.preflight_records({"world": values}, noise, rows, folds)
    assert len(checks) == 40
    assert all(not check["gamma_feasible"] and check["floor_below_all"] for check in checks)
    assert all(12 not in check["row_indices"] for check in checks)
    changed = values.copy()
    changed[[0, 12]] = 50.0
    revised = runner.preflight_records({"world": changed}, noise, rows, folds)
    assert [check for check in checks if check["outer_fold"] == 0] == [
        check for check in revised if check["outer_fold"] == 0
    ]


def test_resume_verifies_outputs_and_preserves_corrupt_receipt(tmp_path: Path) -> None:
    (tmp_path / "values.json").write_text('{"value":1}\n')
    runner.finish(tmp_path, {"cell": 1}, {"identity": 2}, ("values.json",))
    receipt = (tmp_path / "complete.json").read_bytes()
    assert runner.completed(tmp_path, {"cell": 1}, {"identity": 2})
    (tmp_path / "values.json").write_text('{"value":2}\n')
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        runner.completed(tmp_path, {"cell": 1}, {"identity": 2})
    assert (tmp_path / "complete.json").read_bytes() == receipt


def test_draft_cannot_freeze_prepare_or_oracle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "OUTPUT", tmp_path)
    (tmp_path / "registration.json").write_text(json.dumps({"candidate_id": "TP40-034", "status": "draft"}))
    for command in (runner.freeze, runner.prepare, runner.oracle):
        with pytest.raises(ValueError, match="Must register"):
            command()
    assert not (tmp_path / "prepared").exists()
    assert not (tmp_path / "execution_identity.json").exists()
