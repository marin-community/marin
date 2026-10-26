# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Matching-law toy recovery and input-isolation checks; no archived outcomes."""

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_mean_coordinate_oracle_20261015 as runner


@pytest.mark.parametrize("family", runner.FAMILIES)
@pytest.mark.parametrize("strength", (0.0, 0.5))
def test_matching_law_oracle_recovers_each_world_and_keeps_all_candidates(family: str, strength: float) -> None:
    module = runner.ORACLES[family]
    rng = np.random.default_rng(32)
    total = rng.uniform(0.05, 20.0, (90, 2))
    late = total * rng.uniform(0.01, 0.99, total.shape)
    early, alpha = total - late, 0.8
    labels = np.arange(len(early)) % 3
    head = runner.mariner.Head(0.2, np.array([0.8, 1.1, 0.3, 0.7]), 0.6)
    observed = head.predict(module.design(early, late, alpha, runner.SHAPE, strength))
    query_total = rng.uniform(0.05, 20.0, (10, 2))
    query_last = query_total * rng.uniform(0.01, 0.99, query_total.shape)
    query_last[:3] = 0
    query_early = np.r_[query_total - query_last, alpha * query_total, alpha * query_total]
    query_late = np.r_[query_last, (1 - alpha) * query_total, (1 - alpha) * query_total]
    clean = head.predict(module.design(query_early, query_late, alpha, runner.SHAPE, strength))
    fitted = module.oracle_fit(early, late, observed, labels, query_early, query_late, alpha, runner.SHAPE, 0.6)
    diagnostic = module.oracle_diagnostic(fitted, clean, 10, 10)
    assert diagnostic["passed"]
    assert diagnostic["selected_parameters"] == {"lambda": strength}
    assert diagnostic["query_max_abs_error"] < 1e-11
    assert [item["parameters"]["lambda"] for item in fitted["profile"]] == list(runner.law.STRENGTHS)
    for candidate in fitted["profile"]:
        assert len(candidate["inner_heads"]) == 3
        np.testing.assert_allclose(
            candidate["inner_rmse"],
            np.sqrt(np.mean((np.asarray(candidate["inner_predictions"]) - observed) ** 2)),
            atol=1e-15,
        )
        held_rows = []
        for inner in candidate["inner_heads"]:
            assert set(inner["train_indices"]).isdisjoint(inner["held_indices"])
            held_rows.extend(inner["held_indices"])
        assert sorted(held_rows) == list(range(len(early)))
    endpoint = fitted["profile"][-1]
    endpoint_head = runner.mariner.Head(
        endpoint["head"]["intercept"], np.asarray(endpoint["head"]["coefficients"]), endpoint["head"]["floor"]
    )
    # Lambda1 loses all-early benefit but keeps its physical dose-dependent harm.
    expected = runner.mariner.design_matrix(query_early[:3], runner.SHAPE)
    expected[:, :2] = 0
    np.testing.assert_allclose(
        np.asarray(endpoint["query_predictions"])[:3], endpoint_head.predict(expected), atol=1e-14
    )


@pytest.mark.parametrize("family", runner.FAMILIES)
def test_selected_inner_head_excludes_its_held_labels_and_query_geometry(family: str) -> None:
    module = runner.ORACLES[family]
    rng = np.random.default_rng(33)
    early, late = rng.uniform(0.01, 12.0, (60, 2)), rng.uniform(0.01, 4.0, (60, 2))
    head = runner.mariner.Head(0.2, np.array([0.8, 1.1, 0.3, 0.7]), 0.6)
    strength, alpha = 0.5, 0.8
    observed = head.predict(module.design(early, late, alpha, runner.SHAPE, strength))
    labels = np.arange(len(early)) % 3
    original = module.oracle_fit(early, late, observed, labels, early[:3], late[:3], alpha, runner.SHAPE, 0.6)
    altered = observed.copy()
    altered[labels == 0] += 0.01
    changed = module.oracle_fit(early, late, altered, labels, early[:3], late[:3], alpha, runner.SHAPE, 0.6)
    query_changed = module.oracle_fit(
        early, late, observed, labels, early[:3] * 2, late[:3] * 3, alpha, runner.SHAPE, 0.6
    )
    changed_other_fold = False
    for first, second, query in zip(original["profile"], changed["profile"], query_changed["profile"], strict=True):
        assert first["inner_heads"][0]["head"] == second["inner_heads"][0]["head"]
        changed_other_fold |= first["inner_heads"][1]["head"] != second["inner_heads"][1]["head"]
        assert first["head"] == query["head"]
        assert first["inner_heads"] == query["inner_heads"]
        assert first["inner_predictions"] == query["inner_predictions"]
    assert changed_other_fold
    assert original["selected_parameter_index"] == query_changed["selected_parameter_index"]


def test_each_sensitivity_uses_its_law_and_canonical_oracle_is_unchanged() -> None:
    early, late = np.array([[1.0, 2.0], [5.0, 1.0], [10.0, 4.0]]), np.array([[0.2, 1.0], [0.1, 2.0], [3.0, 0.1]])
    head = runner.mariner.Head(0.2, np.array([0.8, 1.1, 0.3, 0.7]), 0.6)
    alpha, strength, step = 0.8, 0.5, 1e-4
    matrices = []
    for family, module in runner.ORACLES.items():
        matrix = module.design(early, late, alpha, runner.SHAPE, strength)
        expected = runner.law.design_and_offset(early, late, alpha, runner.SHAPE, family, {"lambda": strength})[0]
        np.testing.assert_array_equal(matrix, expected)
        matrices.append(matrix)
        report = module.local_sensitivity(early, late, alpha, strength, head, step)
        upper = head.predict(module.design(early, late, alpha, runner.SHAPE, strength + step))
        lower = head.predict(module.design(early, late, alpha, runner.SHAPE, strength - step))
        np.testing.assert_array_equal(report["derivatives"], (upper - lower) / ((strength + step) - (strength - step)))
    assert np.max(abs(matrices[0] - matrices[1])) > 0.01
    physical = runner.physical.design(early, late, alpha, runner.SHAPE, strength)
    expected_physical = runner.law.refresh.design_and_offset(
        early, late, alpha, runner.SHAPE, "late_refresh_hazard", {"lambda": strength}
    )[0]
    np.testing.assert_array_equal(physical, expected_physical)
    assert np.max(abs(physical - matrices[0])) > 0.01


def test_draft_cannot_freeze_or_read_scientific_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "OUTPUT", tmp_path)
    (tmp_path / "registration.json").write_text(json.dumps({"candidate_ids": runner.CANDIDATES, "status": "draft"}))
    for stage in (runner.freeze, runner.preflight, runner.oracle):
        with pytest.raises(ValueError, match="must register"):
            stage()
    assert not (tmp_path / "execution_identity.json").exists()
    assert not (tmp_path / "preflight.json").exists()
    assert not (tmp_path / "oracle_summary.json").exists()
