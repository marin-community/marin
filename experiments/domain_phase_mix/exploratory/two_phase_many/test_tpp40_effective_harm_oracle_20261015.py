# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Private oracle binding, finite-menu recovery and approval-boundary behavior."""

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_effective_harm_oracle_20261015 as oracle


@pytest.mark.parametrize("strength", (0.0, 0.5))
def test_private_oracle_recovers_effective_harm_and_keeps_sink_candidate(strength: float) -> None:
    rng = np.random.default_rng(29)
    total = rng.uniform(0.05, 20.0, (90, 2))
    late = total * rng.uniform(0.01, 0.99, total.shape)
    early, alpha = total - late, 0.8
    labels = np.arange(len(total)) % 3
    head = oracle.mariner.Head(0.2, np.array([0.8, 1.1, 0.3, 0.7]), 0.6)
    observed = head.predict(oracle.design(early, late, alpha, oracle.SHAPE, strength))
    query_total = rng.uniform(0.05, 20.0, (10, 2))
    query_late_base = query_total * rng.uniform(0.01, 0.99, query_total.shape)
    query_late_base[:3] = 0
    query_early = np.r_[query_total - query_late_base, alpha * query_total, alpha * query_total]
    query_late = np.r_[query_late_base, (1 - alpha) * query_total, (1 - alpha) * query_total]
    clean = head.predict(oracle.design(query_early, query_late, alpha, oracle.SHAPE, strength))
    fitted = oracle.oracle_fit(early, late, observed, labels, query_early, query_late, alpha, oracle.SHAPE, 0.6)
    diagnostic = oracle.oracle_diagnostic(fitted, clean, 10, 10)
    assert diagnostic["passed"]
    assert diagnostic["selected_parameters"] == {"lambda": strength}
    assert diagnostic["query_max_abs_error"] < 1e-11
    assert [item["parameters"]["lambda"] for item in fitted["profile"]] == list(oracle.law.STRENGTHS)
    for candidate in fitted["profile"]:
        assert len(candidate["inner_heads"]) == 3
        for inner in candidate["inner_heads"]:
            assert set(inner["train_indices"]).isdisjoint(inner["held_indices"])
    sink = np.asarray(fitted["profile"][-1]["query_predictions"])
    np.testing.assert_array_equal(sink[:3], np.repeat(sink[0], 3))
    # The public physical-harm oracle remains physical after private rebinding.
    physical = oracle.physical.design(early, late, alpha, oracle.SHAPE, 0.5)
    expected = oracle.law.refresh.design_and_offset(
        early, late, alpha, oracle.SHAPE, "late_refresh_hazard", {"lambda": 0.5}
    )[0]
    np.testing.assert_array_equal(physical, expected)
    assert np.max(abs(oracle.design(early, late, alpha, oracle.SHAPE, 0.5)[:, 2:] - physical[:, 2:])) > 0.01


def test_sensitivity_and_inner_heads_use_effective_harm_training_design() -> None:
    rng = np.random.default_rng(20261029)
    early, late = rng.uniform(0.01, 12.0, (60, 2)), rng.uniform(0.01, 4.0, (60, 2))
    head = oracle.mariner.Head(0.2, np.array([0.8, 1.1, 0.3, 0.7]), 0.6)
    strength, alpha, step = 0.5, 0.8, 1e-4
    report = oracle.local_sensitivity(early, late, alpha, strength, head, step)
    upper = head.predict(oracle.design(early, late, alpha, oracle.SHAPE, strength + step))
    lower = head.predict(oracle.design(early, late, alpha, oracle.SHAPE, strength - step))
    denominator = (strength + step) - (strength - step)
    np.testing.assert_array_equal(report["derivatives"], (upper - lower) / denominator)
    physical = oracle.physical.local_sensitivity(early, late, alpha, strength, head, step)
    assert np.max(abs(np.asarray(report["derivatives"]) - physical["derivatives"])) > 0.01
    observed = head.predict(oracle.design(early, late, alpha, oracle.SHAPE, strength))
    labels = np.arange(len(early)) % 3
    original = oracle.oracle_fit(early, late, observed, labels, early[:3], late[:3], alpha, oracle.SHAPE, 0.6)
    altered = observed.copy()
    altered[labels == 0] += 0.01
    changed = oracle.oracle_fit(early, late, altered, labels, early[:3], late[:3], alpha, oracle.SHAPE, 0.6)
    for first, second in zip(original["profile"], changed["profile"], strict=True):
        assert first["inner_heads"][0]["head"] == second["inner_heads"][0]["head"]


def test_draft_and_physical_oracle_approval_do_not_authorize_effective_harm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(oracle, "OUTPUT", tmp_path)
    (tmp_path / "registration.json").write_text(json.dumps({"candidate_id": "TP40-029", "status": "draft"}))
    with pytest.raises(ValueError, match="must register"):
        oracle._ORACLE.freeze()
    with pytest.raises(ValueError, match="must register"):
        oracle._ORACLE.preflight()
    assert not (tmp_path / "execution_identity.json").exists()
    monkeypatch.setattr(oracle, "registration", lambda: {})
    identity = {"sources": {"toy": "fixed"}, "environment": {}}
    monkeypatch.setattr(oracle, "source_identity", lambda: identity)
    (tmp_path / "execution_identity.json").write_text(json.dumps(identity))
    approval = {
        "scope": "late_refresh_known_shape_floor_oracle",
        "registration_sha256": oracle.benchmark.sha256(tmp_path / "registration.json"),
        "execution_identity_sha256": oracle.benchmark.sha256(tmp_path / "execution_identity.json"),
    }
    (tmp_path / "execution_approval.json").write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="Separate exact effective-harm"):
        oracle._ORACLE.preflight()
    approval["scope"] = "effective_harm_known_shape_floor_oracle"
    (tmp_path / "execution_approval.json").write_text(json.dumps(approval))
    assert oracle.execution_identity() == identity
