# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Toy equation recovery and artifact guards; no archived covariates or outcomes."""

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_late_refresh_oracle_20261015 as oracle


@pytest.mark.parametrize("strength", [0.0, 0.5])
def test_toy_oracle_recovers_query_function_and_preserves_inner_training_locality(strength: float) -> None:
    rng = np.random.default_rng(14)
    total = rng.uniform(0.05, 8.0, (90, 2))
    late = total * rng.uniform(0.01, 0.99, total.shape)
    early, alpha = total - late, 0.8
    labels = np.arange(len(total)) % 3
    head = oracle.mariner.Head(0.2, np.array([0.8, 1.1, 0.03, 0.07]), 0.6)
    observed = head.predict(oracle.design(early, late, alpha, oracle.SHAPE, strength))
    query_total = rng.uniform(0.05, 8.0, (10, 2))
    query_late_base = query_total * rng.uniform(0.01, 0.99, query_total.shape)
    query_early = np.r_[query_total - query_late_base, alpha * query_total, alpha * query_total]
    query_late = np.r_[query_late_base, (1 - alpha) * query_total, (1 - alpha) * query_total]
    clean = head.predict(oracle.design(query_early, query_late, alpha, oracle.SHAPE, strength))
    fitted = oracle.oracle_fit(early, late, observed, labels, query_early, query_late, alpha, oracle.SHAPE, 0.6)
    diagnostic = oracle.oracle_diagnostic(fitted, clean, 10, 10)
    assert diagnostic["passed"]
    assert diagnostic["selected_parameters"] == {"lambda": strength}
    assert diagnostic["query_max_abs_error"] < 1e-11
    changed = observed.copy()
    changed[labels == 0] += 0.01
    perturbed = oracle.oracle_fit(early, late, changed, labels, query_early, query_late, alpha, oracle.SHAPE, 0.6)
    for original, altered in zip(fitted["profile"], perturbed["profile"], strict=True):
        record = original["inner_heads"][0]
        assert set(record["train_indices"]).isdisjoint(record["held_indices"])
        np.testing.assert_array_equal(labels[record["held_indices"]], 0)
        assert record["head"] == altered["inner_heads"][0]["head"]
        np.testing.assert_array_equal(
            np.asarray(original["inner_predictions"])[labels == 0],
            np.asarray(altered["inner_predictions"])[labels == 0],
        )


def test_query_ambiguity_blocks_oracle_even_when_selected_candidate_is_exact() -> None:
    clean = np.array([1.0, 1.1, 1.2])
    profile = [
        {"parameters": {"lambda": 0.0}, "inner_rmse": 0.0, "query_predictions": clean.tolist()},
        {"parameters": {"lambda": 0.5}, "inner_rmse": 1e-10, "query_predictions": [1.0, 1.1, 1.3]},
    ]
    fitted = {"selected_parameter_index": 0, "profile": profile}
    result = oracle.oracle_diagnostic(fitted, clean, 1, 1)
    assert not result["passed"]
    assert result["near_exact_cv_max_query_error"] == pytest.approx(0.1)
    profile[1]["query_predictions"] = clean.tolist()
    result = oracle.oracle_diagnostic(fitted, clean, 1, 1)
    assert result["passed"]
    assert not result["kinetics_unique_at_oracle_tolerance"]


def test_draft_protocol_cannot_freeze_or_generate_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oracle, "OUTPUT", tmp_path)
    (tmp_path / "registration.json").write_text(json.dumps({"candidate_id": "TP40-028", "status": "draft"}))
    with pytest.raises(ValueError, match="must register"):
        oracle.freeze()
    with pytest.raises(ValueError, match="must register"):
        oracle.preflight()
    assert not (tmp_path / "execution_identity.json").exists()
    assert not (tmp_path / "preflight.json").exists()


def test_completed_artifact_tampering_is_not_adopted(tmp_path: Path) -> None:
    frozen = {"sources": {"toy": "fixed"}}
    output = tmp_path / "preflight.json"
    output.write_text('{"valid":true}\n')
    receipt = tmp_path / "preflight_receipt.json"
    receipt.write_text(json.dumps({"identity": frozen, "outputs": {output.name: oracle.benchmark.sha256(output)}}))
    assert oracle.verified_record(receipt, frozen, (output.name,))["identity"] == frozen
    output.write_text('{"valid":false}\n')
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        oracle.verified_record(receipt, frozen, (output.name,))
