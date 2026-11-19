# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import asdict

import pytest
from marin.rl.skyrl import SkyRLRun

from experiments.post_training.long_loop.rl_selection import validate_full_selection, validate_selection_receipts
from vendor.skyrl.marinskyrl.checkpoint_paths import ValidationReceipt


def test_selected_export_uses_first_eligible_ema_and_preserves_earliest_tie():
    history = [{"step": step, "reward": 0.5, "attempted": 8, "scored": 8} for step in range(1, 7)]
    best = {
        "model_format": "native",
        "step": 1,
        "ema_window": 5,
        "ema_alpha": 1 / 3,
        "attempted": 8,
        "scored": 8,
        "expected_tasks": 8,
        "minimum_scored_tasks": 8,
        "eligible": True,
        "accuracy": 0.5,
    }
    assert validate_selection_receipts(best, history, 1) == best
    with pytest.raises(ValueError, match="highest EMA"):
        validate_selection_receipts(best | {"step": 6}, history, 6)
    with pytest.raises(ValueError, match="validation window"):
        validate_selection_receipts(best, history[1:], 1)
    with pytest.raises(ValueError, match="coverage"):
        validate_selection_receipts(best, [row | {"scored": 7} if row["step"] == 1 else row for row in history], 1)
    with pytest.raises(ValueError, match="highest EMA"):
        validate_selection_receipts(best, [row | {"reward": 0.9} if row["step"] == 6 else row for row in history], 1)


def test_two_step_cadence_selects_before_five_observations():
    history = [{"step": step, "reward": 0.5, "attempted": 8, "scored": 8} for step in (2, 4, 6, 8, 10)]
    best = {
        "model_format": "native",
        "step": 2,
        "ema_window": 5,
        "ema_alpha": 1 / 3,
        "evaluation_interval": 2,
        "attempted": 8,
        "scored": 8,
        "expected_tasks": 8,
        "minimum_scored_tasks": 8,
        "eligible": True,
        "accuracy": 0.5,
    }
    assert validate_selection_receipts(best, history, 2) == best
    with pytest.raises(ValueError, match="validation window"):
        validate_selection_receipts(best, history[1:], 2)


def test_ninety_percent_coverage_skips_ineligible_gaps_and_accepts_final_odd_step():
    history = [
        {"step": 2, "reward": 0.0, "attempted": 120, "scored": 119},
        {"step": 5, "reward": 1.0, "attempted": 120, "scored": 108},
    ]
    best = {
        "model_format": "native",
        "step": 5,
        "ema_window": 5,
        "ema_alpha": 1 / 3,
        "evaluation_interval": 2,
        "attempted": 120,
        "scored": 108,
        "expected_tasks": 120,
        "minimum_scored_tasks": 108,
        "eligible": True,
        "accuracy": 0.6,
    }
    assert validate_selection_receipts(best, history, 5) == best
    with pytest.raises(ValueError, match="coverage"):
        validate_selection_receipts(best | {"scored": 107}, history, 5)


def test_full_selection_requires_matching_eligibility_receipts_and_committed_payload(tmp_path):
    root = tmp_path / "native"
    checkpoint = root / "global_step_2"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.pt").write_bytes(b"committed")
    best = {
        "model_format": "native",
        "step": 2,
        "model_uri": str(checkpoint / "policy"),
        "ema_window": 5,
        "ema_alpha": 1 / 3,
        "evaluation_interval": 2,
        "attempted": 120,
        "scored": 108,
        "expected_tasks": 120,
        "minimum_scored_tasks": 108,
        "eligible": True,
        "accuracy": 1.0,
    }
    (root / "best_checkpoint.json").write_text(json.dumps(best))
    (root / "validation_selection_history.json").write_text(
        json.dumps(
            [
                {"step": 2, "reward": 1.0, "attempted": 120, "scored": 108},
            ]
        )
    )
    receipt = ValidationReceipt(
        2,
        120,
        108,
        120,
        108,
        12,
        True,
        "best_checkpoint_committed",
        1.0,
        ema_score=1.0,
        improved=True,
        checkpoint_uri=best["model_uri"],
    )
    ledger = root / "validation_selection_receipts.json"
    ledger.write_text(json.dumps([asdict(receipt)]))
    run = SkyRLRun(
        path=str(tmp_path / "run"),
        hf_model_uri=str(tmp_path / "exports/global_step_2/policy"),
        global_step=2,
        tokenizer_uri="fixture",
        tokenizer_revision="a" * 40,
        checkpoint_root=str(root),
        draft_checkpoint_root=None,
        terminal_manifest_uri=str(tmp_path / "terminal.json"),
        iris_job_id="/fixture/rl",
    )
    assert validate_full_selection(run) == best
    ledger.write_text(json.dumps([{**asdict(receipt), "eligible": False}]))
    with pytest.raises(ValueError, match="receipts do not match"):
        validate_full_selection(run)
    ledger.write_text(json.dumps([{**asdict(receipt), "scored_count": 109}]))
    with pytest.raises(ValueError, match="cohort or threshold changed"):
        validate_full_selection(run)
    ledger.write_text(json.dumps([asdict(receipt)]))
    (checkpoint / "trainer_state.pt").unlink()
    with pytest.raises(ValueError, match="not committed"):
        validate_full_selection(run)


def test_final_native_fallback_requires_the_committed_final_marker(tmp_path):
    root = tmp_path / "native"
    checkpoint = root / "global_step_2"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.pt").write_bytes(b"committed trainer state")
    marker = root / "latest_ckpt_global_step.txt"
    marker.write_text("2")
    run = SkyRLRun(
        path=str(tmp_path / "run"),
        hf_model_uri=str(tmp_path / "exports/global_step_2/policy"),
        global_step=2,
        tokenizer_uri="fixture",
        tokenizer_revision="a" * 40,
        checkpoint_root=str(root),
        draft_checkpoint_root=None,
        terminal_manifest_uri=str(tmp_path / "terminal.json"),
        iris_job_id="/fixture/rl",
    )
    receipt = validate_full_selection(run)
    assert receipt == {"selection": "final_native_not_best", "step": 2, "model_uri": str(checkpoint / "policy")}
    marker.write_text("1")
    with pytest.raises(ValueError, match="No committed"):
        validate_full_selection(run)
