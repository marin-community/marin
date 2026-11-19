# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.rl.skyrl import SkyRLRun

from experiments.post_training.long_loop.rl_selection import validate_full_selection, validate_selection_receipts


def test_selected_export_requires_complete_five_step_winning_history():
    history = [{"step": step, "reward": 0.5, "attempted": 8, "scored": 8} for step in range(1, 7)]
    best = {
        "model_format": "native",
        "step": 5,
        "ema_window": 5,
        "ema_alpha": 1 / 3,
        "attempted": 8,
        "scored": 8,
        "accuracy": 0.5,
    }
    assert validate_selection_receipts(best, history, 5) == best
    with pytest.raises(ValueError, match="highest EMA"):
        validate_selection_receipts(best | {"step": 6}, history, 6)
    with pytest.raises(ValueError, match="validation window"):
        validate_selection_receipts(best, history[1:], 5)
    with pytest.raises(ValueError, match="coverage"):
        validate_selection_receipts(best, [row | {"scored": 7} if row["step"] == 1 else row for row in history], 5)
    with pytest.raises(ValueError, match="highest EMA"):
        validate_selection_receipts(best, [row | {"reward": 0.9} if row["step"] == 6 else row for row in history], 5)


def test_two_step_cadence_preserves_five_observation_selection():
    history = [{"step": step, "reward": 0.5, "attempted": 8, "scored": 8} for step in (2, 4, 6, 8, 10)]
    best = {
        "model_format": "native",
        "step": 10,
        "ema_window": 5,
        "ema_alpha": 1 / 3,
        "evaluation_interval": 2,
        "attempted": 8,
        "scored": 8,
        "accuracy": 0.5,
    }
    assert validate_selection_receipts(best, history, 10) == best
    with pytest.raises(ValueError, match="validation window"):
        validate_selection_receipts(best, history[1:], 10)


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
