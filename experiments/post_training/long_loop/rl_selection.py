# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify the selected export against complete, consecutive validation evidence."""

import json
import math

from marin.rl.skyrl import SkyRLRun
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join


def validate_selection_receipts(best: dict, history: list[dict], selected_step: int) -> dict:
    """Reject incomplete coverage, changed lineage, or a nonwinning EMA selection."""
    if best["model_format"] != "native" or best["step"] != selected_step or selected_step < 5:
        raise ValueError("Full RL export must identify a trained five-step selection")
    if best["ema_window"] != 5 or not math.isclose(best["ema_alpha"], 1 / 3):
        raise ValueError("Full RL requires trailing-five adjusted EMA with alpha=1/3")
    expected = best["attempted"]
    if expected < 1 or best["scored"] != expected:
        raise ValueError("Selected validation cohort is incomplete")
    rows = {row["step"]: row for row in history}
    if len(rows) != len(history):
        raise ValueError("Duplicate validation steps")
    scores = {}
    interval = best.get("evaluation_interval", 1)
    weights = [(2 / 3) ** age for age in reversed(range(5))]
    for step in sorted(rows):
        window = [rows.get(index) for index in range(step - 4 * interval, step + 1, interval)]
        if step < 5 * interval or step % interval or any(row is None for row in window):
            continue
        if any(row["attempted"] != expected or row["scored"] != expected for row in window):
            raise ValueError("Validation cohort coverage changed")
        rewards = [row["reward"] for row in window]
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in rewards):
            raise ValueError("Invalid normalized validation reward")
        scores[step] = sum(value * weight for value, weight in zip(rewards, weights, strict=True)) / sum(weights)
    if selected_step not in scores or not math.isclose(scores[selected_step], best["accuracy"], abs_tol=1e-12):
        raise ValueError("Selection score does not match its validation window")
    winner = max(scores, key=lambda step: (scores[step], -step))
    if winner != selected_step:
        raise ValueError("Export is not the highest EMA checkpoint (earliest on ties)")
    return best


def validate_full_selection(run: SkyRLRun) -> dict:
    """Read the durable native selection before admitting its HF export to eval."""
    root = run.checkpoint_root
    fs, key = filesystem_for(prefix_join(root, "best_checkpoint.json"))
    best = json.loads(fs.cat_file(key))
    fs, key = filesystem_for(prefix_join(root, "validation_selection_history.json"))
    history = json.loads(fs.cat_file(key))
    if run.global_step is None or not run.hf_model_uri:
        raise ValueError("Full RL result has no selected HF export")
    if best["model_uri"] != prefix_join(root, f"global_step_{run.global_step}/policy"):
        raise ValueError("Selected native checkpoint lineage differs from this run")
    if not str(StoragePath(run.hf_model_uri)).endswith(f"/global_step_{run.global_step}/policy"):
        raise ValueError("HF export step differs from selected native checkpoint")
    return validate_selection_receipts(best, history, run.global_step)
