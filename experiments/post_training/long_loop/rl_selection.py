# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify the selected export against complete, consecutive validation evidence."""

import json
import math

from marin.rl.skyrl import SkyRLRun
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join

from vendor.skyrl.marinskyrl.checkpoint_paths import BEST_CHECKPOINT_FILENAME, trailing_validation_ema

VALIDATION_EMA_WINDOW = 5
VALIDATION_EMA_ALPHA = 2 / (VALIDATION_EMA_WINDOW + 1)


def validate_selection_receipts(best: dict, history: list[dict], selected_step: int) -> dict:
    """Reject incomplete coverage, changed lineage, or a nonwinning EMA selection."""
    if best["model_format"] != "native" or best["step"] != selected_step or selected_step < VALIDATION_EMA_WINDOW:
        raise ValueError("Full RL export must identify a trained five-step selection")
    if best["ema_window"] != VALIDATION_EMA_WINDOW or not math.isclose(best["ema_alpha"], VALIDATION_EMA_ALPHA):
        raise ValueError("Full RL requires trailing-five adjusted EMA with alpha=1/3")
    expected = best["attempted"]
    if expected < 1 or best["scored"] != expected:
        raise ValueError("Selected validation cohort is incomplete")
    rows = {row["step"]: row for row in history}
    if len(rows) != len(history):
        raise ValueError("Duplicate validation steps")
    scores = {}
    interval = best.get("evaluation_interval", 1)
    for step in sorted(rows):
        if step < VALIDATION_EMA_WINDOW * interval or step % interval:
            continue
        score = trailing_validation_ema(history, step, VALIDATION_EMA_WINDOW, interval=interval)
        if score is None:
            continue
        window = [rows[index] for index in range(step - (VALIDATION_EMA_WINDOW - 1) * interval, step + 1, interval)]
        if any(row["attempted"] != expected or row["scored"] != expected for row in window):
            raise ValueError("Validation cohort coverage changed")
        scores[step] = score
    if selected_step not in scores or not math.isclose(scores[selected_step], best["accuracy"], abs_tol=1e-12):
        raise ValueError("Selection score does not match its validation window")
    winner = max(scores, key=lambda step: (scores[step], -step))
    if winner != selected_step:
        raise ValueError("Export is not the highest EMA checkpoint (earliest on ties)")
    return best


def validate_full_selection(run: SkyRLRun) -> dict:
    """Read the durable native selection before admitting its HF export to eval."""
    root = run.checkpoint_root
    fs, key = filesystem_for(prefix_join(root, BEST_CHECKPOINT_FILENAME))
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
