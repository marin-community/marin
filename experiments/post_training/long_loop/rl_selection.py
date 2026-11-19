# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify the selected export against eligible validation and native-commit receipts."""

import json
import math

from marin.rl.skyrl import SkyRLRun
from rigging.filesystem.storage_path import StoragePath, prefix_join

from vendor.skyrl.marinskyrl.checkpoint_paths import (
    BEST_CHECKPOINT_FILENAME,
    VALIDATION_RECEIPTS_FILENAME,
    ValidationObservation,
    selected_export_step,
    trailing_validation_ema,
)

VALIDATION_EMA_WINDOW = 5
VALIDATION_EMA_ALPHA = 2 / (VALIDATION_EMA_WINDOW + 1)
MINIMUM_VALIDATION_COVERAGE = 0.9


def validate_selection_receipts(best: dict, history: list[dict], selected_step: int) -> dict:
    """Verify the winning EMA over eligible trained validations, including warmup."""
    if best["model_format"] != "native" or best["step"] != selected_step or selected_step < 1:
        raise ValueError("Full RL export must identify a trained validation selection")
    if best["ema_window"] != VALIDATION_EMA_WINDOW or not math.isclose(best["ema_alpha"], VALIDATION_EMA_ALPHA):
        raise ValueError("Full RL requires adjusted EMA window five with alpha=1/3")
    expected = best["expected_tasks"]
    minimum = best["minimum_scored_tasks"]
    if (
        expected < 1
        or not best["eligible"]
        or not math.ceil(MINIMUM_VALIDATION_COVERAGE * expected) <= minimum <= expected
    ):
        raise ValueError("Selected validation threshold does not satisfy 90% coverage")
    if not minimum <= best["scored"] <= best["attempted"] <= expected:
        raise ValueError("Selected validation cohort coverage is below its declared threshold")
    rows = {row["step"]: row for row in history}
    if len(rows) != len(history):
        raise ValueError("Duplicate validation steps")
    scores = {}
    observations = [ValidationObservation(**row) for row in history]
    interval = best.get("evaluation_interval", 1)
    for step in sorted(rows):
        row = rows[step]
        if step < 1 or not minimum <= row["scored"] <= row["attempted"] <= expected:
            raise ValueError("Validation cohort coverage changed")
        score = trailing_validation_ema(observations, step, VALIDATION_EMA_WINDOW, interval=interval)
        if score is None:
            raise ValueError("Eligible validation is missing its EMA observation")
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
    best_path = StoragePath(root) / BEST_CHECKPOINT_FILENAME
    if run.global_step is None or not run.hf_model_uri:
        raise ValueError("Full RL result has no selected HF export")
    if not str(StoragePath(run.hf_model_uri)).endswith(f"/global_step_{run.global_step}/policy"):
        raise ValueError("HF export step differs from selected native checkpoint")
    best = json.loads(best_path.read_bytes()) if best_path.exists() else None
    if best is None or best["model_format"] != "native":
        selected = selected_export_step(root, "best", run.global_step)
        if selected != run.global_step:
            raise ValueError("Final native export does not match its committed checkpoint")
        return {
            "selection": "final_native_not_best",
            "step": selected,
            "model_uri": prefix_join(root, f"global_step_{selected}/policy"),
        }
    history = json.loads((StoragePath(root) / "validation_selection_history.json").read_bytes())
    if run.global_step is None or not run.hf_model_uri:
        raise ValueError("Full RL result has no selected HF export")
    if best["model_uri"] != prefix_join(root, f"global_step_{run.global_step}/policy"):
        raise ValueError("Selected native checkpoint lineage differs from this run")
    if not str(StoragePath(run.hf_model_uri)).endswith(f"/global_step_{run.global_step}/policy"):
        raise ValueError("HF export step differs from selected native checkpoint")
    if selected_export_step(root, "best", run.global_step) != run.global_step:
        raise ValueError("Export does not bind the selected committed best checkpoint")
    receipts = json.loads((StoragePath(root) / VALIDATION_RECEIPTS_FILENAME).read_bytes())
    eligible = {row["step"]: row for row in receipts if row["eligible"] and row["step"] > 0}
    if len({row["step"] for row in receipts}) != len(receipts):
        raise ValueError("Duplicate validation receipts")
    if set(eligible) != {row["step"] for row in history}:
        raise ValueError("Eligible validation receipts do not match the EMA history")
    for row in receipts:
        if (
            row["expected_tasks"] != best["expected_tasks"]
            or row["minimum_scored_tasks"] != best["minimum_scored_tasks"]
            or not 0 <= row["scored_count"] <= row["attempted_count"] <= row["expected_tasks"]
            or row["infrastructure_absent"] != row["expected_tasks"] - row["scored_count"]
        ):
            raise ValueError("Validation receipt cohort or threshold changed")
    for row in history:
        receipt = eligible[row["step"]]
        if (receipt["scored_count"], receipt["attempted_count"], receipt["raw_reward"]) != (
            row["scored"],
            row["attempted"],
            row["reward"],
        ):
            raise ValueError("Validation receipt differs from its EMA observation")
    selected_receipt = eligible[run.global_step]
    if (
        not selected_receipt["improved"]
        or selected_receipt["checkpoint_uri"] != best["model_uri"]
        or not math.isclose(selected_receipt["ema_score"], best["accuracy"], abs_tol=1e-12)
    ):
        raise ValueError("Selected validation receipt does not certify the committed best")
    return validate_selection_receipts(best, history, run.global_step)
