"""Shared checkpoint and export path contracts."""

import json
import logging
import math
from dataclasses import dataclass

from rigging.filesystem.storage_path import StoragePath
from .resource_locator import join_resource_path

GLOBAL_STEP_PREFIX = "global_step_"
LATEST_CHECKPOINT_FILE = "latest_ckpt_global_step.txt"
HF_EXPORT_REQUEST_FILENAME = "hf_export_request.json"
BEST_CHECKPOINT_FILENAME = "best_checkpoint.json"
VALIDATION_RECEIPTS_FILENAME = "validation_selection_receipts.json"
EXPORT_FALLBACK_FILENAME = "export-fallback.json"
POLICY_CHECKPOINT_SUBDIRECTORY = "policy"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationObservation:
    step: int
    reward: float
    attempted: int
    scored: int


@dataclass(frozen=True)
class ValidationReceipt:
    step: int
    expected_tasks: int
    minimum_scored_tasks: int
    attempted_count: int
    scored_count: int
    infrastructure_absent: int
    eligible: bool
    reason: str
    raw_reward: float | None
    ema_score: float | None = None
    improved: bool = False
    checkpoint_uri: str | None = None


def protected_native_checkpoint_steps(checkpoint_root: str) -> set[int]:
    """Protect the selected best and the latest committed native state, including final."""
    root = StoragePath(checkpoint_root)
    protected = set()
    marker = root / LATEST_CHECKPOINT_FILE
    if marker.exists():
        protected.add(int(marker.read_text().strip()))
    best_path = root / BEST_CHECKPOINT_FILENAME
    if best_path.exists():
        best = json.loads(best_path.read_text())
        if best["model_format"] == "native":
            protected.add(int(best["step"]))
    return protected


def policy_export_path(export_root: str, global_step: int) -> str:
    """Return the durable policy export path for one trainer step."""
    return join_resource_path(export_root, f"{GLOBAL_STEP_PREFIX}{global_step}", POLICY_CHECKPOINT_SUBDIRECTORY)


@dataclass(frozen=True)
class ExportCheckpoint:
    step: int
    initializer_uri: str | None = None


def selected_export_checkpoint(checkpoint_root: str, selection: str, latest_step: int) -> ExportCheckpoint | None:
    """Resolve the checkpoint selected for HF export, if one is eligible."""
    if selection == "latest":
        return ExportCheckpoint(latest_step)
    if selection != "best":
        raise ValueError(f"Unknown HF checkpoint selection: {selection}")
    try:
        evidence = (StoragePath(checkpoint_root) / BEST_CHECKPOINT_FILENAME).read_text()
    except FileNotFoundError:
        return _fallback_export_checkpoint(checkpoint_root, latest_step)
    best = json.loads(evidence)
    step = int(best["step"])
    if best["model_format"] == "native":
        committed = StoragePath(
            join_resource_path(checkpoint_root, f"{GLOBAL_STEP_PREFIX}{step}", "trainer_state.pt")
        )
        if not committed.exists():
            raise ValueError(f"Selected best native checkpoint is not committed: step {step}")
        return ExportCheckpoint(step)
    if best["model_format"] == "hf" and step == 0:
        uri = best.get("model_uri")
        if isinstance(uri, str) and uri.strip():
            return ExportCheckpoint(step, initializer_uri=uri)
        raise ValueError("The best step-zero checkpoint requires an initializer HF model URI")
    raise ValueError(f"Unsupported best checkpoint format {best['model_format']!r} at step {step}")


def _fallback_export_checkpoint(checkpoint_root: str, latest_step: int) -> ExportCheckpoint:
    """Refuse a fallback when eligible validation exists; otherwise export the committed final step."""
    receipts_path = StoragePath(join_resource_path(checkpoint_root, VALIDATION_RECEIPTS_FILENAME))
    receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else []
    eligible_steps = {row["step"] for row in receipts if row["eligible"] and row["step"] > 0}
    history_path = StoragePath(join_resource_path(checkpoint_root, "validation_selection_history.json"))
    if history_path.exists():
        eligible_steps.update(row["step"] for row in json.loads(history_path.read_text()) if row["step"] > 0)
    if eligible_steps:
        raise ValueError("Eligible validation exists but no committed native best is selected; refusing final fallback")
    final_state = StoragePath(
        join_resource_path(checkpoint_root, f"{GLOBAL_STEP_PREFIX}{latest_step}", "trainer_state.pt")
    )
    marker = StoragePath(join_resource_path(checkpoint_root, LATEST_CHECKPOINT_FILE))
    if not final_state.exists() or not marker.exists() or int(marker.read_text().strip()) != latest_step:
        raise ValueError("No committed best or final native checkpoint is available")
    logger.warning("FINAL CHECKPOINT FALLBACK: no eligible trained validation; exporting final step %s", latest_step)
    return ExportCheckpoint(latest_step)


def trailing_validation_ema(
    history: list[ValidationObservation], step: int, window: int, *, interval: int = 1,
    selection_eligibility: str = "covered",
) -> float | None:
    """Adjusted EMA of up to the last window eligible observations; alpha=2/(window+1)."""
    if window < 1 or interval < 1:
        raise ValueError("Validation EMA window must be positive")
    if selection_eligibility not in {"covered", "complete_window"}:
        raise ValueError("Unknown checkpoint selection eligibility")
    by_step = {row.step: row.reward for row in history if 0 < row.step <= step}
    if len(by_step) != sum(0 < row.step <= step for row in history):
        raise ValueError("Duplicate validation steps")
    if step not in by_step:
        return None
    if selection_eligibility == "complete_window":
        required = range(step - (window - 1) * interval, step + 1, interval)
        if any(index < 1 or index not in by_step for index in required):
            return None
    rewards = [by_step[index] for index in sorted(by_step)[-window:]]
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in rewards):
        raise ValueError("Validation rewards must be finite and normalized")
    decay = 1 - 2 / (window + 1)
    weights = [decay**age for age in reversed(range(len(rewards)))]
    anchor = rewards[0]
    return anchor + math.fsum(
        (reward - anchor) * weight for reward, weight in zip(rewards, weights, strict=True)
    ) / math.fsum(weights)
