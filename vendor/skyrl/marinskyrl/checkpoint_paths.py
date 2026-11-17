"""Shared checkpoint and export path contracts."""

import json
import math
from dataclasses import dataclass

from rigging.filesystem.storage_path import StoragePath
from .resource_locator import join_resource_path

GLOBAL_STEP_PREFIX = "global_step_"
LATEST_CHECKPOINT_FILE = "latest_ckpt_global_step.txt"
HF_EXPORT_REQUEST_FILENAME = "hf_export_request.json"
BEST_CHECKPOINT_FILENAME = "best_checkpoint.json"
POLICY_CHECKPOINT_SUBDIRECTORY = "policy"


@dataclass(frozen=True)
class ValidationObservation:
    step: int
    reward: float
    attempted: int
    scored: int


def policy_export_path(export_root: str, global_step: int) -> str:
    """Return the durable policy export path for one trainer step."""
    return join_resource_path(export_root, f"{GLOBAL_STEP_PREFIX}{global_step}", POLICY_CHECKPOINT_SUBDIRECTORY)


def selected_export_step(checkpoint_root: str, selection: str, latest_step: int) -> int:
    """Resolve the native checkpoint selected for terminal HF conversion."""
    if selection == "latest":
        return latest_step
    if selection != "best":
        raise ValueError(f"Unknown HF checkpoint selection: {selection}")
    path = StoragePath(join_resource_path(checkpoint_root, BEST_CHECKPOINT_FILENAME))
    if path.exists():
        best = json.loads(path.read_text())
        if best["model_format"] == "native":
            return int(best["step"])
    final_state = StoragePath(join_resource_path(checkpoint_root, f"{GLOBAL_STEP_PREFIX}{latest_step}", "trainer_state.pt"))
    marker = StoragePath(join_resource_path(checkpoint_root, LATEST_CHECKPOINT_FILE))
    if not final_state.exists() or not marker.exists() or int(marker.read_text().strip()) != latest_step:
        raise ValueError("No committed best or final native checkpoint is available")
    return latest_step


def trailing_validation_ema(
    history: list[ValidationObservation], step: int, window: int, *, interval: int = 1
) -> float | None:
    """Adjusted EMA over complete scheduled validation observations; alpha=2/(window+1)."""
    if window < 1 or interval < 1:
        raise ValueError("Validation EMA window must be positive")
    by_step = {row.step: row.reward for row in history}
    steps = list(range(step - (window - 1) * interval, step + 1, interval))
    if steps[0] < 1 or any(s not in by_step for s in steps):
        return None
    rewards = [by_step[s] for s in steps]
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in rewards):
        raise ValueError("Validation rewards must be finite and normalized")
    decay = 1 - 2 / (window + 1)
    weights = [decay**age for age in reversed(range(window))]
    return sum(reward * weight for reward, weight in zip(rewards, weights, strict=True)) / sum(weights)
