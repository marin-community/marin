"""Shared checkpoint and export path contracts."""

import json

from rigging.filesystem.storage_path import StoragePath
from marinskyrl.resource_locator import join_resource_path

GLOBAL_STEP_PREFIX = "global_step_"
LATEST_CHECKPOINT_FILE = "latest_ckpt_global_step.txt"
HF_EXPORT_REQUEST_FILENAME = "hf_export_request.json"
BEST_CHECKPOINT_FILENAME = "best_checkpoint.json"
# Shared with marin.rl.skyrl_records.TRACKING_FILENAME.
TRACKING_FILENAME = "tracking.json"
ATTEMPT_RECEIPT_FILENAME = "receipt.json"
POLICY_CHECKPOINT_SUBDIRECTORY = "policy"


def attempt_directory(export_root: str, attempt_id: str) -> str:
    """Return one native attempt's durable artifact directory."""
    return join_resource_path(export_root, "attempts", attempt_id)


def attempt_receipt_path(export_root: str, attempt_id: str) -> str:
    return join_resource_path(attempt_directory(export_root, attempt_id), ATTEMPT_RECEIPT_FILENAME)


def policy_export_path(export_root: str, global_step: int) -> str:
    """Return the durable policy export path for one trainer step."""
    return join_resource_path(export_root, f"{GLOBAL_STEP_PREFIX}{global_step}", POLICY_CHECKPOINT_SUBDIRECTORY)


def selected_export_step(checkpoint_root: str, selection: str, latest_step: int) -> int:
    """Resolve the native checkpoint selected for terminal HF conversion."""
    if selection == "latest":
        return latest_step
    if selection != "best":
        raise ValueError(f"Unknown HF checkpoint selection: {selection}")
    best = json.loads(StoragePath(join_resource_path(checkpoint_root, BEST_CHECKPOINT_FILENAME)).read_text())
    if best["model_format"] != "native":
        raise ValueError("Best checkpoint is the initial HF model; no trained export is available")
    return int(best["step"])
