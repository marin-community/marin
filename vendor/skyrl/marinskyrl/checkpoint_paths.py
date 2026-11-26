"""Shared checkpoint and export path contracts."""

import json
from dataclasses import dataclass

from rigging.filesystem.storage_path import StoragePath

from marinskyrl.resource_locator import join_resource_path

GLOBAL_STEP_PREFIX = "global_step_"
LATEST_CHECKPOINT_FILE = "latest_ckpt_global_step.txt"
HF_EXPORT_REQUEST_FILENAME = "hf_export_request.json"
BEST_CHECKPOINT_FILENAME = "best_checkpoint.json"
POLICY_CHECKPOINT_SUBDIRECTORY = "policy"


def policy_export_path(export_root: str, global_step: int) -> str:
    """Return the durable policy export path for one trainer step."""
    return join_resource_path(export_root, f"{GLOBAL_STEP_PREFIX}{global_step}", POLICY_CHECKPOINT_SUBDIRECTORY)


@dataclass(frozen=True)
class ExportCheckpoint:
    step: int
    initializer_uri: str | None = None
    used_final_fallback: bool = False


def selected_export_checkpoint(checkpoint_root: str, selection: str, latest_step: int) -> ExportCheckpoint | None:
    """Resolve the checkpoint selected for HF export, if one is eligible."""
    if selection == "latest":
        return ExportCheckpoint(latest_step)
    if selection not in ("best", "best_and_final"):
        raise ValueError(f"Unknown HF checkpoint selection: {selection}")
    try:
        evidence = (StoragePath(checkpoint_root) / BEST_CHECKPOINT_FILENAME).read_text()
    except FileNotFoundError:
        return ExportCheckpoint(latest_step, used_final_fallback=True) if selection == "best_and_final" else None
    best = json.loads(evidence)
    step = int(best["step"])
    if best["model_format"] == "native":
        return ExportCheckpoint(step)
    if best["model_format"] == "hf" and step == 0:
        uri = best.get("model_uri")
        if isinstance(uri, str) and uri.strip():
            return ExportCheckpoint(step, initializer_uri=uri)
        raise ValueError("The best step-zero checkpoint requires an initializer HF model URI")
    raise ValueError(f"Unsupported best checkpoint format {best['model_format']!r} at step {step}")
