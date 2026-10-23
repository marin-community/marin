# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bound SFT disk usage while preserving restart and validation selection."""

import json
import shutil
from pathlib import Path


def prune_checkpoints(output: Path, *, final_saved: bool = False) -> list[str]:
    """Keep latest state and, until export, best weights; ignore incomplete saves.

    Trainer writes trainer_state.json after weights and optimizer state. Only
    checkpoints with that completion marker participate. Call after on_save,
    or from a single external monitor while an existing Trainer is running.
    """
    completed = []
    for checkpoint in output.glob("checkpoint-*"):
        state_path = checkpoint / "trainer_state.json"
        try:
            state = json.loads(state_path.read_text())
            completed.append((int(state["global_step"]), checkpoint, state))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    if not completed:
        return []
    _, latest, state = max(completed, key=lambda item: item[0])
    best = Path(state["best_model_checkpoint"]) if state.get("best_model_checkpoint") else latest
    removed = []
    for _, checkpoint, _ in completed:
        if checkpoint == latest:
            continue
        if checkpoint == best and not final_saved:
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "scaler.pt"):
                path = checkpoint / name
                if path.exists():
                    path.unlink()
                    removed.append(str(path))
        else:
            shutil.rmtree(checkpoint)
            removed.append(str(checkpoint))
    return removed

