# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Disk retention keeps resume state and validation-selected weights."""

import json
from pathlib import Path

from experiments.weight_merging.playground.retention import prune_checkpoints


def checkpoint(root: Path, step: int, best: int) -> Path:
    path = root / f"checkpoint-{step}"
    path.mkdir()
    for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (path / name).touch()
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "best_model_checkpoint": str(root / f"checkpoint-{best}")})
    )
    return path


def test_preserves_best_weights_and_latest_resume_state(tmp_path):
    old = checkpoint(tmp_path, 250, 250)
    best = checkpoint(tmp_path, 500, 500)
    latest = checkpoint(tmp_path, 750, 500)
    incomplete = tmp_path / "checkpoint-1000"
    incomplete.mkdir()
    (incomplete / "optimizer.pt").touch()
    prune_checkpoints(tmp_path)
    assert not old.exists()
    assert (best / "model.safetensors").exists()
    assert not (best / "optimizer.pt").exists()
    assert not (best / "scheduler.pt").exists()
    assert not (best / "rng_state.pth").exists()
    assert (latest / "optimizer.pt").exists()
    assert (incomplete / "optimizer.pt").exists()
    assert prune_checkpoints(tmp_path) == []
    prune_checkpoints(tmp_path, final_saved=True)
    assert not best.exists()
    assert (latest / "optimizer.pt").exists()


def test_latest_is_best(tmp_path):
    old = checkpoint(tmp_path, 250, 250)
    latest = checkpoint(tmp_path, 500, 500)
    prune_checkpoints(tmp_path)
    assert not old.exists()
    assert (latest / "model.safetensors").exists()
    assert (latest / "optimizer.pt").exists()
