# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import jax.numpy as jnp
import pytest

from experiments.grug.checkpointing import restore_grug_state_from_checkpoint
from levanter.checkpoint import CheckpointInterval, Checkpointer


def _write_checkpoint_metadata(checkpoint_dir: Path, *, step: int, timestamp: str) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"step": step, "timestamp": timestamp, "is_temporary": True}
    (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_restore_prefers_highest_step_over_latest_timestamp(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"
    _write_checkpoint_metadata(checkpoint_root / "step-100", step=100, timestamp="2026-03-17T00:00:00")
    _write_checkpoint_metadata(checkpoint_root / "step-20", step=20, timestamp="2026-03-17T12:00:00")

    attempted: list[str] = []

    def fake_load(state, path, *, discover_latest, axis_mapping, mesh, allow_partial):
        attempted.append(path)
        assert discover_latest is False
        return {"loaded_from": path}

    loaded = restore_grug_state_from_checkpoint(
        {"state": "init"},
        checkpoint_path=str(checkpoint_root),
        load_checkpoint_setting=True,
        mesh=None,
        allow_partial=False,
        _load_fn=fake_load,
    )

    assert attempted == [str(checkpoint_root / "step-100")]
    assert loaded == {"loaded_from": str(checkpoint_root / "step-100")}


def test_restore_falls_back_to_older_checkpoint_when_latest_fails(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"
    _write_checkpoint_metadata(checkpoint_root / "step-100", step=100, timestamp="2026-03-17T10:00:00")
    _write_checkpoint_metadata(checkpoint_root / "step-90", step=90, timestamp="2026-03-17T09:00:00")

    attempted: list[str] = []

    def fake_load(state, path, *, discover_latest, axis_mapping, mesh, allow_partial):
        attempted.append(path)
        assert discover_latest is False
        if path.endswith("step-100"):
            raise FileNotFoundError(path)
        return {"loaded_from": path}

    loaded = restore_grug_state_from_checkpoint(
        {"state": "init"},
        checkpoint_path=str(checkpoint_root),
        load_checkpoint_setting=None,
        mesh=None,
        allow_partial=False,
        _load_fn=fake_load,
    )

    assert attempted == [
        str(checkpoint_root / "step-100"),
        str(checkpoint_root / "step-100"),
        str(checkpoint_root / "step-90"),
    ]
    assert loaded == {"loaded_from": str(checkpoint_root / "step-90")}


def test_restore_raises_when_required_and_no_checkpoint_loads(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"
    _write_checkpoint_metadata(checkpoint_root / "step-100", step=100, timestamp="2026-03-17T10:00:00")

    def fake_load(state, path, *, discover_latest, axis_mapping, mesh, allow_partial):
        raise FileNotFoundError(path)

    with pytest.raises(FileNotFoundError, match="Could not load a checkpoint"):
        restore_grug_state_from_checkpoint(
            {"state": "init"},
            checkpoint_path=str(checkpoint_root),
            load_checkpoint_setting=True,
            mesh=None,
            allow_partial=False,
            _load_fn=fake_load,
        )


def test_restore_supports_legacy_wrapped_and_current_checkpoint_formats(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"

    checkpointer = Checkpointer(
        base_path=str(checkpoint_root),
        save_interval=timedelta(seconds=1),
        step_policies=[CheckpointInterval(every=1)],
    )

    template_state = {"step": jnp.array(0, dtype=jnp.int32), "value": jnp.array([0, 0], dtype=jnp.int32)}
    legacy_state = {"step": jnp.array(7, dtype=jnp.int32), "value": jnp.array([7, 7], dtype=jnp.int32)}
    current_state = {"step": jnp.array(8, dtype=jnp.int32), "value": jnp.array([8, 8], dtype=jnp.int32)}

    checkpointer.on_step(tree={"train_state": legacy_state}, step=7, force=True)
    checkpointer.wait_until_finished()

    loaded_legacy = restore_grug_state_from_checkpoint(
        template_state,
        checkpoint_path=str(checkpoint_root),
        load_checkpoint_setting=True,
        mesh=None,
        allow_partial=False,
    )
    assert int(loaded_legacy["step"]) == 7
    assert jnp.array_equal(loaded_legacy["value"], legacy_state["value"])

    checkpointer.on_step(tree=current_state, step=8, force=True)
    checkpointer.wait_until_finished()

    loaded_current = restore_grug_state_from_checkpoint(
        template_state,
        checkpoint_path=str(checkpoint_root),
        load_checkpoint_setting=True,
        mesh=None,
        allow_partial=False,
    )
    assert int(loaded_current["step"]) == 8
    assert jnp.array_equal(loaded_current["value"], current_state["value"])
