# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import datetime
import json
import posixpath
import uuid

import pytest
from rigging.filesystem import url_to_fs

try:
    import marin.rl.weight_transfer.checkpoint as checkpoint_transfer
    from marin.rl.weight_transfer import (
        WeightTransferConfig,
        WeightTransferMode,
        create_weight_transfer_client,
    )
except ImportError:
    pytest.skip("Post training imports unavailable", allow_module_level=True)


@pytest.fixture
def checkpoint_loader(monkeypatch):
    loaded_model = object()
    checkpoint_paths = []

    def fake_load_checkpoint(**kwargs):
        checkpoint_paths.append(kwargs["checkpoint_path"])
        return loaded_model

    monkeypatch.setattr(checkpoint_transfer.levanter_checkpoint, "load_checkpoint", fake_load_checkpoint)
    return checkpoint_paths, loaded_model


@pytest.fixture
def memory_checkpoint_dir():
    checkpoint_dir = f"memory://checkpoint-transfer-{uuid.uuid4().hex}/policy_checkpoints"
    yield checkpoint_dir

    fs, path = url_to_fs(checkpoint_dir)
    root = f"/{path.strip('/').split('/', maxsplit=1)[0]}"
    if fs.exists(root):
        fs.rm(root, recursive=True)


def _checkpoint_config(checkpoint_dir: str) -> WeightTransferConfig:
    return WeightTransferConfig(
        mode=WeightTransferMode.GCS_CHECKPOINT,
        checkpoint_dir=checkpoint_dir,
    )


def _create_checkpoint_entries(
    checkpoint_dir: str,
    *,
    steps: list[int] | None = None,
    directories: list[str] | None = None,
    files: list[str] | None = None,
) -> None:
    fs, path = url_to_fs(checkpoint_dir)
    fs.makedirs(path, exist_ok=True)

    for step in steps or []:
        checkpoint_path = posixpath.join(path, f"step_{step}")
        fs.makedirs(checkpoint_path, exist_ok=True)
        timestamp = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC) + datetime.timedelta(seconds=step)
        with fs.open(posixpath.join(checkpoint_path, "metadata.json"), "w") as f:
            json.dump({"step": step, "timestamp": timestamp.isoformat()}, f)

    for directory in directories or []:
        fs.makedirs(posixpath.join(path, directory), exist_ok=True)

    for filename in files or []:
        fs.touch(posixpath.join(path, filename))


def test_receive_weights_loads_latest_numeric_checkpoint(tmp_path, checkpoint_loader):
    checkpoint_dir = str(tmp_path / "policy_checkpoints")
    _create_checkpoint_entries(
        checkpoint_dir,
        steps=[9, 100, 39],
        directories=["other_dir", "step_not_numeric"],
        files=["config.json"],
    )
    checkpoint_paths, loaded_model = checkpoint_loader
    old_model = object()

    client = create_weight_transfer_client(_checkpoint_config(checkpoint_dir))
    update = client.receive_weights(old_model)

    assert update is not None
    assert update.model is loaded_model
    assert update.weight_id == 100
    assert checkpoint_paths == [str(tmp_path / "policy_checkpoints" / "step_100")]


@pytest.mark.parametrize("create_entries", [False, True])
def test_receive_weights_without_checkpoints_returns_none(tmp_path, checkpoint_loader, create_entries):
    checkpoint_dir = str(tmp_path / "policy_checkpoints")
    if create_entries:
        _create_checkpoint_entries(
            checkpoint_dir,
            directories=["other_dir", "step_not_numeric"],
            files=["config.json"],
        )
    checkpoint_paths, _ = checkpoint_loader

    client = create_weight_transfer_client(_checkpoint_config(checkpoint_dir))

    assert client.receive_weights(object()) is None
    assert checkpoint_paths == []


def test_receive_weights_skips_repeated_checkpoint_and_loads_newer_step(tmp_path, checkpoint_loader):
    checkpoint_dir = str(tmp_path / "policy_checkpoints")
    _create_checkpoint_entries(checkpoint_dir, steps=[3])
    checkpoint_paths, _ = checkpoint_loader
    old_model = object()

    client = create_weight_transfer_client(_checkpoint_config(checkpoint_dir))
    first_update = client.receive_weights(old_model)
    repeated_update = client.receive_weights(old_model)

    _create_checkpoint_entries(checkpoint_dir, steps=[5])
    second_update = client.receive_weights(old_model)

    assert first_update is not None
    assert first_update.weight_id == 3
    assert repeated_update is None
    assert second_update is not None
    assert second_update.weight_id == 5
    assert [posixpath.basename(path) for path in checkpoint_paths] == ["step_3", "step_5"]


def test_receive_weights_supports_fsspec_memory_scheme(memory_checkpoint_dir, checkpoint_loader):
    _create_checkpoint_entries(memory_checkpoint_dir, steps=[42])
    checkpoint_paths, _ = checkpoint_loader
    _, path = url_to_fs(memory_checkpoint_dir)
    expected_checkpoint_path = f"memory://{posixpath.join(path, 'step_42')}"

    client = create_weight_transfer_client(_checkpoint_config(memory_checkpoint_dir))
    update = client.receive_weights(object())

    assert update is not None
    assert update.weight_id == 42
    assert checkpoint_paths == [expected_checkpoint_path]
    assert not checkpoint_paths[0].startswith("memory://memory://")
