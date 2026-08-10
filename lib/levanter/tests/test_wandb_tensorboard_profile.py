# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from rigging.filesystem import StoragePath

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "wandb_tensorboard_profile.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("wandb_tensorboard_profile", SCRIPT_PATH)
assert SCRIPT_SPEC is not None
SCRIPT_MODULE = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC.loader is not None
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)

mirror_profile_dir = SCRIPT_MODULE.mirror_profile_dir
resolve_profile_dir = SCRIPT_MODULE.resolve_profile_dir
resolve_profile_run_id = SCRIPT_MODULE.resolve_profile_run_id


def test_resolve_profile_dir_uses_logged_trainer_log_dir():
    run = SimpleNamespace(
        config={"trainer": {"id": "trainer-456", "log_dir": "gs://bucket/logs"}},
        path=["entity", "project", "run-123"],
    )

    assert resolve_profile_run_id(run) == "trainer-456"
    assert resolve_profile_dir(run) == "gs://bucket/logs/trainer-456/profiler"


@pytest.mark.parametrize("root_factory", [lambda _: None, lambda tmp_path: tmp_path / "logs"])
def test_mirror_profile_dir_returns_existing_local_directory(tmp_path, root_factory):
    source = tmp_path / "logs" / "run-123" / "profiler"
    trace_dir = source / "plugins" / "profile" / "2024_01_01_00_00_00"
    trace_dir.mkdir(parents=True)
    (trace_dir / "perfetto_trace.json.gz").write_bytes(b"trace")

    resolved_root = root_factory(tmp_path)
    mirrored = mirror_profile_dir(str(source), resolved_root, run_id="run-123")

    assert mirrored == source
    assert (mirrored / "plugins" / "profile" / "2024_01_01_00_00_00" / "perfetto_trace.json.gz").exists()


def test_mirror_profile_dir_copies_remote_directory_to_download_root(tmp_path):
    source = "memory://wandb/runs/run-123/profiler"
    source_path = StoragePath(source)
    source_path.mkdirs()
    trace_dir = source_path / "plugins/profile/2024_01_01_00_00_00"
    trace_dir.mkdirs()
    (trace_dir / "perfetto_trace.json.gz").write_bytes(b"trace")

    mirrored = mirror_profile_dir(source, tmp_path / "download", run_id="run-123")

    assert mirrored == tmp_path / "download" / "run-123" / "profiler"
    assert (mirrored / "plugins" / "profile" / "2024_01_01_00_00_00" / "perfetto_trace.json.gz").exists()
