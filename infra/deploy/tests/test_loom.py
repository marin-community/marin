# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for Loom GCE activation."""

import io
import subprocess
from pathlib import Path
from typing import Any

import pytest
from marin_deploy.gce import GceVmTarget
from marin_deploy.loom import activate_loom


def test_activate_loom_restarts_the_startup_unit_before_readiness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    def run(arguments: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        events.append("ssh")
        return subprocess.CompletedProcess(arguments, 0)

    def open_ready(_url: str, **_kwargs: Any) -> io.BytesIO:
        events.append("ready")
        return io.BytesIO(b'{"status":"ready"}')

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("marin_deploy.loom.urlopen", open_ready)
    key_file = tmp_path / "google_compute_engine"
    key_file.write_text("private")
    Path(f"{key_file}.pub").write_text("public")

    activate_loom(
        GceVmTarget(
            project="project",
            zone="us-central1-a",
            instance="loom",
            ssh_key_file=key_file,
        ),
        "https://loom.example/api/ready",
    )

    assert events == ["ssh", "ready"]
