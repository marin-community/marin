# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for Finelog deployment transactions."""

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from finelog.deploy.bootstrap import HEALTH_OK
from marin_deploy import finelog as finelog_deploy
from marin_deploy.cli import cli

OLD_IMAGE = f"ghcr.io/example/finelog@sha256:{'a' * 64}"
NEW_IMAGE = f"ghcr.io/example/finelog@sha256:{'b' * 64}"


def _gce_config(tmp_path: Path) -> Path:
    config = tmp_path / "finelog-test.yaml"
    config.write_text(
        """
name: finelog-test
port: 10001
image: ghcr.io/example/finelog:latest
remote_log_dir: gs://example/finelog/test
deployment:
  gcp:
    project: test-project
    zone: us-central1-a
""".strip()
    )
    return config


def test_gce_rollout_restores_running_digest_when_candidate_bootstrap_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    activations: list[str] = []
    persisted_scripts: list[str] = []

    def run(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if arguments[:4] == ["docker", "buildx", "imagetools", "inspect"]:
            return subprocess.CompletedProcess(
                arguments,
                0,
                stdout=json.dumps({"digest": NEW_IMAGE.rpartition("@")[2]}),
                stderr="",
            )

        if arguments[:3] == ["gcloud", "compute", "ssh"]:
            command = arguments[-1]
            if "docker inspect --format" in command:
                return subprocess.CompletedProcess(arguments, 0, stdout=f"{OLD_IMAGE}|\n", stderr="")
            if command == "bash -s":
                bootstrap = kwargs["input"]
                image = NEW_IMAGE if NEW_IMAGE in bootstrap else OLD_IMAGE
                activations.append(image)
                if image == NEW_IMAGE:
                    raise subprocess.CalledProcessError(17, arguments)
                return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")
            return subprocess.CompletedProcess(arguments, 0, stdout=HEALTH_OK, stderr="")

        if arguments[:4] == ["gcloud", "compute", "instances", "add-metadata"]:
            metadata = next(argument for argument in arguments if argument.startswith("--metadata-from-file="))
            persisted_scripts.append(Path(metadata.split("=", 2)[2]).read_text())
            return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

        raise AssertionError(f"unexpected subprocess: {arguments}")

    state_directory = tmp_path / "deploy-state"
    monkeypatch.setattr(finelog_deploy, "STATE_DIR", state_directory, raising=False)
    monkeypatch.setattr(subprocess, "run", run)

    result = CliRunner().invoke(cli, ["finelog", "rollout", str(_gce_config(tmp_path)), "--no-build"])

    assert result.exit_code == 1
    assert activations == [NEW_IMAGE, OLD_IMAGE]
    assert len(persisted_scripts) == 1
    assert OLD_IMAGE in persisted_scripts[0]
    state = json.loads((state_directory / "finelog-test.json").read_text())
    assert state["previous_digest"] == OLD_IMAGE
    assert state["attempted_digest"] == NEW_IMAGE
    assert state["current_digest"] == OLD_IMAGE
    assert state["rolled_back_from"] == NEW_IMAGE


def test_gce_rollout_stops_before_mutation_when_running_digest_inspection_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []

    def run(arguments: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(arguments)
        raise subprocess.CalledProcessError(255, arguments)

    state_directory = tmp_path / "deploy-state"
    monkeypatch.setattr(finelog_deploy, "STATE_DIR", state_directory)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("rigging.timing.time.sleep", lambda _interval: None)

    result = CliRunner().invoke(cli, ["finelog", "rollout", str(_gce_config(tmp_path)), "--no-build"])

    assert result.exit_code == 1
    assert not state_directory.exists()
    assert commands
    assert all(arguments[:3] == ["gcloud", "compute", "ssh"] for arguments in commands)
    assert all(arguments[-1] != "bash -s" for arguments in commands)


def test_gce_rollout_rejects_fast_profile_without_build(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["finelog", "rollout", str(_gce_config(tmp_path)), "--no-build", "--fast"],
    )

    assert result.exit_code == 1
    assert "--fast requires --build" in result.output
