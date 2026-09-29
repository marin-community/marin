# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Isolation and result behavior at the OCI runtime boundary."""

import base64
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from harbor.environments.base import ExecResult

from taskcompendium.grading import Outcome
from taskcompendium.harbor import script_runtime
from taskcompendium.harbor.workspace import UnsafeWorkspaceError, capture_workspace
from taskcompendium.verifiers.script import PrivateResource, ScriptVerifier


def _config() -> ScriptVerifier:
    script = b"#!/usr/bin/env python3\n"
    reference = b"private answer"
    resources = (
        PrivateResource(
            path="grade.py",
            sha256=hashlib.sha256(script).hexdigest(),
            embedded_base64=base64.b64encode(script).decode(),
            executable=True,
        ),
        PrivateResource(
            path="reference.txt",
            sha256=hashlib.sha256(reference).hexdigest(),
            embedded_base64=base64.b64encode(reference).decode(),
        ),
    )
    return ScriptVerifier(
        entrypoint="grade.py",
        timeout_seconds=2.0,
        runtime_image=f"example/verifier@sha256:{'a' * 64}",
        resources=resources,
    )


def _mounts(command: list[str]) -> dict[str, Path]:
    mounts: dict[str, Path] = {}
    for index, token in enumerate(command):
        if token != "--mount":
            continue
        fields = dict(part.split("=", 1) for part in command[index + 1].split(",") if "=" in part)
        mounts[fields["dst"]] = Path(fields["src"])
    return mounts


def test_script_runtime_stages_private_files_and_isolates_workspace(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "program.txt").write_text("agent final state")
    submission = {"protocol_version": 1, "answer_type": "workspace_state", "convention_id": "workspace", "answer": None}
    verifier_dir = tmp_path / "verifier"

    def fake_docker(command, **kwargs):
        assert command[:2] == ["docker", "run"]
        assert command[command.index("--network") + 1] == "none"
        assert "--pull=never" in command
        assert "--log-driver=none" in command
        assert "--read-only" in command
        assert "--cap-drop=ALL" in command
        assert kwargs["timeout"] == 2.0
        mounts = _mounts(command)
        assert set(mounts) == {"/app", "/tests", "/verifier"}
        assert (mounts["/app"] / "program.txt").read_text() == "agent final state"
        assert not (mounts["/app"] / "reference.txt").exists()
        assert (mounts["/tests"] / "reference.txt").read_bytes() == b"private answer"
        assert json.loads((mounts["/verifier"] / "submission.json").read_text()) == submission
        (mounts["/app"] / "program.txt").write_text("grader mutation")
        (mounts["/verifier"] / "result.json").write_text('{"status":"scored","reward":0.75}')
        return subprocess.CompletedProcess(command, 17)

    monkeypatch.setattr(script_runtime.subprocess, "run", fake_docker)
    result = script_runtime.run_script_verifier(_config(), submission, snapshot, verifier_dir)

    assert result.status == Outcome.GRADED
    assert result.reward == 0.75
    assert (snapshot / "program.txt").read_text() == "agent final state"
    assert json.loads((verifier_dir / "result.json").read_text()) == {"status": "scored", "reward": 0.75}


def test_script_runtime_timeout_stops_container_without_reward(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    commands = []

    def fake_docker(command, **kwargs):
        commands.append(command)
        if command[1] == "run":
            assert kwargs["timeout"] == 900
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(script_runtime.subprocess, "run", fake_docker)
    result = script_runtime.run_script_verifier(
        _config().model_copy(update={"timeout_seconds": 1200.0}),
        {"protocol_version": 1, "answer_type": "text", "convention_id": "plain", "answer": "x"},
        snapshot,
        tmp_path / "verifier",
    )

    assert result.status == Outcome.INFRA_ERROR
    assert result.reward is None
    assert commands[1][:3] == ["docker", "rm", "-f"]
    assert commands[1][3] == commands[0][commands[0].index("--name") + 1]


def test_script_runtime_rejects_unscored_reward(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    def fake_docker(command, **kwargs):
        verifier = _mounts(command)["/verifier"]
        (verifier / "result.json").write_text('{"status":"invalid_task","reward":1}')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(script_runtime.subprocess, "run", fake_docker)
    result = script_runtime.run_script_verifier(
        _config(),
        {"protocol_version": 1, "answer_type": "text", "convention_id": "plain", "answer": "x"},
        snapshot,
        tmp_path / "verifier",
    )

    assert result.status == Outcome.INVALID_TASK
    assert result.reward is None


async def test_workspace_capture_is_independent_and_rejects_links(tmp_path):
    agent_workspace = tmp_path / "agent-workspace"
    agent_workspace.mkdir()
    (agent_workspace / "state.txt").write_text("final agent state")

    class Environment:
        async def exec(self, command):
            if command == "test -d /app && test ! -L /app":
                return ExecResult(return_code=0)
            assert command == "cat /proc/self/mountinfo"
            return ExecResult(stdout="1 0 0:1 / /app rw - ext4 /dev/root rw\n", return_code=0)

        async def download_dir(self, source, target):
            assert source == "/app"
            shutil.copytree(agent_workspace, target, dirs_exist_ok=True, symlinks=True)

    copied = await capture_workspace(Environment(), tmp_path / "snapshot")
    (copied / "state.txt").write_text("grader mutation")
    assert (agent_workspace / "state.txt").read_text() == "final agent state"

    (agent_workspace / "outside").symlink_to(tmp_path)
    with pytest.raises(UnsafeWorkspaceError, match="Unsafe workspace entry"):
        await capture_workspace(Environment(), tmp_path / "unsafe-snapshot")
