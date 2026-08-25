# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for GCE VM activation."""

import subprocess
from pathlib import Path
from typing import Any, cast

import pytest
from marin_deploy.gce import (
    GceVmTarget,
    StartupScriptPersistence,
    activate_startup_script,
    run_remote,
)


def test_activate_startup_script_persists_before_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[list[str], str | None]] = []

    def run(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        stdin = cast(str | None, kwargs.get("input"))
        if "--metadata-from-file" in " ".join(arguments):
            metadata_argument = next(argument for argument in arguments if argument.startswith("--metadata-from-file"))
            script_path = Path(metadata_argument.split("=", 2)[2])
            observed.append((arguments, script_path.read_text()))
        else:
            observed.append((arguments, stdin))
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(subprocess, "run", run)
    target = GceVmTarget(
        project="project",
        zone="zone-a",
        instance="controller",
        impersonate_service_account="deployer@example.com",
    )

    activate_startup_script(
        target,
        "echo activated\n",
        persistence=StartupScriptPersistence.BEFORE_ACTIVATION,
        timeout=600,
    )

    assert [arguments[2:4] for arguments, _stdin in observed] == [
        ["instances", "add-metadata"],
        ["ssh", "controller"],
    ]
    assert [stdin for _arguments, stdin in observed] == ["echo activated\n", "echo activated\n"]


def test_activate_startup_script_after_success_does_not_persist_failed_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[list[str]] = []

    def run(arguments: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        observed.append(arguments)
        raise subprocess.CalledProcessError(17, arguments)

    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(subprocess.CalledProcessError):
        activate_startup_script(
            GceVmTarget(project="project", zone="zone-a", instance="service"),
            "exit 17\n",
            persistence=StartupScriptPersistence.AFTER_SUCCESS,
            timeout=600,
        )

    assert len(observed) == 1
    assert observed[0][2:4] == ["ssh", "service"]


def test_run_remote_applies_explicit_connection_identity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed: list[list[str]] = []

    def run(arguments: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        observed.append(arguments)
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(subprocess, "run", run)
    key_file = tmp_path / "id_rsa"
    target = GceVmTarget(
        project="project",
        zone="zone-a",
        instance="service",
        user="operator",
        ssh_key_file=key_file,
        impersonate_service_account="deployer@example.com",
        tunnel_through_iap=True,
    )

    run_remote(target, "true", timeout=30)

    assert observed == [
        [
            "gcloud",
            "compute",
            "ssh",
            "operator@service",
            "--project=project",
            "--zone=zone-a",
            "--impersonate-service-account=deployer@example.com",
            "--quiet",
            "--ssh-flag=-oBatchMode=yes",
            "--ssh-flag=-oConnectTimeout=15",
            f"--ssh-key-file={key_file}",
            "--tunnel-through-iap",
            "--command",
            "true",
        ]
    ]


def test_run_remote_retries_failed_ssh_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    return_codes = iter((255, 0))
    observed: list[list[str]] = []

    def run(arguments: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        observed.append(arguments)
        return_code = next(return_codes)
        if return_code:
            raise subprocess.CalledProcessError(return_code, arguments)
        return subprocess.CompletedProcess(arguments, return_code)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("rigging.timing.time.sleep", lambda _interval: None)

    run_remote(
        GceVmTarget(project="project", zone="zone-a", instance="service"),
        "true",
        timeout=30,
        attempts=2,
    )

    assert len(observed) == 2
