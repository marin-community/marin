# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for Finelog deployment adapters."""

import subprocess

import pytest
from finelog.deploy.config import Deployment, FinelogConfig, GcpDeployment
from marin_deploy.finelog import activate_gce_bootstrap


def test_failed_finelog_activation_preserves_last_successful_startup_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fail_ssh(arguments: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        commands.append(arguments)
        raise subprocess.CalledProcessError(17, arguments)

    monkeypatch.setattr(subprocess, "run", fail_ssh)
    config = FinelogConfig(
        name="finelog-marin",
        port=10001,
        image="ghcr.io/example/finelog:latest",
        remote_log_dir="gs://bucket/finelog/marin",
        deployment=Deployment(
            gcp=GcpDeployment(
                project="project",
                zone="us-central1-a",
                service_account="deployer@example.com",
            )
        ),
    )

    assert not activate_gce_bootstrap(config, "exit 17\n")
    assert len(commands) == 1
    assert commands[0][2:4] == ["ssh", "finelog-marin"]
    assert "--impersonate-service-account=deployer@example.com" in commands[0]
