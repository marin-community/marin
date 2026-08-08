# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GCE bootstrap rendering in `finelog.deploy._gcp`."""

import subprocess
import time

import click
import pytest
from finelog.deploy._gcp import _wait_health_via_ssh, render_bootstrap_for
from finelog.deploy.bootstrap import HEALTH_OK
from finelog.deploy.config import (
    Deployment,
    FinelogConfig,
    ForwardingConfig,
    GcpDeployment,
)


def test_render_bootstrap_refuses_a_forwarding_config() -> None:
    """The rendered script becomes the instance's startup-script metadata, which the
    metadata server hands to every process on the VM and `instances describe` hands to
    every reader of the project. This backend has nowhere to put a private signing key,
    so it must refuse rather than render one into the clear."""
    cfg = FinelogConfig(
        name="finelog-marin",
        port=10001,
        image="ghcr.io/example/finelog:latest",
        remote_log_dir="gs://bucket/finelog/marin",
        deployment=Deployment(gcp=GcpDeployment(project="proj", zone="us-central1-a")),
        forwarding=ForwardingConfig(
            target="https://finelog.oa.dev",
            cluster="marin",
            signing_key=("env:FINELOG_SIGNING_KEY",),
        ),
    )
    with pytest.raises(click.ClickException, match="forwarding is not supported on the gcp backend"):
        render_bootstrap_for(cfg, "ghcr.io/example/finelog@sha256:abc")


def test_health_wait_holds_out_for_an_ingesting_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """`safe_deploy`'s gate is what makes auto-rollback fire. `/health` answers 200
    while the server is merely listening — it is the liveness probe too — so a
    binary whose schema the catalog rejects would otherwise pass the gate while
    every telemetry write it receives fails."""
    cfg = FinelogConfig(
        name="finelog-marin",
        port=10001,
        image="ghcr.io/example/finelog:latest",
        remote_log_dir="gs://bucket/finelog/marin",
        deployment=Deployment(gcp=GcpDeployment(project="proj", zone="us-central1-a")),
    )
    monkeypatch.setattr(time, "sleep", lambda _: None)

    def answer(*bodies: str) -> None:
        replies = iter(bodies)
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, stdout=next(replies), stderr=""),
        )

    answer("degraded: telemetry_v1: registration pending", HEALTH_OK)
    assert _wait_health_via_ssh(cfg, cfg.port, max_attempts=2) == HEALTH_OK

    # The reason comes back to the caller so a failed deploy names the wedged
    # namespace rather than only reporting an unhealthy server.
    answer("degraded: telemetry_v1: registration failed: column type mismatch")
    assert "telemetry_v1" in _wait_health_via_ssh(cfg, cfg.port, max_attempts=1)
