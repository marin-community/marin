# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GCE bootstrap rendering in `finelog.deploy._gcp`."""

from finelog.deploy._gcp import render_bootstrap_for
from finelog.deploy.config import (
    CidrAuthLayer,
    Deployment,
    FinelogConfig,
    GcpDeployment,
)


def _cfg(auth: tuple = (CidrAuthLayer(cidrs=("10.0.0.0/8", "127.0.0.0/8")),)) -> FinelogConfig:
    return FinelogConfig(
        name="finelog-marin",
        port=10001,
        image="ghcr.io/example/finelog:latest",
        remote_log_dir="gs://bucket/finelog/marin",
        deployment=Deployment(gcp=GcpDeployment(project="proj", zone="us-central1-a")),
        auth=auth,
    )


def test_render_bootstrap_for_inlines_the_configured_auth_policy() -> None:
    """The rendered `docker run` must carry FINELOG_AUTH_POLICY. A bootstrap that
    omits it strands the server on its allow-localhost default, where every
    in-VPC client — including the iris controller relaying through its proxy —
    is rejected with `no auth layer admitted it`."""
    rendered = render_bootstrap_for(_cfg(), "ghcr.io/example/finelog@sha256:abc")

    assert '-e FINELOG_AUTH_POLICY=\'[{"type":"cidr","cidrs":["10.0.0.0/8","127.0.0.0/8"]}]\'' in rendered
    assert "ghcr.io/example/finelog@sha256:abc" in rendered


def test_render_bootstrap_for_omits_policy_env_when_no_auth_configured() -> None:
    """No `auth:` block leaves the env var unset, which the server reads as its
    private allow-localhost default rather than as an empty (lock-out) policy."""
    rendered = render_bootstrap_for(_cfg(auth=()), "img")

    assert "FINELOG_AUTH_POLICY" not in rendered
