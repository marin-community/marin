# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GCE bootstrap rendering in `finelog.deploy._gcp`."""

import click
import pytest
from finelog.deploy._gcp import render_bootstrap_for
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
