# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Operator-facing Finelog deployment operations."""

import click

from finelog.deploy import _k8s
from finelog.deploy.config import FinelogConfig, load_finelog_config


def _k8s_config(name: str, operation: str) -> FinelogConfig:
    config = load_finelog_config(name)
    if config.deployment.k8s is None:
        raise click.ClickException(f"{operation} requires a Kubernetes deployment config")
    return config


def rollout(name: str, *, yes: bool) -> None:
    """Deploy and verify a Pulumi-managed Kubernetes Finelog server."""
    config = _k8s_config(name, "rollout")
    _k8s.k8s_pulumi_rollout(config, stack=name, yes=yes)


def rollback(name: str, *, to_revision: int | None, yes: bool) -> None:
    """Restore a retained Kubernetes Finelog revision."""
    config = _k8s_config(name, "rollback")
    _k8s.k8s_rollback(config, stack=name, to_revision=to_revision, yes=yes)
