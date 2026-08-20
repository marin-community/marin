# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finelog deployment commands."""

import click
from finelog.deploy import _k8s
from finelog.deploy.config import FinelogConfig, load_finelog_config


def _k8s_config(name: str, operation: str) -> FinelogConfig:
    config = load_finelog_config(name)
    if config.deployment.k8s is None:
        raise click.ClickException(f"{operation} requires a Kubernetes deployment config")
    return config


@click.group()
def finelog() -> None:
    """Deploy Finelog servers."""


@finelog.command("rollout")
@click.argument("name")
@click.option("-y", "--yes", is_flag=True, help="Skip Pulumi confirmation.")
def rollout_cmd(name: str, yes: bool) -> None:
    """Deploy and verify a Pulumi-managed Kubernetes Finelog server."""
    config = _k8s_config(name, "rollout")
    _k8s.k8s_pulumi_rollout(config, stack=name, yes=yes)


@finelog.command("rollback")
@click.argument("name")
@click.option("--to-revision", type=int, help="Restore an exact retained Kubernetes Deployment revision.")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation.")
def rollback_cmd(name: str, to_revision: int | None, yes: bool) -> None:
    """Restore a retained Kubernetes Finelog revision."""
    config = _k8s_config(name, "rollback")
    _k8s.k8s_rollback(config, stack=name, to_revision=to_revision, yes=yes)
