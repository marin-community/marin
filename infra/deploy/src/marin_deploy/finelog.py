# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finelog deployment commands."""

import click
from finelog.deploy.operations import rollback as rollback_finelog
from finelog.deploy.operations import rollout as rollout_finelog


@click.group()
def finelog() -> None:
    """Deploy Finelog servers."""


@finelog.command("rollout")
@click.argument("name")
@click.option("-y", "--yes", is_flag=True, help="Skip Pulumi confirmation.")
def rollout_cmd(name: str, yes: bool) -> None:
    """Deploy and verify a Pulumi-managed Kubernetes Finelog server."""
    rollout_finelog(name, yes=yes)


@finelog.command("rollback")
@click.argument("name")
@click.option("--to-revision", type=int, help="Restore an exact retained Kubernetes Deployment revision.")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation.")
def rollback_cmd(name: str, to_revision: int | None, yes: bool) -> None:
    """Restore a retained Kubernetes Finelog revision."""
    rollback_finelog(name, to_revision=to_revision, yes=yes)
