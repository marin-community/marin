# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared commands for Pulumi-managed service deployments."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import click

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class PulumiService:
    name: str
    stack: str

    @property
    def project_directory(self) -> Path:
        return REPOSITORY_ROOT / "infra" / self.name


PULUMI_SERVICES = (
    PulumiService(name="ducky", stack="ducky-marin"),
    PulumiService(name="echo", stack="marin-echo"),
    PulumiService(name="evaldash", stack="marin-evaldash"),
    PulumiService(name="grafana", stack="marin-grafana"),
    PulumiService(name="loom", stack="marin-loom"),
    PulumiService(name="xprof", stack="xprof-marin"),
)


def _rollout(service: PulumiService, *, yes: bool, config: tuple[str, ...]) -> None:
    arguments = ["pulumi", "up", "--stack", service.stack]
    if yes:
        arguments.append("--yes")
    for value in config:
        arguments.extend(("--config", value))

    result = subprocess.run(arguments, cwd=service.project_directory, check=False)
    if result.returncode:
        raise click.exceptions.Exit(result.returncode)


def pulumi_service_group(service: PulumiService) -> click.Group:
    """Create the deployment command group for one Pulumi service project."""
    group = click.Group(service.name, help=f"Deploy the {service.name} service.")

    @click.command("rollout")
    @click.option("-y", "--yes", is_flag=True, help="Skip Pulumi confirmation.")
    @click.option("--config", multiple=True, metavar="KEY=VALUE", help="Set a Pulumi configuration value.")
    def rollout_cmd(yes: bool, config: tuple[str, ...]) -> None:
        """Apply the service Pulumi stack."""
        _rollout(service, yes=yes, config=config)

    group.add_command(rollout_cmd)
    return group
