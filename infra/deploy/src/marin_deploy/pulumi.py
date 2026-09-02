# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared commands for Pulumi-managed service deployments."""

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import click

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
GCP_PROJECT = "hai-gcp-models"


@dataclass(frozen=True)
class SecretEnvironment:
    variable: str
    secret: str
    project: str = GCP_PROJECT


@dataclass(frozen=True)
class PulumiService:
    name: str
    stack: str
    secret_environment: tuple[SecretEnvironment, ...] = ()

    @property
    def project_directory(self) -> Path:
        return REPOSITORY_ROOT / "infra" / self.name

    @property
    def stack_config(self) -> Path:
        return self.project_directory / f"Pulumi.{self.stack}.yaml"


CLOUDFLARE_API_TOKEN = SecretEnvironment(
    variable="CLOUDFLARE_API_TOKEN",
    secret="cloudflare-oa-dns-token",
)


PULUMI_SERVICES = (
    PulumiService(name="ducky", stack="ducky-marin"),
    PulumiService(name="echo", stack="marin-echo", secret_environment=(CLOUDFLARE_API_TOKEN,)),
    PulumiService(name="evaldash", stack="marin-evaldash", secret_environment=(CLOUDFLARE_API_TOKEN,)),
    PulumiService(name="grafana", stack="marin-grafana", secret_environment=(CLOUDFLARE_API_TOKEN,)),
    PulumiService(name="marina", stack="marin-marina"),
    PulumiService(name="loom", stack="marin-loom", secret_environment=(CLOUDFLARE_API_TOKEN,)),
    PulumiService(name="xprof", stack="xprof-marin"),
)


def _secret_value(secret: SecretEnvironment) -> str:
    result = subprocess.run(
        [
            "gcloud",
            "secrets",
            "versions",
            "access",
            "latest",
            f"--secret={secret.secret}",
            f"--project={secret.project}",
        ],
        check=False,
        stdout=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        raise click.exceptions.Exit(result.returncode)
    value = result.stdout.strip()
    if not value:
        raise click.ClickException(f"Secret Manager returned an empty value for {secret.secret}")
    if os.environ.get("GITHUB_ACTIONS") == "true":
        click.echo(f"::add-mask::{value}")
    return value


def _rollout(service: PulumiService, *, yes: bool, config: tuple[str, ...]) -> None:
    environment = os.environ.copy()
    for secret in service.secret_environment:
        environment[secret.variable] = _secret_value(secret)

    arguments = ["pulumi", "up", "--stack", service.stack]
    if yes:
        arguments.append("--yes")

    with tempfile.TemporaryDirectory(prefix="marin-deploy-") as temporary_directory:
        if config:
            config_file = Path(temporary_directory) / service.stack_config.name
            shutil.copyfile(service.stack_config, config_file)
            arguments.extend(("--config-file", str(config_file)))
            for value in config:
                arguments.extend(("--config", value))

        result = subprocess.run(arguments, cwd=service.project_directory, env=environment, check=False)
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
