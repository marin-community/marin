# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin service deployment commands."""

import click

from marin_deploy.finelog import finelog
from marin_deploy.pulumi import PULUMI_SERVICES, pulumi_service_group


@click.group()
def cli() -> None:
    """Deploy Marin-operated services."""


cli.add_command(finelog)
for service in PULUMI_SERVICES:
    cli.add_command(pulumi_service_group(service))
