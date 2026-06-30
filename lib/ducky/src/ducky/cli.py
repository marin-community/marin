# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``ducky`` command group: ``ducky deploy`` and ``ducky query``."""

from __future__ import annotations

import click

from ducky.client import query
from ducky.deploy import cli as deploy_command


@click.group()
def cli() -> None:
    """ducky — ad-hoc DuckDB SQL service."""


cli.add_command(deploy_command, name="deploy")
cli.add_command(query)


if __name__ == "__main__":
    cli()
