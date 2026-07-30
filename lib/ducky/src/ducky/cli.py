# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``ducky`` command group: ``ducky query``.

Deploying is Pulumi's job — see infra/ducky (the ``ducky-marin`` stack) and the Deploy
section of the README.
"""

from __future__ import annotations

import click

from ducky.client import query


@click.group()
def cli() -> None:
    """ducky — ad-hoc DuckDB SQL service."""


cli.add_command(query)


if __name__ == "__main__":
    cli()
