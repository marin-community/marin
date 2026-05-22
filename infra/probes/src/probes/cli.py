# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point. All config via flags or env vars (MARIN_PROBES_*).

The three v1 probes are wired up inline below — there's no registry indirection
because the probe set is small and stable. To add a probe: import its class and
append a ProbeSpec to the list in `_build_specs`."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from iris.cluster.client.remote_client import RemoteClusterClient

from probes.checks.controller_ping import ControllerPing
from probes.checks.finelog_write import FinelogWrite
from probes.checks.iris_job import IrisJobSubmit
from probes.daemon import run_canary
from probes.probe import ProbeSpec


@click.command()
@click.option(
    "--sqlite-path",
    envvar="MARIN_PROBES_SQLITE_PATH",
    default="/var/lib/probes/samples.sqlite",
    type=click.Path(dir_okay=False, path_type=Path),
    show_default=True,
    help="Canonical sample store. Parent dir must exist.",
)
@click.option(
    "--iris-endpoint",
    envvar="MARIN_PROBES_IRIS_ENDPOINT",
    required=True,
    help="Iris controller URL, e.g. https://iris-controller.internal:10001",
)
@click.option(
    "--finelog-endpoint",
    envvar="MARIN_PROBES_FINELOG_ENDPOINT",
    default=None,
    help="Finelog URL for the secondary sink + write probe. Defaults to --iris-endpoint.",
)
@click.option(
    "--zone",
    "zones",
    envvar="MARIN_PROBES_ZONES",
    multiple=True,
    required=True,
    help="GCP zone(s) to submit canary jobs into. Repeat for multiple zones; " "env var is comma-separated.",
)
@click.option(
    "--heartbeat-seconds",
    envvar="MARIN_PROBES_HEARTBEAT_SECONDS",
    default=30,
    show_default=True,
    type=int,
)
@click.option(
    "--once",
    is_flag=True,
    help="Run each probe once (ignoring cadence), flush stores, exit 0.",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
)
def cli(
    sqlite_path: Path,
    iris_endpoint: str,
    finelog_endpoint: str | None,
    zones: tuple[str, ...],
    heartbeat_seconds: int,
    once: bool,
    log_level: str,
) -> None:
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Env var may be a single comma-separated string; click gives us a 1-tuple.
    if len(zones) == 1 and "," in zones[0]:
        zones = tuple(z.strip() for z in zones[0].split(",") if z.strip())

    finelog_endpoint = finelog_endpoint or iris_endpoint

    specs = _build_specs(
        iris_endpoint=iris_endpoint,
        finelog_endpoint=finelog_endpoint,
        zones=zones,
    )

    exit_code = run_canary(
        specs,
        sqlite_path=sqlite_path,
        finelog_endpoint=finelog_endpoint,
        heartbeat_seconds=heartbeat_seconds,
        once=once,
    )
    sys.exit(exit_code)


def _build_specs(
    *,
    iris_endpoint: str,
    finelog_endpoint: str,
    zones: tuple[str, ...],
) -> list[ProbeSpec]:
    iris_client = RemoteClusterClient(controller_address=iris_endpoint)
    specs: list[ProbeSpec] = [
        ProbeSpec(
            name="controller-ping",
            kind="ControllerPing",
            location=None,
            cadence_seconds=60,
            deadline_seconds=5.0,
            probe=ControllerPing(iris_client),
        ),
        ProbeSpec(
            name="finelog-write",
            kind="FinelogWrite",
            location=None,
            cadence_seconds=60,
            deadline_seconds=10.0,
            probe=FinelogWrite(finelog_endpoint),
        ),
    ]
    for zone in zones:
        specs.append(
            ProbeSpec(
                name=f"iris-job-submit/{zone}",
                kind="IrisJobSubmit",
                location=zone,
                cadence_seconds=300,
                deadline_seconds=120.0,
                probe=IrisJobSubmit(iris_client, zone=zone),
            )
        )
    return specs


if __name__ == "__main__":
    cli()
