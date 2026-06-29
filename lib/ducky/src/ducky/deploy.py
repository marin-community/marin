# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deploy ducky as an always-on Iris job.

A routable service needs a *named* Iris port, which ``iris job run`` cannot declare
(no ``--ports`` flag). So deploy goes through the Python submit path
(``client.submit(..., ports=["ducky"])``), the same mechanism the actor worker pool
uses. The job is pinned to a single region and grabs a whole v6e-8 host; the
``DUCKY_*`` credential/scratch env vars are forwarded to the task.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import click
from iris.client.client import IrisClient
from iris.cluster.constraints import region_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, tpu_device

logger = logging.getLogger(__name__)

# A single v6e-8 host (ct6e-standard-8t). DuckDB runs on CPU/RAM only — the TPU sits
# idle; we request the slice to grab the host's large CPU/RAM envelope exclusively.
# TODO(ducky): confirm the exact host vCPU/RAM so the request schedules and gets the
# whole host (over-request → never schedules; under-request → DuckDB capped below host).
TPU_VARIANT = "v6e-8"
ALL_CPU = 180.0
ALL_MEMORY = "1400GB"


def _ducky_env_vars() -> dict[str, str]:
    """Forward all DUCKY_* env vars from this process to the task."""
    return {key: value for key, value in os.environ.items() if key.startswith("DUCKY_")}


@click.command()
@click.option(
    "--controller-url",
    default=lambda: os.environ.get("IRIS_CONTROLLER_URL"),
    required=True,
    help="Iris controller URL (default: $IRIS_CONTROLLER_URL).",
)
@click.option("--region", default="us-east5", show_default=True, help="Region to pin the job to.")
@click.option("--name", default="ducky", show_default=True, help="Job name.")
def cli(controller_url: str, region: str, name: str) -> None:
    """Submit the always-on ducky service to an Iris cluster."""
    logging.basicConfig(level=logging.INFO)
    env_vars = _ducky_env_vars()
    if "DUCKY_SCRATCH_BUCKET" not in env_vars:
        raise click.UsageError("DUCKY_* env vars not set — export ducky config before deploying.")

    client = IrisClient.remote(controller_url, workspace=Path.cwd())
    job = client.submit(
        entrypoint=Entrypoint.from_command("python", "-m", "ducky.server"),
        name=name,
        resources=ResourceSpec(cpu=ALL_CPU, memory=ALL_MEMORY, device=tpu_device(TPU_VARIANT)),
        environment=EnvironmentSpec(env_vars=env_vars),
        ports=[name],
        constraints=[region_constraint([region])],
        # Always-on: ride out preemptions rather than exiting.
        max_retries_preemption=1000,
    )
    logger.info("submitted ducky job %s — reachable at /proxy/%s/ once running", job.job_id, name)


if __name__ == "__main__":
    cli()
