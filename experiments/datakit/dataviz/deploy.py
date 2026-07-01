# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deploy the datakit dataviz dashboard as an always-on Iris job.

Like ducky, a routable service needs a *named* Iris port, which ``iris job run``
can't declare, so we go through the Python submit path
(``client.submit(..., ports=["dataviz"])``). The dashboard is CPU-only — it holds
no data, just resolves lineage once and forwards SQL to the ducky service — so it
asks for a small CPU/RAM slice, not a TPU host.

All ``DATAVIZ_*`` env vars are forwarded to the task; at minimum
``DATAVIZ_STORE`` (the store to explore). Point it at ducky with
``DATAVIZ_DUCKY_URL`` and optionally pin ``quality``/``cluster_assign`` lineage
with ``DATAVIZ_QUALITY_MODEL`` + ``DATAVIZ_DOMAIN_CENTROIDS``.

Connect to the controller with ``--controller-url`` (e.g. a tunnel from
``iris --cluster=<name> ...``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import click
from iris.client.client import IrisClient, Job
from iris.cluster.constraints import region_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec

from experiments.datakit.dataviz.server import PORT_NAME

logger = logging.getLogger(__name__)

DEFAULT_CPU = 2.0
DEFAULT_MEMORY = "8GB"


def _dataviz_env_vars() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key.startswith("DATAVIZ_")}


def submit_dataviz(
    client: IrisClient,
    *,
    name: str,
    region: str,
    cpu: float,
    memory: str,
    env_vars: dict[str, str],
) -> Job:
    """Submit the dataviz service: a region-pinned, port-publishing, always-on job."""
    return client.submit(
        entrypoint=Entrypoint.from_command("python", "-m", "experiments.datakit.dataviz.server"),
        name=name,
        resources=ResourceSpec(cpu=cpu, memory=memory),
        environment=EnvironmentSpec(env_vars=env_vars),
        ports=[PORT_NAME],
        constraints=[region_constraint([region])],
        max_retries_preemption=1000,
    )


@click.command()
@click.option(
    "--controller-url",
    default=lambda: os.environ.get("IRIS_CONTROLLER_URL"),
    required=True,
    help="Iris controller URL, e.g. a tunnel from `iris --cluster=<name>` (default $IRIS_CONTROLLER_URL).",
)
@click.option("--region", default="us-east5", show_default=True, help="Region to pin the job to.")
@click.option("--name", default="dataviz", show_default=True, help="Job name.")
@click.option("--cpu", default=DEFAULT_CPU, show_default=True, type=float, help="CPUs to request.")
@click.option("--memory", default=DEFAULT_MEMORY, show_default=True, help="Memory to request.")
def cli(controller_url: str, region: str, name: str, cpu: float, memory: str) -> None:
    """Submit the always-on dataviz dashboard to an Iris cluster."""
    logging.basicConfig(level=logging.INFO)
    env_vars = _dataviz_env_vars()
    if "DATAVIZ_STORE" not in env_vars:
        raise click.UsageError("DATAVIZ_STORE not set — export the store to explore before deploying.")

    client = IrisClient.remote(controller_url, workspace=Path.cwd())
    job = submit_dataviz(client, name=name, region=region, cpu=cpu, memory=memory, env_vars=env_vars)
    logger.info(
        "submitted dataviz job %s (endpoint %r) — reachable at /proxy/dataviz/ once running",
        job.job_id,
        PORT_NAME,
    )


if __name__ == "__main__":
    cli()
