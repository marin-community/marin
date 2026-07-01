# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deploy ducky as an always-on Iris job.

A routable service needs a *named* Iris port, which ``iris job run`` cannot declare
(no ``--ports`` flag). So deploy goes through the Python submit path
(``client.submit(..., ports=["ducky"])``), the same mechanism the actor worker pool
uses. The job is pinned to a single region and grabs a whole v6e-4 host (or a
smaller CPU-only shape for a smoke); the ``DUCKY_*`` config env vars are forwarded
to the task.

Connect to the controller by passing ``--controller-url`` (e.g. a tunnel opened by
``iris --cluster=<name> ...``). Deploy keeps no cluster-resolution/auth logic of its
own — that lives in the iris CLI.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import click
from iris.client.client import IrisClient, Job
from iris.cluster.constraints import region_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, tpu_device

logger = logging.getLogger(__name__)

# Prod default: a single ct6e-standard-4t v6e host — 180 vCPU / 720 GB advertised,
# ~700 GB allocatable after system reservation. DuckDB runs on CPU/RAM only (the v6e
# chips sit idle); we grab the slice for its large CPU/RAM envelope. v6e-4 is the
# largest SINGLE-VM v6e on marin — v6e-8 is a 2-VM slice and DuckDB, being
# single-node, can only use one VM, so it gains nothing from v6e-8. Memory is set
# below allocatable so the request schedules.
DEFAULT_TPU = "v6e-4"
DEFAULT_CPU = 180.0
DEFAULT_MEMORY = "690GB"

# The Iris named port and endpoint name the server binds/registers. Must match
# DuckyConfig.port_name (the server calls ctx.get_port(config.port_name)), and is
# independent of the job --name. The dashboard resolves to the namespaced endpoint
# wire name "<job-namespace>/ducky" through the controller proxy.
PORT_NAME = "ducky"


def _ducky_env_vars() -> dict[str, str]:
    """Forward all DUCKY_* env vars from this process to the task."""
    return {key: value for key, value in os.environ.items() if key.startswith("DUCKY_")}


def submit_ducky(
    client: IrisClient,
    *,
    name: str,
    region: str,
    tpu: str,
    cpu: float,
    memory: str,
    env_vars: dict[str, str],
) -> Job:
    """Submit the ducky service job: a region-pinned, port-publishing, always-on job."""
    device = tpu_device(tpu) if tpu else None
    return client.submit(
        entrypoint=Entrypoint.from_command("python", "-m", "ducky.server"),
        name=name,
        resources=ResourceSpec(cpu=cpu, memory=memory, device=device),
        environment=EnvironmentSpec(env_vars=env_vars),
        ports=[PORT_NAME],
        constraints=[region_constraint([region])],
        # Always-on: ride out preemptions, and also failures — the in-process supervisor
        # restarts the server on an OOM, but if the whole container is cgroup-killed this
        # lets Iris bring the task back instead of leaving the service down.
        max_retries_preemption=1000,
        max_retries_failure=1000,
    )


@click.command()
@click.option(
    "--controller-url",
    default=lambda: os.environ.get("IRIS_CONTROLLER_URL"),
    required=True,
    help="Iris controller URL, e.g. a tunnel from `iris --cluster=<name>` (default $IRIS_CONTROLLER_URL).",
)
@click.option("--region", default="us-east5", show_default=True, help="Region to pin the job to.")
@click.option("--name", default="ducky", show_default=True, help="Job name.")
@click.option(
    "--tpu",
    default=DEFAULT_TPU,
    show_default=True,
    help="TPU variant for the whole-host grab. Pass empty ('') for a CPU-only smoke deploy.",
)
@click.option("--cpu", default=DEFAULT_CPU, show_default=True, type=float, help="CPUs to request.")
@click.option("--memory", default=DEFAULT_MEMORY, show_default=True, help="Memory to request (e.g. 16GB).")
def cli(controller_url: str, region: str, name: str, tpu: str, cpu: float, memory: str) -> None:
    """Submit the always-on ducky service to an Iris cluster."""
    logging.basicConfig(level=logging.INFO)
    env_vars = _ducky_env_vars()
    if "DUCKY_SCRATCH_BUCKET" not in env_vars:
        raise click.UsageError("DUCKY_* env vars not set — export ducky config before deploying.")

    client = IrisClient.remote(controller_url, workspace=Path.cwd())
    job = submit_ducky(client, name=name, region=region, tpu=tpu, cpu=cpu, memory=memory, env_vars=env_vars)
    logger.info(
        "submitted ducky job %s (endpoint %r) — reachable via the controller endpoint proxy once running",
        job.job_id,
        PORT_NAME,
    )


if __name__ == "__main__":
    cli()
