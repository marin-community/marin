# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deploy ducky as an always-on Iris job.

A routable service needs a *named* Iris port, which ``iris job run`` cannot declare
(no ``--ports`` flag). So deploy goes through the Python submit path
(``client.submit(..., ports=["ducky"])``), the same mechanism the actor worker pool
uses. The job is pinned to a single region and grabs a whole v6e-8 host (or a
smaller CPU-only shape for a smoke); the ``DUCKY_*`` config env vars are forwarded
to the task.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import click
from iris.cli.connect import IRIS_CLUSTER_CONFIG_DIRS
from iris.cli.main import create_client_token_provider
from iris.client.client import IrisClient, Job
from iris.cluster.config import IrisConfig
from iris.cluster.constraints import region_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, tpu_device
from iris.rpc.auth import TokenProvider
from rigging.config_discovery import resolve_cluster_config

logger = logging.getLogger(__name__)

# Prod default: a single v6e-8 host (ct6e-standard-8t). DuckDB runs on CPU/RAM only —
# the TPU sits idle; we request the slice to grab the host's large CPU/RAM envelope.
# TODO(ducky): confirm the exact host vCPU/RAM so the request schedules and gets the
# whole host (over-request → never schedules; under-request → DuckDB capped below host).
DEFAULT_TPU = "v6e-8"
DEFAULT_CPU = 180.0
DEFAULT_MEMORY = "1400GB"


def _ducky_env_vars() -> dict[str, str]:
    """Forward all DUCKY_* env vars from this process to the task."""
    return {key: value for key, value in os.environ.items() if key.startswith("DUCKY_")}


@contextmanager
def _controller_connection(
    cluster: str | None, controller_url: str | None
) -> Generator[tuple[str, TokenProvider | None]]:
    """Yield ``(controller_url, token_provider)``, tunneling to a named cluster if needed.

    Mirrors ``iris.cli.connect.require_controller_url``: a direct ``--controller-url``
    is used as-is; otherwise ``--cluster`` is resolved to its config, authenticated,
    and tunneled for the duration of the submit.
    """
    if controller_url:
        yield controller_url, None
        return
    if not cluster:
        raise click.UsageError("Pass --cluster <name> or --controller-url.")

    config_file = resolve_cluster_config(cluster, dirs=IRIS_CLUSTER_CONFIG_DIRS)
    iris_config = IrisConfig.load(str(config_file))
    proto = iris_config.proto
    token_provider = create_client_token_provider(proto.auth, cluster_name=cluster) if proto.HasField("auth") else None
    bundle = iris_config.provider_bundle()
    controller_address = iris_config.controller_address() or bundle.controller.discover_controller(proto.controller)
    logger.info("Tunneling to %s controller at %s", cluster, controller_address)
    with bundle.controller.tunnel(address=controller_address) as tunnel_url:
        yield tunnel_url, token_provider


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
        ports=[name],
        constraints=[region_constraint([region])],
        # Always-on: ride out preemptions rather than exiting.
        max_retries_preemption=1000,
    )


@click.command()
@click.option("--cluster", default=None, help="Named Iris cluster to deploy to (resolves config + tunnels).")
@click.option(
    "--controller-url",
    default=lambda: os.environ.get("IRIS_CONTROLLER_URL"),
    help="Direct controller URL (alternative to --cluster; default $IRIS_CONTROLLER_URL).",
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
def cli(
    cluster: str | None, controller_url: str | None, region: str, name: str, tpu: str, cpu: float, memory: str
) -> None:
    """Submit the always-on ducky service to an Iris cluster."""
    logging.basicConfig(level=logging.INFO)
    env_vars = _ducky_env_vars()
    if "DUCKY_SCRATCH_BUCKET" not in env_vars:
        raise click.UsageError("DUCKY_* env vars not set — export ducky config before deploying.")

    with _controller_connection(cluster, controller_url) as (url, token_provider):
        client = IrisClient.remote(url, workspace=Path.cwd(), token_provider=token_provider)
        job = submit_ducky(client, name=name, region=region, tpu=tpu, cpu=cpu, memory=memory, env_vars=env_vars)
    logger.info("submitted ducky job %s — reachable at /proxy/%s/ once running", job.job_id, name)


if __name__ == "__main__":
    cli()
