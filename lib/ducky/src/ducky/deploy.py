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
from iris.client.client import IrisClient, Job
from iris.cluster.config import IrisConfig
from iris.cluster.constraints import region_constraint
from iris.cluster.token_store import load_any_token, load_token
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, tpu_device
from iris.rpc import config_pb2
from iris.rpc.auth import GcpAccessTokenProvider, StaticTokenProvider, TokenProvider
from rigging.config_discovery import resolve_cluster_config

logger = logging.getLogger(__name__)

# Prod default: a single v6e-8 host (ct6e-standard-8t). DuckDB runs on CPU/RAM only —
# the TPU sits idle; we request the slice to grab the host's large CPU/RAM envelope.
# TODO(ducky): confirm the exact host vCPU/RAM so the request schedules and gets the
# whole host (over-request → never schedules; under-request → DuckDB capped below host).
DEFAULT_TPU = "v6e-8"
DEFAULT_CPU = 180.0
DEFAULT_MEMORY = "1400GB"

# The Iris named port and endpoint name the server binds/registers. Must match
# DuckyConfig.port_name (the server calls ctx.get_port(config.port_name)), and is
# independent of the job --name. The dashboard resolves to the namespaced endpoint
# wire name "<job-namespace>/ducky" through the controller proxy.
PORT_NAME = "ducky"


def _ducky_env_vars() -> dict[str, str]:
    """Forward all DUCKY_* env vars from this process to the task."""
    return {key: value for key, value in os.environ.items() if key.startswith("DUCKY_")}


def _token_provider(auth_config: config_pb2.AuthConfig, cluster_name: str) -> TokenProvider | None:
    """Build a client TokenProvider for a cluster: a stored `iris login` JWT, else the config's auth method.

    Mirrors the CLI's token resolution without importing the CLI module.
    """
    credential = load_token(cluster_name) or load_any_token()
    if credential is not None:
        return StaticTokenProvider(credential.token)
    provider = auth_config.WhichOneof("provider")
    if provider is None:
        return None
    if provider == "gcp":
        return GcpAccessTokenProvider()
    if provider == "static":
        tokens = dict(auth_config.static.tokens)
        if not tokens:
            raise ValueError("Static auth config requires at least one token")
        return StaticTokenProvider(next(iter(tokens)))
    raise ValueError(f"Unknown auth provider: {provider}")


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
    token_provider = _token_provider(proto.auth, cluster) if proto.HasField("auth") else None
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
        ports=[PORT_NAME],
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
    logger.info(
        "submitted ducky job %s (endpoint %r) — reachable via the controller endpoint proxy once running",
        job.job_id,
        PORT_NAME,
    )


if __name__ == "__main__":
    cli()
