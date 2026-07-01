# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``buoy`` CLI: submit the viewer as a self-registering service job on a cluster.

Resolves the named cluster's controller (auto-tunnel), submits an Iris job whose
entrypoint serves the app and registers ``/buoy``, then waits for the endpoint and
prints the proxy URL. buoy is a uv workspace member, so the container installs it
from the frozen lock and the entrypoint (``serve_in_job``) ships by reference.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import click
from iris.cli.connect import IRIS_CLUSTER_CONFIG_DIRS
from iris.client import IrisClient
from iris.cluster.composer import provider_bundle
from iris.cluster.config import load_config
from iris.cluster.constraints import region_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, is_job_finished
from rigging.config_discovery import resolve_cluster_config
from rigging.connect import proxy_path
from rigging.filesystem import marin_temp_bucket, region_from_prefix
from rigging.timing import Duration

from buoy.config import CACHE_PREFIX, CACHE_TTL_DAYS
from buoy.serve import serve_in_job

logger = logging.getLogger("buoy.launch")


@click.command(context_settings={"show_default": True})
@click.option("--cluster", default="marin", help="Named Iris cluster to submit to.")
@click.option("--endpoint-name", default="/buoy", help="Endpoint to register (absolute, no '.').")
@click.option("--name", default="buoy", help="Iris job name.")
@click.option(
    "--cache-root",
    default="",
    help="GCS prefix for the run cache. Empty = derive a region-local marin_temp_bucket "
    "from the cluster's state region (avoids cross-region reads).",
)
@click.option("--cpu", type=float, default=2.0)
@click.option("--memory", default="12g")
@click.option("--disk", default="50g")
@click.option("--timeout", type=int, default=0, help="Job timeout seconds (0 = no timeout).")
@click.option("--max-retries-preemption", type=int, default=10)
@click.option("--wait-timeout", type=float, default=600.0)
def cli(
    cluster: str,
    endpoint_name: str,
    name: str,
    cache_root: str,
    cpu: float,
    memory: str,
    disk: str,
    timeout: int,
    max_retries_preemption: int,
    wait_timeout: float,
) -> None:
    logging.basicConfig(level=logging.INFO, format="[buoy-launch] %(message)s")
    if not endpoint_name.startswith("/") or "." in endpoint_name:
        raise click.ClickException("--endpoint-name must be absolute (start with '/') and contain no '.'.")

    resolved = resolve_cluster_config(cluster, dirs=IRIS_CLUSTER_CONFIG_DIRS)
    config = load_config(str(resolved))
    dashboard_url = config.dashboard_url or None
    # Pin the cache to the cluster's own region. The state dir's bucket fixes the
    # region directly (no MARIN_PREFIX needed on the worker), so the service never
    # reads/writes the cache cross-region.
    if not cache_root:
        cache_root = marin_temp_bucket(CACHE_TTL_DAYS, CACHE_PREFIX, source_prefix=config.storage.remote_state_dir)
    # Pin the worker to the cache's region so the service never moves history/profile
    # data cross-region (the cache bucket determines the region).
    region = region_from_prefix(cache_root)
    constraints = [region_constraint([region])] if region else None
    logger.info("cache_root %s region %s", cache_root, region)
    bundle = provider_bundle(config)
    controller_address = config.controller_address() or bundle.controller.discover_controller(config.controller)
    logger.info("opening tunnel to controller %s", controller_address)

    with (
        bundle.controller.tunnel(address=controller_address) as controller_url,
        IrisClient.remote(controller_url, workspace=Path.cwd()) as client,
    ):
        job = client.submit(
            entrypoint=Entrypoint.from_callable(serve_in_job, endpoint_name),
            name=name,
            resources=ResourceSpec(cpu=cpu, memory=memory, disk=disk),
            environment=EnvironmentSpec(env_vars={"BUOY_CACHE_ROOT": cache_root}),
            ports=["http"],
            constraints=constraints,
            timeout=Duration.from_seconds(timeout) if timeout else None,
            max_retries_failure=0,
            max_retries_preemption=max_retries_preemption,
        )
        proxy = proxy_path(endpoint_name)
        share = f"{dashboard_url.rstrip('/')}{proxy}/" if dashboard_url else f"{proxy}/"
        logger.info("submitted job %s", job)
        logger.info("endpoint     %s", endpoint_name)
        logger.info("share url    %s", share)
        logger.info("stop with    iris job stop %s --cluster %s", job, cluster)

        deadline = time.monotonic() + wait_timeout
        while time.monotonic() < deadline:
            if is_job_finished(job.state):
                raise click.ClickException(f"job {job} finished before registering; check `iris job logs {job}`")
            endpoints = client._cluster_client.list_endpoints(endpoint_name, exact=True)
            if endpoints:
                logger.info("READY — endpoint at %s", endpoints[0].address)
                logger.info("open: %s", share)
                return
            time.sleep(5)
        logger.warning("timed out waiting for endpoint; job may still be booting (`iris job logs %s`)", job)


if __name__ == "__main__":
    cli()
