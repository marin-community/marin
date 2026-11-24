# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the GLM synthesis serving pool: staging, coordinator, and watchdog jobs.

Run from any machine that can reach the pool cluster's controller (the worktree is
bundled so workers restore this code). The jobs:

- ``stage`` mirrors the pinned HF revision into the cluster's regional storage
  (a cache check when the completion marker already exists).
- the coordinator runs ``serve_pool.serve``: it creates the broker, submits one
  engine worker per instance, registers the endpoint, and self-stops after the
  coordinator timeout.
- the watchdog resubmits the coordinator when the endpoint stays unhealthy, so the
  pool survives engine deaths and coordinator rolls for as long as it is wanted.

Example::

    uv run python -m experiments.glm_synthesis_serving.launch_pool --warm-run
    uv run python -m experiments.glm_synthesis_serving.launch_pool
"""

import argparse
import logging
import time
import uuid

import click
from fray.iris_backend import convert_resources
from fray.types import ResourceConfig
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec, is_job_finished
from rigging.config_discovery import find_project_root
from rigging.timing import Duration

from experiments.glm_synthesis_serving import pool, pool_watchdog, serve_pool, stage_weights

logger = logging.getLogger(__name__)

_STAGE_CPU = 8.0
_STAGE_RAM = "16g"
_STAGE_DISK = "64g"
_STAGE_TIMEOUT = Duration.from_hours(6)
_COORDINATOR_CPU = 2.0
_COORDINATOR_RAM = "8g"
_COORDINATOR_DISK = "20g"
_ENDPOINT_POLL_SECONDS = 15.0
_ENDPOINT_READY_TIMEOUT = Duration.from_hours(3)
_CAPABILITY_TTL_HOURS = 24 * 7


def _pool_cpu_resources(*, cpu: float, ram: str, disk: str) -> ResourceConfig:
    return ResourceConfig.with_cpu(
        cpu=cpu,
        ram=ram,
        disk=disk,
        preemptible=False,
        target_cluster=pool.CLUSTER,
    )


def submit_stage_job(client: IrisClient, suffix: str):
    """Submit the in-region weight staging job and return its handle."""
    return client.submit(
        entrypoint=Entrypoint.from_callable(stage_weights.stage),
        name=f"{pool.JOB_PREFIX}-stage-{suffix}",
        resources=convert_resources(_pool_cpu_resources(cpu=_STAGE_CPU, ram=_STAGE_RAM, disk=_STAGE_DISK)),
        environment=EnvironmentSpec(),
        priority_band=pool.PRIORITY,
        timeout=_STAGE_TIMEOUT,
        max_retries_failure=1,
    )


def submit_coordinator_job(
    client: IrisClient,
    *,
    instances: int,
    suffix: str,
    extra_args: tuple[str, ...] = (),
    endpoint_name: str | None = None,
    worker_env_vars: tuple[tuple[str, str], ...] = (),
):
    """Submit the serving coordinator (broker + engine workers + endpoint registration)."""
    entrypoint_args: tuple = (instances, extra_args)
    if endpoint_name is not None or worker_env_vars:
        entrypoint_args = (*entrypoint_args, endpoint_name)
    if worker_env_vars:
        entrypoint_args = (*entrypoint_args, worker_env_vars)
    return client.submit(
        entrypoint=Entrypoint.from_callable(serve_pool.serve, *entrypoint_args),
        name=f"{pool.JOB_PREFIX}-{suffix}",
        resources=convert_resources(
            _pool_cpu_resources(cpu=_COORDINATOR_CPU, ram=_COORDINATOR_RAM, disk=_COORDINATOR_DISK)
        ),
        environment=EnvironmentSpec(),
        priority_band=pool.PRIORITY,
        ports=["http"],
        timeout=Duration.from_hours(pool.COORDINATOR_TIMEOUT_HOURS),
    )


def submit_watchdog_job(client: IrisClient, suffix: str):
    """Submit the watchdog that keeps the endpoint resubmitted across failures."""
    return client.submit(
        entrypoint=Entrypoint.from_callable(pool_watchdog.watch_forever),
        name=f"{pool.JOB_PREFIX}-watch-{suffix}",
        resources=convert_resources(
            _pool_cpu_resources(cpu=_COORDINATOR_CPU, ram=_COORDINATOR_RAM, disk=_COORDINATOR_DISK)
        ),
        environment=EnvironmentSpec(),
        priority_band=pool.PRIORITY,
        timeout=Duration.from_hours(pool_watchdog.WATCHDOG_GENERATION_HOURS),
    )


def wait_for_endpoint(
    client: IrisClient,
    timeout: Duration = _ENDPOINT_READY_TIMEOUT,
    coordinator=None,
    endpoint_name: str | None = None,
) -> str:
    """Poll the controller registry until the pool endpoint registers; return its URL.

    With `coordinator`, abort as soon as that job reaches a terminal state — a failed
    engine boot must wake the caller instead of burning the whole wait budget.
    """
    name = endpoint_name or pool.ENDPOINT_NAME
    deadline = time.monotonic() + timeout.to_seconds()
    while time.monotonic() < deadline:
        if coordinator is not None:
            # Accept a JobHandle (has .status()) or an already-fetched JobStatus.
            state = coordinator.status().state if callable(getattr(coordinator, "status", None)) else coordinator.state
            if is_job_finished(state):
                raise RuntimeError("coordinator finished before registering the endpoint")
        if client.list_endpoint_instances(name):
            return client.resolve_endpoint(name)
        time.sleep(_ENDPOINT_POLL_SECONDS)
    raise TimeoutError(f"pool endpoint {name} did not register within {timeout}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-run", action="store_true", help="Launch one engine instead of the full pool.")
    parser.add_argument(
        "--instances",
        type=int,
        default=None,
        help="Engine count (default: the full pool; --warm-run overrides to 1).",
    )
    parser.add_argument(
        "--mtp",
        type=int,
        default=None,
        help="Enable MTP speculative decoding with this many speculative tokens (warm-run variant).",
    )
    parser.add_argument(
        "--engine-arg",
        action="append",
        default=None,
        help="Extra raw vLLM argument appended to the engine config (repeatable; variant benching).",
    )
    parser.add_argument(
        "--worker-env",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Environment variable added to the engine workers (repeatable; stability variants).",
    )
    parser.add_argument(
        "--endpoint-name",
        default=None,
        help="Register this endpoint instead of the pool's (variant benching).",
    )
    parser.add_argument("--skip-stage", action="store_true", help="Assume the weights prefix is already staged.")
    parser.add_argument("--no-watchdog", action="store_true", help="Do not start the watchdog job.")
    parser.add_argument(
        "--wait",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait for the endpoint and print capability URL.",
    )
    parser.add_argument(
        "--attach-only",
        action="store_true",
        help="Submit nothing; wait for an already-running pool's endpoint and print the URL.",
    )
    parser.add_argument("--tag", default=None, help="Job name tag (default: short hex).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[launch-pool] %(message)s")
    instances = args.instances or (pool.WARM_RUN_INSTANCES if args.warm_run else pool.POOL_INSTANCES)
    extra_args: tuple[str, ...] = ()
    if args.mtp is not None:
        if args.mtp < 1:
            raise SystemExit("--mtp must be positive")
        extra_args = ("--speculative-config", f'{{"method":"mtp","num_speculative_tokens":{args.mtp}}}')
    if args.engine_arg:
        extra_args = (*extra_args, *args.engine_arg)
    worker_env_vars: tuple[tuple[str, str], ...] = ()
    if args.worker_env:
        worker_env_vars = tuple(
            (pair.partition("=")[0], pair.partition("=")[2]) for pair in args.worker_env
        )
        if any(not key for key, _ in worker_env_vars):
            raise SystemExit("--worker-env entries must be KEY=VALUE")
    suffix = args.tag or uuid.uuid4().hex[:6]
    endpoint_name = args.endpoint_name
    workspace = find_project_root()
    if workspace is None:
        raise SystemExit("launch_pool must run inside a marin checkout (the workspace is bundled for workers).")

    with connect_controller(cluster_name=pool.CLUSTER) as endpoint_info:
        with IrisClient.remote(endpoint_info.url, workspace=workspace, credentials=endpoint_info.credentials) as client:
            if args.attach_only:
                live = client.active_job_names_for_prefix(f"/{pool.JOB_PREFIX}-")
                coordinator_status = next(
                    (
                        client.job_status(job_id)
                        for job_id in live
                        if "-watch-" not in str(job_id) and "-stage-" not in str(job_id)
                    ),
                    None,
                )
                base_url = wait_for_endpoint(client, coordinator=coordinator_status)
                click.echo(f"READY — endpoint: {base_url}")
                return

            if not args.skip_stage:
                stage_job = submit_stage_job(client, suffix)
                logger.info("staging job %s", stage_job.job_id)
                stage_job.wait(timeout=_STAGE_TIMEOUT.to_seconds())
                logger.info("weights staged")

            coordinator = submit_coordinator_job(
                client,
                instances=instances,
                suffix=suffix,
                extra_args=extra_args,
                endpoint_name=endpoint_name,
                worker_env_vars=worker_env_vars,
            )
            logger.info("coordinator job %s (instances=%d)", coordinator.job_id, instances)

            if not args.no_watchdog:
                watchdog = submit_watchdog_job(client, suffix)
                logger.info("watchdog job %s", watchdog.job_id)

            if not args.wait:
                logger.info("submitted without waiting; the endpoint registers when the engines are up")
                return

            base_url = wait_for_endpoint(client, coordinator=coordinator, endpoint_name=endpoint_name)
            click.echo("")
            click.echo(f"READY — endpoint: {base_url}")
            click.echo(f"        model:    {pool.SERVED_MODEL_NAME}  ({pool.MODEL_REPO}@{pool.MODEL_REVISION})")
            click.echo(f"        pool:     {instances} x TP{pool.GPUS_PER_ENGINE} engines on {pool.CLUSTER}")
            response = client.mint_endpoint_token(
                endpoint_name or pool.ENDPOINT_NAME, ttl=Duration.from_hours(_CAPABILITY_TTL_HOURS)
            )
            origin = response.capability_url or ""
            if origin:
                click.echo(f"        shared:   {origin}/v1  (capability URL; expires in {_CAPABILITY_TTL_HOURS}h)")
            click.echo(f"        cancel:   iris --cluster {pool.CLUSTER} job cancel {coordinator.job_id}")


if __name__ == "__main__":
    main()
