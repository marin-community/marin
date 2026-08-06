# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deploy DataFusion-Ballista on Iris: one scheduler + N executors + a query client.

PoC launcher for issue #7266. It stands up a Ballista cluster as three Iris jobs from
a custom image whose binaries are patched for gs:// object-store access (see
``lib/smallquery/docker/`` and ``.agents/projects/smallquery/ballista-poc-phase0.md``).
Preemption and durability are out of scope: the scheduler is pinned non-preemptible and
the executors are a plain sized job.

Discovery is launcher-threaded — no service registry and no Python in the tasks. The
launcher submits the scheduler, reads its routable ``host:port`` from the Iris task
status, and passes that to the executors and the client as argv. Tasks are bare
Ballista binaries run with setup skipped (``EnvironmentSpec(setup_scripts=[])``), so the
image is used as-is; ``bash -c`` expands the ``IRIS_PORT_*`` / ``IRIS_ADVERTISE_HOST``
env vars the worker injects.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import click
from iris.cli.connect import open_iris_client
from iris.client.client import IrisClient, Job
from iris.cluster.constraints import preemptible_constraint, region_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "ghcr.io/marin-community/smallquery-ballista:latest"

# Default scheduler gRPC port
DEFAULT_SCHEDULER_PORT = 50050

# Named Iris ports. The worker injects each as IRIS_PORT_<UPPERCASE>.

SCHEDULER_PORT = "scheduler"  # scheduler gRPC (executors + client connect here)
EXECUTOR_FLIGHT_PORT = "flight"  # executor Arrow Flight (shuffle fetch): --bind-port
EXECUTOR_GRPC_PORT = "grpc"  # executor control gRPC: --bind-grpc-port

_TERMINAL_TASK_STATES = frozenset(
    {
        job_pb2.TASK_STATE_FAILED,
        job_pb2.TASK_STATE_KILLED,
        job_pb2.TASK_STATE_WORKER_FAILED,
        job_pb2.TASK_STATE_UNSCHEDULABLE,
        job_pb2.TASK_STATE_COSCHED_FAILED,
    }
)


@dataclass(frozen=True)
class SchedulerEndpoint:
    """Routable address executors and the client dial to reach the scheduler."""

    host: str
    port: int


def _bare_binary_env() -> EnvironmentSpec:
    """Run the image's binaries as-is (no uv sync), with readable Rust logs."""
    return EnvironmentSpec(setup_scripts=[], env_vars={"RUST_LOG": "info"})


def submit_scheduler(
    client: IrisClient,
    *,
    image: str,
    region: str,
    cpu: float,
    memory: str,
    name: str,
) -> Job:
    """Submit the single, non-preemptible ballista-scheduler task."""
    command = (
        "exec ballista-scheduler"
        " --bind-host 0.0.0.0"
        f" --bind-port {DEFAULT_SCHEDULER_PORT}"
        " --external-host $IRIS_ADVERTISE_HOST"
    )
    return client.submit(
        entrypoint=Entrypoint.from_command("bash", "-c", command),
        name=name,
        resources=ResourceSpec(cpu=cpu, memory=memory),
        environment=_bare_binary_env(),
        constraints=[preemptible_constraint(False), region_constraint([region])],
        task_image=image,
    )


def scheduler_endpoint(job: Job, *, timeout: float = 600.0, poll: float = 2.0) -> SchedulerEndpoint:
    """Block until the scheduler task is RUNNING, then return its routable host:port.

    ``worker_address`` is ``<worker_ip>:<agent_port>``; the agent port is irrelevant, so
    take the host and pair it with the scheduler's allocated named port.
    """
    deadline = time.monotonic() + timeout
    last_log_state = None
    while time.monotonic() < deadline:
        tasks = [t.status() for t in job.tasks()]
        for task in tasks:
            if task.state in _TERMINAL_TASK_STATES:
                raise RuntimeError(f"scheduler task entered {job_pb2.TaskState.Name(task.state)}: {task.error}")
            if task.state != last_log_state:
                logger.info("scheduler task state: %s", job_pb2.TaskState.Name(task.state))
                last_log_state = task.state
            if task.state == job_pb2.TASK_STATE_RUNNING and task.worker_address:
                host = task.worker_address.rsplit(":", 1)[0]
                port = task.ports.get(SCHEDULER_PORT, DEFAULT_SCHEDULER_PORT)
                endpoint = SchedulerEndpoint(host=host, port=port)
                logger.info("scheduler ready at %s:%d", endpoint.host, endpoint.port)
                return endpoint
        time.sleep(poll)
    raise TimeoutError(f"scheduler not RUNNING within {timeout}s")


def submit_executors(
    client: IrisClient,
    *,
    image: str,
    region: str,
    scheduler: SchedulerEndpoint,
    num_workers: int,
    cpu: float,
    memory: str,
    name: str,
) -> Job:
    """Submit a sized job of ballista-executor tasks that register with the scheduler."""
    command = (
        "exec ballista-executor"
        " --bind-host 0.0.0.0"
        " --bind-port $IRIS_PORT_FLIGHT"
        " --bind-grpc-port $IRIS_PORT_GRPC"
        " --external-host $IRIS_ADVERTISE_HOST"
        f" --scheduler-host {scheduler.host}"
        f" --scheduler-port {scheduler.port}"
    )
    return client.submit(
        entrypoint=Entrypoint.from_command("bash", "-c", command),
        name=name,
        resources=ResourceSpec(cpu=cpu, memory=memory),
        environment=_bare_binary_env(),
        ports=[EXECUTOR_FLIGHT_PORT, EXECUTOR_GRPC_PORT],
        constraints=[region_constraint([region])],
        replicas=num_workers,
        task_image=image,
        max_task_failures=max(10, num_workers // 10),
        max_retries_failure=3,
        max_retries_preemption=10,
    )


def wait_for_executors(job: Job, num_workers: int, *, timeout: float = 600.0, poll: float = 5.0) -> None:
    """Block until a sufficient number of executors are running, or a timeout is reached."""
    deadline = time.monotonic() + timeout
    last_running = -1
    first_ready_time = None
    while time.monotonic() < deadline:
        tasks = [t.status() for t in job.tasks()]
        running = sum(1 for task in tasks if task.state == job_pb2.TASK_STATE_RUNNING)
        succeeded = sum(1 for task in tasks if task.state == job_pb2.TASK_STATE_SUCCEEDED)
        failed = sum(1 for task in tasks if task.state in (job_pb2.TASK_STATE_FAILED, job_pb2.TASK_STATE_WORKER_FAILED))

        if running != last_running:
            logger.info(
                "Executors status: %d running, %d succeeded, %d failed (target: %d)",
                running,
                succeeded,
                failed,
                num_workers,
            )
            last_running = running

        if running >= num_workers:
            logger.info("All target executors running!")
            return

        if running >= 1 and first_ready_time is None:
            first_ready_time = time.monotonic()

        # If at least 80% of the executors are running/succeeded, or we have
        # been waiting 45 seconds since the first one became ready:
        if first_ready_time is not None:
            elapsed_since_ready = time.monotonic() - first_ready_time
            if (running + succeeded) >= int(num_workers * 0.8) or elapsed_since_ready > 45.0:
                logger.info(
                    "Proceeding with %d active executors (succeeded: %d, elapsed since ready: %.1fs)",
                    running,
                    succeeded,
                    elapsed_since_ready,
                )
                return

        time.sleep(poll)

    if last_running == 0:
        raise TimeoutError(f"No executors became RUNNING within {timeout}s")


def run_query(
    client: IrisClient,
    *,
    image: str,
    region: str,
    scheduler: SchedulerEndpoint,
    sql: str,
    name: str,
) -> Job:
    """Submit a one-shot ballista-cli task that runs ``sql`` against the scheduler.

    The client runs in-cluster because the scheduler's routable IP is internal. Results
    print to the task's stdout, surfaced via streamed job logs.
    """
    entrypoint = Entrypoint(
        command=["bash", "-c", f"exec ballista-cli --host {scheduler.host} --port {scheduler.port} -f /app/query.sql"],
        workdir_files={"query.sql": sql.encode()},
    )
    return client.submit(
        entrypoint=entrypoint,
        name=name,
        resources=ResourceSpec(cpu=0.1, memory="256Mi"),
        environment=_bare_binary_env(),
        constraints=[region_constraint([region])],
        task_image=image,
    )


def _build_sql(data: str | None, table: str, sql: str) -> str:
    """Optionally prepend a CREATE EXTERNAL TABLE over ``data`` before ``sql``."""
    if not data:
        return sql
    return f"CREATE EXTERNAL TABLE {table} STORED AS PARQUET LOCATION '{data}';\n{sql}"


@click.command()
@click.option("--cluster", default=None, help="Iris cluster to connect to (resolves controller URL + auth).")
@click.option("--controller-url", default=None, help="Explicit Iris controller URL; exclusive with --cluster.")
@click.option("--region", default="us-central2", show_default=True, help="Region to pin all jobs to (data-local).")
@click.option("--image", default=DEFAULT_IMAGE, show_default=True, help="Ballista task image (patched for gs://).")
@click.option("--workers", "-n", default=2, show_default=True, type=int, help="Number of executors.")
@click.option("--data", default=None, help="gs:// Parquet path to register as an external table before the query.")
@click.option("--table", default="t", show_default=True, help="Table name for --data.")
@click.option("--sql", default="SELECT count(*) FROM t", show_default=True, help="SQL to run.")
@click.option("--scheduler-cpu", default=2.0, show_default=True, type=float)
@click.option("--scheduler-memory", default="4g", show_default=True)
@click.option("--executor-cpu", default=4.0, show_default=True, type=float)
@click.option("--executor-memory", default="8g", show_default=True)
@click.option("--keep", is_flag=True, help="Leave the scheduler and executors running after the query.")
def cli(
    cluster: str | None,
    controller_url: str | None,
    region: str,
    image: str,
    workers: int,
    data: str | None,
    table: str,
    sql: str,
    scheduler_cpu: float,
    scheduler_memory: str,
    executor_cpu: float,
    executor_memory: str,
    keep: bool,
) -> None:
    """Deploy Ballista on Iris and run one query. Tears the cluster down unless --keep."""
    logging.basicConfig(level=logging.INFO)
    if cluster and controller_url:
        raise click.UsageError("Pass --cluster or --controller-url, not both.")
    if not cluster and not controller_url:
        raise click.UsageError("Pass --cluster <name> or --controller-url <url>.")

    full_sql = _build_sql(data, table, sql)

    ts = int(time.time())
    with open_iris_client(cluster_name=cluster, controller_url=controller_url, workspace=Path.cwd()) as client:
        scheduler_job = submit_scheduler(
            client,
            image=image,
            region=region,
            cpu=scheduler_cpu,
            memory=scheduler_memory,
            name=f"ballista-scheduler-{ts}",
        )
        executor_job = None
        try:
            endpoint = scheduler_endpoint(scheduler_job)
            executor_job = submit_executors(
                client,
                image=image,
                region=region,
                scheduler=endpoint,
                num_workers=workers,
                cpu=executor_cpu,
                memory=executor_memory,
                name=f"ballista-executors-{ts}",
            )
            wait_for_executors(executor_job, workers)

            client_job = run_query(
                client, image=image, region=region, scheduler=endpoint, sql=full_sql, name=f"ballista-query-{ts}"
            )

            logger.info("running query on %s:%d", endpoint.host, endpoint.port)
            client_job.wait(timeout=600.0, stream_logs=True, raise_on_failure=False)
        finally:
            if keep:
                logger.info(
                    "leaving cluster up (--keep): scheduler=%s executors=%s",
                    scheduler_job.job_id,
                    executor_job.job_id if executor_job else "n/a",
                )
            else:
                if executor_job is not None:
                    executor_job.terminate()
                scheduler_job.terminate()


if __name__ == "__main__":
    cli()
