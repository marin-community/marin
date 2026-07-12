# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deploy ducky as an always-on Iris job.

A routable service needs a *named* Iris port, which ``iris job run`` cannot declare
(no ``--ports`` flag). So deploy goes through the Python submit path
(``client.submit(..., ports=["ducky"])``), the same mechanism the actor worker pool
uses. The job is pinned to a single region and grabs a whole v6e-4 host (or a
smaller CPU-only shape for a smoke); the ``DUCKY_*`` config env vars are forwarded
to the task.

Connect to the controller with ``--cluster <name>`` (resolved through the iris CLI's
``open_iris_client``, which handles both IAP-fronted clusters — a direct HTTPS ingress
with a service-account-minted IAP token — and SSH-tunnelled ones) or an explicit
``--controller-url``. Deploy keeps no cluster-resolution/auth logic of its own — that
lives in the iris CLI.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

import click
from iris.cli.connect import open_iris_client
from iris.client.client import IrisClient, Job
from iris.cluster.constraints import region_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, tpu_device
from iris.rpc import job_pb2

from ducky.config import PORT_NAME

logger = logging.getLogger(__name__)

DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
# The built SPA is gitignored; this glob (relative to the workspace root) tells the Iris
# bundle to ship it to the worker anyway. Owning it here keeps ducky-specific paths out of
# Iris's bundler.
DASHBOARD_DIST_INCLUDE = "lib/ducky/dashboard/dist/**/*"

# Prod default: a single ct6e-standard-4t v6e host — 180 vCPU / 720 GB advertised,
# ~700 GB allocatable after system reservation. DuckDB runs on CPU/RAM only (the v6e
# chips sit idle); we grab the slice for its large CPU/RAM envelope. v6e-4 is the
# largest SINGLE-VM v6e on marin — v6e-8 is a 2-VM slice and DuckDB, being
# single-node, can only use one VM, so it gains nothing from v6e-8. Memory is set
# below allocatable so the request schedules.
DEFAULT_TPU = "v6e-4"
DEFAULT_CPU = 180.0
DEFAULT_MEMORY = "690GB"


def _ducky_env_vars() -> dict[str, str]:
    """Forward all DUCKY_* env vars from this process to the task."""
    return {key: value for key, value in os.environ.items() if key.startswith("DUCKY_")}


# "Effectively unlimited" retry ceiling. Iris has no infinite sentinel and its retry
# budgets are lifetime totals that never reset, so a long-lived service must set a bound
# so large it can't realistically exhaust. Each retry is paced by VM boot / capacity (not a
# tight loop), so a huge value doesn't hot-loop; a permanently-broken deploy is caught by
# CI and the redeploy watchdog, not by letting the job die.
_EFFECTIVELY_UNLIMITED_RETRIES = 1_000_000


def submit_ducky(
    client: IrisClient,
    *,
    name: str,
    region: str,
    tpu: str,
    cpu: float,
    memory: str,
    env_vars: dict[str, str],
    existing_job_policy: job_pb2.ExistingJobPolicy = job_pb2.EXISTING_JOB_POLICY_RECREATE,
) -> Job:
    """Submit the ducky service job: a region-pinned, port-publishing, best-effort-always-on job.

    Best-effort-always-up rests on three layers: the in-process supervisor restarts the
    server on an in-container crash/OOM (no Iris budget spent); Iris task retries bring the
    task back if the whole container dies or the VM is preempted; and all three budgets are
    set effectively unlimited so neither preemptions nor failures ever terminate the job.
    ``max_task_failures`` is the critical one — it's a job-level cumulative budget that
    defaults to 0, so a single hard container failure would otherwise fail the whole job
    regardless of the per-task retry budget.
    """
    device = tpu_device(tpu) if tpu else None
    return client.submit(
        entrypoint=Entrypoint.from_command("python", "-m", "ducky.server"),
        name=name,
        resources=ResourceSpec(cpu=cpu, memory=memory, device=device),
        environment=EnvironmentSpec(env_vars=env_vars),
        ports=[PORT_NAME],
        constraints=[region_constraint([region])],
        max_retries_preemption=_EFFECTIVELY_UNLIMITED_RETRIES,  # never let preemptions end the service
        max_retries_failure=_EFFECTIVELY_UNLIMITED_RETRIES,  # per-task budget for whole-container deaths
        max_task_failures=_EFFECTIVELY_UNLIMITED_RETRIES,  # job-level cumulative budget (defaults to 0!)
        existing_job_policy=existing_job_policy,
    )


def _build_dashboard() -> None:
    """Build the Vue dashboard so the gitignored ``dashboard/dist`` ships in the bundle.

    Runs ``npm install`` (only if deps are missing) then ``npm run build``. The Iris
    bundle re-includes the gitignored ``dist`` via ``GENERATED_ARTIFACT_GLOBS``.
    """
    npm = shutil.which("npm")
    if npm is None:
        raise click.UsageError("npm not found — install Node, or build the dashboard yourself and pass --skip-build.")
    if not (DASHBOARD_DIR / "node_modules").is_dir():
        logger.info("installing dashboard deps (npm install)…")
        subprocess.run([npm, "install"], cwd=DASHBOARD_DIR, check=True)
    logger.info("building dashboard (npm run build)…")
    subprocess.run([npm, "run", "build"], cwd=DASHBOARD_DIR, check=True)


@click.command()
@click.option(
    "--cluster",
    default=None,
    help="Iris cluster to connect to (resolves the controller URL and auth via the iris CLI); "
    "exclusive with --controller-url.",
)
@click.option(
    "--controller-url",
    default=lambda: os.environ.get("IRIS_CONTROLLER_URL"),
    help="Explicit Iris controller URL (default $IRIS_CONTROLLER_URL); mutually exclusive with --cluster.",
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
@click.option("--skip-build", is_flag=True, help="Skip the dashboard `npm run build` (use an already-built dist).")
@click.option(
    "--keep",
    is_flag=True,
    help="Idempotent (watchdog) mode: keep a running instance untouched, only (re)create if it's gone/terminal.",
)
def cli(
    cluster: str | None,
    controller_url: str | None,
    region: str,
    name: str,
    tpu: str,
    cpu: float,
    memory: str,
    skip_build: bool,
    keep: bool,
) -> None:
    """Submit the best-effort-always-on ducky service to an Iris cluster.

    Pass ``--cluster <name>`` to resolve the controller through the iris CLI (IAP ingress or
    SSH tunnel, with auth), or ``--controller-url`` to target one you already have. By default
    a running instance is replaced (RECREATE); ``--keep`` leaves a healthy instance alone and
    only recreates a gone/terminal one — run it on a schedule as a watchdog so the service
    comes back if its Iris job is ever lost.
    """
    logging.basicConfig(level=logging.INFO)
    if cluster and controller_url:
        raise click.UsageError("Pass --cluster or --controller-url, not both.")
    if not cluster and not controller_url:
        raise click.UsageError("Pass --cluster <name>, or --controller-url <url>.")
    env_vars = _ducky_env_vars()
    if "DUCKY_SCRATCH_BUCKET" not in env_vars:
        raise click.UsageError("DUCKY_* env vars not set — export ducky config before deploying.")

    if not skip_build:
        _build_dashboard()

    policy = job_pb2.EXISTING_JOB_POLICY_KEEP if keep else job_pb2.EXISTING_JOB_POLICY_RECREATE

    with open_iris_client(
        cluster_name=cluster,
        controller_url=controller_url,
        workspace=Path.cwd(),
        extra_bundle_includes=[DASHBOARD_DIST_INCLUDE],
    ) as client:
        job = submit_ducky(
            client,
            name=name,
            region=region,
            tpu=tpu,
            cpu=cpu,
            memory=memory,
            env_vars=env_vars,
            existing_job_policy=policy,
        )
        logger.info(
            "submitted ducky job %s (endpoint %r) — reachable via the controller endpoint proxy once running",
            job.job_id,
            PORT_NAME,
        )


if __name__ == "__main__":
    cli()
