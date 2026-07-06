# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deploy the datakit web explorer dashboard as an always-on Iris job.

Like ducky, a routable service needs a *named* Iris port, which ``iris job run``
can't declare, so we go through the Python submit path
(``client.submit(..., ports=["datakit_explorer"])``). The dashboard is CPU-only — it
holds no bulk data: it resolves lineage once, forwards SQL to the ducky service, and
lazily reads the store's tokenized bucket caches only when the store-cache view is
sampled — so it asks for a small CPU/RAM slice, not a TPU host.

The Vue SPA is built here (``npm run build`` into the gitignored ``dashboard/dist``)
and shipped to the worker via an extra Iris bundle include.

All ``WEB_EXPLORER_*`` env vars are forwarded to the task; at minimum
``WEB_EXPLORER_STORE`` (the store to explore). Point it at ducky with
``WEB_EXPLORER_DUCKY_URL`` and optionally pin ``quality``/``cluster_assign`` lineage
with ``WEB_EXPLORER_QUALITY_MODEL`` + ``WEB_EXPLORER_DOMAIN_CENTROIDS``.

Pass ``--cluster <name>`` to auto-open a controller tunnel (via the iris CLI), or
``--controller-url`` to target one you already have.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

import click
from ducky.tunnel import cluster_tunnel
from iris.client.client import IrisClient, Job
from iris.cluster.constraints import region_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2

from experiments.datakit.web_explorer.config import PORT_NAME

logger = logging.getLogger(__name__)

DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"
# The built SPA is gitignored; this glob (relative to the workspace root) tells the Iris
# bundle to ship it to the worker anyway. Owning it here keeps web_explorer-specific
# paths out of Iris's bundler.
DASHBOARD_DIST_INCLUDE = "experiments/datakit/web_explorer/dashboard/dist/**/*"

DEFAULT_CPU = 2.0
DEFAULT_MEMORY = "8GB"

# "Effectively unlimited" retry ceiling. Iris has no infinite sentinel and its retry
# budgets are lifetime totals that never reset, so a long-lived service must set a bound
# so large it can't realistically exhaust. Each retry is paced by VM boot / capacity (not
# a tight loop), so a huge value doesn't hot-loop.
_EFFECTIVELY_UNLIMITED_RETRIES = 1_000_000


def _web_explorer_env_vars() -> dict[str, str]:
    """Forward all WEB_EXPLORER_* env vars from this process to the task."""
    return {key: value for key, value in os.environ.items() if key.startswith("WEB_EXPLORER_")}


def submit_web_explorer(
    client: IrisClient,
    *,
    name: str,
    region: str,
    cpu: float,
    memory: str,
    env_vars: dict[str, str],
    existing_job_policy: job_pb2.ExistingJobPolicy = job_pb2.EXISTING_JOB_POLICY_RECREATE,
) -> Job:
    """Submit the web_explorer service: a region-pinned, port-publishing, always-on job.

    All three retry budgets are effectively unlimited so neither preemptions nor
    container failures ever terminate the service; ``max_task_failures`` is the critical
    one — it's a job-level cumulative budget that defaults to 0, so a single hard
    container failure would otherwise fail the whole job.
    """
    return client.submit(
        entrypoint=Entrypoint.from_command("python", "-m", "experiments.datakit.web_explorer.server"),
        name=name,
        resources=ResourceSpec(cpu=cpu, memory=memory),
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
    bundle re-includes the gitignored ``dist`` via ``DASHBOARD_DIST_INCLUDE``.
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
    help="Iris cluster to auto-tunnel to (opens `iris cluster dashboard`); exclusive with --controller-url.",
)
@click.option(
    "--controller-url",
    default=lambda: os.environ.get("IRIS_CONTROLLER_URL"),
    help="Explicit Iris controller URL (default $IRIS_CONTROLLER_URL); mutually exclusive with --cluster.",
)
@click.option("--region", default="us-east5", show_default=True, help="Region to pin the job to.")
@click.option("--name", default="web_explorer", show_default=True, help="Job name.")
@click.option("--cpu", default=DEFAULT_CPU, show_default=True, type=float, help="CPUs to request.")
@click.option("--memory", default=DEFAULT_MEMORY, show_default=True, help="Memory to request.")
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
    cpu: float,
    memory: str,
    skip_build: bool,
    keep: bool,
) -> None:
    """Submit the always-on web explorer dashboard to an Iris cluster.

    Pass ``--cluster <name>`` to auto-open a controller tunnel, or ``--controller-url``
    to target one you already have. By default a running instance is replaced (RECREATE);
    ``--keep`` leaves a healthy instance alone and only recreates a gone/terminal one.
    """
    logging.basicConfig(level=logging.INFO)
    if cluster and controller_url:
        raise click.UsageError("Pass --cluster or --controller-url, not both.")
    if not cluster and not controller_url:
        raise click.UsageError("Pass --cluster <name> to auto-tunnel, or --controller-url <url>.")
    env_vars = _web_explorer_env_vars()
    if "WEB_EXPLORER_STORE" not in env_vars:
        raise click.UsageError("WEB_EXPLORER_STORE not set — export the store to explore before deploying.")

    if not skip_build:
        _build_dashboard()

    policy = job_pb2.EXISTING_JOB_POLICY_KEEP if keep else job_pb2.EXISTING_JOB_POLICY_RECREATE

    def _submit(url: str) -> None:
        client = IrisClient.remote(url, workspace=Path.cwd(), extra_bundle_includes=[DASHBOARD_DIST_INCLUDE])
        job = submit_web_explorer(
            client,
            name=name,
            region=region,
            cpu=cpu,
            memory=memory,
            env_vars=env_vars,
            existing_job_policy=policy,
        )
        logger.info(
            "submitted web_explorer job %s (endpoint %r) — reachable at /proxy/datakit_explorer/ once running",
            job.job_id,
            PORT_NAME,
        )

    if cluster:
        with cluster_tunnel(cluster) as url:
            _submit(url)
    else:
        assert controller_url is not None  # guarded above
        _submit(controller_url)


if __name__ == "__main__":
    cli()
