# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic infra canary. Health checks against Iris and Finelog plus an
accelerator provisioning-stats gauge, each run as a collector on its own cadence;
samples are logged to stdout (picked up by Cloud Logging on COS) and fanned to the
sinks.
"""

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from pathlib import Path

import click
from cluster import collect_jobs, collect_workers
from finelog.client.log_client import FlushResult, LogClient
from finelog.rpc import logging_pb2
from iris.cluster.client.remote_client import RemoteClusterClient
from iris.cluster.constraints import zone_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, JobName, ResourceSpec
from iris.rpc import job_pb2
from provisioning import collect_provisioning
from rigging.filesystem import REGION_TO_DATA_BUCKET
from rigging.log_setup import configure_logging
from rigging.timing import Duration
from runner import Collector, CollectorRunner, MetricSink, health_collector
from sinks import FinelogTableSink, JsonlGcsSink

logger = logging.getLogger(__name__)

# Iris advertises the finelog log-server under this logical name in its endpoint
# registry; resolve it to a concrete address via list_endpoints (same name the
# iris worker uses).
LOG_SERVER_ENDPOINT_NAME = "/system/log-server"

# Default zones to canary when --zone is not given: the busiest europe-west4 and
# us-west4 zones in the fleet.
DEFAULT_ZONES = ("europe-west4-b", "us-west4-a")

# Provisioning gauge: a trailing window over the controller's iris.provisioning
# namespace, re-emitted each cadence. A 3h window smooths the bursty per-minute
# noise (stockouts persist for hours); 15min cadence is ample resolution and the
# finelog query is sub-second. The timeout covers the query plus aggregation.
PROVISION_WINDOW_HOURS = 3.0
PROVISION_CADENCE = 900.0
PROVISION_TIMEOUT = 60.0

# Cluster-state gauges backing the status page. Workers is a single ListWorkers
# RPC paged client-side; jobs is one raw-SQL GROUP BY. Both are sub-second, so the
# cadences are about freshness (workers churn faster than the 24h job window) and
# the timeouts only cover a slow/hung controller.
WORKERS_CADENCE = 60.0
WORKERS_TIMEOUT = 30.0
JOBS_CADENCE = 120.0
JOBS_TIMEOUT = 30.0

# The iris worker unconditionally runs `uv sync --all-packages --no-group dev`
# against the job's bundle, which fails without a pyproject.toml (and without a
# `dev` group for --no-group to exclude). The canary sleep job needs no deps, so
# ship a throwaway workspace whose only content is a minimal pyproject that
# resolves to an empty venv: an empty dev group satisfies --no-group dev, and
# package=false skips building it as a package.
CANARY_PYPROJECT = """\
[project]
name = "iris-canary"
version = "0"
requires-python = ">=3.12"

[dependency-groups]
dev = []

[tool.uv]
package = false
"""

# finelog-write probe: the key/source the canary writes under. Reads match on
# the KEY column (FetchLogsRequest.source + MatchScope are key matchers despite
# the field name), so the readback queries FINELOG_PROBE_KEY, not the source.
FINELOG_PROBE_KEY = "infra.canary.finelog_probe"
FINELOG_PROBE_SOURCE = "/canary/finelog-write-probe"
# Cap the flush wait: the StatsService write can be slow or hang, and an
# unbounded flush would block the probe to its timeout and leak the worker
# thread. Flush + readback stay under the finelog-write probe timeout.
FINELOG_FLUSH_TIMEOUT = 8.0
FINELOG_READBACK_TIMEOUT = 5.0
FINELOG_READBACK_POLL_INTERVAL = 0.25

# Where each Sample is persisted (beyond the stdout log line). The local dir is
# the VM's /var/lib/probes host mount; finished daily files roll up to GCS in the
# same region as the VM (no cross-region egress).
PROBE_RESULTS_DIR = Path("/var/lib/probes")
PROBE_RESULTS_GCS_PREFIX = f"gs://{REGION_TO_DATA_BUCKET['us-central1']}/infra/probes"
PROBE_RESULTS_NAMESPACE = "infra.canary.metrics"


# ---- health checks --------------------------------------------------------


def probe_controller_ping(iris: RemoteClusterClient) -> bool:
    iris.list_workers()
    return True


def probe_iris_job_submit(iris: RemoteClusterClient, zone: str) -> bool:
    job_id = JobName.root("infra-probes", f"canary-{zone}-{int(time.time())}")
    submitted = iris.submit_job(
        job_id=job_id,
        entrypoint=Entrypoint.from_command("python", "-c", "import time; time.sleep(1)"),
        resources=ResourceSpec(cpu=1.0, memory="256m").to_proto(),
        environment=EnvironmentSpec().to_proto(),
        constraints=[zone_constraint(zone).to_proto()],
        max_retries_failure=0,
        max_retries_preemption=0,
        timeout=Duration.from_seconds(60),
    )
    status = iris.wait_for_job(submitted, timeout=100.0)
    return status.state == job_pb2.JOB_STATE_SUCCEEDED


def probe_finelog_write(finelog: LogClient) -> bool:
    nonce = uuid.uuid4().hex
    ts_ms = int(time.time() * 1000)
    finelog.write_batch(
        key=FINELOG_PROBE_KEY,
        messages=[
            logging_pb2.LogEntry(
                timestamp=logging_pb2.Timestamp(epoch_ms=ts_ms),
                source=FINELOG_PROBE_SOURCE,
                data=nonce,
                level=logging_pb2.LOG_LEVEL_INFO,
            )
        ],
    )
    if finelog.flush(timeout=FINELOG_FLUSH_TIMEOUT) != FlushResult.SUCCEEDED:
        return False
    # Re-read our own write until the nonce shows up or the readback budget is
    # spent: the write is durable now but the index lags it, so a single fetch races.
    # source here matches the KEY column (EXACT), not the entry's source field.
    deadline = time.monotonic() + FINELOG_READBACK_TIMEOUT
    while True:
        response = finelog.fetch_logs(
            logging_pb2.FetchLogsRequest(
                source=FINELOG_PROBE_KEY,
                match_scope=logging_pb2.MATCH_SCOPE_EXACT,
                since_ms=ts_ms - 1000,
                max_lines=64,
            )
        )
        if any(e.data == nonce for e in response.entries):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(FINELOG_READBACK_POLL_INTERVAL)


# ---- entrypoint -----------------------------------------------------------


def resolve_finelog_address(iris: RemoteClusterClient, name: str) -> str:
    """Resolve the finelog log-server address from iris's endpoint registry."""
    endpoints = iris.list_endpoints(name, exact=True)
    if not endpoints:
        raise ConnectionError(f"no {name!r} endpoint registered on the iris controller")
    return endpoints[0].address


def make_canary_workspace() -> Path:
    """Create the throwaway workspace bundled with the iris-job-submit canary so
    the worker's `uv sync` build step has a pyproject.toml to resolve against."""
    workspace = Path(tempfile.mkdtemp(prefix="iris-canary-workspace-"))
    (workspace / "pyproject.toml").write_text(CANARY_PYPROJECT)
    return workspace


def build_sinks(finelog: LogClient) -> list[MetricSink]:
    """Construct the sample sinks, skipping any that fail to initialize so the
    canary still runs (and reports samples) on a sink-side fault."""
    sinks: list[MetricSink] = []
    try:
        sinks.append(JsonlGcsSink(PROBE_RESULTS_DIR, PROBE_RESULTS_GCS_PREFIX))
    except Exception:
        logger.exception("failed to init JSONL/GCS sink; continuing without it")
    try:
        sinks.append(FinelogTableSink(finelog, PROBE_RESULTS_NAMESPACE))
    except Exception:
        logger.exception("failed to init finelog sink; continuing without it")
    return sinks


def build_collectors(iris: RemoteClusterClient, finelog: LogClient, zones: tuple[str, ...]) -> list[Collector]:
    """Health checks plus the provisioning gauge, each on its own cadence."""
    collectors = [
        health_collector("controller-ping", lambda: probe_controller_ping(iris), timeout=5.0, cadence=60.0),
        health_collector("finelog-write", lambda: probe_finelog_write(finelog), timeout=15.0, cadence=60.0),
        Collector(
            name="provisioning",
            collect=lambda: collect_provisioning(finelog, window_hours=PROVISION_WINDOW_HOURS),
            timeout=PROVISION_TIMEOUT,
            cadence=PROVISION_CADENCE,
        ),
        Collector(
            name="workers",
            collect=lambda: collect_workers(iris),
            timeout=WORKERS_TIMEOUT,
            cadence=WORKERS_CADENCE,
        ),
        Collector(
            name="jobs",
            collect=lambda: collect_jobs(iris),
            timeout=JOBS_TIMEOUT,
            cadence=JOBS_CADENCE,
        ),
    ]
    for zone in zones:
        collectors.append(
            health_collector(
                f"iris-job-submit/{zone}",
                lambda z=zone: probe_iris_job_submit(iris, z),
                timeout=120.0,
                cadence=300.0,
            )
        )
    return collectors


@click.command()
@click.option("--iris-endpoint", required=True, help="controller RPC, e.g. http://10.128.0.3:10000")
@click.option(
    "--zone",
    "zones",
    multiple=True,
    help=f"GCP zone for iris-job-submit; repeat for multiple (default: {', '.join(DEFAULT_ZONES)})",
)
def main(iris_endpoint: str, zones: tuple[str, ...]) -> None:
    zones = zones or DEFAULT_ZONES

    iris = RemoteClusterClient(controller_address=iris_endpoint, workspace=make_canary_workspace())
    finelog = LogClient.connect(
        LOG_SERVER_ENDPOINT_NAME,
        resolver=lambda name: resolve_finelog_address(iris, name),
    )

    runner = CollectorRunner(sinks=build_sinks(finelog))
    for collector in build_collectors(iris, finelog, zones):
        runner.add(collector)
    runner.run()


if __name__ == "__main__":
    configure_logging()
    main()
