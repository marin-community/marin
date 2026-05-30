# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic infra canary. Three probes against Iris and Finelog, run on a
fixed cadence, results logged to stdout (picked up by Cloud Logging on COS).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from finelog.client.log_client import FlushResult, LogClient
from finelog.rpc import logging_pb2
from iris.cluster.client.remote_client import RemoteClusterClient
from iris.cluster.constraints import zone_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, JobName, ResourceSpec
from iris.rpc import job_pb2
from rigging.log_setup import configure_logging
from rigging.timing import Duration

logger = logging.getLogger("probes")

# Iris advertises the finelog log-server under this logical name in its endpoint
# registry; resolve it to a concrete address via list_endpoints (same name the
# iris worker uses).
LOG_SERVER_ENDPOINT_NAME = "/system/log-server"

# Default zones to canary when --zone is not given: the busiest europe-west4 and
# us-west4 zones in the fleet.
DEFAULT_ZONES = ("europe-west4-b", "us-west4-a")

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
requires-python = ">=3.11"

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


@dataclass
class ProbeResult:
    # name and started_at are supplied by the runner, which owns this metadata;
    # the probe fn only reports is_success. wall_time is filled in once the run
    # completes.
    is_success: bool
    name: str
    started_at: datetime  # UTC wall-clock time at probe start
    wall_time: float | None = None


# A probe fn reports only whether the probe succeeded; the runner stamps the
# rest of the ProbeResult.
ProbeFn = Callable[[], bool]


@dataclass
class Probe:
    """A registered probe: a callable to run, a name to report it under, and
    timing (per-run timeout, between-runs cadence)."""

    name: str
    fn: ProbeFn
    timeout: float
    cadence: float


class ProbeRunner:
    """Register probes, then ``run()`` to execute each one forever on its own
    cadence. Ctrl-C kills the process — there is no graceful shutdown path;
    samples are stateless so there's nothing to clean up. Each result is
    logged as ``probe <name>: ok|fail [<wall_ms>ms] start=<utc-iso>`` and
    that's the only output — operator log aggregation does the rest."""

    def __init__(self) -> None:
        self._probes: list[Probe] = []

    def add_probe(self, name: str, fn: ProbeFn, *, timeout: float, cadence: float) -> None:
        self._probes.append(Probe(name, fn, timeout, cadence))

    def run(self) -> None:
        if not self._probes:
            raise ValueError("no probes registered")
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        await asyncio.gather(*(self._run_probe(probe) for probe in self._probes))

    async def _run_probe(self, probe: Probe) -> None:
        while True:
            started_at = datetime.now(timezone.utc)
            start = time.monotonic()
            try:
                is_success = await asyncio.wait_for(asyncio.to_thread(probe.fn), timeout=probe.timeout)
            except asyncio.TimeoutError:
                is_success = False
            except Exception:
                logger.exception("probe %s raised", probe.name)
                is_success = False
            result = ProbeResult(is_success=is_success, name=probe.name, started_at=started_at)
            result.wall_time = time.monotonic() - start
            level = logging.INFO if result.is_success else logging.ERROR
            status = "ok" if result.is_success else "fail"
            logger.log(
                level,
                "probe %s: %s [%dms] start=%s",
                result.name,
                status,
                result.wall_time * 1000,
                started_at.isoformat(timespec="milliseconds"),
            )
            await asyncio.sleep(probe.cadence)


# ---- probes ---------------------------------------------------------------


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


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="probes")
    p.add_argument("--iris-endpoint", required=True, help="e.g. http://10.128.0.3:10001")
    p.add_argument(
        "--zone",
        action="append",
        help=f"GCP zone for iris-job-submit; repeat for multiple (default: {', '.join(DEFAULT_ZONES)})",
    )
    args = p.parse_args(argv)
    # append default would accumulate onto args.zone, so fall back here instead.
    zones = args.zone or list(DEFAULT_ZONES)

    iris = RemoteClusterClient(controller_address=args.iris_endpoint, workspace=make_canary_workspace())
    finelog = LogClient.connect(
        LOG_SERVER_ENDPOINT_NAME,
        resolver=lambda name: resolve_finelog_address(iris, name),
    )

    runner = ProbeRunner()
    runner.add_probe("controller-ping", lambda: probe_controller_ping(iris), timeout=5.0, cadence=60.0)
    runner.add_probe("finelog-write", lambda: probe_finelog_write(finelog), timeout=15.0, cadence=60.0)
    for zone in zones:
        runner.add_probe(
            f"iris-job-submit/{zone}",
            lambda z=zone: probe_iris_job_submit(iris, z),
            timeout=120.0,
            cadence=300.0,
        )
    runner.run()


if __name__ == "__main__":
    configure_logging()
    main()
