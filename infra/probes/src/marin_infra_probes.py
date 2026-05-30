# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic infra canary. Three probes against Iris and Finelog, run on a
fixed cadence, results logged to stdout (picked up by Cloud Logging on COS).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from finelog.client.log_client import LogClient
from finelog.rpc import logging_pb2
from iris.cluster.client.remote_client import RemoteClusterClient
from iris.cluster.constraints import zone_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, JobName, ResourceSpec
from iris.rpc import job_pb2
from rigging.log_setup import configure_logging
from rigging.timing import Duration

logger = logging.getLogger("probes")


@dataclass
class ProbeResult:
    is_success: bool
    wall_time: float | None = None


ProbeFn = Callable[[], ProbeResult]


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
    logged as ``probe <name>: ok|fail [<wall_ms>ms]`` and that's the only
    output — operator log aggregation does the rest."""

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
            start = time.monotonic()
            try:
                result = await asyncio.wait_for(asyncio.to_thread(probe.fn), timeout=probe.timeout)
            except asyncio.TimeoutError:
                result = ProbeResult(is_success=False)
            except Exception:
                logger.exception("probe %s raised", probe.name)
                result = ProbeResult(is_success=False)
            result.wall_time = time.monotonic() - start
            level = logging.INFO if result.is_success else logging.ERROR
            status = "ok" if result.is_success else "fail"
            logger.log(level, "probe %s: %s [%dms]", probe.name, status, result.wall_time * 1000)
            await asyncio.sleep(probe.cadence)


# ---- probes ---------------------------------------------------------------


def probe_controller_ping(iris: RemoteClusterClient) -> ProbeResult:
    iris.list_workers()
    return ProbeResult(is_success=True)


def probe_iris_job_submit(iris: RemoteClusterClient, zone: str) -> ProbeResult:
    job_id = JobName.root("probes", f"canary-{zone}-{int(time.time())}")
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
    return ProbeResult(is_success=status.state == job_pb2.JOB_STATE_SUCCEEDED)


def probe_finelog_write(finelog: LogClient) -> ProbeResult:
    nonce = uuid.uuid4().hex
    ts_ms = int(time.time() * 1000)
    finelog.write_batch(
        key="marin.canary.probe",
        messages=[
            logging_pb2.LogEntry(
                timestamp=logging_pb2.Timestamp(epoch_ms=ts_ms),
                source="/canary/finelog-write-probe",
                data=nonce,
                level=logging_pb2.LOG_LEVEL_INFO,
                key=nonce,
            )
        ],
    )
    response = finelog.fetch_logs(
        logging_pb2.FetchLogsRequest(source="/canary/finelog-write-probe", since_ms=ts_ms - 1000, max_lines=64)
    )
    return ProbeResult(is_success=any(e.data == nonce for e in response.entries))


# ---- entrypoint -----------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="probes")
    p.add_argument("--iris-endpoint", required=True, help="e.g. https://iris-controller.internal:10001")
    p.add_argument("--finelog-endpoint", help="defaults to --iris-endpoint")
    p.add_argument("--zone", action="append", required=True, help="GCP zone for iris-job-submit; repeat for multiple")
    args = p.parse_args(argv)

    iris = RemoteClusterClient(controller_address=args.iris_endpoint)
    finelog = LogClient.connect(args.finelog_endpoint or args.iris_endpoint)

    runner = ProbeRunner()
    runner.add_probe("controller-ping", lambda: probe_controller_ping(iris), timeout=5.0, cadence=60.0)
    runner.add_probe("finelog-write", lambda: probe_finelog_write(finelog), timeout=10.0, cadence=60.0)
    for zone in args.zone:
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
