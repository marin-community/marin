# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic infra canary. Three probes against Iris and Finelog, run on a
fixed cadence, results logged to stdout (picked up by Cloud Logging on COS).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
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
from rigging.timing import Duration

logger = logging.getLogger("probes")


@dataclass
class ProbeResult:
    is_success: bool
    wall_time: float | None = None


ProbeFn = Callable[[], ProbeResult]


@dataclass
class _Spec:
    name: str
    fn: ProbeFn
    timeout: float
    cadence: float


class ProbeRunner:
    """Register probes, then ``run()`` to execute each one forever on its own
    cadence. SIGTERM/SIGINT (or ``stop()`` from another thread) stops cleanly.
    Results are emitted via ``on_result(name, result)``; default just logs."""

    def __init__(self, on_result: Callable[[str, ProbeResult], None] | None = None):
        self._specs: list[_Spec] = []
        self._on_result = on_result or _log_result
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown: asyncio.Event | None = None

    def add_probe(self, name: str, fn: ProbeFn, *, timeout: float, cadence: float) -> None:
        self._specs.append(_Spec(name, fn, timeout, cadence))

    def stop(self) -> None:
        """Request graceful shutdown; safe to call from any thread."""
        if self._loop is not None and self._shutdown is not None:
            self._loop.call_soon_threadsafe(self._shutdown.set)

    def run(self) -> None:
        if not self._specs:
            raise ValueError("no probes registered")
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._shutdown = asyncio.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._loop.add_signal_handler(sig, self._shutdown.set)
            except (NotImplementedError, ValueError, RuntimeError):
                pass  # non-main thread, or platform doesn't support it
        await asyncio.gather(*(self._run_probe(spec) for spec in self._specs))

    async def _run_probe(self, spec: _Spec) -> None:
        assert self._shutdown is not None
        while not self._shutdown.is_set():
            start = time.monotonic()
            try:
                result = await asyncio.wait_for(asyncio.to_thread(spec.fn), timeout=spec.timeout)
            except asyncio.TimeoutError:
                result = ProbeResult(is_success=False)
            except Exception:
                logger.exception("probe %s raised", spec.name)
                result = ProbeResult(is_success=False)
            result.wall_time = time.monotonic() - start
            self._on_result(spec.name, result)
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=spec.cadence)
            except asyncio.TimeoutError:
                pass


def _log_result(name: str, result: ProbeResult) -> None:
    status = "ok" if result.is_success else "fail"
    wt = f"{result.wall_time * 1000:.0f}ms" if result.wall_time is not None else "-"
    level = logging.INFO if result.is_success else logging.ERROR
    logger.log(level, "probe %s: %s [%s]", name, status, wt)


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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="probes")
    p.add_argument("--iris-endpoint", required=True, help="e.g. https://iris-controller.internal:10001")
    p.add_argument("--finelog-endpoint", help="defaults to --iris-endpoint")
    p.add_argument("--zone", action="append", required=True, help="GCP zone for iris-job-submit; repeat for multiple")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

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
    return 0
