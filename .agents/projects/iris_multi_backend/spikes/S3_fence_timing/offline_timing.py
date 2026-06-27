#!/usr/bin/env python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SPIKE S3 — offline fence/reuse timing harness.

Measures the terms of the skew-safe reuse invariant against the REAL iris worker
code over in-memory fakes. No network, no cluster, no docker — every number here
comes from driving production code paths (``Worker.handle_reconcile``,
``TaskAttempt.kill``, the task monitor loop, ``Worker._serve``) with a fake
container substrate.

The invariant (design.md / spec.md §1):

    root_reuse_time >= send_time + lease_duration + max_skew + transport_grace + kill_grace

i.e. the root may re-place a task (a *new* attempt) only after it is provable the
agent has self-fenced the old runner. This harness pins:

  * kill_grace      — M1: time from kill-initiated to attempt-terminal.
  * reconcile RTT   — M2: server-side ``handle_reconcile`` cost + reconcile->kill.
  * transport_grace — M3: the per-worker reconcile RPC bound (RECONCILE_RPC_TIMEOUT).
  * self-fence      — M4: the worker's own lost-contact self-reset latency.
  * the formula     — M5: plug measured + derived terms into root_reuse_time.

Run:  .venv/bin/python offline_timing.py
      .venv/bin/python offline_timing.py --quick   # skip the slow 5s-poll cells

Nothing here touches a live cluster. See partition_harness.py for the gated
live runs that measure the two terms this harness CANNOT: a real worker-daemon
self-fence over a real partition, and k8s pod-create / pod-self-fence latency.
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import Mock

from iris.cluster.bundle import BundleStore
from iris.cluster.runtime.docker import DockerRuntime
from iris.cluster.runtime.types import ContainerPhase, ContainerStats, ContainerStatus
from iris.cluster.types import Entrypoint, JobName
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.cluster.worker.worker_types import LogLine
from iris.rpc import job_pb2, worker_pb2
from iris.time_proto import duration_to_proto
from rigging.timing import Duration

# Real production constants, imported so the harness tracks the source of truth.
from iris.cluster.backends.rpc.backend import RECONCILE_RPC_TIMEOUT  # noqa: E402

RUNNING = job_pb2.TASK_STATE_RUNNING
TERMINAL = (
    job_pb2.TASK_STATE_SUCCEEDED,
    job_pb2.TASK_STATE_FAILED,
    job_pb2.TASK_STATE_KILLED,
    job_pb2.TASK_STATE_WORKER_FAILED,
)


# --------------------------------------------------------------------------- #
# Minimal in-memory substrate (self-contained; mirrors the worker test conftest
# but without the pytest dependency).
# --------------------------------------------------------------------------- #
@dataclass
class _FakeLogReader:
    def read(self) -> list[LogLine]:
        return []

    def read_all(self) -> list[LogLine]:
        return []


class FakeContainerHandle:
    """In-memory ContainerHandle. ``sigterm_stops`` toggles a cooperative vs a
    wedged container; ``exit_delay`` models a container that ignores even SIGKILL
    for a while (stress the post-kill exit wait)."""

    def __init__(
        self,
        *,
        sigterm_stops: bool = False,
        exit_delay: float = 0.0,
    ) -> None:
        self.sigterm_stops = sigterm_stops
        self._exit_delay = exit_delay
        self._kill_requested_at: float | None = None
        self._killed = False
        self.stop_calls: list[dict[str, object]] = []

    @property
    def container_id(self) -> str | None:
        return "container123"

    def build(self, on_logs: Callable[[list[LogLine]], None] | None = None) -> list[LogLine]:
        return []

    def run(self) -> None:
        return None

    def stop(self, force: bool = False) -> None:
        self.stop_calls.append({"force": force, "t": time.monotonic()})
        if force or self.sigterm_stops:
            self._kill_requested_at = time.monotonic()
            if self._exit_delay == 0.0:
                self._killed = True

    def status(self) -> ContainerStatus:
        if not self._killed and self._kill_requested_at is not None and self._exit_delay > 0.0:
            if time.monotonic() - self._kill_requested_at >= self._exit_delay:
                self._killed = True
        if self._killed:
            return ContainerStatus(phase=ContainerPhase.STOPPED, exit_code=137)
        return ContainerStatus(phase=ContainerPhase.RUNNING)

    def log_reader(self) -> _FakeLogReader:
        return _FakeLogReader()

    def stats(self) -> ContainerStats:
        return ContainerStats(memory_mb=100, cpu_millicores=500, process_count=5, available=True)

    def disk_usage_mb(self) -> int:
        return 0

    def profile(self, *a: object, **k: object) -> bytes:
        raise RuntimeError("profiling not supported in FakeContainerHandle")

    def cleanup(self) -> None:
        pass


def make_worker(handle: FakeContainerHandle, poll_interval: float) -> Worker:
    tmp = Path(tempfile.mkdtemp())
    bundle = Mock(spec=BundleStore)
    bundle.extract_bundle_to = Mock()
    runtime = Mock(spec=DockerRuntime)
    runtime.create_container = Mock(return_value=handle)
    runtime.stage_bundle = Mock()
    runtime.list_iris_containers = Mock(return_value=[])
    runtime.remove_all_iris_containers = Mock(return_value=0)
    runtime.remove_containers = Mock(return_value=0)
    runtime.discover_containers = Mock(return_value=[])
    runtime.cleanup = Mock()
    config = WorkerConfig(
        port=0,
        port_range=(50000, 50100),
        poll_interval=Duration.from_seconds(poll_interval),
        cache_dir=tmp / "cache",
        default_task_image="mock-image",
    )
    return Worker(config, bundle_store=bundle, container_runtime=runtime)


def _run_task(worker: Worker, attempt_uid: str = "uid-aaaa0000") -> tuple[str, object]:
    def fn() -> None:
        print("hi")

    request = job_pb2.RunTaskRequest(
        task_id=JobName.root("u", "t").task(0).to_wire(),
        num_tasks=1,
        attempt_id=0,
        attempt_uid=attempt_uid,
        entrypoint=Entrypoint.from_callable(fn).to_proto(),
        environment=job_pb2.EnvironmentConfig(env_vars={}, setup_scripts=["uv sync\n"]),
        bundle_id="a" * 64,
        resources=job_pb2.ResourceSpecProto(cpu_millicores=2000, memory_bytes=4 * 1024**3),
        ports=[],
        task_image="",
    )
    request.timeout.CopyFrom(duration_to_proto(Duration.from_seconds(300)))
    task_id = worker.submit_task(request)
    task = worker.get_task(task_id)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if task.status == RUNNING and task.container_id:
            break
        time.sleep(0.005)
    return task_id, task


def _wait_terminal(task: object, timeout: float = 40.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if task.status in TERMINAL:  # type: ignore[attr-defined]
            return True
        time.sleep(0.005)
    return False


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    label: str
    value: str
    note: str = ""


@dataclass
class Section:
    title: str
    rows: list[Row] = field(default_factory=list)

    def add(self, label: str, value: str, note: str = "") -> None:
        self.rows.append(Row(label, value, note))


def _print(section: Section) -> None:
    print(f"\n=== {section.title} ===")
    width = max((len(r.label) for r in section.rows), default=0)
    for r in section.rows:
        suffix = f"   # {r.note}" if r.note else ""
        print(f"  {r.label:<{width}}  {r.value}{suffix}")


# --------------------------------------------------------------------------- #
# M1 — kill latency (kill_grace)
# --------------------------------------------------------------------------- #
def measure_kill_latency(poll_intervals: list[float]) -> Section:
    sec = Section("M1: kill latency (kill_grace) — kill-initiated -> attempt terminal")
    # (container, term_timeout_ms, poll_interval)
    cases = []
    for poll in poll_intervals:
        cases.append(("cooperative", True, 5000, poll, 0.0))
        cases.append(("wedged", False, 500, poll, 0.0))
        cases.append(("wedged", False, 5000, poll, 0.0))
    # one slow-exit case: container ignores SIGKILL for 1.5s (post-kill exit wait)
    cases.append(("slow-exit-1.5s", False, 5000, 0.1, 1.5))

    for kind, sigterm_stops, term_ms, poll, exit_delay in cases:
        handle = FakeContainerHandle(sigterm_stops=sigterm_stops, exit_delay=exit_delay)
        worker = make_worker(handle, poll_interval=poll)
        _, task = _run_task(worker)
        t0 = time.monotonic()
        worker.kill_task(task.task_id.to_wire(), term_timeout_ms=term_ms)  # type: ignore[attr-defined]
        terminal = _wait_terminal(task)
        dt = time.monotonic() - t0
        sigterm_at = handle.stop_calls[0]["t"] - t0 if handle.stop_calls else float("nan")  # type: ignore[operator]
        state = job_pb2.TaskState.Name(task.status)  # type: ignore[attr-defined]
        sec.add(
            f"{kind:<14} term={term_ms:>4}ms poll={poll:>3}s",
            f"terminal={dt:6.3f}s  sigterm@{sigterm_at:6.3f}s  -> {state}",
            "" if terminal else "DID NOT TERMINATE",
        )
    return sec


# --------------------------------------------------------------------------- #
# M2 — reconcile zombie-kill (server-side reconcile RTT + reconcile->kill)
# --------------------------------------------------------------------------- #
def measure_reconcile_zombie_kill(poll_intervals: list[float]) -> Section:
    sec = Section("M2: reconcile zombie-kill — handle_reconcile(empty) cost + observed death")
    for poll in poll_intervals:
        handle = FakeContainerHandle(sigterm_stops=False, exit_delay=0.0)
        worker = make_worker(handle, poll_interval=poll)
        _, task = _run_task(worker)
        # An empty desired set: the running attempt becomes a zombie and is killed.
        t0 = time.monotonic()
        worker.handle_reconcile(worker_pb2.Worker.ReconcileRequest(desired=[]))
        handler_dt = time.monotonic() - t0
        terminal = _wait_terminal(task)
        kill_dt = time.monotonic() - t0
        sigterm_at = handle.stop_calls[0]["t"] - t0 if handle.stop_calls else float("nan")  # type: ignore[operator]
        sec.add(
            f"poll={poll:>3}s",
            f"handler_return={handler_dt * 1000:7.2f}ms  sigterm@{sigterm_at:6.3f}s  terminal@{kill_dt:6.3f}s",
            "" if terminal else "ZOMBIE SURVIVED",
        )
    return sec


# --------------------------------------------------------------------------- #
# M3 — transport_grace: the per-worker reconcile RPC bound
# --------------------------------------------------------------------------- #
def measure_reconcile_rpc_timeout() -> Section:
    sec = Section("M3: transport_grace — RECONCILE_RPC_TIMEOUT enforcement (hung worker)")

    async def _hung_reconcile() -> None:
        await asyncio.sleep(3600)  # a partitioned/hung worker never answers

    async def _drive() -> tuple[bool, float]:
        # Exactly the pattern in RpcTaskBackend._reconcile_one (backend.py:316).
        t0 = time.monotonic()
        timed_out = False
        try:
            await asyncio.wait_for(_hung_reconcile(), timeout=RECONCILE_RPC_TIMEOUT.to_seconds())
        except (TimeoutError, asyncio.TimeoutError):
            timed_out = True
        return timed_out, time.monotonic() - t0

    timed_out, dt = asyncio.run(_drive())
    sec.add(
        "RECONCILE_RPC_TIMEOUT (source)",
        f"{RECONCILE_RPC_TIMEOUT.to_seconds():.1f}s",
        "imported from backends/rpc/backend.py",
    )
    sec.add(
        "measured wait_for bound",
        f"{dt:.3f}s  (timed_out={timed_out})",
        "one partitioned worker costs one such window per reconcile round",
    )
    return sec


# --------------------------------------------------------------------------- #
# M4 — worker self-fence latency (lost-contact self-reset)
# --------------------------------------------------------------------------- #
def measure_self_fence(heartbeat_timeouts: list[float]) -> Section:
    sec = Section("M4: worker self-fence — _serve() return after lost controller contact")
    from iris.cluster.worker.worker import WorkerConfig as _WC  # local: only to rebuild config

    for hb in heartbeat_timeouts:
        handle = FakeContainerHandle()
        worker = make_worker(handle, poll_interval=0.1)
        # Shrink only the heartbeat_timeout; everything else is production code.
        worker._config = _WC(
            port=worker._config.port,
            port_range=worker._config.port_range,
            poll_interval=worker._config.poll_interval,
            cache_dir=worker._config.cache_dir,
            default_task_image=worker._config.default_task_image,
            heartbeat_timeout=Duration.from_seconds(hb),
        )
        stop_event = threading.Event()
        t0 = time.monotonic()
        worker._serve(stop_event)  # returns when no contact for heartbeat_timeout
        dt = time.monotonic() - t0
        sec.add(
            f"heartbeat_timeout={hb:>4}s",
            f"_serve returned @ {dt:6.3f}s",
            "detection granularity = the 1s serve-loop poll",
        )
    sec.add(
        "PRODUCTION DEFAULT",
        "heartbeat_timeout = 600.0s",
        "worker.py:83 — today's worker self-fence ~= 600s + <=1s + reset",
    )
    return sec


# --------------------------------------------------------------------------- #
# M5 — plug the terms into the invariant
# --------------------------------------------------------------------------- #
def evaluate_formula() -> Section:
    sec = Section("M5: root_reuse_time = lease + max_skew + transport_grace + kill_grace")

    def reuse(lease: float, skew: float, transport: float, kill: float) -> float:
        return lease + skew + transport + kill

    # max_skew is DERIVED, not measured: leases are monotonic-duration (Deadline.from_seconds
    # uses time.monotonic, timing.py:60), so only clock-RATE skew over the lease window matters.
    # NTP-disciplined hosts drift < ~250ppm typical / < ~500ppm worst-case; over a 60s lease that
    # is < 30ms. We bound max_skew conservatively at 1.0s (engineering margin, ~30x worst-case).
    skew = 1.0

    # "Today" — if the agent self-fence were the existing worker heartbeat_timeout (600s) and
    # kill_grace the monitor poll (5s). transport_grace = the reconcile RPC bound (3s).
    today = reuse(lease=600.0, skew=skew, transport=3.0, kill=5.0)
    sec.add(
        "TODAY (heartbeat_timeout self-fence)",
        f"lease=600 + skew={skew} + transport=3 + kill=5  = {today:.0f}s  (~{today / 60:.1f} min)",
        "today's worker self-reset is the only lost-contact fence -> MINUTES",
    )

    # "Proposed fast lease" — a dedicated short agent->worker lease for the self-fence path.
    fast = reuse(lease=20.0, skew=skew, transport=3.0, kill=5.0)
    sec.add(
        "PROPOSED (20s dedicated lease)",
        f"lease=20 + skew={skew} + transport=3 + kill=5  = {fast:.0f}s",
        "requires a NEW short lease + faster worker self-fence than 600s",
    )

    aggressive = reuse(lease=10.0, skew=skew, transport=3.0, kill=5.0)
    sec.add(
        "AGGRESSIVE (10s lease, 5s kill)",
        f"lease=10 + skew={skew} + transport=3 + kill=5  = {aggressive:.0f}s",
        "floor is set by kill_grace (monitor poll) + transport, ~8s irreducible",
    )
    return sec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="skip the slow 5s-poll cells")
    args = parser.parse_args()

    polls = [0.1, 1.0] if args.quick else [0.1, 1.0, 5.0]
    hb = [0.1, 1.5, 2.5]

    print("SPIKE S3 — offline fence/reuse timing (real iris code over in-memory fakes)")
    print("No cluster, no network, no docker. Wall-clock numbers are machine-dependent;")
    print("the GRANULARITY they expose (poll intervals, RPC bound) is what matters.")

    _print(measure_kill_latency(polls))
    _print(measure_reconcile_zombie_kill(polls))
    _print(measure_reconcile_rpc_timeout())
    _print(measure_self_fence(hb))
    _print(evaluate_formula())
    print("\nDone. See SPIKE.md for interpretation and the gated live-run plan.")


if __name__ == "__main__":
    main()
