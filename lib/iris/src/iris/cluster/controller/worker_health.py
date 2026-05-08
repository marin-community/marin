# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-memory worker health and liveness tracking.

A single tracker owns every transient per-worker signal:

- ``last_heartbeat_ms``: bumped on each successful heartbeat / ping.
- ``healthy`` / ``active``: liveness verdict; flipped to false when the worker
  is marked unhealthy or removed.
- ``consecutive_ping_failures``: incremented by failed ping/heartbeat RPCs,
  reset on success. Ten consecutive failures trip the termination threshold.
- ``build_failures``: monotonic counter for BUILDING→FAILED transitions. Ten
  build failures trip the termination threshold independently.

All four pieces of state used to live in dedicated columns on the ``workers``
SQLite row, rewritten on every heartbeat / ping batch. That made the heartbeat
path a SQLite writer transaction, bloating the WAL and starving dashboard
reads. The tracker keeps that data in memory; the SQLite ``workers`` row only
records durable identity / capability metadata.

Crash recovery: a fresh controller starts with an empty tracker. Until each
worker re-establishes contact (one ping cycle, ~10s) it appears unhealthy /
inactive — the same trade-off the ping-failure counter has always made.

Thread-safe: written from ping/heartbeat and task-update threads, read from
the reaper, scheduler, and RPC handler threads.
"""

import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass

from iris.cluster.types import WorkerId, get_gpu_count, get_tpu_count
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

PING_FAILURE_THRESHOLD = 10
BUILD_FAILURE_THRESHOLD = 10


@dataclass(slots=True)
class CommittedResources:
    """Sum of resources currently committed to a worker by active tasks."""

    cpu_millicores: int = 0
    memory_bytes: int = 0
    gpu: int = 0
    tpu: int = 0


class WorkerCommitTracker:
    """In-memory ``{worker_id: CommittedResources}`` map.

    The scheduler increments and decrements committed resources per
    assignment / completion / preemption. Previously this was rewritten on
    the ``workers`` SQLite row inside every assignment transaction, which
    serialized all writes through the SQLite writer connection. Holding it
    in memory keeps assignment hot paths writer-free while still providing
    the available-capacity arithmetic the scheduler needs.

    Crash recovery: the tracker is repopulated at controller boot from a
    single ``GROUP BY current_worker_id`` aggregation across active tasks.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._committed: dict[WorkerId, CommittedResources] = {}

    def reset(self, committed: dict[WorkerId, CommittedResources]) -> None:
        """Replace the entire map. Used on boot to prime from SQL."""
        with self._lock:
            self._committed = dict(committed)

    def add(self, worker_id: WorkerId, resources: job_pb2.ResourceSpecProto) -> None:
        with self._lock:
            entry = self._committed.setdefault(worker_id, CommittedResources())
            entry.cpu_millicores += int(resources.cpu_millicores)
            entry.memory_bytes += int(resources.memory_bytes)
            entry.gpu += int(get_gpu_count(resources.device))
            entry.tpu += int(get_tpu_count(resources.device))

    def subtract(self, worker_id: WorkerId, resources: job_pb2.ResourceSpecProto) -> None:
        with self._lock:
            entry = self._committed.setdefault(worker_id, CommittedResources())
            entry.cpu_millicores = max(0, entry.cpu_millicores - int(resources.cpu_millicores))
            entry.memory_bytes = max(0, entry.memory_bytes - int(resources.memory_bytes))
            entry.gpu = max(0, entry.gpu - int(get_gpu_count(resources.device)))
            entry.tpu = max(0, entry.tpu - int(get_tpu_count(resources.device)))

    def get(self, worker_id: WorkerId) -> CommittedResources:
        with self._lock:
            entry = self._committed.get(worker_id)
            if entry is None:
                return CommittedResources()
            return CommittedResources(
                cpu_millicores=entry.cpu_millicores,
                memory_bytes=entry.memory_bytes,
                gpu=entry.gpu,
                tpu=entry.tpu,
            )

    def forget(self, worker_id: WorkerId) -> None:
        with self._lock:
            self._committed.pop(worker_id, None)

    def all(self) -> dict[WorkerId, CommittedResources]:
        with self._lock:
            return {
                wid: CommittedResources(
                    cpu_millicores=v.cpu_millicores,
                    memory_bytes=v.memory_bytes,
                    gpu=v.gpu,
                    tpu=v.tpu,
                )
                for wid, v in self._committed.items()
            }


@dataclass(slots=True)
class WorkerLiveness:
    """Snapshot of a worker's transient state."""

    last_heartbeat_ms: int
    healthy: bool
    active: bool
    consecutive_ping_failures: int
    build_failures: int


@dataclass(slots=True)
class _WorkerState:
    last_heartbeat_ms: int = 0
    healthy: bool = True
    active: bool = True
    consecutive_ping_failures: int = 0
    build_failures: int = 0


class WorkerHealthTracker:
    """In-memory source of truth for worker liveness and failure counters."""

    def __init__(
        self,
        *,
        ping_threshold: int = PING_FAILURE_THRESHOLD,
        build_threshold: int = BUILD_FAILURE_THRESHOLD,
    ) -> None:
        assert ping_threshold > 0
        assert build_threshold > 0
        self._ping_threshold = ping_threshold
        self._build_threshold = build_threshold
        self._lock = threading.Lock()
        self._states: dict[WorkerId, _WorkerState] = {}

    # -- Registration / heartbeat -------------------------------------------

    def register(self, worker_id: WorkerId, *, now_ms: int) -> None:
        """Mark a worker as live with a fresh heartbeat. Resets failure counters."""
        with self._lock:
            state = self._states.setdefault(worker_id, _WorkerState())
            state.last_heartbeat_ms = now_ms
            state.healthy = True
            state.active = True
            state.consecutive_ping_failures = 0

    def heartbeat(self, worker_ids: Iterable[WorkerId], now_ms: int) -> None:
        """Record a successful heartbeat batch — bumps last_heartbeat_ms and resets health."""
        with self._lock:
            for wid in worker_ids:
                state = self._states.setdefault(wid, _WorkerState())
                state.last_heartbeat_ms = now_ms
                state.healthy = True
                state.active = True
                state.consecutive_ping_failures = 0

    def bump_heartbeat(self, worker_ids: Iterable[WorkerId], now_ms: int) -> None:
        """Record a successful ping batch — bumps last_heartbeat_ms only.

        Does not reset healthy/active/consecutive_ping_failures. The ping path
        records failures separately via :meth:`ping`.
        """
        with self._lock:
            for wid in worker_ids:
                state = self._states.setdefault(wid, _WorkerState())
                state.last_heartbeat_ms = now_ms

    def ping(self, worker_id: WorkerId, *, healthy: bool) -> None:
        """Record a ping outcome. A healthy ping resets the consecutive failure count."""
        with self._lock:
            state = self._states.setdefault(worker_id, _WorkerState())
            if healthy:
                state.consecutive_ping_failures = 0
            else:
                state.consecutive_ping_failures += 1
            failures = state.consecutive_ping_failures
        logger.debug(
            "Worker %s ping=%s consecutive_ping_failures=%d",
            worker_id,
            "ok" if healthy else "fail",
            failures,
        )

    def build_failed(self, worker_id: WorkerId) -> None:
        """Record a BUILDING→FAILED transition."""
        with self._lock:
            state = self._states.setdefault(worker_id, _WorkerState())
            state.build_failures += 1
            failures = state.build_failures
        logger.debug("Worker %s build_failures=%d", worker_id, failures)

    def mark_unhealthy(self, worker_id: WorkerId) -> None:
        """Force the worker into the unhealthy verdict (used by failure cascade)."""
        with self._lock:
            state = self._states.get(worker_id)
            if state is None:
                return
            state.healthy = False

    # -- Reads --------------------------------------------------------------

    def get(self, worker_id: WorkerId) -> WorkerLiveness | None:
        with self._lock:
            state = self._states.get(worker_id)
            if state is None:
                return None
            return WorkerLiveness(
                last_heartbeat_ms=state.last_heartbeat_ms,
                healthy=state.healthy,
                active=state.active,
                consecutive_ping_failures=state.consecutive_ping_failures,
                build_failures=state.build_failures,
            )

    def all(self) -> dict[WorkerId, WorkerLiveness]:
        with self._lock:
            return {
                wid: WorkerLiveness(
                    last_heartbeat_ms=s.last_heartbeat_ms,
                    healthy=s.healthy,
                    active=s.active,
                    consecutive_ping_failures=s.consecutive_ping_failures,
                    build_failures=s.build_failures,
                )
                for wid, s in self._states.items()
            }

    def workers_over_threshold(self) -> list[WorkerId]:
        """Return IDs of workers that have exceeded a termination threshold."""
        with self._lock:
            return [
                wid
                for wid, s in self._states.items()
                if s.consecutive_ping_failures >= self._ping_threshold or s.build_failures >= self._build_threshold
            ]

    # -- Eviction -----------------------------------------------------------

    def forget(self, worker_id: WorkerId) -> None:
        with self._lock:
            self._states.pop(worker_id, None)

    def forget_many(self, worker_ids: Iterable[WorkerId]) -> None:
        with self._lock:
            for wid in worker_ids:
                self._states.pop(wid, None)

    def snapshot(self) -> dict[WorkerId, tuple[int, int]]:
        """Current (consecutive_ping_failures, build_failures) per worker (for diagnostics)."""
        with self._lock:
            return {wid: (s.consecutive_ping_failures, s.build_failures) for wid, s in self._states.items()}

    # -- Test helpers -------------------------------------------------------

    def set_health_for_test(self, worker_id: WorkerId, healthy: bool) -> None:
        """Test helper: overwrite the healthy verdict."""
        with self._lock:
            state = self._states.setdefault(worker_id, _WorkerState())
            state.healthy = healthy
            if healthy:
                state.consecutive_ping_failures = 0
            else:
                state.consecutive_ping_failures = max(state.consecutive_ping_failures, 1)

    def set_consecutive_failures_for_test(self, worker_id: WorkerId, count: int) -> None:
        """Test helper: overwrite consecutive_ping_failures directly."""
        with self._lock:
            state = self._states.setdefault(worker_id, _WorkerState())
            state.consecutive_ping_failures = count

    def set_last_heartbeat_for_test(self, worker_id: WorkerId, last_heartbeat_ms: int) -> None:
        """Test helper: backdate the last heartbeat for prune-window tests."""
        with self._lock:
            state = self._states.setdefault(worker_id, _WorkerState())
            state.last_heartbeat_ms = last_heartbeat_ms
