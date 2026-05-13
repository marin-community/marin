# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-memory worker liveness tracking.

Per-worker signals:

- ``last_heartbeat_ms``: bumped on each successful heartbeat / ping.
- ``healthy`` / ``active``: liveness verdict; flipped to false when the worker
  is marked unhealthy or removed.
- ``consecutive_failures``: incremented by failed ping/heartbeat RPCs, reset
  on success. ``PING_FAILURE_THRESHOLD`` consecutive failures trip
  termination.
- ``build_failures``: monotonic counter for BUILDING→FAILED transitions.
  ``BUILD_FAILURE_THRESHOLD`` build failures trip termination independently.

Thread-safe: written from ping/heartbeat threads, read from the reaper,
scheduler, and RPC handler threads.
"""

import dataclasses
import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass

from iris.cluster.types import WorkerId

logger = logging.getLogger(__name__)

PING_FAILURE_THRESHOLD = 10
BUILD_FAILURE_THRESHOLD = 10


@dataclass(slots=True)
class WorkerLiveness:
    """Public snapshot of a worker's transient liveness state.

    Mutated in place by the tracker under its lock during heartbeat/ping
    updates. Readers receive copies via :meth:`WorkerHealthTracker.liveness`.
    """

    healthy: bool = False
    active: bool = False
    consecutive_failures: int = 0
    last_heartbeat_ms: int = 0
    build_failures: int = 0


class WorkerHealthTracker:
    """In-memory source of truth for worker liveness."""

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
        self._states: dict[WorkerId, WorkerLiveness] = {}

    # -- Registration / heartbeat -------------------------------------------

    def register(self, worker_id: WorkerId, *, now_ms: int) -> None:
        """Mark a worker as live with a fresh heartbeat. Resets failure counters."""
        with self._lock:
            state = self._states.setdefault(worker_id, WorkerLiveness())
            state.last_heartbeat_ms = now_ms
            state.healthy = True
            state.active = True
            state.consecutive_failures = 0

    def heartbeat(self, worker_ids: Iterable[WorkerId], now_ms: int) -> None:
        """Record a successful heartbeat batch — bumps last_heartbeat_ms and resets health."""
        with self._lock:
            for wid in worker_ids:
                state = self._states.setdefault(wid, WorkerLiveness())
                state.last_heartbeat_ms = now_ms
                state.healthy = True
                state.active = True
                state.consecutive_failures = 0

    def bump_heartbeat(self, worker_ids: Iterable[WorkerId], now_ms: int) -> None:
        """Record a successful ping batch — bumps last_heartbeat_ms only.

        Does not reset healthy/active/consecutive_failures. The ping path
        records failures separately via :meth:`ping`.
        """
        with self._lock:
            for wid in worker_ids:
                state = self._states.setdefault(wid, WorkerLiveness())
                state.last_heartbeat_ms = now_ms

    def ping(self, worker_id: WorkerId, *, healthy: bool) -> None:
        """Record a ping outcome. A healthy ping resets the consecutive failure count."""
        with self._lock:
            state = self._states.setdefault(worker_id, WorkerLiveness())
            if healthy:
                state.consecutive_failures = 0
            else:
                state.consecutive_failures += 1
            failures = state.consecutive_failures
        logger.debug(
            "Worker %s ping=%s consecutive_failures=%d",
            worker_id,
            "ok" if healthy else "fail",
            failures,
        )

    def build_failed(self, worker_id: WorkerId) -> None:
        """Record a BUILDING→FAILED transition."""
        with self._lock:
            state = self._states.setdefault(worker_id, WorkerLiveness())
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

    def liveness(self, worker_id: WorkerId) -> WorkerLiveness:
        """Return a copy of the worker's current liveness snapshot.

        Returns a default-constructed ``WorkerLiveness`` if the worker isn't
        tracked yet. The returned dataclass is a copy — callers may read but
        should not mutate.
        """
        with self._lock:
            state = self._states.get(worker_id)
            return WorkerLiveness() if state is None else dataclasses.replace(state)

    def liveness_many(self, worker_ids: Iterable[WorkerId]) -> dict[WorkerId, WorkerLiveness]:
        """Return a copy of liveness for each requested worker."""
        with self._lock:
            return {wid: dataclasses.replace(self._states.get(wid, WorkerLiveness())) for wid in worker_ids}

    def all(self) -> dict[WorkerId, WorkerLiveness]:
        with self._lock:
            return {wid: dataclasses.replace(state) for wid, state in self._states.items()}

    def workers_over_threshold(self) -> list[WorkerId]:
        """Return IDs of workers that have exceeded a termination threshold."""
        with self._lock:
            return [
                wid
                for wid, s in self._states.items()
                if s.consecutive_failures >= self._ping_threshold or s.build_failures >= self._build_threshold
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
        """Current ``(consecutive_failures, build_failures)`` per worker (for diagnostics)."""
        with self._lock:
            return {wid: (s.consecutive_failures, s.build_failures) for wid, s in self._states.items()}

    # -- Test helpers -------------------------------------------------------

    def set_health_for_test(self, worker_id: WorkerId, healthy: bool) -> None:
        """Test helper: overwrite the healthy verdict."""
        with self._lock:
            state = self._states.setdefault(worker_id, WorkerLiveness())
            state.healthy = healthy
            if healthy:
                state.consecutive_failures = 0
            else:
                state.consecutive_failures = max(state.consecutive_failures, 1)

    def set_consecutive_failures_for_test(self, worker_id: WorkerId, count: int) -> None:
        """Test helper: overwrite consecutive_failures directly."""
        with self._lock:
            state = self._states.setdefault(worker_id, WorkerLiveness())
            state.consecutive_failures = count

    def set_last_heartbeat_for_test(self, worker_id: WorkerId, last_heartbeat_ms: int) -> None:
        """Test helper: backdate the last heartbeat for prune-window tests."""
        with self._lock:
            state = self._states.setdefault(worker_id, WorkerLiveness())
            state.last_heartbeat_ms = last_heartbeat_ms
