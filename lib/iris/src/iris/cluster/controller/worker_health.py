# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-memory worker health tracking with ping-based decay.

Tracks two independent failure modes per worker:

- Consecutive ping failures: incremented by any failed ping or heartbeat RPC,
  reset to zero by any successful ping. Ten consecutive failures trip the
  termination threshold.
- Build failures: monotonic counter for BUILDING→FAILED transitions. Ten build
  failures trip the termination threshold independently.

The ping-based decay means no clock management is needed: healthy pings
naturally reset the failure count, and the tracker needs no time parameters.

Lives entirely in memory. A failing worker recurs within one ping cycle,
so losing evidence on controller restart doesn't meaningfully delay termination.
"""

import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass

from iris.cluster.types import WorkerId

logger = logging.getLogger(__name__)

PING_FAILURE_THRESHOLD = 10
BUILD_FAILURE_THRESHOLD = 10


@dataclass(slots=True)
class _WorkerState:
    consecutive_ping_failures: int = 0
    build_failures: int = 0


class WorkerHealthTracker:
    """Tracks per-worker failure counts for termination decisions.

    Thread-safe: written from ping/heartbeat and task-update threads,
    read from the reaper thread.
    """

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

    def workers_over_threshold(self) -> list[WorkerId]:
        """Return IDs of workers that have exceeded a termination threshold."""
        with self._lock:
            return [
                wid
                for wid, s in self._states.items()
                if s.consecutive_ping_failures >= self._ping_threshold or s.build_failures >= self._build_threshold
            ]

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
