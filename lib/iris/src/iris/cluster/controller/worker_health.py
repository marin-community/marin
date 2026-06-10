# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-memory worker liveness, owned by the controller and folded from backend-observed events.

The backend never decides a worker is dead. Each reconcile tick it *observes*
its own I/O outcomes and emits :class:`WorkerHealthEvent`s; the controller folds
them through the single :meth:`WorkerHealthTracker.apply` site, which accumulates
the per-worker counters and applies the termination thresholds. ``apply`` is the
sole liveness-accounting mutation site. The only other writes are lifecycle:
startup seeding + worker registration (``heartbeat``/``register``) and removal
(``forget``/``forget_many``).

Per-worker signals:

- ``last_heartbeat_ms``: bumped each time the backend reaches the worker.
- ``healthy`` / ``active``: liveness verdict; set true on REACHED, dropped on
  removal.
- ``consecutive_failures``: incremented per UNREACHABLE event, reset on REACHED.
  ``PING_FAILURE_THRESHOLD`` consecutive failures trip termination.
- ``build_failures``: monotonic counter incremented per BUILD_FAILED event.
  ``BUILD_FAILURE_THRESHOLD`` build failures trip termination independently.

Thread-safe: ``apply`` runs on the reconcile thread; reads come from the
scheduler and RPC handler threads.
"""

import dataclasses
import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from iris.cluster.types import WorkerId

logger = logging.getLogger(__name__)

PING_FAILURE_THRESHOLD = 10
BUILD_FAILURE_THRESHOLD = 10


class WorkerHealthEventKind(StrEnum):
    """A backend-observed liveness signal for a single worker."""

    REACHED = "reached"
    """The backend reached the worker this tick — bump heartbeat, reset failures."""

    UNREACHABLE = "unreachable"
    """The backend could not reach the worker — increment consecutive failures."""

    BUILD_FAILED = "build_failed"
    """The worker failed to launch/build an attempt — increment build failures."""


@dataclass(frozen=True, slots=True)
class WorkerHealthEvent:
    """One backend-observed health signal the controller folds via :meth:`apply`."""

    worker_id: WorkerId
    kind: WorkerHealthEventKind


@dataclass(slots=True)
class WorkerLiveness:
    """Public snapshot of a worker's transient liveness state.

    Mutated in place by the tracker under its lock. Readers receive copies via
    :meth:`WorkerHealthTracker.liveness`.
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

    # -- Registration / seeding ---------------------------------------------

    def register(self, worker_id: WorkerId, *, now_ms: int) -> None:
        """Seed a newly-joined worker as live with a fresh heartbeat."""
        self.heartbeat([worker_id], now_ms)

    def heartbeat(self, worker_ids: Iterable[WorkerId], now_ms: int) -> None:
        """Seed/refresh liveness for a batch of workers (startup + registration).

        Bumps ``last_heartbeat_ms``, marks healthy/active, and resets the
        consecutive-failure counter. Steady-state liveness goes through
        :meth:`apply`; this is the lifecycle seed.
        """
        with self._lock:
            for wid in worker_ids:
                state = self._states.setdefault(wid, WorkerLiveness())
                state.last_heartbeat_ms = now_ms
                state.healthy = True
                state.active = True
                state.consecutive_failures = 0

    # -- The single liveness-accounting mutation site -----------------------

    def apply(self, events: Iterable[WorkerHealthEvent], *, now_ms: int) -> list[WorkerId]:
        """Fold backend-observed health events; return workers over a termination threshold.

        REACHED bumps the heartbeat and resets the failure count; UNREACHABLE
        increments consecutive failures; BUILD_FAILED increments build failures.
        Returns every worker currently at/over the ping- or build-failure
        threshold so the controller fails and tears them down — workers are
        forgotten on removal, so a returned worker does not repeat once gone.
        """
        with self._lock:
            for event in events:
                state = self._states.setdefault(event.worker_id, WorkerLiveness())
                if event.kind is WorkerHealthEventKind.REACHED:
                    state.last_heartbeat_ms = now_ms
                    state.healthy = True
                    state.active = True
                    state.consecutive_failures = 0
                elif event.kind is WorkerHealthEventKind.UNREACHABLE:
                    state.consecutive_failures += 1
                elif event.kind is WorkerHealthEventKind.BUILD_FAILED:
                    state.build_failures += 1
            over = [wid for wid, s in self._states.items() if self._over_threshold(s)]
        if over:
            logger.warning("Workers over health threshold: %s", [str(wid) for wid in over[:10]])
        return over

    def _over_threshold(self, state: WorkerLiveness) -> bool:
        return state.consecutive_failures >= self._ping_threshold or state.build_failures >= self._build_threshold

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
