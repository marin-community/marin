# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-memory heartbeat store for the controller ping loop.

When enabled (via ``IRIS_HEARTBEAT_INMEMORY=1``), the ping loop records
per-worker liveness here instead of writing to the ``workers`` table on
every ping. The DB is only touched when a worker crosses the failure
threshold and gets cleaned up via ``fail_workers_batch``.

Design notes:
- Pure in-process state behind a ``threading.Lock``. Critical sections are
  minimal (single-field reads/writes).
- On controller restart the map is empty; liveness state is seeded from
  the ``workers`` table at boot so freshly-restarted controllers do not
  incorrectly reap workers that were healthy before the restart.
- The reconciler cadence (when to re-check for stale workers) is identical
  to the existing ping-loop cadence; the ping loop itself is the reconciler.
  A separate reconciler thread would only introduce race windows.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass


def heartbeat_inmemory_enabled() -> bool:
    """Whether the in-memory heartbeat path is active.

    Gate keyed once per caller. Default off keeps the DB-only behavior.
    """
    return os.environ.get("IRIS_HEARTBEAT_INMEMORY", "0") == "1"


@dataclass
class HeartbeatEntry:
    """One row of the in-memory heartbeat map."""

    last_seen_ns: int
    # Consecutive failure count tracked alongside liveness so reads can
    # distinguish "recently pinged" from "threshold exceeded". The ping
    # loop is the sole writer.
    consecutive_failures: int = 0


class HeartbeatManager:
    """Thread-safe in-memory liveness store.

    Write path: ping loop records successes via ``record_alive`` and
    failures via ``record_failure``. Failure threshold crossings are
    surfaced by ``drain_dead(threshold)`` so the caller can hand off to
    ``fail_workers_batch`` exactly once per worker.

    Read path: ``_reap_stale_workers`` asks ``last_seen_ms`` / ``age_ms``
    to decide whether to fail workers that have not heartbeated recently.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._map: dict[str, HeartbeatEntry] = {}
        # Metrics (monotonic counters) — read without locking since they
        # are independent ints and readers accept eventual consistency.
        self._writes_avoided: int = 0
        self._db_failure_writes: int = 0

    def seed_from_db(self, entries: list[tuple[str, int]]) -> None:
        """Populate the map at boot from ``(worker_id, last_heartbeat_ms)`` rows.

        Called once before the ping loop starts. After this, the DB rows
        are frozen (from the ping loop's perspective) until a worker
        crosses the failure threshold.
        """
        now_ns = time.monotonic_ns()
        # Convert DB epoch_ms -> monotonic_ns delta. Since the DB's epoch
        # and our monotonic clock disagree, we compute the age and subtract.
        # On restart, a worker seen 5 s ago in wall time should look 5 s
        # ago in monotonic time too.
        wall_now_ms = int(time.time() * 1000)
        with self._lock:
            for wid, last_ms in entries:
                if last_ms is None:
                    # Unseen workers default to "just seen" to avoid a
                    # post-restart reap storm before the first ping lands.
                    self._map[wid] = HeartbeatEntry(last_seen_ns=now_ns)
                    continue
                age_ms = max(0, wall_now_ms - int(last_ms))
                seeded_ns = now_ns - age_ms * 1_000_000
                self._map[wid] = HeartbeatEntry(last_seen_ns=seeded_ns)

    def record_alive(self, worker_id: str) -> None:
        now_ns = time.monotonic_ns()
        with self._lock:
            entry = self._map.get(worker_id)
            if entry is None:
                self._map[worker_id] = HeartbeatEntry(last_seen_ns=now_ns)
            else:
                entry.last_seen_ns = now_ns
                entry.consecutive_failures = 0
            self._writes_avoided += 1

    def record_failure(self, worker_id: str) -> int:
        """Increment failure counter; returns new value."""
        with self._lock:
            entry = self._map.get(worker_id)
            if entry is None:
                entry = HeartbeatEntry(last_seen_ns=time.monotonic_ns())
                self._map[worker_id] = entry
            entry.consecutive_failures += 1
            return entry.consecutive_failures

    def age_ms(self, worker_id: str) -> int | None:
        """Monotonic-ns age since last successful ping; None if unknown."""
        with self._lock:
            entry = self._map.get(worker_id)
            if entry is None:
                return None
            return (time.monotonic_ns() - entry.last_seen_ns) // 1_000_000

    def remove(self, worker_id: str) -> None:
        with self._lock:
            self._map.pop(worker_id, None)

    def retain_only(self, active_ids: set[str]) -> None:
        """Drop entries for workers no longer in the active set."""
        with self._lock:
            stale = [wid for wid in self._map if wid not in active_ids]
            for wid in stale:
                del self._map[wid]

    def note_db_failure_write(self, n: int = 1) -> None:
        """Record that we wrote N failed workers to the DB."""
        self._db_failure_writes += n

    @property
    def writes_avoided(self) -> int:
        """Number of successful pings that did NOT write to the DB."""
        return self._writes_avoided

    @property
    def db_failure_writes(self) -> int:
        """Number of DB writes from the reconciler (failed workers only)."""
        return self._db_failure_writes

    def snapshot(self) -> dict[str, int]:
        """Return {worker_id: age_ms} for debugging / tests."""
        now_ns = time.monotonic_ns()
        with self._lock:
            return {wid: (now_ns - e.last_seen_ns) // 1_000_000 for wid, e in self._map.items()}
