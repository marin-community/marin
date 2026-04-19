# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-memory (RAM) heartbeat store for the controller ping loop.

When enabled (via ``IRIS_HEARTBEAT_INMEMORY=1``) the ping loop records
per-worker liveness in this process-local map instead of writing to the
``workers`` table on every successful ping. The DB is touched only when a
worker crosses the consecutive-failure threshold and is handed off to
``fail_workers_batch``.

Semantics (these are load-bearing):

* Absence of a worker from the map means "no data yet" — it is NOT a
  failure and MUST NOT cause the reaper to mark the worker dead. A worker
  is only failed when it misses ``threshold`` consecutive pings in a row.
* On controller restart the map starts empty — the manager is not seeded
  from the DB. Workers reappear in the map on their first successful ping.
* Successful pings NEVER write to the DB while this manager is active.
  The ping loop's only DB write in this mode is ``fail_workers_batch``.
"""

import os
import threading
import time
from dataclasses import dataclass


def heartbeat_inmemory_enabled() -> bool:
    """Whether the in-memory heartbeat path is active.

    Truthy values: "1", "true", "yes" (case-insensitive). Default off.
    """
    raw = os.environ.get("IRIS_HEARTBEAT_INMEMORY", "0").strip().lower()
    return raw in ("1", "true", "yes")


@dataclass
class HeartbeatEntry:
    """One row of the in-memory heartbeat map."""

    last_seen_ns: int
    consecutive_failures: int = 0


class HeartbeatManager:
    """Thread-safe in-memory liveness store.

    The ping loop is the sole writer. ``record_alive`` resets the failure
    counter; ``record_failure`` increments it and returns the new count so
    the caller can compare against the configured threshold. Readers
    observe state via ``age_ms`` / ``snapshot``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._map: dict[str, HeartbeatEntry] = {}
        # Monotonic metrics counters — independent ints, read/written
        # outside the lock; readers accept eventual consistency.
        self.writes_avoided: int = 0
        self.db_failure_writes: int = 0

    def record_alive(self, worker_id: str) -> None:
        """Mark the worker as just seen; reset consecutive failures."""
        now_ns = time.monotonic_ns()
        with self._lock:
            entry = self._map.get(worker_id)
            if entry is None:
                self._map[worker_id] = HeartbeatEntry(last_seen_ns=now_ns)
            else:
                entry.last_seen_ns = now_ns
                entry.consecutive_failures = 0
        self.writes_avoided += 1

    def record_failure(self, worker_id: str) -> int:
        """Increment the failure counter and return the new value.

        If no entry exists we create one with ``last_seen_ns=now`` so a
        missing entry never looks stale; the only failure signal is
        ``threshold`` consecutive failures.
        """
        with self._lock:
            entry = self._map.get(worker_id)
            if entry is None:
                entry = HeartbeatEntry(last_seen_ns=time.monotonic_ns())
                self._map[worker_id] = entry
            entry.consecutive_failures += 1
            return entry.consecutive_failures

    def age_ms(self, worker_id: str) -> int | None:
        """Milliseconds since the last successful ping; ``None`` if unknown."""
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
        self.db_failure_writes += n

    def snapshot(self) -> dict[str, int]:
        """Return ``{worker_id: age_ms}`` for debugging / tests."""
        now_ns = time.monotonic_ns()
        with self._lock:
            return {wid: (now_ns - e.last_seen_ns) // 1_000_000 for wid, e in self._map.items()}
