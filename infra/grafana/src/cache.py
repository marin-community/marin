# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A TTL cache that also collapses concurrent misses on one key into one call.

Grafana's own query caching is an Enterprise feature, so this is what keeps a
shared, auto-refreshing dashboard from multiplying panel renders straight through
to the finelog hub. Coalescing matters as much as the TTL: N viewers opening the
same dashboard at once should cost one query, not N.

Entries are pruned on write, since callers key on a rotating time bucket and the
process is long-lived.
"""

import threading
import time
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Generic, TypeVar

V = TypeVar("V")


@dataclass
class _Entry(Generic[V]):
    value: V
    expires_at: float


class TtlCache(Generic[V]):
    """Cache values under a key for ``ttl`` seconds, coalescing concurrent misses.

    A miss holds a per-key lock while it computes, so concurrent callers for the
    same key wait and then read the fresh value instead of each issuing a query.
    Different keys never block one another.
    """

    def __init__(self, ttl: float) -> None:
        self._ttl = ttl
        self._entries: dict[Hashable, _Entry[V]] = {}
        self._key_locks: dict[Hashable, threading.Lock] = {}
        self._guard = threading.Lock()

    def _lock_for(self, key: Hashable) -> threading.Lock:
        with self._guard:
            return self._key_locks.setdefault(key, threading.Lock())

    def _live(self, key: Hashable) -> _Entry[V] | None:
        with self._guard:
            entry = self._entries.get(key)
        if entry is not None and entry.expires_at > time.monotonic():
            return entry
        return None

    def _store(self, key: Hashable, value: V) -> None:
        """Cache ``value`` under ``key`` for the TTL, dropping every expired entry."""
        now = time.monotonic()
        with self._guard:
            self._entries[key] = _Entry(value=value, expires_at=now + self._ttl)
            expired = [k for k, e in self._entries.items() if e.expires_at <= now]
            for k in expired:
                del self._entries[k]
                # Dropping the lock alongside the entry can race a refresh that
                # holds it: the next caller then makes a fresh lock and both
                # compute. That costs one duplicate query and resolves itself,
                # whereas acquiring key locks under _guard here would invert
                # get_or_compute's lock order and deadlock.
                self._key_locks.pop(k, None)

    def get_or_compute(self, key: Hashable, compute: Callable[[], V]) -> V:
        """Return the cached value for ``key``, computing it if absent or stale."""
        entry = self._live(key)
        if entry is not None:
            return entry.value

        with self._lock_for(key):
            # Another caller may have populated it while we waited for the lock.
            entry = self._live(key)
            if entry is not None:
                return entry.value
            value = compute()
            self._store(key, value)
            return value

    def __len__(self) -> int:
        with self._guard:
            return len(self._entries)
