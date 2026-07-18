# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A TTL cache that coalesces concurrent misses on one key into a single call.

N callers that miss the same key at once run compute once and share the result.
Entries are pruned on write.
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
    """Cache values under a key for ttl seconds, coalescing concurrent misses.

    A miss holds a per-key lock while it computes; concurrent callers for the same
    key wait and read the fresh value. Different keys do not block one another.
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
                # Dropping a key's lock here can race a concurrent refresh holding
                # it, so both compute and one query is duplicated. Acquiring key
                # locks under _guard would invert get_or_compute's lock order and
                # deadlock.
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
