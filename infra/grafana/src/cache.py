# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A TTL cache with per-key coordination and an optional size limit.

Concurrent callers share a result or an expected, message-only failure. Entries
are pruned on write and may be evicted before their TTL to meet a size budget.
"""

import sys
import threading
import time
from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generic, TypeVar

from errors import UpstreamError
from finelog.errors import StatsError

V = TypeVar("V")


@dataclass
class _Entry(Generic[V]):
    value: V
    expires_at: float
    size: int


@dataclass
class _Failure:
    error: Exception
    expires_at: float
    size: int


@dataclass
class _KeyLock:
    lock: threading.Lock
    users: int = 0


def _cacheable_error(error: Exception) -> Exception | None:
    # Reconstruct only errors whose payload is a small message. Copying arbitrary
    # exceptions can retain HTTP responses, JSON documents, and traceback locals.
    if isinstance(error, UpstreamError):
        return UpstreamError(error.source, str(error), status_code=error.status_code)
    if isinstance(error, StatsError) or type(error) in (ValueError, RuntimeError):
        return type(error)(str(error))
    return None


class TtlCache(Generic[V]):
    """Cache outcomes for up to ttl seconds, subject to the size budget.

    A miss holds a per-key lock while it computes; concurrent callers for the same
    key wait and reuse the outcome if it remains cached. Cached failures suppress
    repeated upstream work on retries. Different keys do not block one another.
    """

    def __init__(
        self,
        ttl: float,
        *,
        max_size: int = sys.maxsize,
        get_size: Callable[[V], int] = sys.getsizeof,
    ) -> None:
        self._ttl = ttl
        self._max_size = max_size
        self._get_size = get_size
        self._entries: dict[Hashable, _Entry[V] | _Failure] = {}
        self._key_locks: dict[Hashable, _KeyLock] = {}
        self._guard = threading.Lock()

    @contextmanager
    def _key_lock(self, key: Hashable) -> Iterator[None]:
        with self._guard:
            entry = self._key_locks.setdefault(key, _KeyLock(threading.Lock()))
            entry.users += 1
        try:
            with entry.lock:
                yield
        finally:
            with self._guard:
                entry.users -= 1
                if entry.users == 0:
                    del self._key_locks[key]

    def _live(self, key: Hashable) -> _Entry[V] | _Failure | None:
        with self._guard:
            entry = self._entries.get(key)
        if entry is not None and entry.expires_at > time.monotonic():
            return entry
        return None

    def _store(self, key: Hashable, entry: _Entry[V] | _Failure) -> None:
        """Cache ``entry`` under ``key``, dropping every expired entry."""
        now = time.monotonic()
        with self._guard:
            self._entries[key] = entry
            expired = [k for k, e in self._entries.items() if e.expires_at <= now]
            for k in expired:
                del self._entries[k]

            size = sum(entry.size for entry in self._entries.values())
            while size > self._max_size:
                oldest = next(iter(self._entries))
                removed = self._entries.pop(oldest)
                size -= removed.size

    @staticmethod
    def _resolve(entry: _Entry[V] | _Failure) -> V:
        if isinstance(entry, _Failure):
            error = _cacheable_error(entry.error)
            assert error is not None
            raise error from None
        return entry.value

    def get_or_compute(self, key: Hashable, compute: Callable[[], V]) -> V:
        """Return the cached outcome for ``key``, computing it if absent or stale."""
        entry = self._live(key)
        if entry is not None:
            return self._resolve(entry)

        with self._key_lock(key):
            # Another caller may have populated an outcome while we waited.
            entry = self._live(key)
            if entry is not None:
                return self._resolve(entry)
            try:
                value = compute()
            except Exception as error:
                cached = _cacheable_error(error)
                if cached is not None:
                    self._store(
                        key,
                        _Failure(cached, time.monotonic() + self._ttl, sys.getsizeof(str(cached))),
                    )
                raise
            self._store(
                key,
                _Entry(
                    value=value,
                    expires_at=time.monotonic() + self._ttl,
                    size=self._get_size(value),
                ),
            )
            return value

    def __len__(self) -> int:
        with self._guard:
            return len(self._entries)
