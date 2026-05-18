# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Simple readers-writer lock used by the log store.

Multiple readers can hold the lock concurrently; writers wait for all readers
to release before acquiring exclusive access. The store uses it to keep
queries (which open Parquet files lazily inside DuckDB) from observing a
mid-rename / mid-delete file ops.
"""

from __future__ import annotations

from threading import Condition, Lock


class RWLock:
    """Reader-preference rwlock with no priority inversion safeguards.

    Acceptable for finelog because writers (compaction commit, GC, drop) are
    rare and brief; readers (queries) are the common case.
    """

    def __init__(self) -> None:
        self._cond = Condition(Lock())
        self._readers = 0
        self._writer = False
        # Number of threads currently blocked inside ``write_acquire``
        # waiting for readers/writer to drain. Exposed for tests that need
        # to assert "writer is queued" without timing dependence.
        self._pending_writers = 0

    def read_acquire(self) -> None:
        with self._cond:
            while self._writer:
                self._cond.wait()
            self._readers += 1

    def read_release(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def write_acquire(self) -> None:
        with self._cond:
            self._pending_writers += 1
            # Wake any waiters that care about pending-writer state
            # (tests use this to assert "writer queued" without polling).
            self._cond.notify_all()
            try:
                while self._writer or self._readers > 0:
                    self._cond.wait()
                self._writer = True
            finally:
                self._pending_writers -= 1

    def write_release(self) -> None:
        with self._cond:
            self._writer = False
            self._cond.notify_all()
