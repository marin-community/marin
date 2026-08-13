# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Named byte caches backed by FineStore transactions."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_for_futures

from rigging.filesystem import StoragePath, marin_temp_bucket

from finestore.reader import ReadView
from finestore.store import DataStore

_CACHE_TTL_DAYS = 30
_MAX_TRANSACTION_BYTES = 100 * 1024 * 1024
_MAX_TRANSACTION_OBJECTS = 65_536

logger = logging.getLogger(__name__)


class PersistentKvCache:
    """An in-process memory tier over a FineStore named-object table."""

    def __init__(self, resolve_root: Callable[[], str] | None = None) -> None:
        self._resolve_root = resolve_root
        self._root: str | None = None
        self._store: DataStore | None = None
        self._root_lock = threading.Lock()
        self._lock = threading.Lock()
        self._memory: dict[str, bytes] = {}
        self._remote_pending: dict[str, bytes] = {}
        self._remote_write_scheduled = False
        self._background_lock = threading.Lock()
        self._background_executor: ThreadPoolExecutor | None = None
        self._pending_writes: set[Future] = set()

    @classmethod
    def at(cls, root: str) -> PersistentKvCache:
        return cls(lambda: root)

    @classmethod
    def in_memory(cls) -> PersistentKvCache:
        return cls()

    @classmethod
    def for_prefix(cls, prefix: str) -> PersistentKvCache:
        return cls(lambda: marin_temp_bucket(_CACHE_TTL_DAYS, prefix))

    def location(self) -> str | None:
        return self._resolve_root() if self._resolve_root is not None else None

    def load(self, key: str) -> bytes | None:
        """Return ``key`` from memory or the latest committed FineStore view."""
        with self._lock:
            if key in self._memory:
                return self._memory[key]
        if self._resolve_root is None:
            return None
        try:
            value = ReadView(self._storage_root()).read_blob(key)
        except OSError as exc:
            logger.warning("FineStore cache is unreadable, treating %s as a miss: %s", key, exc)
            return None
        if value is not None:
            with self._lock:
                self._memory.setdefault(key, value)
        return value

    def store(self, key: str, value: bytes) -> None:
        """Publish ``value`` under ``key``; remote commits run in the background."""
        with self._lock:
            self._memory[key] = value
        if self._resolve_root is None:
            return
        if StoragePath(self._storage_root()).is_remote:
            self._queue_remote_write(key, value)
        else:
            try:
                self._write(key, value)
            except OSError as exc:
                logger.warning("FineStore cache is unwritable, not storing %s: %s", key, exc)

    def close(self) -> None:
        """Close the lazy writer after its explicit commits have completed."""
        self._flush_background_writes()
        with self._background_lock:
            executor = self._background_executor
            self._background_executor = None
        if executor is not None:
            executor.shutdown(wait=True)
        with self._root_lock:
            if self._store is not None:
                self._store.close()
                self._store = None

    def _storage_root(self) -> str:
        with self._root_lock:
            if self._root is None:
                assert self._resolve_root is not None
                self._root = self._resolve_root()
            return self._root

    def _writer(self) -> DataStore:
        with self._root_lock:
            if self._store is None:
                self._store = DataStore.open(self._storage_root_unlocked())
            return self._store

    def _storage_root_unlocked(self) -> str:
        if self._root is None:
            assert self._resolve_root is not None
            self._root = self._resolve_root()
        return self._root

    def _write(self, key: str, value: bytes) -> None:
        writer = self._writer()
        writer.write_object(key, value)
        writer.flush()

    def _queue_remote_write(self, key: str, value: bytes) -> None:
        with self._lock:
            self._remote_pending[key] = value
            if self._remote_write_scheduled:
                return
            self._remote_write_scheduled = True
        self._submit_background_write(self._drain_remote_writes)

    def _submit_background_write(self, write: Callable[[], None]) -> None:
        with self._background_lock:
            if self._background_executor is None:
                self._background_executor = ThreadPoolExecutor(1, thread_name_prefix="finestore-cache-write")
            future = self._background_executor.submit(write)
            self._pending_writes.add(future)
        future.add_done_callback(self._forget_pending_write)

    def _forget_pending_write(self, future: Future) -> None:
        with self._background_lock:
            self._pending_writes.discard(future)
        error = future.exception()
        if error is not None:
            logger.warning("FineStore cache write failed: %s", error)

    def _flush_background_writes(self) -> None:
        while True:
            with self._background_lock:
                pending = list(self._pending_writes)
            if not pending:
                return
            wait_for_futures(pending)

    def _drain_remote_writes(self) -> None:
        """Publish every queued burst as a bounded multi-object transaction."""
        while True:
            with self._lock:
                selected = []
                selected_bytes = 0
                for key, value in self._remote_pending.items():
                    if selected and (
                        len(selected) >= _MAX_TRANSACTION_OBJECTS or selected_bytes + len(value) > _MAX_TRANSACTION_BYTES
                    ):
                        break
                    selected.append(key)
                    selected_bytes += len(value)
                if not selected:
                    self._remote_write_scheduled = False
                    return
                pending = {key: self._remote_pending.pop(key) for key in selected}
            try:
                with self._writer().transaction() as transaction:
                    for key, value in pending.items():
                        transaction.write_object(key, value)
            except Exception:
                with self._lock:
                    self._remote_write_scheduled = False
                    retry = bool(self._remote_pending)
                    if retry:
                        self._remote_write_scheduled = True
                if retry:
                    self._submit_background_write(self._drain_remote_writes)
                raise
