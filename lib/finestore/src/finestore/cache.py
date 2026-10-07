# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Named byte caches backed by FineStore transactions."""

from __future__ import annotations

import atexit
import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

from rigging import telemetry
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import StoragePath

from finestore.reader import BlobReadDiagnostics, ReadView
from finestore.store import DataStore

_CACHE_TTL_DAYS = 30
_EXIT_FLUSH_TIMEOUT = 10.0
_MAX_BATCH_DATA_BYTES = 100 * 1024 * 1024
_MAX_BATCH_OBJECTS = 65_536

logger = logging.getLogger(__name__)

_CACHE_LOAD_CALLS = telemetry.counter("finestore_cache_load_calls", unit="{call}")
_CACHE_REQUESTED_KEYS = telemetry.counter("finestore_cache_requested_keys", unit="{key}")
_CACHE_MEMORY_HITS = telemetry.counter("finestore_cache_memory_hits", unit="{key}")
_CACHE_STORAGE_HITS = telemetry.counter("finestore_cache_storage_hits", unit="{key}")
_CACHE_MISSES = telemetry.counter("finestore_cache_misses", unit="{key}")


@dataclass
class CacheReadDiagnostics:
    """Cumulative cache probes and underlying FineStore batch reads."""

    load_calls: int = 0
    requested_keys: int = 0
    memory_hits: int = 0
    storage_hits: int = 0
    misses: int = 0
    blob_reads: BlobReadDiagnostics = field(default_factory=BlobReadDiagnostics)


class PersistentKvCache:
    """An in-process memory tier over best-effort FineStore named-object writes.

    ``is_writer`` is evaluated when storing so a distributed process can resolve its
    rank after constructing the cache. Non-writers retain values in memory and keep
    read-through access to the shared archive.
    """

    def __init__(
        self,
        resolve_root: Callable[[], str] | None = None,
        *,
        is_writer: Callable[[], bool] | None = None,
    ) -> None:
        self._resolve_root = resolve_root
        self._is_writer = is_writer or (lambda: True)
        self._root: str | None = None
        self._store: DataStore | None = None
        self._root_lock = threading.Lock()
        self._lock = threading.Lock()
        self._memory: dict[str, bytes] = {}
        self._read_diagnostics = CacheReadDiagnostics()
        self._remote_pending: dict[str, bytes] = {}
        self._background_thread: threading.Thread | None = None
        self._exit_registered = False
        self._closed = False

    @classmethod
    def at(cls, root: str) -> PersistentKvCache:
        return cls(lambda: root)

    @classmethod
    def in_memory(cls) -> PersistentKvCache:
        return cls()

    @classmethod
    def for_prefix(cls, prefix: str, *, is_writer: Callable[[], bool] | None = None) -> PersistentKvCache:
        return cls(lambda: marin_temp_bucket(_CACHE_TTL_DAYS, prefix), is_writer=is_writer)

    def location(self) -> str | None:
        return self._resolve_root() if self._resolve_root is not None else None

    def load(self, key: str) -> bytes | None:
        """Return ``key`` from memory or the latest committed FineStore view."""
        return self.load_many((key,)).get(key)

    def read_diagnostics(self) -> CacheReadDiagnostics:
        """Return a snapshot of cumulative cache probes and FineStore reads."""
        with self._lock:
            return replace(self._read_diagnostics, blob_reads=replace(self._read_diagnostics.blob_reads))

    def load_many(self, keys: Sequence[str]) -> dict[str, bytes]:
        """Read a batch from memory and one committed view, omitting cache misses."""
        unique_keys = list(dict.fromkeys(keys))
        with self._lock:
            values = {key: self._memory[key] for key in unique_keys if key in self._memory}
        memory_hits = len(values)
        missing = [key for key in unique_keys if key not in values]
        loaded: dict[str, bytes] = {}
        view: ReadView | None = None
        if missing and self._resolve_root is not None:
            # The cache is best-effort: an unreadable, inconsistent, or corrupt archive is a miss.
            try:
                view = ReadView(self._storage_root())
                loaded = view.read_blobs(missing)
            except Exception as exc:
                logger.warning("FineStore cache is unreadable, treating %d keys as misses: %s", len(missing), exc)
        with self._lock:
            for key, value in loaded.items():
                self._memory.setdefault(key, value)
            result = {key: self._memory[key] for key in unique_keys if key in self._memory}
            diagnostics = self._read_diagnostics
            diagnostics.load_calls += 1
            diagnostics.requested_keys += len(unique_keys)
            diagnostics.memory_hits += memory_hits
            diagnostics.storage_hits += len(loaded)
            diagnostics.misses += len(unique_keys) - len(result)
            if view is not None:
                diagnostics.blob_reads.add(view.read_diagnostics())
        _CACHE_LOAD_CALLS.add()
        _CACHE_REQUESTED_KEYS.add(len(unique_keys))
        _CACHE_MEMORY_HITS.add(memory_hits)
        _CACHE_STORAGE_HITS.add(len(loaded))
        _CACHE_MISSES.add(len(unique_keys) - len(result))
        return result

    def store(self, key: str, value: bytes) -> None:
        """Remember ``value`` and persist it synchronously or through the remote write queue."""
        with self._lock:
            if self._closed:
                raise RuntimeError("cache is closed")
            self._memory[key] = value
        if self._resolve_root is None or not self._is_writer():
            return
        if StoragePath(self._storage_root()).is_remote:
            self._queue_remote_write(key, value)
        else:
            if not self._ensure_exit_cleanup():
                return
            try:
                self._write(key, value)
            except Exception as exc:
                logger.warning("FineStore cache is unwritable, not storing %s: %s", key, exc)

    def close(self) -> None:
        """Drain queued cache writes and close the lazy writer."""
        self._close(timeout=None)
        with self._lock:
            registered = self._exit_registered
            self._exit_registered = False
        if registered:
            atexit.unregister(self._close_at_exit)

    def _close_at_exit(self) -> None:
        if not self._close(timeout=_EXIT_FLUSH_TIMEOUT):
            logger.warning(
                "FineStore cache write did not finish within %.1f seconds; abandoning it at process exit",
                _EXIT_FLUSH_TIMEOUT,
            )

    def _close(self, *, timeout: float | None) -> bool:
        """Close the cache, returning false when a queued write exceeds ``timeout``."""
        with self._lock:
            if self._closed:
                return True
            self._closed = True
            thread = self._background_thread
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                return False
        with self._root_lock:
            if self._store is not None:
                try:
                    self._store.close()
                except Exception as exc:
                    logger.warning("FineStore cache writer failed while closing: %s", exc)
                self._store = None
        return True

    def _storage_root(self) -> str:
        with self._root_lock:
            return self._storage_root_unlocked()

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
            if self._closed:
                return
            self._remote_pending[key] = value
            if self._background_thread is None:
                self._start_background_thread_locked()

    def _ensure_exit_cleanup(self) -> bool:
        with self._lock:
            if self._closed:
                return False
            self._register_exit_cleanup_locked()
            return True

    def _register_exit_cleanup_locked(self) -> None:
        if not self._exit_registered:
            atexit.register(self._close_at_exit)
            self._exit_registered = True

    def _start_background_thread_locked(self) -> None:
        self._register_exit_cleanup_locked()
        # Cache commits may be abandoned. The exit handler gives this daemon a bounded
        # drain window without making interpreter shutdown wait indefinitely on storage.
        thread = threading.Thread(target=self._drain_remote_writes, name="finestore-cache-write", daemon=True)
        self._background_thread = thread
        thread.start()

    def _drain_remote_writes(self) -> None:
        """Publish queued values in batches bounded by data bytes and object count."""
        try:
            while pending := self._take_remote_batch():
                try:
                    with self._writer().unbounded_transaction() as transaction:
                        for key, value in pending.items():
                            transaction.write_object(key, value)
                except Exception as exc:
                    logger.warning("FineStore cache write failed: %s", exc)
        finally:
            with self._lock:
                if self._background_thread is threading.current_thread():
                    self._background_thread = None
                    if self._remote_pending and not self._closed:
                        self._start_background_thread_locked()

    def _take_remote_batch(self) -> dict[str, bytes]:
        with self._lock:
            selected = []
            selected_bytes = 0
            for key, value in self._remote_pending.items():
                object_bytes = len(value)
                if selected and (
                    len(selected) >= _MAX_BATCH_OBJECTS or selected_bytes + object_bytes > _MAX_BATCH_DATA_BYTES
                ):
                    break
                selected.append(key)
                selected_bytes += object_bytes
            if not selected:
                self._background_thread = None
                return {}
            return {key: self._remote_pending.pop(key) for key in selected}
