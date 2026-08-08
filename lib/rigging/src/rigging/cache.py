# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist content-addressed compiled artifacts across processes.

Compiled artifacts that a caller can address by a single key and serialize to
bytes — autotuned kernel block sizes, CuTeDSL kernel object code — share one
storage shape: a key that folds the build environment into a content address, an
atomic write so a reader never sees a half-written value, and many processes that
build and store the same keys at once. :class:`PersistentKvCache` owns that shape,
one object per key, written through :func:`rigging.filesystem.atomic_rename`.

A cache spans up to three tiers, fastest first: an in-process memory tier that
answers a repeated key without touching a directory; an optional node-local
directory that persists across the processes on one node and is written inline;
and an optional object-store directory that persists across nodes and is written
in the background so an upload never blocks the thread that just built the value.
A load reads the tiers in order and remembers the first hit in memory; a store
writes every tier. :func:`marin_kv_cache` assembles the standard stack for a
logical prefix; :meth:`PersistentKvCache.at` pins a single inline directory and
:meth:`PersistentKvCache.in_memory` keeps a value for the life of the process.

Each directory resolves on first use, not at construction, so a caller may build
a cache at import time before the active cluster config is available. An
unreachable directory — a mount that has not arrived, a location the task cannot
write — degrades to the next tier with a warning rather than failing the caller.

A whole compiler cache directory that a non-Python consumer manages — XLA's
on-disk autotune cache, which its C++ opens directly — is a different shape (a
mirrored tree, not a keyed value). :class:`SyncedDirectory` covers that: it
mirrors a local directory to and from an object store so the consumer keeps
opening a local path while the contents persist across nodes.
"""

import atexit
import logging
import os
import pathlib
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_for_futures

from rigging.filesystem import StoragePath, atomic_rename, marin_local_cache, marin_temp_bucket, prefix_join

logger = logging.getLogger(__name__)

# Compiled artifacts that outlive a single run but not a code change; the object
# store reclaims a key that stops being written after this many days.
_CACHE_TTL_DAYS = 30
_BACKGROUND_WRITERS = 4
# How often a SyncedDirectory mirrors newly written files up while a run is live.
_SYNC_FLUSH_INTERVAL = 120.0

_background_lock = threading.Lock()
_background_executor: ThreadPoolExecutor | None = None
_pending_writes: set[Future] = set()


def _submit_background_write(write: Callable[[], None]) -> Future:
    """Run ``write`` on the shared background pool and track it until it finishes."""
    global _background_executor
    with _background_lock:
        if _background_executor is None:
            _background_executor = ThreadPoolExecutor(_BACKGROUND_WRITERS, thread_name_prefix="kv-cache-write")
            atexit.register(_shutdown_background_writes)
        future = _background_executor.submit(write)
        _pending_writes.add(future)
    future.add_done_callback(_forget_pending_write)
    return future


def _shutdown_background_writes() -> None:
    """Drain pending writes and close the pool cleanly at process exit."""
    flush_background_writes()
    with _background_lock:
        executor = _background_executor
    if executor is not None:
        executor.shutdown(wait=True)


def _forget_pending_write(future: Future) -> None:
    with _background_lock:
        _pending_writes.discard(future)
    error = future.exception()
    if error is not None:
        logger.warning("background cache write failed: %s", error)


def flush_background_writes(timeout: float | None = None) -> None:
    """Block until every in-flight background object-store write has finished.

    Registered on process exit so a run that just built a batch of kernels
    persists them before the interpreter tears down its threads.
    """
    with _background_lock:
        pending = list(_pending_writes)
    if pending:
        wait_for_futures(pending, timeout=timeout)


class _Directory:
    """One tier of a cache: objects named ``{key}{suffix}`` under a resolved directory."""

    def __init__(self, resolve_directory: Callable[[], str], suffix: str) -> None:
        self._resolve_directory = resolve_directory
        self._suffix = suffix
        self._lock = threading.Lock()
        self._root: StoragePath | None = None

    def location(self) -> str:
        return self._resolve_directory()

    def read(self, key: str) -> bytes | None:
        try:
            obj = self._object(key)
            return obj.read_bytes() if obj.exists() else None
        except OSError as exc:
            logger.warning("cache directory unreadable, treating as a miss: %s", exc)
            return None

    def write(self, key: str, value: bytes) -> None:
        try:
            with atomic_rename(str(self._object(key))) as staged:
                StoragePath(staged).write_bytes(value)
        except OSError as exc:
            logger.warning("cache directory unwritable, not storing %s: %s", key, exc)

    def _object(self, key: str) -> StoragePath:
        with self._lock:
            if self._root is None:
                root = StoragePath(self._resolve_directory())
                root.mkdirs()
                self._root = root
        return self._root / f"{key}{self._suffix}"


class PersistentKvCache:
    """A content-addressed byte cache over memory, a node-local dir, and an object store.

    A store writes to a staged sibling and renames it into place, so a concurrent
    reader sees either the previous value or the new one and never a partial write.
    When several processes store the same key at once the last writer wins, which
    is correct for a content-addressed key because the bytes are identical.

    The key must be a path-safe token; a caller that keys on free-form text hashes
    it first. ``suffix`` is appended to each object's name, e.g. ``".o"``.
    """

    def __init__(
        self,
        *,
        local: Callable[[], str] | None = None,
        remote: Callable[[], str] | None = None,
        suffix: str = "",
    ) -> None:
        self._local = _Directory(local, suffix) if local is not None else None
        self._remote = _Directory(remote, suffix) if remote is not None else None
        self._lock = threading.Lock()
        self._memory: dict[str, bytes] = {}

    @classmethod
    def at(cls, directory: str, *, suffix: str = "") -> "PersistentKvCache":
        """A cache backed by memory and one inline directory."""
        return cls(local=lambda: directory, suffix=suffix)

    @classmethod
    def in_memory(cls) -> "PersistentKvCache":
        """A cache with no directory: values live only for the life of the process."""
        return cls()

    def location(self) -> str | None:
        """The object-store directory if present, else the node-local one, else ``None``."""
        tier = self._remote or self._local
        return tier.location() if tier is not None else None

    def load(self, key: str) -> bytes | None:
        """Return the stored value for ``key``, or ``None`` when no tier holds it."""
        with self._lock:
            if key in self._memory:
                return self._memory[key]
        for tier in (self._local, self._remote):
            if tier is None:
                continue
            value = tier.read(key)
            if value is not None:
                with self._lock:
                    self._memory.setdefault(key, value)
                return value
        return None

    def store(self, key: str, value: bytes) -> None:
        """Persist ``value`` under ``key`` across every tier; a reader never sees a partial write."""
        with self._lock:
            self._memory[key] = value
        if self._local is not None:
            self._local.write(key, value)
        if self._remote is not None:
            remote = self._remote
            _submit_background_write(lambda: remote.write(key, value))


def marin_kv_cache(prefix: str, *, suffix: str = "") -> PersistentKvCache:
    """The standard cache stack for ``prefix``: memory, node-local disk, and object store.

    The node-local tier lands under :func:`rigging.filesystem.marin_local_cache`
    when a task's runtime provides one, and is skipped otherwise. The object-store
    tier lands under a region-local temp prefix with a lifecycle TTL, so a key that
    stops being written is reclaimed; it resolves through the active cluster config
    on first use, so this is safe to construct before that config loads.
    """
    local_root = marin_local_cache()
    local = (lambda: prefix_join(local_root, prefix)) if local_root is not None else None
    return PersistentKvCache(
        local=local,
        remote=lambda: marin_temp_bucket(_CACHE_TTL_DAYS, prefix),
        suffix=suffix,
    )


class SyncedDirectory:
    """Mirror a whole local directory to an object-store directory, both ways.

    A consumer that can only open a local path — XLA's per-fusion autotune cache,
    which its C++ opens through ``tsl::Env`` and cannot point at an object store —
    is the case :class:`PersistentKvCache` does not cover: the contents are a
    mirrored tree, not keyed values a caller serializes. On construction this
    stages the object-store directory down into the local one so the consumer
    starts warm, then mirrors newly written files back up on a daemon thread and
    once more at process exit. A file already staged down is not re-uploaded.

    Every transfer is best-effort: an unreachable object store degrades to a
    node-local directory with a warning rather than failing the caller. The remote
    directory resolves on first use, so this is safe to build before the active
    cluster config loads.
    """

    def __init__(self, remote: Callable[[], str], local: str, *, flush_interval: float = _SYNC_FLUSH_INTERVAL) -> None:
        self._remote = remote
        self._local = local
        self._flush_interval = flush_interval
        self._flush_lock = threading.Lock()
        self._uploaded: set[tuple[str, int]] = set()
        os.makedirs(local, exist_ok=True)
        self._stage_down()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="synced-dir-mirror", daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def flush(self) -> None:
        """Upload every local file not already mirrored up."""
        base = pathlib.Path(self._local)
        with self._flush_lock:
            remote_root = StoragePath(self._remote())
            for path in sorted(base.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(base).as_posix()
                fingerprint = (relative, path.stat().st_size)
                if fingerprint in self._uploaded:
                    continue
                target = remote_root / relative
                try:
                    target.parent.mkdirs()
                    target.upload_from(str(path))
                except OSError as exc:
                    logger.warning("synced cache upload failed for %s: %s", relative, exc)
                    return
                self._uploaded.add(fingerprint)

    def close(self) -> None:
        """Stop the mirror thread and flush a final time."""
        if self._stop.is_set():
            return
        atexit.unregister(self.close)
        self._stop.set()
        self._thread.join()
        self.flush()

    def _run(self) -> None:
        while not self._stop.wait(self._flush_interval):
            self.flush()

    def _stage_down(self) -> None:
        remote_root = StoragePath(self._remote())
        try:
            if not remote_root.exists():
                return
            for parent, _dirs, files in remote_root.walk():
                for name in files:
                    remote_file = parent / name
                    relative = remote_file.relative_to(remote_root)
                    local_file = pathlib.Path(self._local) / relative
                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    remote_file.download_to(str(local_file))
                    self._uploaded.add((pathlib.PurePosixPath(relative).as_posix(), local_file.stat().st_size))
        except OSError as exc:
            logger.warning("synced cache stage-down failed, starting cold: %s", exc)


def sync_kv_cache(remote: Callable[[], str], local: str) -> SyncedDirectory:
    """Stage the object-store directory ``remote`` into ``local`` and mirror it back.

    Returns the :class:`SyncedDirectory` handle so a caller may flush or close it
    explicitly; it also mirrors periodically and once on process exit.
    """
    return SyncedDirectory(remote, local)
