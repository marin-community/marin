# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist content-addressed compiled artifacts across processes.

Compiled artifacts that a caller can address by a single key and serialize to
bytes — autotuned kernel block sizes, CuTeDSL kernel object code — share one
storage shape: a key that folds the build environment into a content address, an
atomic write so a reader never sees a half-written value, and many processes that
build and store the same keys at once. :class:`PersistentKvCache` owns that shape,
one object per key over an in-process memory tier and a directory. It answers a
repeated key from memory; it writes an object-store directory in the background so
an upload never blocks the thread that built the value. :meth:`for_prefix`
assembles the standard cache on region-local temp storage.

A whole compiler cache directory that a non-Python consumer manages — XLA's
on-disk autotune cache, which its C++ opens directly — is a different shape (a
mirrored tree, not a keyed value). :class:`SyncedDirectory` covers that: it stages
an object-store directory down into a local one so the consumer starts warm, then
mirrors newly written files back up. :func:`sync_kv_cache` namespaces it per build.
"""

import atexit
import hashlib
import logging
import os
import pathlib
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_for_futures
from typing import Protocol

from rigging.config_discovery import find_project_root
from rigging.filesystem import StoragePath, atomic_rename, marin_temp_bucket, prefix_join
from rigging.provenance import launch_provenance

logger = logging.getLogger(__name__)

# Compiled artifacts that outlive a single run but not a code change; the object
# store reclaims a key that stops being written after this many days.
_CACHE_TTL_DAYS = 30
# Bound the at-exit drain so a slow upload cannot hang a shutting-down process.
_EXIT_FLUSH_TIMEOUT = 10.0
# How often a SyncedDirectory mirrors newly written files up while a run is live.
_SYNC_FLUSH_INTERVAL = 120.0

_background_lock = threading.Lock()
_background_executor: ThreadPoolExecutor | None = None
_pending_writes: set[Future] = set()


class _Digest(Protocol):
    def update(self, value: bytes, /) -> None: ...


def file_content_hash(path: pathlib.Path) -> str:
    """Return a checkout-independent SHA-256 identity for one file's bytes."""
    _reject_missing_or_symlink(path)
    if not path.is_file():
        raise ValueError(f"content-hash input is not a regular file: {path}")
    digest = hashlib.sha256()
    _hash_component(digest, "schema", b"rigging-file-content-v1")
    _hash_component(digest, "content", path.read_bytes())
    return digest.hexdigest()


def directory_content_hash(directory: pathlib.Path) -> str:
    """Return a checkout-independent SHA-256 identity for a directory tree.

    Relative paths and bytes are significant. Absolute roots and mtimes are
    excluded. Derived Python bytecode is excluded; missing paths, non-regular
    entries, and symlinks are rejected because they make the boundary ambiguous.
    """
    _reject_missing_or_symlink(directory)
    root = directory.resolve()
    if not root.is_dir():
        raise ValueError(f"content-hash input is not a directory: {directory}")

    digest = hashlib.sha256()
    _hash_component(digest, "schema", b"rigging-directory-content-v1")
    for candidate in sorted(root.rglob("*")):
        relative = candidate.relative_to(root)
        _reject_missing_or_symlink(candidate)
        if _is_derived_bytecode(relative):
            continue
        if candidate.is_file():
            _hash_component(digest, relative.as_posix(), candidate.read_bytes())
        elif not candidate.is_dir():
            raise ValueError(f"content-hash input is not a regular file or directory: {candidate}")
    return digest.hexdigest()


def combined_content_hash(components: Sequence[str]) -> str:
    """Combine ordered, labeled identities into one unambiguous SHA-256 address."""
    digest = hashlib.sha256()
    _hash_component(digest, "schema", b"rigging-combined-content-v1")
    for index, component in enumerate(components):
        _hash_component(digest, str(index), component.encode())
    return digest.hexdigest()


def workspace_lock_hash(start: pathlib.Path) -> str:
    """Return the Marin ``uv.lock`` content identity visible from ``start``.

    A missing workspace or lockfile raises so callers can disable shared
    persistence when the external dependency generation is unknown.
    """
    workspace = find_project_root(start)
    if workspace is None:
        raise ValueError(f"Marin workspace is unavailable from {start}")
    lockfile = workspace / "uv.lock"
    if not lockfile.is_file():
        raise ValueError(f"Marin dependency lockfile is unavailable: {lockfile}")
    return file_content_hash(lockfile)


def _hash_component(digest: _Digest, name: str, value: bytes) -> None:
    name_bytes = name.encode()
    digest.update(len(name_bytes).to_bytes(8, "big"))
    digest.update(name_bytes)
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _reject_missing_or_symlink(path: pathlib.Path) -> None:
    if path.is_symlink():
        raise ValueError(f"compile-cache inputs cannot be symlinks: {path}")
    if not path.exists():
        raise ValueError(f"compile-cache input does not exist: {path}")


def _is_derived_bytecode(path: pathlib.PurePath) -> bool:
    return "__pycache__" in path.parts or path.suffix == ".pyc"


def _submit_background_write(write: Callable[[], None]) -> None:
    """Run ``write`` on the shared background writer and track it until it finishes."""
    global _background_executor
    with _background_lock:
        if _background_executor is None:
            _background_executor = ThreadPoolExecutor(1, thread_name_prefix="cache-write")
            atexit.register(_drain_pending_writes)
        future = _background_executor.submit(write)
        _pending_writes.add(future)
    future.add_done_callback(_forget_pending_write)


def _forget_pending_write(future: Future) -> None:
    with _background_lock:
        _pending_writes.discard(future)
    error = future.exception()
    if error is not None:
        logger.warning("background cache write failed: %s", error)


def _drain_pending_writes() -> None:
    """Best-effort at exit: wait briefly for in-flight writes, but do not hang shutdown."""
    flush_background_writes(_EXIT_FLUSH_TIMEOUT)


def flush_background_writes(timeout: float | None = None) -> None:
    """Block until in-flight background writes finish (or ``timeout`` elapses)."""
    with _background_lock:
        pending = list(_pending_writes)
    if pending:
        wait_for_futures(pending, timeout=timeout)


class PersistentKvCache:
    """A content-addressed byte cache over an in-process memory tier and a directory.

    A load answers from memory, else reads the directory and remembers the value. A
    store writes memory inline; it writes an object-store directory in the
    background, a local directory inline. Writes go through
    :func:`rigging.filesystem.atomic_rename`, so a concurrent reader never sees a
    partial value, and two processes writing the same key are safe because a
    content-addressed key means identical bytes.

    The key must be a path-safe token; a caller that keys on free-form text hashes
    it first. The directory resolves on first use, so a cache can be built before
    the active cluster config loads; an unreachable directory degrades to memory
    with a warning.
    """

    def __init__(self, resolve_directory: Callable[[], str] | None = None) -> None:
        self._resolve_directory = resolve_directory
        self._root: StoragePath | None = None
        self._root_lock = threading.Lock()
        self._lock = threading.Lock()
        self._memory: dict[str, bytes] = {}

    @classmethod
    def at(cls, directory: str) -> "PersistentKvCache":
        """A cache backed by memory and a fixed directory."""
        return cls(lambda: directory)

    @classmethod
    def in_memory(cls) -> "PersistentKvCache":
        """A cache with no directory: values live only for the life of the process."""
        return cls()

    @classmethod
    def for_prefix(cls, prefix: str) -> "PersistentKvCache":
        """The standard cache for ``prefix``, on region-local temp object storage.

        The object store reclaims a key that stops being written after the TTL. The
        location resolves through the active cluster config on first use, so this is
        safe to construct before that config loads.
        """
        return cls(lambda: marin_temp_bucket(_CACHE_TTL_DAYS, prefix))

    def location(self) -> str | None:
        """The resolved directory without creating it, or ``None`` when memory-only."""
        return self._resolve_directory() if self._resolve_directory is not None else None

    def load(self, key: str) -> bytes | None:
        """Return the stored value for ``key``, or ``None`` when it is absent or unreadable."""
        with self._lock:
            if key in self._memory:
                return self._memory[key]
        value = self._read(key)
        if value is not None:
            with self._lock:
                self._memory.setdefault(key, value)
        return value

    def store(self, key: str, value: bytes) -> None:
        """Persist ``value`` under ``key``; a reader never sees a partial write."""
        with self._lock:
            self._memory[key] = value
        if self._resolve_directory is None:
            return
        obj = self._object(key)
        if obj.is_remote:
            _submit_background_write(lambda: _write_object(obj, value))
        else:
            _write_object(obj, value)

    def _read(self, key: str) -> bytes | None:
        if self._resolve_directory is None:
            return None
        try:
            obj = self._object(key)
            return obj.read_bytes() if obj.exists() else None
        except OSError as exc:
            logger.warning("cache directory unreadable, treating as a miss: %s", exc)
            return None

    def _object(self, key: str) -> StoragePath:
        with self._root_lock:
            if self._root is None:
                self._root = StoragePath(self._resolve_directory())  # pyrefly: ignore  # guarded by callers
        return self._root / key


def _write_object(obj: StoragePath, value: bytes) -> None:
    try:
        obj.parent.mkdirs()
        with atomic_rename(str(obj)) as staged:
            StoragePath(staged).write_bytes(value)
    except OSError as exc:
        logger.warning("cache directory unwritable, not storing %s: %s", obj.name, exc)


class SyncedDirectory:
    """Mirror a whole local directory to and from an object-store directory.

    A consumer that can only open a local path — XLA's per-fusion autotune cache,
    which its C++ opens through ``tsl::Env`` and cannot point at an object store —
    is the case :class:`PersistentKvCache` does not cover: the contents are a
    mirrored tree, not keyed values a caller serializes. On construction this stages
    the object-store directory down into the local one so the consumer starts warm,
    then mirrors files written since on a daemon thread. Mirroring is best-effort
    and never blocks shutdown: on a hard exit the last unflushed files are simply
    rebuilt next run.

    Every transfer degrades to a warning rather than failing the caller. The remote
    directory resolves on first use, so this is safe to build before the active
    cluster config loads.
    """

    def __init__(self, remote: Callable[[], str], local: str, *, flush_interval: float = _SYNC_FLUSH_INTERVAL) -> None:
        self._remote = remote
        self._local = local
        self._flush_interval = flush_interval
        self._flush_lock = threading.Lock()
        self._known: set[str] = set()
        os.makedirs(local, exist_ok=True)
        self._fetch_remote()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="synced-dir-mirror", daemon=True)
        self._thread.start()

    def flush(self) -> None:
        """Upload the local files written since the last mirror."""
        base = pathlib.Path(self._local)
        with self._flush_lock:
            present = {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()}
            remote_root = StoragePath(self._remote())
            for relative in sorted(present - self._known):
                target = remote_root / relative
                try:
                    target.parent.mkdirs()
                    target.upload_from(str(base / relative))
                except OSError as exc:
                    logger.warning("synced cache upload failed for %s: %s", relative, exc)
                    return
                self._known.add(relative)

    def close(self) -> None:
        """Stop the mirror thread and flush a final time."""
        self._stop.set()
        self._thread.join()
        self.flush()

    def _run(self) -> None:
        while not self._stop.wait(self._flush_interval):
            self.flush()

    def _fetch_remote(self) -> None:
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
                    self._known.add(pathlib.PurePosixPath(relative).as_posix())
        except OSError as exc:
            logger.warning("synced cache fetch failed, starting cold: %s", exc)


def sync_kv_cache(prefix: str, local: str) -> SyncedDirectory | None:
    """Mirror ``local`` against a per-build object-store directory for ``prefix``.

    Namespaces the object-store directory by the launch tree hash, so a build's
    files are shared across its nodes and reclaimed once the build ages out of the
    temp prefix. Returns ``None`` — leaving the directory node-local — when there is
    no tree hash to namespace by.
    """
    tree_hash = launch_provenance().tree_hash
    if not tree_hash:
        return None
    return SyncedDirectory(lambda: prefix_join(marin_temp_bucket(_CACHE_TTL_DAYS, prefix), tree_hash), local)
