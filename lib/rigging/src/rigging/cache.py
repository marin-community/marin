# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist content-addressed compiled artifacts across processes.

Compiled artifacts that a caller can address by a single key and serialize to
bytes — autotuned kernel block sizes, CuTeDSL kernel object code — share one
storage shape: a key that folds the build environment into a content address, an
atomic write so a reader never sees a half-written value, and many processes that
build and store the same keys at once. :class:`PersistentKvCache` owns that shape,
one object per key, written through :func:`rigging.filesystem.atomic_rename`.

Every cache keeps an in-process memory tier: a repeated key is answered without
re-reading the directory, and a cache built with no directory is memory-only,
which a caller uses to hold a value for the life of a process without persisting
it. The directory tier persists across processes and resolves on first use, not
at construction, so a caller may build one at import time before the active
cluster config is available. :func:`marin_kv_cache` maps a logical prefix onto
the region-local marin store; :meth:`PersistentKvCache.at` pins a fixed
directory; :meth:`PersistentKvCache.in_memory` has no directory at all.

A whole compiler cache directory that a non-Python consumer manages — vLLM's or
XLA's on-disk caches — is a different shape (a mirrored tree, not a keyed value)
and belongs behind a separate directory-sync primitive rather than this one.
"""

import logging
import threading
from collections.abc import Callable

from rigging.filesystem import StoragePath, atomic_rename, marin_prefix, prefix_join

logger = logging.getLogger(__name__)


class PersistentKvCache:
    """One object per key, over an in-process memory tier and an optional directory.

    A load answers from memory when the key was seen in this process, otherwise it
    reads the directory and remembers what it found; a store writes both tiers. A
    cache with no directory holds its values only for the life of the process.

    A store writes to a staged sibling and renames it into place, so a concurrent
    reader sees either the previous value or the new one and never a partial write.
    When several processes store the same key at once the last writer wins, which
    is correct for a content-addressed key because the bytes are identical.

    An unreachable directory — a mount that has not arrived, a location the task
    cannot write — degrades to the memory tier with a warning rather than failing
    the caller, so a best-effort cache never aborts the work it was meant to speed
    up.

    The key must be a path-safe token; a caller that keys on free-form text hashes
    it first. ``suffix`` is appended to each object's name, e.g. ``".o"``.
    """

    def __init__(self, resolve_directory: Callable[[], str] | None, *, suffix: str = "") -> None:
        self._resolve_directory = resolve_directory
        self._suffix = suffix
        self._root: StoragePath | None = None
        self._lock = threading.Lock()
        self._memory: dict[str, bytes] = {}

    @classmethod
    def at(cls, directory: str, *, suffix: str = "") -> "PersistentKvCache":
        """A cache backed by a fixed ``directory``."""
        return cls(lambda: directory, suffix=suffix)

    @classmethod
    def in_memory(cls) -> "PersistentKvCache":
        """A cache with no directory: values live only for the life of the process."""
        return cls(None)

    def location(self) -> str | None:
        """The resolved store directory without creating it, or ``None`` when memory-only."""
        return self._resolve_directory() if self._resolve_directory is not None else None

    def load(self, key: str) -> bytes | None:
        """Return the stored value for ``key``, or ``None`` when it is absent or unreadable."""
        with self._lock:
            if key in self._memory:
                return self._memory[key]
        value = self._read_directory(key)
        if value is not None:
            with self._lock:
                self._memory.setdefault(key, value)
        return value

    def store(self, key: str, value: bytes) -> None:
        """Persist ``value`` under ``key``; a reader never sees a partial write."""
        with self._lock:
            self._memory[key] = value
        self._write_directory(key, value)

    def _read_directory(self, key: str) -> bytes | None:
        if self._resolve_directory is None:
            return None
        try:
            path = self._object(key)
            return path.read_bytes() if path.exists() else None
        except OSError as exc:
            logger.warning("persistent cache unreadable, treating as a miss: %s", exc)
            return None

    def _write_directory(self, key: str, value: bytes) -> None:
        if self._resolve_directory is None:
            return
        try:
            with atomic_rename(str(self._object(key))) as staged:
                StoragePath(staged).write_bytes(value)
        except OSError as exc:
            logger.warning("persistent cache unwritable, not storing %s: %s", key, exc)

    def _object(self, key: str) -> StoragePath:
        assert self._resolve_directory is not None
        if self._root is None:
            root = StoragePath(self._resolve_directory())
            root.mkdirs()
            self._root = root
        return self._root / f"{key}{self._suffix}"


def marin_kv_cache(prefix: str, *, suffix: str = "") -> PersistentKvCache:
    """A cache under the region-local marin prefix, namespaced by ``prefix``.

    The location resolves through :func:`rigging.filesystem.marin_prefix` on first
    use, so this is safe to construct before the active cluster config is loaded.
    """
    return PersistentKvCache(lambda: prefix_join(marin_prefix(), prefix), suffix=suffix)
