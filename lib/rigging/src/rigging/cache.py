# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist content-addressed compiled artifacts across processes.

Compiled artifacts that a caller can address by a single key and serialize to
bytes — autotuned kernel block sizes, CuTeDSL kernel object code — share one
storage shape: a key that folds the build environment into a content address, an
atomic write so a reader never sees a half-written value, and many processes that
build and store the same keys at once. :class:`PersistentKvCache` owns that shape,
one object per key, written through :func:`rigging.filesystem.atomic_rename`.

The store resolves its directory on first use, not at construction, so a caller
may build one at import time before the active cluster config is available.
:func:`marin_kv_cache` maps a logical prefix onto the region-local marin store;
:meth:`PersistentKvCache.at` pins a fixed directory.

A whole compiler cache directory that a non-Python consumer manages — vLLM's or
XLA's on-disk caches — is a different shape (a mirrored tree, not a keyed value)
and belongs behind a separate directory-sync primitive rather than this one.
"""

from collections.abc import Callable

from rigging.filesystem import StoragePath, atomic_rename, marin_prefix, prefix_join


class PersistentKvCache:
    """One object per key under a directory resolved on first use.

    A store writes to a staged sibling and renames it into place, so a concurrent
    reader sees either the previous value or the new one and never a partial write.
    When several processes store the same key at once the last writer wins, which
    is correct for a content-addressed key because the bytes are identical.

    The key must be a path-safe token; a caller that keys on free-form text hashes
    it first. ``suffix`` is appended to each object's name, e.g. ``".o"``.
    """

    def __init__(self, resolve_directory: Callable[[], str], *, suffix: str = "") -> None:
        self._resolve_directory = resolve_directory
        self._suffix = suffix
        self._root: StoragePath | None = None

    @classmethod
    def at(cls, directory: str, *, suffix: str = "") -> "PersistentKvCache":
        """A cache at a fixed ``directory``."""
        return cls(lambda: directory, suffix=suffix)

    def location(self) -> str:
        """The resolved store directory, without creating it."""
        return self._resolve_directory()

    def load(self, key: str) -> bytes | None:
        """Return the stored value for ``key``, or ``None`` when it is absent."""
        path = self._object(key)
        return path.read_bytes() if path.exists() else None

    def store(self, key: str, value: bytes) -> None:
        """Persist ``value`` under ``key``; a reader never sees a partial write."""
        with atomic_rename(str(self._object(key))) as staged:
            StoragePath(staged).write_bytes(value)

    def _object(self, key: str) -> StoragePath:
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
