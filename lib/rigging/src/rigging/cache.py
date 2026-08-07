# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist content-addressed compiled artifacts across processes.

Compiled artifacts that a caller can address by a single key and serialize to
bytes — autotuned kernel block sizes, CuTeDSL kernel object code — share one
storage shape: a key that folds the build environment into a content address, an
atomic write so a reader never sees a half-written value, and many processes that
build and store the same keys at once. :class:`PersistentKvCache` owns that shape,
one object per key, written through :func:`rigging.filesystem.atomic_rename`.

A whole compiler cache directory that a non-Python consumer manages — vLLM's or
XLA's on-disk caches — is a different shape (a mirrored tree, not a keyed value)
and belongs behind a separate directory-sync primitive rather than this one.

The store is deliberately dumb about what a value means. A caller keys on
whatever identifies its artifact — a source digest, a device architecture, a set
of package versions — and serializes the value to bytes. Keeping the environment
in the key is the caller's job; the store only guarantees that a stored value is
read back whole.
"""

import dataclasses
import functools

from rigging.filesystem import StoragePath, atomic_rename


@dataclasses.dataclass(frozen=True)
class PersistentKvCache:
    """One object per key on local or remote storage, safe for concurrent writers.

    A store writes to a staged sibling and renames it into place, so a concurrent
    reader sees either the previous value or the new one and never a partial write.
    When several processes store the same key at once the last writer wins, which
    is correct for a content-addressed key because the bytes are identical.

    Args:
        directory: Store location, a local path or any fsspec URL.
        suffix: Appended to each object's name, e.g. ``".o"``. The key must be a
            path-safe token; a caller that keys on free-form text hashes it first.
    """

    directory: str
    suffix: str = ""

    def load(self, key: str) -> bytes | None:
        """Return the stored value for ``key``, or ``None`` when it is absent."""
        path = self._object(key)
        return path.read_bytes() if path.exists() else None

    def store(self, key: str, value: bytes) -> None:
        """Persist ``value`` under ``key``; a reader never sees a partial write."""
        path = str(self._object(key))
        with atomic_rename(path) as staged:
            StoragePath(staged).write_bytes(value)

    def _object(self, key: str) -> StoragePath:
        return _root(self.directory) / f"{key}{self.suffix}"


@functools.cache
def _root(directory: str) -> StoragePath:
    path = StoragePath(directory)
    path.mkdirs()
    return path
