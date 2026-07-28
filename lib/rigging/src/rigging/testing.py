# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test-support doubles for services built on rigging.

Shipped alongside the library (not under ``tests/``) so both rigging's own suite
and downstream consumers such as iris import one copy rather than each carrying
its own.
"""

from collections.abc import Callable
from typing import Any

from fsspec.implementations.memory import MemoryFileSystem
from fsspec.spec import AbstractFileSystem

from rigging.filesystem.storage_path import StoragePath
from rigging.server_auth import VerifiedIdentity

type FileSystemResolver = Callable[..., tuple[AbstractFileSystem, str]]


def memory_filesystem_and_resolver(
    protocol: str,
    fallback: FileSystemResolver,
) -> tuple[MemoryFileSystem, FileSystemResolver]:
    """Return an isolated memory filesystem and resolver for one URL scheme."""

    class RemoteMemoryFileSystem(MemoryFileSystem):
        pass

    RemoteMemoryFileSystem.protocol = protocol
    RemoteMemoryFileSystem.store = {}
    RemoteMemoryFileSystem.pseudo_dirs = [""]
    remote_fs = RemoteMemoryFileSystem()

    def resolve(url: str, **kwargs: Any) -> tuple[AbstractFileSystem, str]:
        path = StoragePath(url)
        if path.scheme != protocol:
            return fallback(url, **kwargs)
        return remote_fs, "/".join(part for part in (path.netloc, path.key) if part)

    return remote_fs, resolve


class MockVerifier:
    """Deterministic :class:`~rigging.server_auth.TokenVerifier` double: maps fixed tokens to identities.

    A stand-in for a real service verifier (which checks JWTs it signed) in tests
    that only need a bearer token to resolve to a known user. Not a production auth
    mechanism.

    Args:
        tokens: Mapping of token string to username. Every user gets role ``"user"``.
    """

    def __init__(self, tokens: dict[str, str]):
        self._tokens = tokens

    def verify(self, token: str) -> VerifiedIdentity:
        user = self._tokens.get(token)
        if user is None:
            raise ValueError("Invalid token")
        return VerifiedIdentity(user_id=user, role="user")
