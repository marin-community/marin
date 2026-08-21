# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Content identities shared by compiled-artifact caches."""

import hashlib
import pathlib
from collections.abc import Sequence
from typing import Protocol

from rigging.config_discovery import find_project_root

_HASH_COMPONENT_LENGTH_BYTES = 8


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

    Relative paths and bytes are significant. Absolute roots and mtimes are excluded.
    Derived Python bytecode is excluded; missing paths, non-regular entries, and symlinks
    are rejected because they make the boundary ambiguous.
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
    """Return the Marin ``uv.lock`` content identity visible from ``start``."""
    workspace = find_project_root(start)
    if workspace is None:
        raise ValueError(f"Marin workspace is unavailable from {start}")
    lockfile = workspace / "uv.lock"
    if not lockfile.is_file():
        raise ValueError(f"Marin dependency lockfile is unavailable: {lockfile}")
    return file_content_hash(lockfile)


def _hash_component(digest: _Digest, name: str, value: bytes) -> None:
    name_bytes = name.encode()
    digest.update(len(name_bytes).to_bytes(_HASH_COMPONENT_LENGTH_BYTES, "big"))
    digest.update(name_bytes)
    digest.update(len(value).to_bytes(_HASH_COMPONENT_LENGTH_BYTES, "big"))
    digest.update(value)


def _reject_missing_or_symlink(path: pathlib.Path) -> None:
    if path.is_symlink():
        raise ValueError(f"compile-cache inputs cannot be symlinks: {path}")
    if not path.exists():
        raise ValueError(f"compile-cache input does not exist: {path}")


def _is_derived_bytecode(path: pathlib.PurePath) -> bool:
    return "__pycache__" in path.parts or path.suffix == ".pyc"
