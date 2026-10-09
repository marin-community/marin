# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Relative POSIX paths and file collision checks."""

from collections.abc import Iterable
from pathlib import PurePosixPath


def validate_relative_file_path(path: str) -> PurePosixPath:
    """Require a normalized relative POSIX path with no traversal."""
    if not path or "\x00" in path:
        raise ValueError(f"Invalid relative path: {path!r}")
    if path.startswith("/") or any(part in ("", ".", "..") for part in path.split("/")):
        raise ValueError(f"Path must be normalized and relative: {path!r}")
    return PurePosixPath(path)


def validate_relative_file_paths(paths: Iterable[str]) -> None:
    """Reject duplicate files and file/directory collisions on POSIX hosts."""
    files: set[str] = set()
    directories: set[str] = set()
    for path in paths:
        parts = validate_relative_file_path(path).parts
        key = "/".join(parts)
        parents = {"/".join(parts[:index]) for index in range(1, len(parts))}
        if key in files or key in directories or parents & files:
            raise ValueError(f"Path collision: {path}")
        files.add(key)
        directories.update(parents)
