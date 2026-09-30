# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Portable file paths for pinned provider source packages."""

import unicodedata
from collections.abc import Iterable
from pathlib import PurePosixPath


def validate_relative_file_path(path: str) -> PurePosixPath:
    """Require a portable normalized relative path with no traversal."""
    if not path or "\\" in path or "\x00" in path or unicodedata.normalize("NFC", path) != path:
        raise ValueError(f"Invalid relative path: {path!r}")
    if path.startswith("/") or any(part in ("", ".", "..") for part in path.split("/")):
        raise ValueError(f"Path must be normalized and relative: {path!r}")
    if ":" in path.split("/")[0]:
        raise ValueError(f"Path must be normalized and relative: {path!r}")
    return PurePosixPath(path)


def validate_relative_file_paths(paths: Iterable[str]) -> None:
    """Reject normalized relative paths that collide on case-insensitive hosts."""
    files: set[str] = set()
    for path in paths:
        parts = validate_relative_file_path(path).parts
        folded = tuple(part.casefold() for part in parts)
        key = "/".join(folded)
        if key in files or any("/".join(folded[:index]) in files for index in range(1, len(folded))):
            raise ValueError(f"Path collision: {path}")
        if any(existing.startswith(f"{key}/") for existing in files):
            raise ValueError(f"Path collision: {path}")
        files.add(key)
