# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small structured helpers for inspecting an already-created sandbox."""

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class RemotePathInfo:
    """A sandbox file-system entry returned by an inspection command."""

    path: str
    size: int
    is_directory: bool


def normalize_remote_paths(entries: Iterable[object]) -> list[RemotePathInfo]:
    """Normalize Daytona SDK file metadata without depending on its model types."""

    normalized = [
        RemotePathInfo(
            path=str(getattr(entry, "path", entry)),
            size=int(getattr(entry, "size", 0) or 0),
            is_directory=bool(getattr(entry, "is_dir", False)),
        )
        for entry in entries
    ]
    return sorted(normalized, key=lambda entry: entry.path)
