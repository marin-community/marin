# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded concurrent glob expansion across routed filesystems."""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import fsspec
from braceexpand import braceexpand

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath

DEFAULT_GLOB_WORKERS = 128


@dataclass(frozen=True)
class GlobEntry:
    """One matched file and the size returned by its listing request."""

    path: str
    size: int


def _glob_pattern(pattern: str) -> list[GlobEntry]:
    fs, fs_pattern = filesystem_for(pattern)
    protocol = fsspec.core.split_protocol(pattern)[0]
    entries = []
    for path, info in fs.glob(fs_pattern, detail=True).items():
        full_path = f"{protocol}://{path}" if protocol and "://" not in path else path
        entries.append(GlobEntry(path=full_path, size=int(info.get("size") or 0)))
    return entries


def glob_with_metadata(
    patterns: Sequence[str],
    *,
    workers: int = DEFAULT_GLOB_WORKERS,
) -> list[GlobEntry]:
    """Expand file patterns concurrently and return unique path/size entries."""
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")

    expanded_patterns = [expanded for pattern in patterns for expanded in braceexpand(StoragePath.normalize(pattern))]
    if not expanded_patterns:
        return []

    entries_by_path: dict[str, GlobEntry] = {}
    with ThreadPoolExecutor(
        max_workers=min(workers, len(expanded_patterns)),
        thread_name_prefix="filesystem-glob",
    ) as pool:
        for entries in pool.map(_glob_pattern, expanded_patterns):
            for entry in entries:
                entries_by_path[entry.path] = entry
    return sorted(entries_by_path.values(), key=lambda entry: entry.path)
