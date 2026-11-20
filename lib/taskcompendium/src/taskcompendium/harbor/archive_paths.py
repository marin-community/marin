# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve archived evidence paths without changing their recorded identities."""

from collections.abc import Mapping
from pathlib import Path

from rigging.filesystem.storage_path import rebase_file_path


def resolve_archive_path(path: Path, roots: Mapping[str, str]) -> Path:
    """Locate an archived path beneath its restored root, preserving its relative key."""
    for original in sorted(roots, key=lambda root: len(Path(root).parts), reverse=True):
        if path.is_relative_to(original):
            return Path(rebase_file_path(original, str(path), roots[original]))
    return path
