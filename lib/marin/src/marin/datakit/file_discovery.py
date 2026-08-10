# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recursive discovery of zephyr-readable data files under a staged input tree.

Shared by ``normalize`` (which sizes its input to pick a shard count) and
``decon`` (which reads an eval corpus), so both agree on what counts as a data
file and neither pays for a second metadata pass over the tree.
"""

import os
from collections.abc import Iterator
from dataclasses import dataclass

from rigging.filesystem import url_to_fs
from zephyr.readers import SUPPORTED_EXTENSIONS


@dataclass(frozen=True)
class DataFile:
    """A discovered data file and its size in bytes, both taken from one directory walk."""

    path: str
    size: int


def walk_data_files(
    root: str,
    *,
    extensions: tuple[str, ...] = SUPPORTED_EXTENSIONS,
    exclude_dir_names: frozenset[str] = frozenset(),
) -> Iterator[DataFile]:
    """Yield every data file under *root* whose name ends in one of *extensions*.

    Dotfiles and everything under a hidden directory are skipped, so the sidecars that
    routinely sit beside staged data (``.provenance.json``, ``README``, ``_SUCCESS``,
    ``.metrics/``, ``.executor_info/``) never reach ``zephyr.readers.load_file``, which
    would reject their extension and fail the whole step.

    ``exclude_dir_names`` drops any file whose immediate parent directory name is in the
    set. The eval-corpus layout is ``<root>/<split>/<task>/<file>``, so this excludes
    named tasks at read time without regenerating the corpus.

    Sizes come from the walk's own listing rather than a per-file stat: ``fsspec`` fetches
    that metadata to enumerate the tree either way, and statting each result again costs
    one round trip per file on object storage.
    """
    fs, resolved = url_to_fs(root)
    protocol = root.split("://")[0] if "://" in root else ""

    for dirpath, _dirs, files in fs.walk(resolved, detail=True):
        relative = os.path.relpath(dirpath, resolved)
        if relative != "." and any(part.startswith(".") for part in relative.split(os.sep)):
            continue
        if os.path.basename(dirpath.rstrip("/")) in exclude_dir_names:
            continue
        for name, info in files.items():
            if name.startswith(".") or not name.endswith(extensions):
                continue
            path = os.path.join(dirpath, name)
            yield DataFile(path=f"{protocol}://{path}" if protocol else path, size=info["size"])
