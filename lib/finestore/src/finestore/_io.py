# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shard write primitive: an Arrow table to one immutable, atomically-published Parquet object.

Shared by the writer and the compactor. A shard is written to a temp sibling key and renamed into
place, so a reader listing the directory never observes a half-written object.
"""

from __future__ import annotations

import os

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath, atomic_rename

# zstd matches finelog's segment encoding: strong ratio on the repetitive token-id / logprob / JSON
# columns eval data is dominated by, for negligible CPU at write time.
_COMPRESSION = "zstd"


def ensure_parent(path: str) -> None:
    """Create the parent directory of ``path`` (a no-op on object stores that lack directories)."""
    parent = path.rsplit("/", 1)[0]
    if "://" in path:
        StoragePath(parent).mkdirs()
    elif parent:
        os.makedirs(parent, exist_ok=True)


def write_table(path: str, table: pa.Table) -> None:
    """Write ``table`` to the shard object ``path`` atomically, compressed with zstd.

    Writes go through ``atomic_rename`` and ``StoragePath.open``, which resolve the filesystem through
    rigging's factory at call time, so a test that routes an object-store scheme to an in-memory
    filesystem sees these writes too.
    """
    ensure_parent(path)
    with atomic_rename(path) as temp_path:
        with StoragePath(temp_path).open("wb") as handle:
            pq.write_table(table, handle, compression=_COMPRESSION)
