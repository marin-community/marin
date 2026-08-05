# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compaction: merge a table's small level-0 shards into one sorted, deduplicated shard.

Compaction is an optimization, not a correctness requirement — a reader deduplicates regardless. It
reads the table's current live rows (already deduplicated), writes them sorted by the primary key to
the next generation, and optionally deletes the shards it superseded. Because the merged rows keep
their original ``_seq`` and the new shard has a higher generation, a reader prefers it; a crash
between the write and the delete leaves both, and dedup still returns one row per key.
"""

from __future__ import annotations

import logging
import uuid

from rigging.filesystem import url_to_fs

from finestore import _io
from finestore.layout import GEN_COLUMN, SEQ_COLUMN, shard_path
from finestore.reader import CompositeReader

logger = logging.getLogger(__name__)

# The writer identity stamped on compacted shards, distinct from any live append writer.
_COMPACTOR = "compactor"


def compact(root: str, table: str, *, delete_source: bool = True) -> int:
    """Compact ``table`` under ``root`` into one sorted shard; return the row count written.

    Returns 0 (writing nothing) when the table is empty or has no shards. When ``delete_source`` is
    set, shards from generations below the new one are removed after the merged shard is published.
    """
    reader = CompositeReader(root)
    shards = reader.list_shards(table)
    if not shards:
        return 0
    pk = reader.primary_key(table)
    next_generation = max(shard.generation for shard in shards) + 1

    data = reader.scan(table, primary_key=pk, dedup=True)
    if data is None or data.num_rows == 0:
        return 0
    if GEN_COLUMN in data.column_names:
        data = data.drop_columns([GEN_COLUMN])
    sort_keys = [(name, "ascending") for name in pk if name in data.column_names]
    if sort_keys:
        data = data.sort_by(sort_keys)

    min_seq = min(data.column(SEQ_COLUMN).to_pylist()) if SEQ_COLUMN in data.column_names else 0
    out_path = shard_path(root, table, _COMPACTOR, next_generation, min_seq, uuid.uuid4().hex[:8])
    _io.write_table(out_path, data)
    logger.info("finestore compacted %s to generation %d (%d rows)", table, next_generation, data.num_rows)

    if delete_source:
        fs, _ = url_to_fs(root)
        for shard in shards:
            if shard.generation < next_generation:
                fs.rm(shard.path)
    return data.num_rows
