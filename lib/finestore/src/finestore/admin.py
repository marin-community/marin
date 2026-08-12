# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit metadata operations for archive migrations."""

from __future__ import annotations

from finestore.commit import CommitCoordinator, CommitDelta, write_schema
from finestore.layout import CommitToken, FineStoreLayout, TableMetadata


def set_table_metadata(root: str, table: str, metadata: TableMetadata) -> CommitToken:
    """Replace one table's logical metadata without changing its visible shards."""
    layout = FineStoreLayout(root.rstrip("/"))
    coordinator = CommitCoordinator(layout)
    metadata_path = write_schema(layout, metadata)
    return coordinator.commit(CommitDelta(metadata_updates={table: metadata_path}))


def drop_table(root: str, table: str) -> CommitToken:
    """Remove a table from the next manifest while retaining its immutable objects."""
    layout = FineStoreLayout(root.rstrip("/"))
    return CommitCoordinator(layout).commit(CommitDelta(removals=frozenset({table})))
