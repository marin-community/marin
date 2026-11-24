# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit metadata operations for archive migrations."""

from __future__ import annotations

from finestore.commit import CommitCoordinator, CommitDelta, write_schema
from finestore.layout import CommitToken, FineStoreLayout, TableMetadata, TableVersion


def set_table_metadata(root: str, table: str, metadata: TableMetadata) -> CommitToken:
    """Replace one table's logical metadata without changing its visible shards."""
    layout = FineStoreLayout(root)
    coordinator = CommitCoordinator(layout)
    metadata_path = write_schema(layout, metadata)
    return coordinator.commit(CommitDelta(metadata_updates={table: metadata_path}))


def drop_table(root: str, table: str, *, expected: TableVersion | None = None) -> CommitToken:
    """Drop the expected table version while retaining its immutable objects."""
    coordinator = CommitCoordinator(FineStoreLayout(root))
    snapshot = coordinator.snapshot()
    version = expected if expected is not None else snapshot.manifest.table_version(table)
    return coordinator.commit(
        CommitDelta(reset_tables=frozenset({table}), expected_tables={table: version}), base=snapshot
    )
