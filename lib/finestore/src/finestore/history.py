# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Commit history and row versions, read from an archive's immutable manifests.

Every manifest names its parent, so the chain from ``HEAD`` back to the first commit is the
archive's history. FineStore never deletes data files, so the rows a commit wrote stay
readable through that commit's manifest after later commits supersede them and compaction
rewrites their files. Walking the history reads one manifest per commit.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from finestore.commit import ArchiveSnapshot, read_snapshot, read_snapshot_at
from finestore.layout import RESERVED_COLUMNS, CommitInfo, FineStoreLayout
from finestore.reader import ReadView


@dataclass(frozen=True)
class CommitRecord:
    """One published commit."""

    commit_id: str
    sequence: int
    parent_commit_id: str | None
    info: CommitInfo | None
    """Who published the commit and why, or ``None`` when the commit recorded neither."""


@dataclass(frozen=True)
class RowVersion:
    """One committed version of a row and the commit that wrote it."""

    commit: CommitRecord
    row: dict
    """The row's user columns, without FineStore's system columns."""


def commit_log(root: str) -> Iterator[CommitRecord]:
    """Yield the commits of the archive at ``root`` from the newest back to the first."""
    for snapshot in _snapshots(root):
        yield _record(ReadView(root, snapshot), snapshot)


def row_history(root: str, table: str, key: tuple) -> Iterator[RowVersion]:
    """Yield every committed version of the row with primary key ``key``, newest first.

    Pass a version's ``commit.commit_id`` to ``ReadView.at`` to read the whole archive as
    that commit left it. A commit is checked only when it published a shard whose key range
    may hold ``key``, so a commit that wrote other rows costs its manifest read.
    """
    key = tuple(key)
    if any(value is None for value in key):
        raise ValueError("row_history does not support null key values")
    for snapshot in _snapshots(root):
        state = snapshot.manifest.tables.get(table)
        if state is None:
            continue
        sequence = snapshot.manifest.sequence
        own_shards = [
            shard
            for shard in state.shards
            if shard.row_commit_upper_bound >= sequence
            and (shard.key_range is None or shard.key_range.may_contain_prefix(key))
        ]
        if not own_shards:
            continue
        view = ReadView(root, snapshot)
        primary_key = view.primary_key(table)
        if len(key) != len(primary_key):
            raise ValueError(f"row_history needs a value for each primary-key column {primary_key}")
        if not view.keys_committed_after(table, [key], sequence - 1):
            continue
        row = view.rows_by_key(table, [key])[key]
        yield RowVersion(
            commit=_record(view, snapshot),
            row={name: value for name, value in row.items() if name not in RESERVED_COLUMNS},
        )


def _snapshots(root: str) -> Iterator[ArchiveSnapshot]:
    layout = FineStoreLayout(root)
    snapshot = read_snapshot(layout)
    while snapshot.token is not None:
        yield snapshot
        parent = snapshot.manifest.parent_commit_id
        if parent is None:
            return
        snapshot = read_snapshot_at(layout, parent)


def _record(view: ReadView, snapshot: ArchiveSnapshot) -> CommitRecord:
    manifest = snapshot.manifest
    return CommitRecord(
        commit_id=manifest.commit_id,
        sequence=manifest.sequence,
        parent_commit_id=manifest.parent_commit_id,
        info=view.commit_info,
    )
