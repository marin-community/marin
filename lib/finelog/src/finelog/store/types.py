# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Leaf vocabulary for the store layer.

Shared types referenced by both :mod:`finelog.store.catalog` and
:mod:`finelog.store.log_namespace`. This module has no dependencies on
other ``finelog.store`` modules so it sits below them in the import
graph and breaks the catalog ↔ namespace cycle structurally.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import pyarrow as pa

from finelog.rpc import logging_pb2
from finelog.store.schema import AlignedBatch, Schema
from finelog.types import LogReadResult


class SegmentLocation(StrEnum):
    """Where a segment's bytes currently live.

    Every catalog row is part of the table; ``location`` says whether the
    bytes are reachable from local disk, the remote bucket, or both. The
    sync loop reconciles remote with catalog; eviction flips ``BOTH`` →
    ``REMOTE`` rather than dropping the row, so a durable archive cannot
    be confused with an orphan compaction input.
    """

    LOCAL = "LOCAL"
    REMOTE = "REMOTE"
    BOTH = "BOTH"


@dataclass(frozen=True)
class SegmentRow:
    """One persisted row in the segments catalog table.

    ``level`` is the segment's tier in the leveled compaction scheme (0 =
    freshly flushed; promoted to ``level + 1`` when the planner picks it
    up). ``min_key_value`` / ``max_key_value`` carry the parquet footer's
    column statistics for the namespace's declared ``Schema.key_column``.
    They are ``None`` for namespaces whose schema has no ``key_column``,
    or for empty segments where no statistics exist.
    """

    namespace: str
    path: str
    level: int
    min_seq: int
    max_seq: int
    row_count: int
    byte_size: int
    created_at_ms: int
    location: SegmentLocation = SegmentLocation.LOCAL
    min_key_value: str | None = None
    max_key_value: str | None = None


@dataclass(frozen=True)
class NamespaceStats:
    """Aggregate counters for one namespace's persisted segments.

    Live (in-RAM) buffer counts are layered on top of these by the namespace
    in :meth:`finelog.store.log_namespace.DiskLogNamespace.stats`.
    """

    row_count: int
    byte_size: int
    min_seq: int
    max_seq: int
    segment_count: int

    @classmethod
    def empty(cls) -> NamespaceStats:
        return cls(row_count=0, byte_size=0, min_seq=0, max_seq=0, segment_count=0)


@dataclass
class LocalSegment:
    path: str
    size_bytes: int
    level: int
    min_seq: int
    max_seq: int
    row_count: int
    # ``created_at_ms`` is stamped once at flush/merge time and preserved
    # across level bumps and catalog reconcile. Currently informational
    # only — the planner promotes by byte target or segment count, not
    # age — but kept as the catalog's canonical birth time for ops
    # tools / future age-based policies.
    created_at_ms: int = 0
    # Typed key-column bounds (Python int / str / float / bool / bytes
    # depending on schema). Stringified only at the catalog boundary in
    # ``_segment_to_row``; held typed in memory so ``aggregate_key_bounds``
    # can compare numeric keys with native ordering.
    min_key_value: object | None = None
    max_key_value: object | None = None
    # Where the bytes live. The deque only ever holds ``LOCAL`` or ``BOTH``
    # entries; eviction flips to ``REMOTE`` and removes the entry from the
    # deque (the row stays in the catalog as a durable-archive pointer).
    location: SegmentLocation = SegmentLocation.LOCAL


class LogNamespaceProtocol(Protocol):
    name: str
    schema: Schema

    def append_log_batch(self, items: list[tuple[str, list]]) -> int: ...

    def append_aligned_batch(self, aligned: AlignedBatch) -> int: ...

    def max_persisted_seq(self) -> int: ...

    def get_logs(
        self,
        key: str,
        *,
        match_scope: int = logging_pb2.MATCH_SCOPE_EXACT,
        since_ms: int = 0,
        cursor: int = 0,
        substring_filter: str = "",
        max_lines: int = 0,
        tail: bool = False,
        min_level: str = "",
    ) -> LogReadResult: ...

    def query_snapshot(self) -> tuple[list[LocalSegment], list[pa.Table]]: ...

    def all_segments_unlocked(self) -> list[LocalSegment]: ...

    def update_schema(self, new_schema: Schema) -> None: ...

    def evict_segment(self, path: str) -> int: ...

    def remove_local_storage(self) -> None: ...

    def close(self) -> None: ...

    def stop_and_join(self) -> None: ...

    def stats(self) -> NamespaceStats: ...

    def ram_bytes(self) -> int: ...

    def chunk_count(self) -> int: ...
