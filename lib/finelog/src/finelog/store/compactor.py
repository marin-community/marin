# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Leveled compaction planner for finelog parquet segments.

The :class:`Compactor` is stateless: the per-namespace bg thread owns
locks, catalog mutation, and the actual COPY execution; this module
decides *which* segments to merge and *into what file*. Splitting the
policy out lets us unit-test ``plan`` over fixtures without a running
store.

Levels are time-ordered: every fresh flush emits an L0 segment. The
planner promotes L_n → L_{n+1} when the longest contiguous run of L_n
segments hits the byte target for that tier (or, for L0 only, when the
oldest run has been sitting longer than ``max_l0_age_sec`` — the lever
that keeps low-volume namespaces from leaking small files).

The terminal level is ``len(level_targets)``: segments at that tier
never re-compact. They are also the only tier eligible for eviction
(see :meth:`finelog.store.catalog.Catalog.select_eviction_candidate`),
so the layout converges to a small fanout of large terminal-level files
plus a short tail of in-flight intermediates.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from finelog.store.catalog import SegmentRow
from finelog.store.schema import IMPLICIT_SEQ_COLUMN, Schema, duckdb_type_for
from finelog.store.sql_escape import quote_ident, quote_literal

_MiB = 1024 * 1024

# Row group size for compacted parquet output. Count-based; matches the
# size used by the flush path so reads see a uniform layout.
_ROW_GROUP_SIZE = 16_384


def compaction_sort_keys(schema: Schema) -> tuple[str, ...]:
    """Sort keys used by both flush-time sort and compaction's ORDER BY.

    ``key_column`` first (so range scans on it prune row groups efficiently),
    then the implicit ``seq``.
    """
    if schema.key_column:
        return (schema.key_column, IMPLICIT_SEQ_COLUMN)
    return (IMPLICIT_SEQ_COLUMN,)


@dataclass(frozen=True)
class CompactionConfig:
    """Tuning knobs for the leveled compaction policy.

    ``level_targets[n]`` is the summed byte size at which the longest
    contiguous run of L_n segments is promoted to L_{n+1}. The terminal
    level is ``len(level_targets)``; segments at that tier are never
    re-compacted (and become eviction candidates).
    """

    level_targets: tuple[int, ...] = (64 * _MiB, 256 * _MiB)
    check_interval_sec: float = 30.0
    max_l0_age_sec: float = 300.0
    max_segments_per_namespace: int = 1000
    max_bytes_per_namespace: int = 100 * 1024**3


@dataclass(frozen=True)
class CompactionJob:
    """One pending merge: ``len(inputs)`` segments → one ``output_level`` segment."""

    inputs: tuple[SegmentRow, ...]
    output_level: int
    output_min_seq: int
    output_max_seq: int


class Compactor:
    """Plans level promotions and emits the merge SQL.

    Construct one per namespace; instances are cheap and stateless. The
    schema is supplied at ``merge_sql`` call time so post-evolution writes
    use the current column list without any side-channel mutation.
    """

    def __init__(self, config: CompactionConfig) -> None:
        self.config = config

    @property
    def terminal_level(self) -> int:
        """Segments at this level are never re-compacted."""
        return len(self.config.level_targets)

    def plan(self, segments: Sequence[SegmentRow], now_ms: int) -> CompactionJob | None:
        """Return the next merge job, or ``None`` if nothing is due.

        Walks tiers from L0 upward and returns the first promotable run.
        Doing the lowest tier first keeps ``len(_local_segments)`` small
        in steady state, which dominates per-read fanout cost.
        """
        for n, target in enumerate(self.config.level_targets):
            at_level = sorted([s for s in segments if s.level == n], key=lambda s: s.min_seq)
            if not at_level:
                continue
            for run in _contiguous_runs(at_level):
                if _run_bytes(run) >= target:
                    return _build_job(run, output_level=n + 1)
                if n == 0 and self.config.max_l0_age_sec > 0:
                    oldest_age_sec = (now_ms - run[0].created_at_ms) / 1000.0
                    if oldest_age_sec >= self.config.max_l0_age_sec:
                        return _build_job(run, output_level=n + 1)
        return None

    def merge_sql(self, job: CompactionJob, *, schema: Schema, staging_path: Path) -> str:
        """Build the DuckDB ``COPY`` that merges ``job.inputs`` to ``staging_path``.

        Uses ``read_parquet([...], union_by_name=true)`` so columns missing
        from any individual input come back as NULL, then projects through
        ``schema`` with explicit ``NULL::TYPE AS name`` for columns added
        since some inputs were written.
        """
        input_paths = [Path(seg.path) for seg in job.inputs]
        present = _present_input_columns(input_paths)
        select_exprs = [
            quote_ident(col.name) if col.name in present else f"NULL::{duckdb_type_for(col)} AS {quote_ident(col.name)}"
            for col in schema.columns
        ]
        order_keys = compaction_sort_keys(schema)
        order_clause = "ORDER BY " + ", ".join(quote_ident(k) for k in order_keys)
        paths_sql = ", ".join(quote_literal(str(p)) for p in input_paths)
        select_clause = ", ".join(select_exprs)
        return (
            f"COPY (SELECT {select_clause} "
            f"FROM read_parquet([{paths_sql}], union_by_name=true) "
            f"{order_clause}) "
            f"TO {quote_literal(str(staging_path))} "
            f"(FORMAT 'parquet', ROW_GROUP_SIZE {_ROW_GROUP_SIZE}, "
            f"COMPRESSION 'zstd', COMPRESSION_LEVEL 1, WRITE_BLOOM_FILTER true)"
        )


def _contiguous_runs(segments: Sequence[SegmentRow]) -> list[list[SegmentRow]]:
    """Group ``segments`` (sorted by ``min_seq``) into adjacency runs.

    Adjacency means ``prev.max_seq + 1 == next.min_seq``. With FIFO-by-seq
    eviction, the live set is always a contiguous suffix of the global seq
    range, so this should produce a single run; the multi-run branch is
    defensive against an externally introduced gap.
    """
    if not segments:
        return []
    runs: list[list[SegmentRow]] = [[segments[0]]]
    for seg in segments[1:]:
        if runs[-1][-1].max_seq + 1 == seg.min_seq:
            runs[-1].append(seg)
        else:
            runs.append([seg])
    return runs


def _run_bytes(run: Sequence[SegmentRow]) -> int:
    return sum(s.byte_size for s in run)


def _build_job(run: Sequence[SegmentRow], *, output_level: int) -> CompactionJob:
    return CompactionJob(
        inputs=tuple(run),
        output_level=output_level,
        output_min_seq=min(s.min_seq for s in run),
        output_max_seq=max(s.max_seq for s in run),
    )


def aggregate_key_bounds(
    bounds: Iterable[tuple[object | None, object | None]],
) -> tuple[object | None, object | None]:
    """Fold per-input ``(min, max)`` key tuples into a single ``(min, max)``.

    Operates on the typed Python value (int, str, float, bool, bytes,
    datetime — whatever ``pyarrow`` decoded the key column as) so numeric
    keys keep native ordering. Stringification happens later, at the
    catalog write boundary (``DiskLogNamespace._segment_to_row``); callers
    must not stringify before passing through here or ``"10" < "2"``
    flips the bound for an int64 key column.

    Inputs whose values are ``None`` (empty segment / no stats) are skipped.
    Returns ``(None, None)`` if every input was skipped.
    """
    overall_min: object | None = None
    overall_max: object | None = None
    for lo, hi in bounds:
        if lo is not None and (overall_min is None or lo < overall_min):  # type: ignore[operator]
            overall_min = lo
        if hi is not None and (overall_max is None or hi > overall_max):  # type: ignore[operator]
            overall_max = hi
    return overall_min, overall_max


def _present_input_columns(input_paths: list[Path]) -> set[str]:
    present: set[str] = set()
    for p in input_paths:
        present.update(pq.read_schema(p).names)
    return present


def seg_filename(level: int, min_seq: int) -> str:
    """Filename for a segment at ``level`` whose smallest seq is ``min_seq``."""
    return f"seg_L{level}_{min_seq:019d}.parquet"


_SEG_FILENAME_RE = re.compile(r"^seg_L(?P<level>\d+)_(?P<seq>\d+)\.parquet$")


def parse_seg_filename(name: str) -> tuple[int, int] | None:
    """Recover ``(level, min_seq)`` from a segment filename, or ``None``."""
    match = _SEG_FILENAME_RE.match(name)
    if match is None:
        return None
    return int(match.group("level")), int(match.group("seq"))
