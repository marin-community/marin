# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the leveled compaction planner.

These exercise ``Compactor.plan`` and ``aggregate_key_bounds`` directly,
without spinning up a full ``DuckDBLogStore``. The store-level integration
tests (test_duckdb_store, test_query, test_eviction) cover the wired-up
behavior; this file isolates the policy decisions so a regression in the
planner shows up with a clear failure.
"""

from __future__ import annotations

from finelog.store.compactor import (
    CompactionConfig,
    Compactor,
    aggregate_key_bounds,
)
from finelog.store.types import SegmentRow


def _row(
    *,
    level: int,
    min_seq: int,
    max_seq: int,
    byte_size: int,
    created_at_ms: int = 0,
) -> SegmentRow:
    return SegmentRow(
        namespace="ns",
        path=f"/x/seg_L{level}_{min_seq:019d}.parquet",
        level=level,
        min_seq=min_seq,
        max_seq=max_seq,
        row_count=max_seq - min_seq + 1,
        byte_size=byte_size,
        created_at_ms=created_at_ms,
    )


def test_plan_returns_none_when_under_target():
    compactor = Compactor(CompactionConfig(level_targets=(1024,), max_segments_per_level=1024))
    rows = [_row(level=0, min_seq=1, max_seq=1, byte_size=128)]
    assert compactor.plan(rows) is None


def test_plan_promotes_when_byte_target_reached():
    compactor = Compactor(CompactionConfig(level_targets=(1024,), max_segments_per_level=1024))
    rows = [
        _row(level=0, min_seq=1, max_seq=1, byte_size=512),
        _row(level=0, min_seq=2, max_seq=2, byte_size=512),
    ]
    job = compactor.plan(rows)
    assert job is not None
    assert job.output_level == 1
    assert [r.min_seq for r in job.inputs] == [1, 2]


def test_plan_promotes_at_segment_count_below_byte_target():
    """Run promotes once it reaches ``max_segments_per_level`` even if bytes are tiny.

    The lever that keeps slow / bursty namespaces from leaking small
    files at any non-terminal tier (and bounds per-read parquet-open
    fanout).
    """
    compactor = Compactor(CompactionConfig(level_targets=(1 << 30,), max_segments_per_level=3))
    rows = [
        _row(level=0, min_seq=1, max_seq=1, byte_size=128),
        _row(level=0, min_seq=2, max_seq=2, byte_size=128),
        _row(level=0, min_seq=3, max_seq=3, byte_size=128),
    ]
    job = compactor.plan(rows)
    assert job is not None
    assert job.output_level == 1
    assert [r.min_seq for r in job.inputs] == [1, 2, 3]


def test_plan_does_not_count_promote_terminal_level():
    compactor = Compactor(CompactionConfig(level_targets=(1024,), max_segments_per_level=2))
    rows = [
        _row(level=1, min_seq=1, max_seq=1, byte_size=128),
        _row(level=1, min_seq=2, max_seq=2, byte_size=128),
        _row(level=1, min_seq=3, max_seq=3, byte_size=128),
    ]
    assert compactor.plan(rows) is None


def test_plan_count_promotes_non_terminal_l1_below_byte_target():
    """L1 promotes by count when L2 is non-terminal.

    Regression: prior to the count trigger, L1 only promoted on byte
    target, so namespaces whose L0 flushes were small accumulated
    hundreds of sub-target L1 files indefinitely.
    """
    compactor = Compactor(CompactionConfig(level_targets=(64, 1 << 30), max_segments_per_level=2))
    rows = [
        _row(level=1, min_seq=1, max_seq=1, byte_size=8),
        _row(level=1, min_seq=2, max_seq=2, byte_size=8),
    ]
    job = compactor.plan(rows)
    assert job is not None
    assert job.output_level == 2
    assert [r.min_seq for r in job.inputs] == [1, 2]


def test_aggregate_key_bounds_preserves_numeric_ordering():
    """Regression: stringified ``"10" < "2"`` would invert numeric bounds."""
    lo, hi = aggregate_key_bounds([(2, 10), (5, 7)])
    assert lo == 2
    assert hi == 10


def test_aggregate_key_bounds_handles_strings():
    lo, hi = aggregate_key_bounds([("alice", "bob"), ("carol", "dave")])
    assert lo == "alice"
    assert hi == "dave"


def test_aggregate_key_bounds_skips_none_inputs():
    lo, hi = aggregate_key_bounds([(None, None), (3, 9)])
    assert lo == 3
    assert hi == 9


def test_aggregate_key_bounds_all_none():
    assert aggregate_key_bounds([(None, None), (None, None)]) == (None, None)
