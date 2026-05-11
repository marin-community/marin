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

from finelog.store.catalog import SegmentRow
from finelog.store.compactor import (
    CompactionConfig,
    Compactor,
    aggregate_key_bounds,
)


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
    compactor = Compactor(CompactionConfig(level_targets=(1024,), max_l0_age_sec=0))
    rows = [_row(level=0, min_seq=1, max_seq=1, byte_size=128)]
    assert compactor.plan(rows, now_ms=0) is None


def test_plan_promotes_when_byte_target_reached():
    compactor = Compactor(CompactionConfig(level_targets=(1024,), max_l0_age_sec=0))
    rows = [
        _row(level=0, min_seq=1, max_seq=1, byte_size=512),
        _row(level=0, min_seq=2, max_seq=2, byte_size=512),
    ]
    job = compactor.plan(rows, now_ms=0)
    assert job is not None
    assert job.output_level == 1
    assert [r.min_seq for r in job.inputs] == [1, 2]


def test_plan_promotes_aged_l0_below_target():
    """L0 segments past ``max_l0_age_sec`` promote regardless of byte target.

    Regression for the bug where ``_segment_to_row`` overwrote
    ``created_at_ms`` with ``time.time()`` on every tick — under that bug
    this case would never fire because each plan tick would observe a
    "freshly created" row.
    """
    compactor = Compactor(CompactionConfig(level_targets=(1 << 30,), max_l0_age_sec=300.0))
    rows = [
        _row(level=0, min_seq=1, max_seq=1, byte_size=128, created_at_ms=0),
        _row(level=0, min_seq=2, max_seq=2, byte_size=128, created_at_ms=0),
    ]
    # Five minutes after birth — exactly at the threshold.
    job = compactor.plan(rows, now_ms=300_000)
    assert job is not None
    assert job.output_level == 1
    assert [r.min_seq for r in job.inputs] == [1, 2]


def test_plan_does_not_age_terminal_level():
    compactor = Compactor(CompactionConfig(level_targets=(1024,), max_l0_age_sec=300.0))
    rows = [_row(level=1, min_seq=1, max_seq=1, byte_size=128, created_at_ms=0)]
    assert compactor.plan(rows, now_ms=10**12) is None


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
