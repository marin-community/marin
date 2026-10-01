# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import datetime as dt

from finelog.client import DirectQuerySegment

from scripts.ops.cross_region import TimeWindow, choose_log_segments

WINDOW_START = dt.datetime(2026, 10, 1, 0, 0, tzinfo=dt.UTC)
WINDOW = TimeWindow(start=WINDOW_START, end=WINDOW_START + dt.timedelta(hours=24))


def _segment(name: str, created_at: dt.datetime) -> DirectQuerySegment:
    return DirectQuerySegment(
        object_uri=f"gs://bucket/finelog/_finelog/tables/log/objects/{name}.parquet",
        byte_size=1,
        row_count=1,
        created_at_ms=int(created_at.timestamp() * 1000),
    )


def test_choose_log_segments_keeps_segments_created_after_the_lookback_cutoff() -> None:
    stale = _segment("stale", WINDOW_START - dt.timedelta(hours=12, seconds=1))
    at_cutoff = _segment("at-cutoff", WINDOW_START - dt.timedelta(hours=12))
    during = _segment("during", WINDOW_START + dt.timedelta(hours=3))
    after = _segment("after", WINDOW.end + dt.timedelta(hours=1))

    chosen = choose_log_segments([after, stale, during, at_cutoff], WINDOW, lookback_hours=12.0)

    assert chosen == [at_cutoff, during, after]


def test_choose_log_segments_returns_nothing_when_every_segment_predates_the_window() -> None:
    old = [_segment(f"old-{i}", WINDOW_START - dt.timedelta(days=i + 1)) for i in range(3)]

    assert choose_log_segments(old, WINDOW, lookback_hours=12.0) == []
