# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import datetime as dt
import itertools
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from finelog.client import DirectQuerySegment

from scripts.ops.cross_region import TimeWindow, analyze, choose_log_segments, iter_log_rows

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


TASK_ID = "/alice/train-job/0"
CENTRAL_BUCKET_PATH = "gs://marin-us-central2/checkpoints/model.safetensors"
EUROPE_BUCKET_PATH = "gs://marin-eu-west4/tokenized/shard.parquet"


def _controller_db(path: Path) -> Path:
    """Write a controller DB with one attempt of TASK_ID on a us-central2 worker spanning WINDOW."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE task_attempts (task_id TEXT, attempt_id INTEGER, worker_id TEXT, "
        "started_at_ms INTEGER, finished_at_ms INTEGER)"
    )
    conn.execute("CREATE TABLE worker_attributes (worker_id TEXT, key TEXT, str_value TEXT)")
    conn.execute("INSERT INTO task_attempts VALUES (?, 0, 'w-1', ?, ?)", (TASK_ID, WINDOW.start_ms, WINDOW.end_ms))
    conn.execute("INSERT INTO worker_attributes VALUES ('w-1', 'region', 'us-central2')")
    conn.commit()
    conn.close()
    return path


def _log_rows() -> list[tuple[str, int, str]]:
    """In-window log rows, oldest first: more cross-region reads than the sample caps, each logged twice."""
    rows = []
    for minute in range(120):
        epoch_ms = WINDOW.start_ms + minute * 60_000
        path = EUROPE_BUCKET_PATH if minute % 3 else CENTRAL_BUCKET_PATH
        line = (f"{TASK_ID}:0", epoch_ms, f"Reading {path}")
        rows += [line, line]
    return rows


def test_analyze_report_is_independent_of_how_rows_split_across_segments(tmp_path: Path) -> None:
    db_path = _controller_db(tmp_path / "controller.sqlite3")
    rows = _log_rows()
    newest_first = sorted(rows, key=lambda row: -row[1])
    # Out-of-order segments with each duplicate pair split between them.
    segments = [rows[1::2], list(reversed(rows[0::2]))]

    single_scan = analyze(newest_first, db_path, WINDOW)
    segmented_scan = analyze(itertools.chain.from_iterable(segments), db_path, WINDOW)

    assert segmented_scan == single_scan
    assert segmented_scan["cross_region_path_mentions"] == 160
    sample_times = [sample["timestamp_ms"] for sample in segmented_scan["samples"]]
    assert sample_times == sorted({row[1] for row in rows if EUROPE_BUCKET_PATH in row[2]}, reverse=True)[:50]


def test_iter_log_rows_streams_in_window_gs_rows_and_leaves_no_segment_on_disk(tmp_path: Path) -> None:
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    in_window = (f"{TASK_ID}:0", WINDOW.start_ms + 1, f"Reading {EUROPE_BUCKET_PATH}")
    segments = []
    for name, rows in {
        "seg-a": [in_window, (f"{TASK_ID}:0", WINDOW.start_ms - 1, f"Reading {EUROPE_BUCKET_PATH}")],
        "seg-b": [(f"{TASK_ID}:0", WINDOW.start_ms + 2, "no paths here"), ("worker-log", WINDOW.start_ms + 3, "gs://x")],
    }.items():
        object_path = remote_dir / f"{name}.parquet"
        keys, times, data = zip(*rows, strict=True)
        pq.write_table(pa.table({"key": keys, "epoch_ms": times, "data": data}), object_path)
        segments.append(
            DirectQuerySegment(
                object_uri=str(object_path),
                byte_size=object_path.stat().st_size,
                row_count=len(rows),
                created_at_ms=WINDOW.start_ms,
            )
        )
    scratch_dir = tmp_path / "scratch"

    streamed = list(iter_log_rows(segments, WINDOW, scratch_dir, max_segment_bytes=2**20))

    assert streamed == [in_window]
    assert list(scratch_dir.iterdir()) == []
