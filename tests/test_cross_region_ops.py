# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/iris/cross_region_ops.analyze.

The core regression these tests pin is the false-positive cross-region flag
triggered when a task gets resubmitted with the same name: ``attempt_id`` gets
reused across the old (deleted) and new attempts, so log lines from the prior
execution must be rejected by the time-window filter rather than re-attributed
to the new attempt's region.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import pytest

# Both are hard deps of the script under test but they live behind optional
# extras in the workspace (iris[controller]), so skip cleanly if missing.
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("duckdb")

from scripts.iris.cross_region_ops import (  # noqa: E402
    ATTEMPT_WINDOW_GRACE_MS,
    TimeWindow,
    analyze,
    load_attempt_windows,
)


def _make_controller_db(path: Path, rows: list[dict]) -> None:
    """Write a minimal `task_attempts`/`workers`/`worker_attributes` schema.

    `rows` is a list of dicts with keys: task_id, attempt_id, worker_id,
    started_at_ms, finished_at_ms, region. `region` may be None to simulate a
    worker whose attributes were cascaded away.
    """
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE workers (worker_id TEXT PRIMARY KEY);
        CREATE TABLE worker_attributes (
            worker_id TEXT,
            key TEXT,
            str_value TEXT,
            PRIMARY KEY (worker_id, key)
        );
        CREATE TABLE task_attempts (
            task_id TEXT NOT NULL,
            attempt_id INTEGER NOT NULL,
            worker_id TEXT,
            started_at_ms INTEGER,
            finished_at_ms INTEGER,
            PRIMARY KEY (task_id, attempt_id)
        );
        """
    )
    for row in rows:
        if row["worker_id"] is not None:
            conn.execute(
                "INSERT OR IGNORE INTO workers (worker_id) VALUES (?)",
                (row["worker_id"],),
            )
            if row["region"] is not None:
                conn.execute(
                    "INSERT OR IGNORE INTO worker_attributes (worker_id, key, str_value) VALUES (?, 'region', ?)",
                    (row["worker_id"], row["region"]),
                )
        conn.execute(
            "INSERT INTO task_attempts (task_id, attempt_id, worker_id, started_at_ms, finished_at_ms)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                row["task_id"],
                row["attempt_id"],
                row["worker_id"],
                row["started_at_ms"],
                row["finished_at_ms"],
            ),
        )
    conn.commit()
    conn.close()


def _make_log_parquet(path: Path, rows: list[tuple[str, int, str]]) -> None:
    """Write a parquet with the columns the script reads: key/epoch_ms/data."""
    keys = [r[0] for r in rows]
    epochs = [r[1] for r in rows]
    datas = [r[2] for r in rows]
    table = pa.table(
        {
            "key": pa.array(keys, type=pa.string()),
            "epoch_ms": pa.array(epochs, type=pa.int64()),
            "data": pa.array(datas, type=pa.string()),
        }
    )
    pq.write_table(table, path)


def _window_from_ms(start_ms: int, end_ms: int) -> TimeWindow:
    return TimeWindow(
        start=dt.datetime.fromtimestamp(start_ms / 1000, tz=dt.timezone.utc),
        end=dt.datetime.fromtimestamp(end_ms / 1000, tz=dt.timezone.utc),
    )


def test_load_attempt_windows_skips_never_started(tmp_path: Path) -> None:
    db = tmp_path / "controller.sqlite3"
    _make_controller_db(
        db,
        [
            {
                "task_id": "/u/job/t",
                "attempt_id": 0,
                "worker_id": "w1",
                "started_at_ms": 1_000,
                "finished_at_ms": 2_000,
                "region": "us-central1",
            },
            {
                # Never started — must not appear.
                "task_id": "/u/job/t",
                "attempt_id": 1,
                "worker_id": "w1",
                "started_at_ms": None,
                "finished_at_ms": None,
                "region": "us-central1",
            },
        ],
    )
    attempts = load_attempt_windows(db)
    assert list(attempts.keys()) == [("/u/job/t", 0)]


def test_load_attempt_windows_worker_cascaded(tmp_path: Path) -> None:
    """Attempt with no `worker_attributes.region` still appears — with region=None."""
    db = tmp_path / "controller.sqlite3"
    _make_controller_db(
        db,
        [
            {
                "task_id": "/u/job/t",
                "attempt_id": 0,
                "worker_id": "w-gone",
                "started_at_ms": 1_000,
                "finished_at_ms": 2_000,
                "region": None,  # simulate cascaded-away worker_attributes row
            },
        ],
    )
    attempts = load_attempt_windows(db)
    assert attempts[("/u/job/t", 0)].region is None


def test_analyze_flags_cross_region_inside_window(tmp_path: Path) -> None:
    """Baseline: a log line inside the attempt's window, writing to another
    region, is correctly flagged as cross-region."""
    db = tmp_path / "controller.sqlite3"
    _make_controller_db(
        db,
        [
            {
                "task_id": "/u/job/t",
                "attempt_id": 0,
                "worker_id": "w1",
                "started_at_ms": 1_000,
                "finished_at_ms": 2_000,
                "region": "us-central1",
            },
        ],
    )
    parquet = tmp_path / "log.parquet"
    _make_log_parquet(
        parquet,
        [("/u/job/t:0", 1_500, "saving checkpoint to gs://marin-us-east5/foo/bar.safetensors")],
    )

    summary = analyze([parquet], db, _window_from_ms(0, 3_000))

    assert summary["cross_region_log_lines"] == 1
    assert summary["cross_region_path_mentions"] == 1
    assert summary["cross_region_region_pairs"] == {"us-central1->us-east5": 1}


def test_analyze_rejects_log_line_outside_attempt_window(tmp_path: Path) -> None:
    """The bug Larry hit: attempt_id=0 was reused after a resubmission in a
    different region. The parquet still has log lines from the *prior* attempt
    keyed ``:0``; without the time-window filter those get re-attributed to the
    new attempt's region and flagged as cross-region."""
    db = tmp_path / "controller.sqlite3"
    _make_controller_db(
        db,
        [
            {
                # Only the current (us-central1) attempt is in the DB — the
                # prior us-east5 attempt was deleted on resubmission.
                "task_id": "/larry/job/t",
                "attempt_id": 0,
                "worker_id": "w-new",
                "started_at_ms": 10_000,
                "finished_at_ms": 20_000,
                "region": "us-central1",
            },
        ],
    )
    parquet = tmp_path / "log.parquet"
    _make_log_parquet(
        parquet,
        [
            # From the prior us-east5 attempt that was deleted. epoch_ms is
            # before the current attempt's `started_at_ms`, so it must not
            # count as cross-region.
            ("/larry/job/t:0", 5_000, "Saved checkpoint to gs://marin-us-east5/x/step-1"),
            # From the current us-central1 attempt writing to an in-region
            # bucket — also not cross-region.
            ("/larry/job/t:0", 15_000, "Saved checkpoint to gs://marin-us-central1/x/step-2"),
        ],
    )

    summary = analyze([parquet], db, _window_from_ms(0, 30_000))

    assert summary["cross_region_log_lines"] == 0
    assert summary["cross_region_path_mentions"] == 0
    # The pre-window log line is surfaced in the coverage counter.
    assert summary["lines_outside_attempt_window"] == 1
    assert summary["lines_no_attempt_match"] == 0


def test_analyze_grace_margin_covers_late_flush(tmp_path: Path) -> None:
    """A log line arriving a few seconds after ``finished_at_ms`` (async flush)
    still matches, as long as it's within ``ATTEMPT_WINDOW_GRACE_MS``."""
    db = tmp_path / "controller.sqlite3"
    _make_controller_db(
        db,
        [
            {
                "task_id": "/u/job/t",
                "attempt_id": 0,
                "worker_id": "w1",
                "started_at_ms": 1_000,
                "finished_at_ms": 2_000,
                "region": "us-central1",
            },
        ],
    )
    parquet = tmp_path / "log.parquet"
    # 1s after finished_at_ms — inside the 30s grace window.
    _make_log_parquet(
        parquet,
        [("/u/job/t:0", 2_000 + 1_000, "saving gs://marin-us-east5/x/y.safetensors")],
    )
    summary = analyze([parquet], db, _window_from_ms(0, 10_000))
    assert summary["cross_region_log_lines"] == 1
    assert summary["lines_outside_attempt_window"] == 0

    # Well past the grace window — rejected.
    _make_log_parquet(
        parquet,
        [("/u/job/t:0", 2_000 + ATTEMPT_WINDOW_GRACE_MS + 1_000, "saving gs://marin-us-east5/x/y.safetensors")],
    )
    summary = analyze([parquet], db, _window_from_ms(0, 60_000))
    assert summary["cross_region_log_lines"] == 0
    assert summary["lines_outside_attempt_window"] == 1


def test_analyze_worker_region_unknown_is_not_crossregion(tmp_path: Path) -> None:
    """When the attempt matches but the worker's region is unknown (cascaded
    away), we must not fall back to any other region — the old bug 2 from the
    issue analysis."""
    db = tmp_path / "controller.sqlite3"
    _make_controller_db(
        db,
        [
            {
                "task_id": "/u/job/t",
                "attempt_id": 0,
                "worker_id": "w-gone",
                "started_at_ms": 1_000,
                "finished_at_ms": 2_000,
                "region": None,
            },
        ],
    )
    parquet = tmp_path / "log.parquet"
    _make_log_parquet(
        parquet,
        [("/u/job/t:0", 1_500, "Saved checkpoint to gs://marin-us-east5/x/y.safetensors")],
    )
    summary = analyze([parquet], db, _window_from_ms(0, 3_000))

    assert summary["cross_region_log_lines"] == 0
    assert summary["lines_matched_attempt_worker_unknown"] == 1


def test_analyze_no_attempt_in_db_falls_into_unmatched(tmp_path: Path) -> None:
    """A log for a ``(task_id, attempt_id)`` the DB doesn't know about goes
    into ``lines_no_attempt_match`` / ``unmatched_tasks_top25``, not into the
    cross-region counters."""
    db = tmp_path / "controller.sqlite3"
    _make_controller_db(db, [])
    parquet = tmp_path / "log.parquet"
    _make_log_parquet(
        parquet,
        [("/u/job/t:7", 1_500, "Saved checkpoint to gs://marin-us-east5/x/y.safetensors")],
    )
    summary = analyze([parquet], db, _window_from_ms(0, 3_000))

    assert summary["cross_region_log_lines"] == 0
    assert summary["lines_no_attempt_match"] == 1
    assert summary["unmatched_tasks_top25"] == {"/u/job/t": 1}


@pytest.mark.parametrize(
    "bucket, region, expect_cross",
    [
        ("marin-us-east5", "us-central1", True),
        # Same region → not cross-region even though the bucket has the marin-
        # prefix form.
        ("marin-us-central1", "us-central1", False),
    ],
)
def test_analyze_parametrized_region_match(tmp_path: Path, bucket: str, region: str, expect_cross: bool) -> None:
    db = tmp_path / "controller.sqlite3"
    _make_controller_db(
        db,
        [
            {
                "task_id": "/u/job/t",
                "attempt_id": 0,
                "worker_id": "w1",
                "started_at_ms": 1_000,
                "finished_at_ms": 2_000,
                "region": region,
            }
        ],
    )
    parquet = tmp_path / "log.parquet"
    _make_log_parquet(
        parquet,
        [("/u/job/t:0", 1_500, f"saving gs://{bucket}/x/y.safetensors")],
    )
    summary = analyze([parquet], db, _window_from_ms(0, 3_000))
    assert bool(summary["cross_region_log_lines"]) == expect_cross
