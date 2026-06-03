# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the week-over-week snapshot/diff logic in report.py."""

from __future__ import annotations

import pytest

from scripts.ops.storage import report

GiB = 1024**3


def _row(bucket: str, prefix: str, total_bytes: int) -> dict:
    return {
        "bucket": bucket,
        "dir_prefix": prefix,
        "object_count": 1,
        "total_bytes": total_bytes,
        "monthly_cost": float(total_bytes) / GiB,
    }


def test_compute_changes_classifies_and_filters_by_threshold():
    previous = [
        _row("b", "grew/a", 300 * GiB),
        _row("b", "shrank/a", 400 * GiB),
        _row("b", "gone/a", 200 * GiB),
        _row("b", "stable/a", 150 * GiB),
    ]
    current = [
        _row("b", "grew/a", 560 * GiB),  # +260
        _row("b", "shrank/a", 250 * GiB),  # -150
        _row("b", "new/a", 250 * GiB),  # +250 (absent before)
        _row("b", "stable/a", 160 * GiB),  # +10, below threshold
        # gone/a absent now -> -200
    ]
    changes = report.compute_changes(current, previous, threshold_bytes=100 * GiB)
    by_prefix = {c["dir_prefix"]: c for c in changes}

    assert set(by_prefix) == {"grew/a", "shrank/a", "new/a", "gone/a"}
    assert by_prefix["grew/a"]["status"] == "grew"
    assert by_prefix["shrank/a"]["status"] == "shrank"
    assert by_prefix["new/a"]["status"] == "new"
    assert by_prefix["gone/a"]["status"] == "gone"
    assert by_prefix["new/a"]["was_bytes"] == 0
    assert by_prefix["gone/a"]["now_bytes"] == 0
    # "stable/a" moved only +10 GiB and must be filtered out.
    assert "stable/a" not in by_prefix


def test_compute_changes_sorted_by_absolute_delta():
    previous = [_row("b", "p1", 100 * GiB), _row("b", "p2", 100 * GiB)]
    current = [_row("b", "p1", 600 * GiB), _row("b", "p2", 450 * GiB)]
    changes = report.compute_changes(current, previous, threshold_bytes=100 * GiB)
    assert [c["dir_prefix"] for c in changes] == ["p1", "p2"]  # +500 before +350


def test_render_changes_section_baseline_when_no_previous_date():
    out = report.render_changes_section([], previous_date=None, threshold_bytes=100 * GiB)
    assert "baseline" in out.lower()
    assert "Δ Size" not in out


def test_render_changes_section_table_and_threshold_note():
    changes = report.compute_changes(
        [_row("marin-eu-west4", "datakit/big", 250 * GiB)],
        [],
        threshold_bytes=100 * GiB,
    )
    out = report.render_changes_section(changes, previous_date="2026-06-02", threshold_bytes=100 * GiB)
    assert "since 2026-06-02" in out
    assert "datakit/big" in out
    assert "marin-eu-west4" in out
    assert "new" in out


def test_snapshot_roundtrip_and_find_latest(tmp_path):
    rows = [_row("b", "p/a", 5 * GiB), _row("b", "p/b", 9 * GiB)]
    history = str(tmp_path)

    report.write_snapshot(rows, report.snapshot_path(history, "2026-05-26"))
    report.write_snapshot(rows, report.snapshot_path(history, "2026-06-02"))

    assert report.read_snapshot(report.snapshot_path(history, "2026-06-02")) == rows

    # Most recent strictly before today's date.
    found = report.find_latest_snapshot(history, before_date="2026-06-09")
    assert found is not None and found[1] == "2026-06-02"

    # Same-day exclusion: a run on 2026-06-02 must not diff against its own snapshot.
    older = report.find_latest_snapshot(history, before_date="2026-06-02")
    assert older is not None and older[1] == "2026-05-26"


def test_find_latest_snapshot_none_when_empty(tmp_path):
    assert report.find_latest_snapshot(str(tmp_path)) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
