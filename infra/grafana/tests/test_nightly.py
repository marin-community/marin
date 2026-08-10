# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for the nightly-lane regression matrix projection.

Ported from the deleted TypeScript status page's nightlyProjection.test.ts. v1
drops the run-attempt "recovery" pass (attemptsByRunId/selectedReruns/recovered),
so the recovery-history test case isn't ported. Everything here exercises
project_nightlies directly against hand-built lanes and snapshots — no network.
"""

import dataclasses
from datetime import UTC, datetime

from nightly import NightlyLaneSnapshot, NightlyRun, project_nightlies
from nightly_config import NightlyLane

_BASE_LANE = NightlyLane(
    id="lane",
    label="Lane",
    short_label="Lane",
    group="marin",
    subgroup="training",
    repository="marin-community/marin",
    workflow_file="nightly.yaml",
    branch="main",
    weekdays=(0, 1, 2, 3, 4, 5, 6),
    hour=10,
    minute=0,
    active_from=None,
    active_until=None,
    overdue_grace_minutes=240,
    expected_min_seconds=None,
    expected_max_seconds=None,
)

_BASE_RUN = NightlyRun(
    id=1,
    status="completed",
    conclusion="success",
    sha="1234567890abcdef",
    created_at="2026-07-17T11:00:00.000Z",
    run_started_at="2026-07-17T11:00:00.000Z",
    updated_at="2026-07-17T11:10:00.000Z",
    url="https://github.com/marin-community/marin/actions/runs/1",
)


def _lane(**overrides) -> NightlyLane:
    return dataclasses.replace(_BASE_LANE, **overrides)


def _run(**overrides) -> NightlyRun:
    return dataclasses.replace(_BASE_RUN, **overrides)


def _snapshot(lane_id: str, runs: list[NightlyRun] | None = None, error: str | None = None) -> NightlyLaneSnapshot:
    return NightlyLaneSnapshot(lane_id=lane_id, runs=runs or [], error=error)


def _cell(rows: list[dict], lane: NightlyLane, date: str) -> dict:
    (row,) = [r for r in rows if r["lane"] == lane.short_label and r["date"] == date]
    return row


def test_weekly_lane_shows_one_missing_occurrence_and_six_quiet_non_occurrences():
    weekly = _lane(id="weekly", short_label="Weekly", weekdays=(1,), hour=1, minute=0, overdue_grace_minutes=480)
    now = datetime(2026, 7, 17, 15, 0, 0, tzinfo=UTC)

    rows = project_nightlies([weekly], [_snapshot(weekly.id)], now)

    assert [(row["date"], row["state"]) for row in rows] == [
        ("2026-07-17", "not-scheduled"),
        ("2026-07-16", "not-scheduled"),
        ("2026-07-15", "not-scheduled"),
        ("2026-07-14", "not-scheduled"),
        ("2026-07-13", "missing"),
        ("2026-07-12", "not-scheduled"),
        ("2026-07-11", "not-scheduled"),
    ]

    missing = _cell(rows, weekly, "2026-07-13")
    assert missing["due"] is True
    assert missing["healthy"] is False
    assert missing["status_code"] == 4
    assert missing["duration_state"] == "not-applicable"

    not_scheduled = _cell(rows, weekly, "2026-07-17")
    assert not_scheduled["due"] is False
    assert not_scheduled["status_code"] is None


def test_lifecycle_distinguishes_not_introduced_from_not_yet_due():
    new_lane = _lane(id="new-lane", active_from="2026-07-17", hour=7, minute=30, overdue_grace_minutes=300)
    now = datetime(2026, 7, 17, 8, 0, 0, tzinfo=UTC)

    rows = project_nightlies([new_lane], [_snapshot(new_lane.id)], now)

    today = _cell(rows, new_lane, "2026-07-17")
    assert today["state"] == "not-yet-due"
    assert today["due"] is False
    assert today["status_code"] == 6

    yesterday = _cell(rows, new_lane, "2026-07-16")
    assert yesterday["state"] == "not-introduced"
    assert yesterday["due"] is False
    assert yesterday["status_code"] is None


def test_too_short_success_is_excluded_from_health_but_baseline_pending_is_healthy():
    bounded = _lane(id="bounded", short_label="Bounded", expected_min_seconds=360, expected_max_seconds=900)
    pending = _lane(id="pending", short_label="Pending", workflow_file="pending.yaml")
    now = datetime(2026, 7, 17, 15, 0, 0, tzinfo=UTC)

    rows = project_nightlies(
        [bounded, pending],
        [
            _snapshot(
                bounded.id,
                [_run(id=10, run_started_at="2026-07-17T11:00:00.000Z", updated_at="2026-07-17T11:01:11.000Z")],
            ),
            _snapshot(pending.id, [_run(id=11)]),
        ],
        now,
    )

    bounded_today = _cell(rows, bounded, "2026-07-17")
    assert bounded_today["conclusion"] == "success"
    assert bounded_today["duration_seconds"] == 71
    assert bounded_today["duration_state"] == "too-short"
    assert bounded_today["healthy"] is False
    assert bounded_today["status_code"] == 3  # run, unhealthy

    pending_today = _cell(rows, pending, "2026-07-17")
    assert pending_today["duration_state"] == "baseline-pending"
    assert pending_today["healthy"] is True
    assert pending_today["status_code"] == 1  # run, healthy, no baseline yet

    # Matches the TS response.today aggregate: 1 of the 2 due lanes healthy.
    today_rows = [row for row in rows if row["date"] == "2026-07-17"]
    assert sum(row["due"] for row in today_rows) == 2
    assert sum(row["healthy"] for row in today_rows) == 1


def test_projection_exposes_stable_lane_order_and_workflow_link():
    healthy = _lane(id="healthy", short_label="Healthy")
    quiet = _lane(id="quiet", short_label="Quiet", weekdays=(1,))  # only ever due on Tuesday
    now = datetime(2026, 7, 17, 15, 0, 0, tzinfo=UTC)

    long_rows = project_nightlies([healthy, quiet], [_snapshot(healthy.id, [_run()]), _snapshot(quiet.id)], now)
    assert {row["lane_order"] for row in long_rows if row["lane_id"] == "healthy"} == {0}
    assert {row["lane_order"] for row in long_rows if row["lane_id"] == "quiet"} == {1}
    assert long_rows[0]["workflow_url"] == "https://github.com/marin-community/marin/actions/workflows/nightly.yaml"
