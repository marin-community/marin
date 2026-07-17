# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project nightly-lane GitHub run snapshots onto a 7-day regression matrix.

For each (lane, day) over the trailing 7 UTC days, project_nightlies matches the
lane's fetched runs against its schedule to classify one cell: whether a run was
due, whether it ran, and — if it ran — whether it finished healthy and on time.
The projection is pure: it takes the current instant as an argument rather than
reading the clock, so it is deterministic and unit-testable.
"""

import dataclasses
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from nightly_config import NightlyLane

# Duration past which a "slow" run instead counts as "very-slow" (multiple of the
# expected max).
_VERY_SLOW_FACTOR = 1.5


class NightlyCellState(StrEnum):
    """The classification of one (lane, day) cell."""

    NOT_SCHEDULED = "not-scheduled"
    NOT_INTRODUCED = "not-introduced"
    RETIRED = "retired"
    NOT_YET_DUE = "not-yet-due"
    MISSING = "missing"
    UNAVAILABLE = "unavailable"
    RUN = "run"


class NightlyDurationState(StrEnum):
    """How a run's duration compares to its lane's expected range."""

    NOT_APPLICABLE = "not-applicable"
    BASELINE_PENDING = "baseline-pending"
    TOO_SHORT = "too-short"
    NORMAL = "normal"
    SLOW = "slow"
    VERY_SLOW = "very-slow"


@dataclasses.dataclass(frozen=True)
class NightlyRun:
    """One GitHub Actions workflow run, as reported by the runs-list API."""

    id: int
    status: str
    conclusion: str | None
    sha: str
    created_at: str
    run_started_at: str | None
    updated_at: str
    url: str


@dataclasses.dataclass(frozen=True)
class NightlyLaneSnapshot:
    """One lane's fetched runs, or the error that prevented fetching them."""

    lane_id: str
    runs: list[NightlyRun]
    error: str | None


def _parse_instant(value: str) -> datetime:
    """Parse a GitHub API ISO-8601 instant (Z-suffixed) to a UTC datetime."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _utc_day_start(instant: datetime) -> datetime:
    """Truncate a UTC instant to that day's 00:00:00."""
    return instant.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


def _date_key(day: datetime) -> str:
    return day.strftime("%Y-%m-%d")


def _expected_at(lane: NightlyLane, day: datetime) -> datetime | None:
    """The lane's scheduled run instant on `day`, or None if not scheduled that day.

    day.weekday() is Monday=0..Sunday=6; lane.weekdays is Sunday=0..Saturday=6.
    """
    sunday_based_weekday = (day.weekday() + 1) % 7
    if sunday_based_weekday not in lane.weekdays:
        return None
    return day.replace(hour=lane.hour, minute=lane.minute, second=0, microsecond=0)


def _next_expected_at(lane: NightlyLane, expected: datetime) -> datetime:
    """The lane's next scheduled instant strictly after `expected`."""
    day_start = _utc_day_start(expected)
    for offset in range(1, 8):
        candidate = _expected_at(lane, day_start + timedelta(days=offset))
        if candidate is not None:
            return candidate
    raise ValueError(f"{lane.id}: schedule has no next occurrence")


def _duration_state(duration_seconds: int | None, lane: NightlyLane, *, completed: bool) -> NightlyDurationState:
    if duration_seconds is None:
        return NightlyDurationState.NOT_APPLICABLE
    if lane.expected_min_seconds is None or lane.expected_max_seconds is None:
        return NightlyDurationState.BASELINE_PENDING
    if completed and duration_seconds < lane.expected_min_seconds:
        return NightlyDurationState.TOO_SHORT
    if duration_seconds <= lane.expected_max_seconds:
        return NightlyDurationState.NORMAL
    if duration_seconds > lane.expected_max_seconds * _VERY_SLOW_FACTOR:
        return NightlyDurationState.VERY_SLOW
    return NightlyDurationState.SLOW


def _run_duration_seconds(run: NightlyRun, now: datetime) -> int | None:
    if not run.run_started_at:
        return None
    start = _parse_instant(run.run_started_at)
    end = _parse_instant(run.updated_at) if run.status == "completed" else now
    return max(0, round((end - start).total_seconds()))


def _is_successful_run(run: NightlyRun) -> bool:
    return run.status == "completed" and run.conclusion == "success"


def _status_code(state: NightlyCellState, *, healthy: bool, duration_state: NightlyDurationState) -> int | None:
    """The panel's coloring code for one cell, or None where no color applies."""
    if state == NightlyCellState.RUN:
        if not healthy:
            return 3
        slow_states = (NightlyDurationState.SLOW, NightlyDurationState.VERY_SLOW)
        return 2 if duration_state in slow_states else 1
    if state == NightlyCellState.MISSING:
        return 4
    if state == NightlyCellState.UNAVAILABLE:
        return 5
    if state == NightlyCellState.NOT_YET_DUE:
        return 6
    return None


def _row(
    lane: NightlyLane,
    date: str,
    ts: int,
    *,
    state: NightlyCellState,
    due: bool,
    healthy: bool,
    duration_state: NightlyDurationState,
    duration_seconds: int | None,
    conclusion: str | None,
    url: str | None,
) -> dict:
    return {
        "ts": ts,
        "date": date,
        "lane_id": lane.id,
        "lane": lane.short_label,
        "label": lane.label,
        "group": lane.group,
        "subgroup": lane.subgroup,
        "repository": lane.repository,
        "workflow": lane.workflow_file,
        "state": state,
        "status_code": _status_code(state, healthy=healthy, duration_state=duration_state),
        "healthy": healthy,
        "due": due,
        "duration_state": duration_state,
        "duration_seconds": duration_seconds,
        "conclusion": conclusion,
        "url": url,
    }


def _empty_row(
    lane: NightlyLane,
    date: str,
    ts: int,
    expected: datetime | None,
    snapshot: NightlyLaneSnapshot,
    now: datetime,
) -> dict:
    """A (lane, day) cell with no matched run: not-scheduled through missing/unavailable."""
    if expected is None:
        state, due = NightlyCellState.NOT_SCHEDULED, False
    elif lane.active_from and date < lane.active_from:
        state, due = NightlyCellState.NOT_INTRODUCED, False
    elif lane.active_until and date > lane.active_until:
        state, due = NightlyCellState.RETIRED, False
    else:
        due_at = expected + timedelta(minutes=lane.overdue_grace_minutes)
        if now < due_at:
            state, due = NightlyCellState.NOT_YET_DUE, False
        elif snapshot.error:
            state, due = NightlyCellState.UNAVAILABLE, True
        else:
            state, due = NightlyCellState.MISSING, True
    return _row(
        lane,
        date,
        ts,
        state=state,
        due=due,
        healthy=False,
        duration_state=NightlyDurationState.NOT_APPLICABLE,
        duration_seconds=None,
        conclusion=None,
        url=None,
    )


def _run_row(lane: NightlyLane, date: str, ts: int, candidates: Sequence[NightlyRun], now: datetime) -> dict:
    """A (lane, day) cell matched to one or more runs; the latest-created one wins."""
    run = max(candidates, key=lambda candidate: _parse_instant(candidate.created_at))
    completed = run.status == "completed"
    duration_seconds = _run_duration_seconds(run, now)
    duration_state = _duration_state(duration_seconds, lane, completed=completed)
    healthy = _is_successful_run(run) and duration_state != NightlyDurationState.TOO_SHORT
    return _row(
        lane,
        date,
        ts,
        state=NightlyCellState.RUN,
        due=True,
        healthy=healthy,
        duration_state=duration_state,
        duration_seconds=duration_seconds,
        conclusion=run.conclusion,
        url=run.url,
    )


def project_nightlies(
    lanes: Sequence[NightlyLane], snapshots: Sequence[NightlyLaneSnapshot], now: datetime
) -> list[dict]:
    """Project each lane's fetched runs onto the trailing 7 UTC days (today + 6 back).

    Returns one flat dict per (lane, day), ordered by lane (config order) then day
    descending (today first). `now` must be UTC; the function reads no clock.
    """
    snapshot_by_lane = {snapshot.lane_id: snapshot for snapshot in snapshots}
    today_start = _utc_day_start(now)

    rows: list[dict] = []
    for lane in lanes:
        snapshot = snapshot_by_lane.get(lane.id) or NightlyLaneSnapshot(
            lane_id=lane.id, runs=[], error="No source snapshot"
        )
        for offset in range(7):
            day = today_start - timedelta(days=offset)
            date = _date_key(day)
            ts = round(day.timestamp() * 1000)
            expected = _expected_at(lane, day)
            introduced = not (lane.active_from and date < lane.active_from)
            retired = bool(lane.active_until and date > lane.active_until)
            if expected is None or not introduced or retired:
                rows.append(_empty_row(lane, date, ts, expected, snapshot, now))
                continue

            next_expected = _next_expected_at(lane, expected)
            candidates = [run for run in snapshot.runs if expected <= _parse_instant(run.created_at) < next_expected]
            if candidates:
                rows.append(_run_row(lane, date, ts, candidates, now))
            else:
                rows.append(_empty_row(lane, date, ts, expected, snapshot, now))
    return rows


def nightly_matrix(rows: Sequence[dict]) -> list[dict]:
    """Pivot per-cell rows into one wide row per day for the state-timeline panel.

    Each output row is `{ts, date, <lane_id>: status_code, ...}` — one key per lane
    — ordered by day ascending so the panel's time axis reads left to right. A cell
    with no color code (a day the lane was not scheduled or not yet introduced)
    carries `None`, which the panel renders as a gap. State-timeline needs one
    series per lane, which is a numeric field per lane; the long per-cell rows do
    not split into series, so the endpoint serves this wide view instead.
    """
    by_day: dict[int, dict] = {}
    for row in rows:
        day = by_day.setdefault(row["ts"], {"ts": row["ts"], "date": row["date"]})
        day[row["lane_id"]] = row["status_code"]
    return [by_day[ts] for ts in sorted(by_day)]
