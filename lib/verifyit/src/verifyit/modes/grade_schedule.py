# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a final calendar JSON value against TaskTrove scheduling constraints."""

import json
import re
import unicodedata
from collections.abc import Mapping
from itertools import pairwise

from verifyit.grade import Reward, scored

_TIME_RE = re.compile(r"(\d{2}):(\d{2})")
_CLOCK = r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?"
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def grade_schedule_candidate(expected: Mapping[str, Mapping[str, object]], answer: str) -> Reward:
    """Parse a submitted JSON schedule, including the source's fenced format."""
    fence = _FENCE_RE.search(answer)
    try:
        events = json.loads(fence.group(1) if fence else answer)
    except json.JSONDecodeError:
        return scored(0.0)
    return score_schedule(expected, events)


def normalized_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(unicodedata.normalize("NFC", value).split())
    return normalized or None


def parse_time(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    match = _TIME_RE.fullmatch(value.strip())
    if match is None:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour * 60 + minute


def _clock_minutes(hour: str, minute: str | None, ampm: str | None) -> int | None:
    """Return valid clock time as minutes after midnight, or None."""
    h, m = int(hour), int(minute or 0)
    if not (0 <= m <= 59):
        return None
    if ampm is None:
        return h * 60 + m if 0 <= h <= 23 else None
    if not (1 <= h <= 12):
        return None
    return (h % 12 + (12 if ampm == "pm" else 0)) * 60 + m


def _constraint_holds(constraint: object, start: int, end: int) -> bool:
    if constraint is None or constraint == "":
        return True
    if not isinstance(constraint, str):
        return False
    text = " ".join(constraint.strip().lower().split())
    match = re.fullmatch(rf"before\s+{_CLOCK}", text)
    if match is not None:
        limit = _clock_minutes(*match.groups())
        return limit is not None and end <= limit
    match = re.fullmatch(rf"after\s+{_CLOCK}", text)
    if match is not None:
        limit = _clock_minutes(*match.groups())
        return limit is not None and start >= limit
    match = re.fullmatch(rf"at\s+{_CLOCK}", text)
    if match is not None:
        exact = _clock_minutes(*match.groups())
        return exact is not None and start == exact
    match = re.fullmatch(rf"between\s+{_CLOCK}\s+and\s+{_CLOCK}", text)
    if match is not None:
        groups = match.groups()
        lower = _clock_minutes(*groups[:3])
        upper = _clock_minutes(*groups[3:])
        return lower is not None and upper is not None and start >= lower and end <= upper
    return False


def score_schedule(expected: Mapping[str, Mapping[str, object]], events: object) -> Reward:
    errors: list[str] = []
    if not isinstance(events, list):
        return scored(0.0, error="answer must be a JSON list")

    actual_by_id: dict[int, dict] = {}
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"event at index {index} is not an object")
            continue
        event_id = event.get("event_id")
        if not isinstance(event_id, int) or isinstance(event_id, bool):
            errors.append(f"event at index {index} has invalid event_id")
            continue
        if event_id in actual_by_id:
            errors.append(f"duplicate event_id {event_id}")
            continue
        actual_by_id[event_id] = event

    expected_ids = {int(key) for key in expected}
    for event_id in sorted(expected_ids - actual_by_id.keys()):
        errors.append(f"missing event_id {event_id}")
    for event_id in sorted(actual_by_id.keys() - expected_ids):
        errors.append(f"unexpected event_id {event_id}")

    intervals: list[tuple[int, int, int]] = []
    for event_id in sorted(expected_ids & actual_by_id.keys()):
        spec = expected[str(event_id)]
        actual = actual_by_id[event_id]
        expected_name = normalized_name(spec.get("event_name"))
        if expected_name is None or normalized_name(actual.get("event_name")) != expected_name:
            errors.append(f"event {event_id} name mismatch")

        duration = actual.get("duration")
        if (
            not isinstance(duration, int)
            or isinstance(duration, bool)
            or duration <= 0
            or duration != spec.get("duration")
        ):
            errors.append(f"event {event_id} duration mismatch")
            continue
        start = parse_time(actual.get("start_time"))
        minimum = parse_time(spec.get("min_time"))
        maximum = parse_time(spec.get("max_time"))
        if start is None or minimum is None or maximum is None:
            errors.append(f"event {event_id} has invalid time data")
            continue
        end = start + duration
        if start < minimum or end > maximum:
            errors.append(f"event {event_id} is outside its allowed window")
        if not _constraint_holds(spec.get("constraint"), start, end):
            errors.append(f"event {event_id} violates its declared constraint")
        intervals.append((start, end, event_id))

    intervals.sort()
    for previous, current in pairwise(intervals):
        if current[0] < previous[1]:
            errors.append(f"events {previous[2]} and {current[2]} overlap")
    return scored(float(not errors), error="; ".join(errors) or None)
