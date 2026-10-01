# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade the real TaskTrove final-calendar contract, accepting alternative schedules.

The scoring rules are retained from the TaskTrove cleanup's standalone calendar
checker: IDs, names, durations, explicit windows, supported text constraints, and
pairwise overlap. This is a final-answer adapter, not an interactive environment.
"""

import json
import re
import unicodedata
from itertools import pairwise

from pydantic import JsonValue, model_validator

from taskcompendium.grading import GradeResult, GradingAttempt, Outcome, Verifier
from taskcompendium.submission import extract_answer

_TIME_RE = re.compile(r"(\d{2}):(\d{2})")

_CLOCK = r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?"

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _normalized_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(unicodedata.normalize("NFC", value).split())
    return normalized or None


def _parse_time(value: object) -> int | None:
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


def _score(expected: dict, events: object) -> tuple[int, list[str]]:
    errors: list[str] = []
    if not isinstance(events, list):
        return 0, ["answer must be a JSON list"]

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
        expected_name = _normalized_name(spec.get("event_name"))
        if expected_name is None or _normalized_name(actual.get("event_name")) != expected_name:
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
        start = _parse_time(actual.get("start_time"))
        minimum = _parse_time(spec.get("min_time"))
        maximum = _parse_time(spec.get("max_time"))
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
    return (1 if not errors else 0), errors


def _extract_json(raw: str) -> object:
    fence = _FENCE_RE.search(raw)
    return json.loads(fence.group(1) if fence else raw)


class ScheduleAnswerVerifier(Verifier):
    expected_events: dict[str, dict[str, JsonValue]]

    @model_validator(mode="after")
    def validate_expected_events(self) -> "ScheduleAnswerVerifier":
        if not self.expected_events:
            raise ValueError("Expected calendar events are required")
        for key, event in self.expected_events.items():
            int(key)
            duration = event.get("duration")
            if (
                not isinstance(duration, int)
                or isinstance(duration, bool)
                or duration <= 0
                or _normalized_name(event.get("event_name")) is None
                or _parse_time(event.get("min_time")) is None
                or _parse_time(event.get("max_time")) is None
            ):
                raise ValueError(f"Malformed calendar event {key}")
        return self

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        try:
            answer = extract_answer(attempt.conversation[-1], attempt.convention)
        except (ValueError, TypeError) as error:
            return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
        try:
            events = _extract_json(answer)
        except json.JSONDecodeError:
            return GradeResult(Outcome.GRADED, 0.0)
        reward, errors = _score(self.expected_events, events)
        return GradeResult(Outcome.GRADED, float(reward), "; ".join(errors) or None)
