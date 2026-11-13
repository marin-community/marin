# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-owned calendar assertions through the existing ScriptSpec verdict channel."""

from __future__ import annotations

import json
import math
import re
import os
from pathlib import Path
from typing import Any


def _time_to_minutes(value: str) -> int:
    if not isinstance(value, str):
        raise ValueError("clock time must be text")
    match = re.fullmatch(r"(\d{1,2})(?::(\d{1,2}))?\s*(am|pm)?", value.strip())
    if match is None:
        raise ValueError("invalid clock time")
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    suffix = match.group(3)
    if suffix is None and match.group(2) is None:
        raise ValueError("24-hour clock requires minutes")
    if not 0 <= minute < 60 or not (1 <= hour <= 12 if suffix else 0 <= hour < 24):
        raise ValueError("clock time outside valid range")
    if suffix:
        hour = hour % 12 + (12 if suffix == "pm" else 0)
    return hour * 60 + minute


def _duration(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError("duration must be nonnegative numeric data")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("duration must be finite")


def _validate_reference(reference: dict) -> None:
    _duration(reference["duration"])
    low = _time_to_minutes(reference["min_time"])
    high = _time_to_minutes(reference["max_time"])
    if low > high:
        raise ValueError("calendar window is reversed")
    constraint = reference["constraint"]
    if constraint is None:
        return
    if not isinstance(constraint, str):
        raise ValueError("invalid calendar constraint")
    for prefix in ("before ", "after ", "at "):
        if constraint.startswith(prefix):
            _time_to_minutes(constraint.removeprefix(prefix))
            return
    if constraint.startswith("between "):
        lower, upper = constraint.removeprefix("between ").split(" and ")
        if _time_to_minutes(lower) <= _time_to_minutes(upper):
            return
    raise ValueError("invalid calendar constraint")


def _conflicts(events: list[dict[str, Any]], event: dict[str, Any]) -> bool:
    start = _time_to_minutes(event["start_time"])
    end = start + event["duration"]
    for other in events:
        if other is event:
            continue
        other_start = _time_to_minutes(other["start_time"])
        other_end = other_start + other["duration"]
        if not (end <= other_start or start >= other_end):
            return True
    return False


def _satisfies(event: dict[str, Any], expected: dict[str, Any]) -> bool:
    if event["duration"] != expected["duration"]:
        return False
    start = _time_to_minutes(event["start_time"])
    end = start + event["duration"]
    if start < _time_to_minutes(expected["min_time"]) or end > _time_to_minutes(expected["max_time"]):
        return False
    constraint = expected["constraint"]
    if constraint is None:
        return True
    if constraint.startswith("before "):
        return end <= _time_to_minutes(constraint.removeprefix("before "))
    if constraint.startswith("after "):
        return start >= _time_to_minutes(constraint.removeprefix("after "))
    if constraint.startswith("between "):
        low, high = constraint.removeprefix("between ").split(" and ")
        return start >= _time_to_minutes(low) and end <= _time_to_minutes(high)
    if constraint.startswith("at "):
        return start == _time_to_minutes(constraint.removeprefix("at "))
    raise RuntimeError(f"Unknown calendar constraint: {constraint!r}")


def _check(events: object, expected: dict, think: bool) -> dict:
    # Validate trusted constraints before interpreting candidate evidence.
    try:
        if not isinstance(expected, dict):
            raise ValueError("calendar references must be an object")
        for reference in expected.values():
            _validate_reference(reference)
    except (KeyError, TypeError, ValueError, AttributeError, RuntimeError):
        return {"status": "invalid_task", "reward": 0.0, "detail": {"reason": "invalid_calendar_reference"}}
    reason = "pass"
    try:
        if think:
            reason = "think_found"
        elif not expected:
            pass
        elif not events:
            reason = "no_json_list"
        else:
            for event in events:
                _duration(event["duration"])
                _time_to_minutes(event["start_time"])
            by_id = {str(event["event_id"]): event for event in events}
            if len(by_id) != len(expected):
                reason = "different_number_of_events"
            elif any(_conflicts(events, event) for event in events):
                reason = "conflicting_events"
            elif any(not _satisfies(by_id[event_id], expected[event_id]) for event_id in expected):
                reason = "constraint_violated"
    except (KeyError, TypeError, ValueError, AttributeError):
        reason = "invalid_response"
    return {"status": "scored", "reward": float(reason == "pass"), "detail": {"reason": reason}}


def _grade_calendar_verifyit(response: str, expected: dict) -> tuple[float, str]:
    from tempfile import TemporaryDirectory

    from verifyit.grade import Status, run
    from verifyit.spec import ScriptSpec, render_spec
    from skyrl_gym.envs.nemotron_ultra.calendar import _extract_json_list

    with TemporaryDirectory(prefix="skyrl-calendar-") as directory:
        root = Path(directory)
        (root / "checker.py").write_text(Path(__file__).read_text())
        events = _extract_json_list(response)
        (root / "data.json").write_text(
            json.dumps({"events": events, "expected": expected, "think": "<think>" in response}, allow_nan=False)
        )
        (root / "verifier.toml").write_text(
            render_spec(ScriptSpec(path="checker.py", verdict_file="calendar-result.json"))
        )
        verdict = run(root / "verifier.toml", root)
    if verdict.status is not Status.SCORED:
        raise RuntimeError("calendar verification failed")
    return verdict.reward, verdict.detail["reason"]


def grade_calendar_verifyit(response: str, expected: dict) -> tuple[float, str]:
    try:
        return _grade_calendar_verifyit(response, expected)
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError("calendar verification failed") from error


if __name__ == "__main__":
    tests = Path(os.environ["VERIFYIT_TESTS_DIR"])
    logs = Path(os.environ["VERIFYIT_LOGS_DIR"])
    data = json.loads((tests / "data.json").read_text())
    verdict = _check(data["events"], data["expected"], data["think"])
    (logs / "calendar-result.json").write_text(json.dumps(verdict, allow_nan=False))
