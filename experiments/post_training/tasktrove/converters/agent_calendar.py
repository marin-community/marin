# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Calendar tool-use tasks graded on final state.

The agent writes its final calendar to ``/app/answer.txt`` as a JSON list of events; the old
grader (``tests/verifier.py``) checked event ids, names, durations, per-event time windows,
natural-language constraints (``"before 11am"``, ``"after 2pm"``, ``"at 11:15am"``,
``"between 10am and 1pm"``) and pairwise overlap. No built-in mode covers that shape, so this maps
onto :class:`ScriptSpec`: a new, self-contained checker under ``tests/`` reimplements the same
final-state contract without importing the old grader.
"""

import json

from tasktrove_verify.spec import ScriptSpec

from experiments.post_training.tasktrove.converters.converted_task import (
    ConvertedTask,
    Converter,
    ConverterKey,
    ConvertStatus,
    Rejected,
)
from experiments.post_training.tasktrove.converters.nemotron_data import verifier_data
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, INSTRUCTION, SOLUTION_DIR, TaskFiles

CHECKER_NAME = "agent_calendar_checker.py"
DATA_NAME = "expected_events.json"

CHECKER_PY = (
    '''\
#!/usr/bin/env python3
"""Score a final calendar state against one task's expected events.

Reads the expected events from ``expected_events.json`` beside this script (under
``$TASKTROVE_TESTS_DIR``), the agent's calendar from ``$TASKTROVE_WORKSPACE/answer.txt``, and
reports the reward through ``$TASKTROVE_LOGS_DIR/reward.json``. Self-contained: it does not import
the original dataset's grader.
"""

import json
import os
import re
import sys
import unicodedata
from pathlib import Path

_TIME_RE = re.compile(r"(\\d{2}):(\\d{2})")
_CLOCK = r"(\\d{1,2})(?::(\\d{2}))?\\s*(am|pm)?"
_FENCE_RE = re.compile(r"```(?:json)?\\s*(.*?)```", re.DOTALL)


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
    match = re.fullmatch(rf"before\\s+{_CLOCK}", text)
    if match is not None:
        limit = _clock_minutes(*match.groups())
        return limit is not None and end <= limit
    match = re.fullmatch(rf"after\\s+{_CLOCK}", text)
    if match is not None:
        limit = _clock_minutes(*match.groups())
        return limit is not None and start >= limit
    match = re.fullmatch(rf"at\\s+{_CLOCK}", text)
    if match is not None:
        exact = _clock_minutes(*match.groups())
        return exact is not None and start == exact
    match = re.fullmatch(rf"between\\s+{_CLOCK}\\s+and\\s+{_CLOCK}", text)
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
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1]:
            errors.append(f"events {previous[2]} and {current[2]} overlap")
    return (1 if not errors else 0), errors


def _extract_json(raw: str) -> object:
    fence = _FENCE_RE.search(raw)
    return json.loads(fence.group(1) if fence else raw)


def main() -> int:
    tests_dir = Path(os.environ["TASKTROVE_TESTS_DIR"])
    workspace = Path(os.environ["TASKTROVE_WORKSPACE"])
    logs_dir = Path(os.environ["TASKTROVE_LOGS_DIR"])
    expected = json.loads((tests_dir / "'''
    + DATA_NAME
    + """").read_text())
    answer_path = workspace / "answer.txt"

    if not answer_path.exists():
        reward, errors = 0, ["answer.txt missing"]
    else:
        try:
            events = _extract_json(answer_path.read_text(errors="replace"))
        except json.JSONDecodeError as error:
            reward, errors = 0, [f"answer parse error: {error}"]
        else:
            reward, errors = _score(expected, events)

    for error in errors:
        print(error, file=sys.stderr)
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "reward.json").write_text(json.dumps({"reward": reward}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
)


def _expected_events(data: dict) -> dict | None:
    events = data.get("expected_events")
    if not isinstance(events, dict) or not events:
        return None
    for key, spec in events.items():
        if not isinstance(spec, dict):
            return None
        try:
            int(key)
        except (TypeError, ValueError):
            return None
        if not {"event_name", "duration", "min_time", "max_time"} <= spec.keys():
            return None
    return events


def convert_agent_calendar(task: TaskFiles) -> ConvertedTask | Rejected:
    """Calendar scheduling: ``{"expected_events": {"<id>": {event_name, duration, min_time, max_time,
    constraint}, ...}}``. Graded by a from-scratch checker script, not the old grader."""
    data = verifier_data(task)
    expected = _expected_events(data)
    if expected is None:
        return Rejected(ConvertStatus.NULL_GRADER, "expected_events is missing, empty, or malformed")
    return ConvertedTask(
        instruction=task.text(INSTRUCTION),
        spec=ScriptSpec(path=CHECKER_NAME),
        dockerfile=task.text(DOCKERFILE),
        tags=("tool-use", "calendar", "scheduling", "state-tracking", "nemotron"),
        data_files={
            f"tests/{CHECKER_NAME}": CHECKER_PY.encode(),
            f"tests/{DATA_NAME}": json.dumps(expected).encode(),
        },
        solution_files=task.under(SOLUTION_DIR),
    )


CONVERTER = Converter(
    name="agent_calendar",
    keys=(ConverterKey("tool-use", frozenset({"tests/test.sh", "tests/verifier.py"})),),
    convert=convert_agent_calendar,
)
