# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Calendar tasks of the Snowball seed as generator entries, for the generator box.

A calendar task is a conversation in which the user adds events and constraints, graded by NeMo Gym's calendar scorer
against the ``exp_cal_state`` of its grader contract: the events of the final reply must not overlap, and each must
keep its duration, its time window and its constraint. An entry is the conversation and that state. A generated entry
is valid when the conversation names each event and a schedule that the scorer accepts exists. The scorer is the one
that a calendar task ships with its grader, so a generated task is checked by the code that grades it.
"""

import base64
import json
import re
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from taskcompendium.models import TaskSpec

from experiments.post_training.rsi.gap import CONFIG_PATH, contract_resource
from experiments.post_training.rsi.seed import host_graded

CALENDAR = "calendar"
CALENDAR_DATASET = "calendar"
SCORER_PATH = "skyrl_gym/envs/nemotron_ultra/calendar.py"
SLOT_MINUTES = 5
"""The granularity at which ``schedule`` searches start times."""
ROLES = ("system", "user", "assistant")


def _verifier_file(payload: dict, path: str) -> str | None:
    file = next((file for file in payload.get("resources", {}).get("verifier") or () if file["path"] == path), None)
    return base64.b64decode(file["source"]["content_base64"]).decode() if file is not None else None


def calendar_entry(task: TaskSpec) -> dict | None:
    """The conversation and expected calendar state of a calendar task; ``None`` for any other task."""
    payload = json.loads(task.model_dump_json())
    if _verifier_file(payload, CONFIG_PATH) is None:
        return None
    terms = json.loads(base64.b64decode(contract_resource(payload)["source"]["content_base64"])).get("contract") or {}
    if CALENDAR_DATASET not in str(terms.get("dataset")) or "exp_cal_state" not in terms:
        return None
    messages = [{"role": event["role"], "content": event["content"]} for event in payload["context"]["events"]]
    return {"messages": messages, "exp_cal_state": terms["exp_cal_state"]}


@dataclass(frozen=True)
class CalendarScorer:
    """The calendar scorer that a calendar task ships with its grader."""

    module: types.ModuleType

    @classmethod
    def of(cls, task: TaskSpec) -> "CalendarScorer":
        source = _verifier_file(json.loads(task.model_dump_json()), SCORER_PATH)
        if source is None:
            raise ValueError(f"{task.id} ships no calendar scorer")
        module = types.ModuleType("calendar_scorer")
        exec(compile(source, SCORER_PATH, "exec"), module.__dict__)
        return cls(module)

    def minutes(self, value: str) -> int:
        return self.module._time_to_minutes(value)

    def satisfies(self, event: dict, expected: dict) -> bool:
        return self.module._satisfies(event, expected)

    def reward(self, reply: str, expected: Mapping[str, dict]) -> float:
        return self.module.grade_calendar(reply, dict(expected))[0]


def schedule(expected: Mapping[str, dict], scorer: CalendarScorer) -> list[dict] | None:
    """A schedule of the expected events that keeps each one's window and constraint, without overlaps, found by
    backtracking over start times; ``None`` when there is none."""
    events = sorted(expected.values(), key=lambda event: event["duration"], reverse=True)
    placed: list[tuple[int, int, dict]] = []

    def place(index: int) -> bool:
        if index == len(events):
            return True
        event = events[index]
        low, high = scorer.minutes(event["min_time"]), scorer.minutes(event["max_time"])
        for start in range(low, high - event["duration"] + 1, SLOT_MINUTES):
            end = start + event["duration"]
            if any(start < other_end and other_start < end for other_start, other_end, _ in placed):
                continue
            entry = {"event_id": event["event_id"], "start_time": f"{start // 60:02d}:{start % 60:02d}"}
            entry["duration"] = event["duration"]
            if not scorer.satisfies(entry, event):
                continue
            placed.append((start, end, entry))
            if place(index + 1):
                return True
            placed.pop()
        return False

    return [entry for _, _, entry in placed] if place(0) else None


def valid_calendar(entry: Mapping, scorer: CalendarScorer) -> bool:
    """Whether a generated entry is a conversation that ends with a user turn and names every expected event with its
    duration, and whose expected state has a schedule that the calendar scorer accepts."""
    try:
        messages, expected = entry["messages"], entry["exp_cal_state"]
        if not messages or messages[-1]["role"] != "user" or any(m["role"] not in ROLES for m in messages):
            return False
        text = " ".join(str(m["content"]) for m in messages)
        for event_id, event in expected.items():
            if str(event["event_id"]) != str(event_id) or not re.search(rf"\b{event['duration']}\b", text):
                return False
            if not re.search(rf"event[ _-]?id:?\s*{event_id}\b", text, re.IGNORECASE):
                return False
        solution = schedule(expected, scorer)
        return solution is not None and scorer.reward(json.dumps(solution), expected) == 1.0
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError):
        # A malformed entry, including a constraint that the scorer does not know, is an invalid instance.
        return False


def calendar_task(template: TaskSpec, task_id: str, entry: Mapping, capability: str, tags: Sequence[str]) -> TaskSpec:
    """A task in the form of the calendar task ``template``, with the conversation and expected state of ``entry``."""
    task = json.loads(template.model_dump_json())
    resource = contract_resource(task)
    config = json.loads(base64.b64decode(resource["source"]["content_base64"]))
    config["contract"]["exp_cal_state"] = dict(entry["exp_cal_state"])
    for key in ("pass_rate", "pass_rate_passed", "pass_rate_total", "uuid"):
        config["contract"].pop(key, None)
    resource["source"]["content_base64"] = base64.b64encode(json.dumps(config).encode()).decode()
    shape = task["context"]["events"][0]
    task["context"]["events"] = [{**shape, "role": m["role"], "content": m["content"]} for m in entry["messages"]]
    task["id"] = task_id
    task["tags"] = [*tags, capability]
    return host_graded(TaskSpec.model_validate(task))
