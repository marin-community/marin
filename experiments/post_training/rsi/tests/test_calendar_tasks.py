# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A generated calendar entry is valid only when the scorer that grades it can accept some schedule."""

import pytest

pytest.importorskip("taskforge")

import base64
import json
import time
from pathlib import Path

from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)

from experiments.post_training.rsi.calendar_tasks import (
    SCORER_PATH,
    CalendarScorer,
    calendar_entry,
    calendar_task,
    schedule,
    valid_calendar,
)

SCORER = Path(__file__).parents[4] / "vendor/skyrl/skyrl-gym" / SCORER_PATH
STATE = {
    "0": {"event_id": 0, "duration": 60, "constraint": "before 11:00am", "min_time": "10:00", "max_time": "16:00"},
    "1": {"event_id": 1, "duration": 45, "constraint": "at 11:15am", "min_time": "10:00", "max_time": "16:00"},
    "2": {"event_id": 2, "duration": 30, "constraint": None, "min_time": "10:00", "max_time": "16:00"},
}
MESSAGES = [
    {"role": "system", "content": "Keep my calendar."},
    {"role": "user", "content": "Add Standup for 60 minutes, done before 11am (event id: 0)."},
    {"role": "assistant", "content": "[]"},
    {"role": "user", "content": "Add Review, 45 minutes at 11:15am (event id: 1), and Lunch, 30 minutes (event id: 2)."},
]


def inline(path: str, content: str) -> dict:
    return {
        "path": path,
        "source": {"kind": "inline_file", "content_base64": base64.b64encode(content.encode()).decode()},
    }


def calendar_template(state: dict) -> TaskSpec:
    """A calendar task in the seed's form: its conversation, its scorer and its grader contract."""
    spec = TaskSpec(
        id="calendar-0",
        context=ConversationInput(events=tuple(TextMessage(role=m["role"], content=m["content"]) for m in MESSAGES)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="test"),
        source=Source(dataset="test", revision="1", row="calendar-0", importer_revision="1"),
    )
    contract = {"contract": {"dataset": "ultra_sft_step3200_calendar_v2", "exp_cal_state": state, "uuid": "u"}}
    verifier = [inline(SCORER_PATH, SCORER.read_text()), inline("config.json", json.dumps(contract))]
    return TaskSpec.model_validate({**spec.model_dump(), "resources": {"verifier": verifier}})


@pytest.fixture
def scorer() -> CalendarScorer:
    return CalendarScorer.of(calendar_template(STATE))


def test_a_solvable_entry_is_valid_and_its_schedule_passes_the_scorer(scorer):
    entry = calendar_entry(calendar_template(STATE))
    assert entry == {"messages": MESSAGES, "exp_cal_state": STATE}
    assert valid_calendar(entry, scorer)
    solution = schedule(STATE, scorer)
    assert scorer.reward(json.dumps(solution), STATE) == 1.0


def test_an_entry_without_a_schedule_or_naming_no_event_is_invalid(scorer):
    # Event 0 must start at 10:00 or later and end by 11:00, so it has no room once it needs 70 minutes.
    crowded = {**STATE, "0": {**STATE["0"], "duration": 70}}
    assert schedule(crowded, scorer) is None
    assert not valid_calendar({"messages": MESSAGES, "exp_cal_state": crowded}, scorer)
    unnamed = {**STATE, "3": {**STATE["2"], "event_id": 3}}
    assert not valid_calendar({"messages": MESSAGES, "exp_cal_state": unnamed}, scorer)
    ends_with_reply = [*MESSAGES, {"role": "assistant", "content": "[]"}]
    assert not valid_calendar({"messages": ends_with_reply, "exp_cal_state": STATE}, scorer)


def test_a_generated_task_keeps_the_template_form_with_the_new_state():
    moved = {**STATE, "2": {**STATE["2"], "constraint": "after 2:00pm"}}
    task = calendar_task(
        calendar_template(STATE),
        "calendar.generator-0",
        {"messages": MESSAGES, "exp_cal_state": moved},
        "capability:x",
        ("synth",),
    )
    assert task.id == "calendar.generator-0"
    assert task.tags == ("synth", "capability:x")
    assert calendar_entry(task) == {"messages": MESSAGES, "exp_cal_state": moved}


def test_a_constraint_that_the_user_never_asked_for_is_invalid(scorer):
    # The state says 11:15, the conversation asks for 11:45: a program bug that would train a wrong answer.
    unasked = {**STATE, "1": {**STATE["1"], "constraint": "at 11:45am"}}
    assert schedule(unasked, scorer) is not None
    assert not valid_calendar({"messages": MESSAGES, "exp_cal_state": unasked}, scorer)
    noon = {**STATE, "2": {**STATE["2"], "constraint": "after 12pm"}}
    by_noon = [*MESSAGES[:-1], {"role": "user", "content": MESSAGES[-1]["content"] + " Lunch starts at noon or later."}]
    assert valid_calendar({"messages": by_noon, "exp_cal_state": noon}, scorer)


def test_an_unschedulable_entry_is_rejected_quickly(scorer):
    # Nine 50-minute events cannot fit in six hours; an exhaustive search would try every arrangement.
    crowded = {
        str(k): {"event_id": k, "duration": 50, "constraint": None, "min_time": "10:00", "max_time": "16:00"}
        for k in range(9)
    }
    started = time.monotonic()
    assert schedule(crowded, scorer) is None
    assert time.monotonic() - started < 5
