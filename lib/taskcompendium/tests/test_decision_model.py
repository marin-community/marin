# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decision-model graders: the declared kind and grading through a caller's read."""

import json

import pytest
from pydantic import TypeAdapter, ValidationError

from taskcompendium.decision_model import DecisionModelConnection, DecisionModelRequest, grade_decision_model_answer
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    DECISION_LEVELS,
    DECISION_QUESTION,
    AnswerType,
    ConversationInput,
    ConversationTrace,
    DecisionModelGrader,
    EnvironmentRequirements,
    Grader,
    GradingAttempt,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.task_grading import grade_task

MODEL = "decision-model@fixture-3"
GRADER = DecisionModelGrader(
    deliverable="The boiling point of water at sea level.", rubric="Celsius or Kelvin.", reference="100 C", cut=3.0
)


def _task(grader: Grader = GRADER) -> TaskSpec:
    return TaskSpec(
        id="decision-model",
        context=ConversationInput(events=(TextMessage(role="user", content="At what temperature does water boil?"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=grader,
        source=Source(dataset="test", revision="1", row="0", importer_revision="1"),
    )


def _attempt(task: TaskSpec, answer: str) -> GradingAttempt:
    return GradingAttempt(
        ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=answer)))
    )


def _connection(level, requests: list[DecisionModelRequest] | None = None) -> DecisionModelConnection:
    async def read(request: DecisionModelRequest) -> float:
        if requests is not None:
            requests.append(request)
        if isinstance(level, Exception):
            raise level
        return level

    return DecisionModelConnection(MODEL, read)


def test_a_decision_model_task_round_trips_through_json():
    task = _task()
    restored = TaskSpec.model_validate_json(task.model_dump_json())
    assert restored == task
    assert json.loads(task.model_dump_json())["grader"]["kind"] == "decision_model"
    assert "decision_model" in json.dumps(TaskSpec.model_json_schema())


@pytest.mark.parametrize("cut", [1.5, 3.75, float("nan")])
def test_a_cut_outside_the_level_band_is_rejected(cut):
    with pytest.raises(ValidationError):
        TypeAdapter(Grader).validate_python({**GRADER.model_dump(), "cut": cut})


@pytest.mark.parametrize(
    "level,reward",
    [(0, 0.0), (1, 0.0), (2, 0.0), (2.99, 0.0), (3, 1.0), (3.4, 1.0), (4, 1.0)],
)
async def test_the_level_maps_through_the_authors_cut(level, reward):
    task = _task()
    grade = await grade_decision_model_answer(task, _attempt(task, "212 F, which is 100 C."), _connection(level))
    assert (grade.status, grade.reward) == (Outcome.GRADED, reward)
    assert grade.detail == {"decision_model": {"model": MODEL, "level": level, "cut": 3.0}}


async def test_the_read_request_carries_exactly_what_identifies_the_read():
    task = _task()
    requests: list[DecisionModelRequest] = []
    await grade_decision_model_answer(task, _attempt(task, "100 degrees Celsius"), _connection(4, requests))
    assert requests == [
        DecisionModelRequest(
            model=MODEL,
            question=DECISION_QUESTION,
            levels=DECISION_LEVELS,
            deliverable=GRADER.deliverable,
            rubric=GRADER.rubric,
            reference=GRADER.reference,
            candidate="100 degrees Celsius",
        )
    ]


@pytest.mark.parametrize(
    "level,message",
    [
        pytest.param(5, "returned 5, not a level in 0..4", id="above"),
        pytest.param(-0.5, "returned -0.5, not a level in 0..4", id="below"),
        pytest.param(float("nan"), "returned nan, not a level in 0..4", id="nan"),
        pytest.param(None, "returned None, not a level in 0..4", id="missing"),
        pytest.param(True, "returned True, not a level in 0..4", id="bool"),
        pytest.param(ConnectionError("clef unavailable"), "read failed: ConnectionError: clef unavailable", id="raises"),
    ],
)
async def test_a_read_without_a_declared_level_is_an_infrastructure_error(level, message):
    task = _task()
    grade = await grade_decision_model_answer(task, _attempt(task, "100 C"), _connection(level))
    assert (grade.status, grade.reward) == (Outcome.INFRA_ERROR, None)
    assert grade.error == f"Decision model {MODEL!r} {message}"


async def test_a_blank_answer_fails_submission_without_a_read():
    task = _task()
    requests: list[DecisionModelRequest] = []
    grade = await grade_decision_model_answer(task, _attempt(task, "  "), _connection(4, requests))
    assert (grade.status, grade.reward, requests) == (Outcome.SUBMISSION_FAILURE, 0.0, [])


def test_grading_without_a_connection_is_refused():
    task = _task()
    with pytest.raises(TypeError, match="needs a connection"):
        grade_task(task, _attempt(task, "100 C"))
