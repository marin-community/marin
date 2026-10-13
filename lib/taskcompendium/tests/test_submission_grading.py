# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed submissions and in-process grading without an execution provider."""

import json

import pytest
from verifyit.json_comparison import NumericTypePolicy
from verifyit.spec import ExactSpec, McqSpec, NumericSpec, ScriptSpec, Spec, StructuredExactSpec

from taskcompendium.grader import verifyit_package
from taskcompendium.grading import grade_answer
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    ConversationTrace,
    EnvironmentRequirements,
    Grader,
    GradingAttempt,
    JsonAnswer,
    JsonValueAnswer,
    NoGrader,
    PlainText,
    Source,
    StateSubmission,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.models import RuntimeEvidence, grading_attempt
from taskcompendium.runtime.task_grading import grade_task
from taskcompendium.submission import submission_compatibility

GRADING_ENVIRONMENT = EnvironmentRequirements(
    docker_image="private/grader@sha256:" + "a" * 64, command_semantics=CommandSemantics.LINUX_PROCESS
)


def _task(grader: Grader, answer_type=AnswerType.TEXT, answer_format=PlainText()) -> TaskSpec:
    return TaskSpec(
        id="submission-grading",
        context=ConversationInput(events=(TextMessage(role="user", content="Complete the task."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=answer_type,
        answer_format=answer_format,
        grader=grader,
        source=Source(dataset="test", revision="1", row="0", importer_revision="1"),
    )


def _grader(spec: Spec) -> Grader:
    return verifyit_package(spec).grader


def _conversation(task: TaskSpec, content: str) -> ConversationTrace:
    return ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=content)))


def _attempt(task, content, state=None):
    return GradingAttempt(conversation=_conversation(task, content), state=state)


@pytest.mark.parametrize(
    "actual,reward",
    [
        ({"nested": {"right": [1, True, "x"], "left": None}}, 1.0),
        ({"nested": {"right": [True, True, "x"], "left": None}}, 0.0),
        ({"nested": {"right": [True, 1, "x"], "left": None}}, 0.0),
        ({"nested": {"right": [1, True, "x", 2], "left": None}}, 0.0),
    ],
)
def test_structured_exact_compares_json_types_and_order(actual, reward):
    task = _task(
        _grader(StructuredExactSpec(expected={"nested": {"left": None, "right": [1, True, "x"]}})),
        AnswerType.JSON,
        JsonValueAnswer(),
    )
    result = grade_answer(task, _attempt(task, json.dumps(actual)))
    assert (result.status, result.reward) == (Outcome.GRADED, reward)


@pytest.mark.parametrize(
    "actual,policy,reward",
    [
        ({"value": 16.0}, NumericTypePolicy.VALUE, 1.0),
        ({"value": 17}, NumericTypePolicy.VALUE, 0.0),
        ({"value": 16.0}, NumericTypePolicy.STRICT, 0.0),
    ],
)
def test_json_answer_and_captured_state_share_structured_grading(actual, policy, reward):
    grader = _grader(StructuredExactSpec(expected={"value": 16}, numeric_types=policy))
    chat_task = TaskSpec.model_validate_json(_task(grader, AnswerType.JSON, JsonValueAnswer()).model_dump_json())
    state_task = TaskSpec.model_validate_json(_task(grader, AnswerType.STATE).model_dump_json())
    chat_result = grade_answer(chat_task, _attempt(chat_task, json.dumps(actual)))
    state_result = grade_answer(state_task, _attempt(state_task, "Done.", StateSubmission(actual)))
    assert (chat_result.status, chat_result.reward) == (Outcome.GRADED, reward)
    assert state_result == chat_result


@pytest.mark.parametrize(
    "actual,status,reward",
    [("yes", Outcome.GRADED, 1.0), ("no", Outcome.GRADED, 0.0), (12, Outcome.SUBMISSION_FAILURE, 0.0)],
)
def test_exact_grader_accepts_string_json_and_captured_state(actual, status, reward):
    grader = _grader(ExactSpec(expected=("yes",)))
    chat_task = _task(grader, AnswerType.JSON, JsonValueAnswer())
    state_task = _task(grader, AnswerType.STATE)
    chat_result = grade_answer(chat_task, _attempt(chat_task, json.dumps(actual)))
    state_result = grade_answer(state_task, _attempt(state_task, "Done.", StateSubmission(actual)))
    assert (chat_result.status, chat_result.reward) == (status, reward)
    assert state_result == chat_result


@pytest.mark.parametrize(
    "mode,spec,answer_type,correct,wrong",
    [
        ("exact", ExactSpec(expected=("yes",)), AnswerType.TEXT, "yes", "no"),
        ("numeric", NumericSpec("12.0", tolerance_abs=0.0, tolerance_rel=0.0), AnswerType.NUMBER, "12", "13"),
        ("mcq", McqSpec(expected="B", options=3), AnswerType.TEXT, "B", "A"),
    ],
)
def test_verifyit_modes_serialize_load_and_grade(mode, spec, answer_type, correct, wrong):
    serialized = json.loads(_task(_grader(spec), answer_type).model_dump_json())
    assert (serialized["grader"]["kind"], serialized["grader"]["mode"]) == ("verifyit", mode)
    loaded = TaskSpec.model_validate_json(json.dumps(serialized))
    correct_result = grade_answer(loaded, _attempt(loaded, correct))
    wrong_result = grade_answer(loaded, _attempt(loaded, wrong))
    assert (correct_result.status, correct_result.reward) == (Outcome.GRADED, 1.0)
    assert (wrong_result.status, wrong_result.reward) == (Outcome.GRADED, 0.0)


@pytest.mark.parametrize("content", ['{"answer":"yes","answer":"no"}', '{"answer":"yes","meta":{"x":1,"x":2}}'])
def test_duplicate_json_answer_is_submission_failure(content):
    task = _task(_grader(ExactSpec(expected=("yes",))), answer_format=JsonAnswer())
    result = grade_answer(task, _attempt(task, content))
    assert (result.status, result.reward) == (Outcome.SUBMISSION_FAILURE, 0.0)


def test_captured_null_state_is_a_submission_but_missing_state_is_not():
    task = _task(_grader(StructuredExactSpec(expected=None)), AnswerType.STATE)
    missing = grade_answer(task, _attempt(task, "Done."))
    captured = grade_answer(task, _attempt(task, "Done.", StateSubmission(None)))
    assert (missing.status, missing.reward) == (Outcome.SUBMISSION_FAILURE, 0.0)
    assert (captured.status, captured.reward) == (Outcome.GRADED, 1.0)


@pytest.mark.parametrize("state,reward", [("null", 1.0), ('{"value": 1}', 0.0)])
def test_runtime_passes_captured_state_to_in_process_grader(state, reward):
    task = _task(_grader(StructuredExactSpec(expected=None)), AnswerType.STATE)
    attempt = grading_attempt(_conversation(task, "unused"), RuntimeEvidence({}, state))
    result = grade_task(task, attempt)
    assert (result.status, result.reward) == (Outcome.GRADED, reward)


@pytest.mark.parametrize("state", ['{"x":1,"x":2}', "NaN", "1e1000", "{"])
def test_invalid_captured_state_never_becomes_a_submission(state):
    task = _task(_grader(StructuredExactSpec(expected=None)), AnswerType.STATE)
    with pytest.raises(ValueError):
        grading_attempt(_conversation(task, "unused"), RuntimeEvidence({}, state))


@pytest.mark.parametrize(
    "spec",
    [
        ExactSpec(expected=("done",)),
        NumericSpec("12", tolerance_abs=0.0, tolerance_rel=0.0),
        ScriptSpec(path="grade.py"),
    ],
)
def test_sandbox_grader_never_awards_in_process_credit(spec):
    task = _task(verifyit_package(spec, environment=GRADING_ENVIRONMENT).grader)
    task = TaskSpec.model_validate_json(task.model_dump_json())
    attempt = _attempt(task, "done")
    # The answer format can carry the answer, but only the grading environment may score it.
    assert submission_compatibility(task).compatible
    with pytest.raises(TypeError):
        grade_answer(task, attempt)
    result = grade_task(task, attempt)
    assert (result.status, result.reward) == (Outcome.INFRA_ERROR, None)


def test_task_without_a_grader_is_unavailable_with_its_reason():
    task = _task(NoGrader(reason="Source evaluator is unavailable", contract={"evaluator": "llm_judge"}))
    task = TaskSpec.model_validate_json(task.model_dump_json())
    attempt = _attempt(task, "done")
    with pytest.raises(TypeError):
        grade_answer(task, attempt)
    result = grade_task(task, attempt)
    assert (result.status, result.reward, result.error) == (Outcome.UNAVAILABLE, None, "Source evaluator is unavailable")
