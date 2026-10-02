# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed submissions and private grading without an execution provider."""

import json
from typing import Literal

import pytest
from pydantic import PrivateAttr, TypeAdapter

from taskcompendium.grading import Outcome, exact_answer, numeric_answer, structured_exact
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    ConversationTrace,
    EnvironmentRequirements,
    Source,
    TaskSpec,
    TextMessage,
    VerifierKind,
    VerifierSpec,
)
from taskcompendium.submission import (
    AnswerFormat,
    GradingAttempt,
    JsonAnswer,
    PlainText,
    StateSubmission,
    SubmissionConvention,
    TextSubmission,
)
from taskcompendium.verifier_registry import grade_answer, resolve_verifier
from taskcompendium.verifiers.multiple_choice import multiple_choice_answer


def _task(verifier, answer_type=AnswerType.TEXT):
    return TaskSpec(
        id="submission-grading",
        context=ConversationInput(events=(TextMessage(role="user", content="Complete the task."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=answer_type,
        verifier=verifier,
        source=Source(dataset="test", revision="1", row="0", importer_revision="1"),
    )


def _attempt(task, content):
    return GradingAttempt(
        conversation=ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=content))),
        workspace=object(),
    )


@pytest.mark.parametrize(
    "actual,reward",
    [
        ({"nested": {"right": [1, True, "x"], "left": None}}, 1.0),
        ({"nested": {"right": [True, True, "x"], "left": None}}, 0.0),
        ({"nested": {"right": [True, 1, "x"], "left": None}}, 0.0),
        ({"nested": {"right": [1, True, "x", 2], "left": None}}, 0.0),
    ],
)
async def test_structured_exact_compares_json_types_and_order(actual, reward):
    task = _task(structured_exact({"nested": {"left": None, "right": [1, True, "x"]}}), AnswerType.STATE)
    result = await resolve_verifier(task.verifier).grade(StateSubmission(actual), attempt=_attempt(task, "Done."))
    assert (result.status, result.reward) == (Outcome.GRADED, reward)


class ChangingText(PlainText):
    answer_format: Literal[AnswerFormat.PLAIN] = AnswerFormat.PLAIN
    _submitted: bool = PrivateAttr(default=False)

    async def extract(self, attempt: GradingAttempt) -> TextSubmission:
        value = "different" if self._submitted else "first"
        self._submitted = True
        return TextSubmission(value)


async def test_verifier_grades_the_single_extracted_submission():
    task = _task(exact_answer("first"))
    result = await grade_answer(task, ChangingText(id="changing"), _attempt(task, "Done."))
    assert (result.status, result.reward) == (Outcome.GRADED, 1.0)


async def test_serialized_json_convention_extracts_answer_and_scores_invalid_submission():
    task = _task(exact_answer("yes"))
    convention = TypeAdapter(SubmissionConvention).validate_json(JsonAnswer(id="json").model_dump_json())
    valid = await grade_answer(task, convention, _attempt(task, '{"answer":"yes"}'))
    invalid = await grade_answer(task, convention, _attempt(task, '{"answer":'))
    assert (valid.status, valid.reward) == (Outcome.GRADED, 1.0)
    assert (invalid.status, invalid.reward) == (Outcome.SUBMISSION_FAILURE, 0.0)


async def test_invalid_private_verifier_is_not_scored_as_agent_failure():
    task = _task(VerifierSpec(kind=VerifierKind.EXACT_ANSWER, parameters_json="{}"))
    with pytest.raises(ValueError, match="Invalid 'exact' verifier parameters"):
        await grade_answer(task, JsonAnswer(id="json"), _attempt(task, '{"answer":'))


@pytest.mark.parametrize(
    "wire_kind,verifier,answer_type,correct,wrong",
    [
        ("exact", exact_answer("yes"), AnswerType.TEXT, "yes", "no"),
        ("numeric", numeric_answer(12.0, tolerance_abs=0.0, tolerance_rel=0.0), AnswerType.NUMBER, "12", "13"),
        ("mcq", multiple_choice_answer("B", 3), AnswerType.TEXT, "B", "A"),
    ],
)
async def test_schema_canonical_verifier_kinds_serialize_load_and_grade(
    wire_kind, verifier, answer_type, correct, wrong
):
    serialized = json.loads(_task(verifier, answer_type).model_dump_json())
    assert serialized["verifier"]["kind"] == wire_kind
    loaded = TaskSpec.model_validate_json(json.dumps(serialized))
    convention = PlainText(id="plain")
    correct_result = await grade_answer(loaded, convention, _attempt(loaded, correct))
    wrong_result = await grade_answer(loaded, convention, _attempt(loaded, wrong))
    assert (correct_result.status, correct_result.reward) == (Outcome.GRADED, 1.0)
    assert (wrong_result.status, wrong_result.reward) == (Outcome.GRADED, 0.0)
