# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed submissions and private grading without an execution provider."""

import json

import pytest
from pydantic import PrivateAttr
from verifyit.json_comparison import NumericTypePolicy
from verifyit.spec import Mode

from taskcompendium.grading import Outcome, exact_answer, grade_answer, numeric_answer, structured_exact
from taskcompendium.grading_contract import GradingAttempt, StateSubmission, SubmissionFailure, TextSubmission
from taskcompendium.models import (
    SCHEMA_VERSION,
    AnswerType,
    ConversationInput,
    ConversationTrace,
    EnvironmentRequirements,
    Source,
    TaskSpec,
    TextMessage,
    VerifierSpec,
)
from taskcompendium.runtime.models import RuntimeEvidence
from taskcompendium.runtime.task_grading import grade_task
from taskcompendium.submission import (
    JsonAnswer,
    JsonValueAnswer,
    PlainText,
    SubmissionConvention,
)
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


def _attempt(task, content, state=None):
    return GradingAttempt(
        conversation=ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=content))),
        state=state,
    )


class StateAnswer(SubmissionConvention):
    submission_types = (StateSubmission,)

    def supports(self, answer_type: AnswerType) -> bool:
        return answer_type == AnswerType.STATE

    def extract(self, attempt: GradingAttempt) -> StateSubmission:
        if attempt.state is None:
            raise SubmissionFailure("Missing captured state")
        return attempt.state


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
    task = _task(structured_exact({"nested": {"left": None, "right": [1, True, "x"]}}), AnswerType.STATE)
    result = grade_answer(task, StateAnswer(id="state"), _attempt(task, "Done.", StateSubmission(actual)))
    assert (result.status, result.reward) == (Outcome.GRADED, reward)


@pytest.mark.parametrize(
    "actual,policy,reward",
    [
        ({"value": 16.0}, NumericTypePolicy.VALUE, 1.0),
        ({"value": 17}, NumericTypePolicy.VALUE, 0.0),
        ({"value": 16.0}, NumericTypePolicy.STRICT, 0.0),
    ],
)
def test_json_answer_and_acquired_state_share_structured_grading(actual, policy, reward):
    verifier = structured_exact({"value": 16}, numeric_types=policy)
    chat_task = TaskSpec.model_validate_json(_task(verifier, AnswerType.JSON).model_dump_json())
    state_task = TaskSpec.model_validate_json(_task(verifier, AnswerType.STATE).model_dump_json())
    chat_result = grade_answer(chat_task, JsonValueAnswer(id="json-value"), _attempt(chat_task, json.dumps(actual)))
    state_result = grade_answer(
        state_task, StateAnswer(id="state"), _attempt(state_task, "Done.", StateSubmission(actual))
    )
    assert (chat_result.status, chat_result.reward) == (Outcome.GRADED, reward)
    assert state_result == chat_result


@pytest.mark.parametrize(
    "actual,status,reward",
    [("yes", Outcome.GRADED, 1.0), ("no", Outcome.GRADED, 0.0), (12, Outcome.SUBMISSION_FAILURE, 0.0)],
)
def test_exact_verifier_accepts_string_json_and_acquired_state(actual, status, reward):
    verifier = exact_answer("yes")
    chat_task = _task(verifier, AnswerType.JSON)
    state_task = _task(verifier, AnswerType.STATE)
    chat_result = grade_answer(chat_task, JsonValueAnswer(id="json-value"), _attempt(chat_task, json.dumps(actual)))
    state_result = grade_answer(
        state_task, StateAnswer(id="state"), _attempt(state_task, "Done.", StateSubmission(actual))
    )
    assert (chat_result.status, chat_result.reward) == (status, reward)
    assert state_result == chat_result


class ChangingText(PlainText):
    _submitted: bool = PrivateAttr(default=False)

    def extract(self, _attempt: GradingAttempt) -> TextSubmission:
        value = "different" if self._submitted else "first"
        self._submitted = True
        return TextSubmission(value)


def test_verifier_grades_the_single_extracted_submission():
    task = _task(exact_answer("first"))
    result = grade_answer(task, ChangingText(id="changing"), _attempt(task, "Done."))
    assert (result.status, result.reward) == (Outcome.GRADED, 1.0)


def test_serialized_json_convention_extracts_answer_and_scores_invalid_submission():
    task = _task(exact_answer("yes"))
    convention = JsonAnswer.model_validate_json(JsonAnswer(id="json").model_dump_json())
    valid = grade_answer(task, convention, _attempt(task, '{"answer":"yes"}'))
    invalid = grade_answer(task, convention, _attempt(task, '{"answer":'))
    assert (valid.status, valid.reward) == (Outcome.GRADED, 1.0)
    assert (invalid.status, invalid.reward) == (Outcome.SUBMISSION_FAILURE, 0.0)


def test_invalid_private_verifier_is_not_scored_as_agent_failure():
    task = _task(VerifierSpec(kind=Mode.EXACT, parameters_json="{}"))
    with pytest.raises(ValueError):
        grade_answer(task, JsonAnswer(id="json"), _attempt(task, '{"answer":'))


@pytest.mark.parametrize(
    "wire_kind,verifier,answer_type,correct,wrong",
    [
        ("exact", exact_answer("yes"), AnswerType.TEXT, "yes", "no"),
        ("numeric", numeric_answer("12.0", tolerance_abs=0.0, tolerance_rel=0.0), AnswerType.NUMBER, "12", "13"),
        ("mcq", multiple_choice_answer("B", 3), AnswerType.TEXT, "B", "A"),
    ],
)
def test_canonical_verifier_kinds_serialize_load_and_grade(wire_kind, verifier, answer_type, correct, wrong):
    serialized = json.loads(_task(verifier, answer_type).model_dump_json())
    assert serialized["verifier"]["kind"] == wire_kind
    assert serialized["schema_version"] == SCHEMA_VERSION
    loaded = TaskSpec.model_validate_json(json.dumps(serialized))
    convention = PlainText(id="plain")
    correct_result = grade_answer(loaded, convention, _attempt(loaded, correct))
    wrong_result = grade_answer(loaded, convention, _attempt(loaded, wrong))
    assert (correct_result.status, correct_result.reward) == (Outcome.GRADED, 1.0)
    assert (wrong_result.status, wrong_result.reward) == (Outcome.GRADED, 0.0)


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
def test_structured_exact_rejects_nonfinite_gold_before_serialization_but_preserves_null(number):
    with pytest.raises(ValueError):
        structured_exact({"nested": [number]})
    task = _task(structured_exact(None), AnswerType.STATE)
    result = grade_answer(task, StateAnswer(id="state"), _attempt(task, "Done.", StateSubmission(None)))
    assert (result.status, result.reward) == (Outcome.GRADED, 1.0)


@pytest.mark.parametrize("content", ['{"answer":"yes","answer":"no"}', '{"answer":"yes","meta":{"x":1,"x":2}}'])
def test_duplicate_json_answer_is_submission_failure(content):
    task = _task(exact_answer("yes"))
    result = grade_answer(task, JsonAnswer(id="json"), _attempt(task, content))
    assert (result.status, result.reward) == (Outcome.SUBMISSION_FAILURE, 0.0)


def test_invalid_private_reference_precedes_one_time_submission_acquisition():
    task = _task(
        VerifierSpec(kind="numeric", parameters_json='{"expected":"bad","tolerance_abs":"0","tolerance_rel":"0"}')
    )
    convention = ChangingText(id="changing")
    with pytest.raises(ValueError):
        grade_answer(task, convention, _attempt(task, "first"))
    valid_task = _task(exact_answer("first"))
    result = grade_answer(valid_task, convention, _attempt(valid_task, "unused"))
    assert (result.status, result.reward) == (Outcome.GRADED, 1.0)


@pytest.mark.parametrize("parameters", ['{"expected":"yes","expected":"no"}', '{"expected":{"key":1,"\\u006bey":2}}'])
def test_ambiguous_private_verifier_json_rejects_duplicate_keys(parameters):
    with pytest.raises(ValueError):
        VerifierSpec(kind="exact", parameters_json=parameters)


def test_captured_null_state_is_a_submission_but_missing_state_is_not():
    task = _task(structured_exact(None), AnswerType.STATE)
    convention = StateAnswer(id="state")
    missing = grade_answer(task, convention, _attempt(task, "Done."))
    captured = grade_answer(task, convention, _attempt(task, "Done.", StateSubmission(None)))
    assert (missing.status, missing.reward) == (Outcome.SUBMISSION_FAILURE, 0.0)
    assert (captured.status, captured.reward) == (Outcome.GRADED, 1.0)


class FileAnswer(PlainText):
    path: str

    def extract(self, attempt: GradingAttempt) -> TextSubmission:
        if self.path not in attempt.files:
            raise SubmissionFailure("Missing captured answer file")
        return TextSubmission(attempt.files[self.path].decode())


@pytest.mark.parametrize("files,reward", [({"/app/answer.txt": b"yes"}, 1.0), ({"/app/answer.txt": b"no"}, 0.0)])
def test_runtime_passes_captured_files_to_candidate_convention(files, reward):
    task = _task(exact_answer("yes"))
    result = grade_task(
        task,
        FileAnswer(id="file", path="/app/answer.txt"),
        _attempt(task, "unused").conversation,
        RuntimeEvidence(files, "null"),
    )
    assert (result.status, result.reward) == (Outcome.GRADED, reward)


@pytest.mark.parametrize("state,reward", [("null", 1.0), ('{"value": 1}', 0.0)])
def test_runtime_passes_captured_state_to_candidate_convention(state, reward):
    task = _task(structured_exact(None), AnswerType.STATE)
    result = grade_task(
        task,
        StateAnswer(id="state"),
        _attempt(task, "unused").conversation,
        RuntimeEvidence({}, state),
    )
    assert (result.status, result.reward) == (Outcome.GRADED, reward)


@pytest.mark.parametrize("state", ['{"x":1,"x":2}', "NaN", "1e1000", "{"])
def test_invalid_captured_state_remains_an_infrastructure_error(state):
    task = _task(structured_exact(None), AnswerType.STATE)
    result = grade_task(
        task,
        StateAnswer(id="state"),
        _attempt(task, "unused").conversation,
        RuntimeEvidence({}, state),
    )
    assert (result.status, result.reward) == (Outcome.INFRA_ERROR, None)
    assert "Invalid captured state" in result.error
