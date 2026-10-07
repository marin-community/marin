# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submission extraction and model-visible requests preserve task contracts."""

import json

import pytest

from taskcompendium.grading import Outcome, exact_answer, grade_answer, numeric_answer, structured_exact
from taskcompendium.grading_contract import GradingAttempt
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    ConversationInput,
    ConversationToolCall,
    ConversationTrace,
    EnvironmentRequirements,
    FunctionDefinition,
    Source,
    TaskSpec,
    TextMessage,
    VerifierSpec,
)
from taskcompendium.submission import (
    AnswerCall,
    JsonAnswer,
    JsonValueAnswer,
    PlainText,
    chat_request,
    render_instruction,
    submission_compatibility,
)


def _attempt(specification, response):
    return GradingAttempt(ConversationTrace(events=(*specification.context.events, response)))


def _answer_action(answer: str, name: str = "submit_answer") -> AssistantToolCalls:
    return AssistantToolCalls(
        calls=(ConversationToolCall(call_id="call-answer", name=name, arguments={"answer": answer}),)
    )


@pytest.fixture
def specification() -> TaskSpec:
    return TaskSpec(
        id="arithmetic-7-plus-5",
        context=ConversationInput(events=(TextMessage(role="user", content="What is 7 + 5?"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.NUMBER,
        verifier=numeric_answer("12.0", tolerance_abs=0.0, tolerance_rel=0.0),
        source=Source(dataset="hand-authored", revision="2026-09-16", row="arithmetic-7-plus-5", importer_revision="1"),
    )


@pytest.mark.parametrize(
    "convention,response,status,reward",
    [
        (PlainText(id="plain"), "not a number", Outcome.SUBMISSION_FAILURE, 0.0),
        (JsonAnswer(id="json"), '{"answer":"12"}', Outcome.GRADED, 1.0),
        (JsonAnswer(id="json"), '{"answer":"13"}', Outcome.GRADED, 0.0),
        (JsonAnswer(id="json"), '{"answer":"12"', Outcome.SUBMISSION_FAILURE, 0.0),
    ],
)
def test_answer_conventions_distinguish_wrong_and_malformed_submissions(
    specification, convention, response, status, reward
):
    result = grade_answer(
        specification, convention, _attempt(specification, TextMessage(role="assistant", content=response))
    )
    assert (result.status, result.reward) == (status, reward)


def test_plain_text_rejects_tool_call_evidence(specification):
    convention = PlainText(id="plain")
    attempt = _attempt(specification, _answer_action("12"))
    result = grade_answer(specification, convention, attempt)
    assert (result.status, result.reward) == (Outcome.SUBMISSION_FAILURE, 0.0)


@pytest.mark.parametrize(
    "kind,parameters,private_image",
    [
        ("numeric", {"expected": "12", "tolerance_abs": 0, "tolerance_rel": 0}, True),
        ("script", {"path": "/tests/grade.py"}, True),
        ("script", {"path": "/tests/grade.py"}, False),
    ],
)
def test_submission_compatibility_accepts_private_runtime_verifiers_without_executing_them(
    specification, kind, parameters, private_image
):
    task = specification.model_copy(
        update={
            "verifier": VerifierSpec(
                kind=kind,
                parameters_json=json.dumps(parameters),
                environment_requirements=EnvironmentRequirements(
                    docker_image="fixture@sha256:" + "0" * 64 if private_image else None
                ),
            )
        }
    )
    convention = PlainText(id="plain")
    assert submission_compatibility(task, convention).compatible
    with pytest.raises(NotImplementedError):
        grade_answer(task, convention, _attempt(task, TextMessage(role="assistant", content="12")))


@pytest.mark.parametrize(
    "answer_type,verifier,response",
    [
        (AnswerType.TEXT, exact_answer("12"), "12"),
        (AnswerType.NUMBER, numeric_answer("12", tolerance_abs=0.0, tolerance_rel=0.0), "12.0"),
    ],
)
def test_answer_call_grades_semantic_answers(specification, answer_type, verifier, response):
    task = specification.model_copy(update={"answer_type": answer_type, "verifier": verifier})
    convention = AnswerCall(id="answer-call")
    correct = grade_answer(task, convention, _attempt(task, _answer_action(response)))
    wrong = grade_answer(task, convention, _attempt(task, _answer_action("13")))
    invalid = _answer_action(response, name="lookup")
    rejected = grade_answer(task, convention, _attempt(task, invalid))
    assert (correct.status, correct.reward) == (Outcome.GRADED, 1.0)
    assert (wrong.status, wrong.reward) == (Outcome.GRADED, 0.0)
    assert (rejected.status, rejected.reward) == (Outcome.SUBMISSION_FAILURE, 0.0)


@pytest.mark.parametrize(
    "convention,tool_names,tool_choice",
    [
        (PlainText(id="plain"), ["lookup"], None),
        (JsonAnswer(id="json"), ["lookup"], None),
        (AnswerCall(id="answer-call"), ["lookup", "submit_answer"], "required"),
    ],
)
def test_submission_request_preserves_tools_and_keeps_answer_private(specification, convention, tool_names, tool_choice):
    task = specification.model_copy(
        update={"final_tools": (FunctionDefinition(name="lookup", parameters={"type": "object"}),)}
    )
    request = chat_request(task, convention)
    assert request["tools"][0] == {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
    assert [tool["function"]["name"] for tool in request["tools"]] == tool_names
    assert request.get("tool_choice") == tool_choice
    if tool_choice == "required":
        assert request["parallel_tool_calls"] is False
    assert "12" not in render_instruction(task, convention)


def test_answer_call_name_collision_cannot_change_source_tool(specification):
    task = specification.model_copy(
        update={"final_tools": (FunctionDefinition(name="submit_answer", parameters={"type": "object"}),)}
    )
    convention = AnswerCall(id="answer-call")
    assert not submission_compatibility(task, convention).compatible
    with pytest.raises(ValueError):
        chat_request(task, convention)


@pytest.mark.parametrize(
    "content",
    [
        '{"value":16,"value":17}',
        '{"value":16,"nested":{"x":1,"\\u0078":2}}',
        '{"value":NaN}',
        '{"value":1e400}',
        '{"value":',
    ],
)
def test_json_value_rejects_ambiguous_and_nonfinite_submissions(specification, content):
    task = specification.model_copy(
        update={"answer_type": AnswerType.JSON, "verifier": structured_exact({"value": 16, "nested": [True, None]})}
    )
    convention = JsonValueAnswer(id="json-value")
    result = grade_answer(task, convention, _attempt(task, TextMessage(role="assistant", content=content)))
    assert (result.status, result.reward) == (Outcome.SUBMISSION_FAILURE, 0.0)


def test_incompatible_result_convention_and_verifier_cannot_form_chat_request(specification):
    task = specification.model_copy(update={"verifier": structured_exact({"value": 12}), "answer_type": AnswerType.TEXT})
    convention = PlainText(id="plain")
    assert not submission_compatibility(task, convention).compatible
    with pytest.raises(ValueError):
        chat_request(task, convention)


def test_direct_chat_cannot_acquire_state_for_structured_verifier(specification):
    task = specification.model_copy(
        update={"verifier": structured_exact({"value": 12}), "answer_type": AnswerType.STATE}
    )
    with pytest.raises(NotImplementedError):
        chat_request(task, JsonValueAnswer(id="json-value"))
