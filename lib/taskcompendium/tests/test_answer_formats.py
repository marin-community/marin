# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Answer formats extract submissions and shape model-visible requests without revealing the answer."""

import pytest
from verifyit.spec import ExactSpec, NumericSpec, StructuredExactSpec

from taskcompendium.grader import verifyit_package
from taskcompendium.grading import grade_answer
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    AnswerCall,
    AnswerType,
    AssistantToolCalls,
    Boxed,
    ConversationInput,
    ConversationToolCall,
    ConversationTrace,
    EnvironmentRequirements,
    FunctionDefinition,
    GradingAttempt,
    JsonAnswer,
    JsonValueAnswer,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.submission import chat_request, render_instruction, submission_compatibility


def _revised(task: TaskSpec, **update) -> TaskSpec:
    """Rebuild a task with changed fields, applying TaskSpec validation."""
    return TaskSpec.model_validate({**dict(task), **update})


def _attempt(task, response):
    return GradingAttempt(ConversationTrace(events=(*task.context.events, response)))


def _reply(content: str) -> TextMessage:
    return TextMessage(role="assistant", content=content)


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
        answer_format=PlainText(),
        grader=verifyit_package(NumericSpec("12.0", tolerance_abs=0.0, tolerance_rel=0.0)).grader,
        source=Source(dataset="hand-authored", revision="2026-09-16", row="arithmetic-7-plus-5", importer_revision="1"),
    )


@pytest.mark.parametrize(
    "answer_format,response,status,reward",
    [
        (PlainText(), "12", Outcome.GRADED, 1.0),
        (PlainText(), "not a number", Outcome.SUBMISSION_FAILURE, 0.0),
        (JsonAnswer(), '{"answer":"12"}', Outcome.GRADED, 1.0),
        (JsonAnswer(), '{"answer":"13"}', Outcome.GRADED, 0.0),
        (JsonAnswer(), '{"answer":"12"', Outcome.SUBMISSION_FAILURE, 0.0),
        (JsonAnswer(), '{"answer":[12]}', Outcome.SUBMISSION_FAILURE, 0.0),
    ],
)
def test_answer_formats_distinguish_wrong_and_malformed_submissions(
    specification, answer_format, response, status, reward
):
    task = _revised(specification, answer_format=answer_format)
    result = grade_answer(task, _attempt(task, _reply(response)))
    assert (result.status, result.reward) == (status, reward)


@pytest.mark.parametrize(
    "answer_format,response,reward",
    [
        (Boxed(), r"The answer is \boxed{\frac{1}{2}}.", 1.0),
        (Boxed(), r"Not \boxed{1}; finally \boxed{\frac{1}{2}}", 1.0),
        (Boxed(), r"\frac{1}{2}", 1.0),
        (Boxed(), r"The answer is \boxed{\frac{1}{3}}.", 0.0),
        (PlainText(), r"The answer is \boxed{\frac{1}{2}}.", 0.0),
    ],
)
def test_boxed_format_grades_the_last_balanced_box_or_the_whole_message(specification, answer_format, response, reward):
    task = _revised(
        specification,
        answer_type=AnswerType.TEXT,
        answer_format=answer_format,
        grader=verifyit_package(ExactSpec(expected=(r"\frac{1}{2}",))).grader,
    )
    result = grade_answer(task, _attempt(task, _reply(response)))
    assert (result.status, result.reward) == (Outcome.GRADED, reward)


# The reply GLM-5.3 gave to a number task under JsonAnswer in #9758.
FENCED_NUMBER_REPLY = '```json\n{"answer": 42}\n```'
EXPECT_42 = NumericSpec("42", tolerance_abs=0.0, tolerance_rel=0.0)


@pytest.mark.parametrize(
    "answer_type,spec,response,status,reward",
    [
        (AnswerType.NUMBER, EXPECT_42, FENCED_NUMBER_REPLY, Outcome.GRADED, 1.0),
        (AnswerType.NUMBER, EXPECT_42, '{"answer": 42}', Outcome.GRADED, 1.0),
        (AnswerType.NUMBER, EXPECT_42, '{"answer": 41.5}', Outcome.GRADED, 0.0),
        (AnswerType.NUMBER, EXPECT_42, '```json\n{"answer": "42"}\n```', Outcome.GRADED, 1.0),
        (AnswerType.TEXT, ExactSpec(expected=("12",)), '```json\n{"answer": "12"}\n```', Outcome.GRADED, 1.0),
        (
            AnswerType.NUMBER,
            NumericSpec("1", tolerance_abs=0.0, tolerance_rel=0.0),
            '{"answer": true}',
            Outcome.SUBMISSION_FAILURE,
            0.0,
        ),
        (AnswerType.TEXT, ExactSpec(expected=("42",)), FENCED_NUMBER_REPLY, Outcome.SUBMISSION_FAILURE, 0.0),
        (AnswerType.NUMBER, EXPECT_42, f"The answer is below.\n{FENCED_NUMBER_REPLY}", Outcome.SUBMISSION_FAILURE, 0.0),
    ],
)
def test_json_answer_accepts_enclosing_fence_and_number_for_numeric_task(
    specification, answer_type, spec, response, status, reward
):
    task = _revised(
        specification, answer_type=answer_type, answer_format=JsonAnswer(), grader=verifyit_package(spec).grader
    )
    result = grade_answer(task, _attempt(task, _reply(response)))
    assert (result.status, result.reward) == (status, reward)


def test_plain_text_rejects_tool_call_evidence(specification):
    result = grade_answer(specification, _attempt(specification, _answer_action("12")))
    assert (result.status, result.reward) == (Outcome.SUBMISSION_FAILURE, 0.0)


@pytest.mark.parametrize(
    "answer_type,spec,response",
    [
        (AnswerType.TEXT, ExactSpec(expected=("12",)), "12"),
        (AnswerType.NUMBER, NumericSpec("12", tolerance_abs=0.0, tolerance_rel=0.0), "12.0"),
    ],
)
def test_answer_call_grades_semantic_answers(specification, answer_type, spec, response):
    task = _revised(
        specification, answer_type=answer_type, answer_format=AnswerCall(), grader=verifyit_package(spec).grader
    )
    correct = grade_answer(task, _attempt(task, _answer_action(response)))
    wrong = grade_answer(task, _attempt(task, _answer_action("13")))
    rejected = grade_answer(task, _attempt(task, _answer_action(response, name="lookup")))
    assert (correct.status, correct.reward) == (Outcome.GRADED, 1.0)
    assert (wrong.status, wrong.reward) == (Outcome.GRADED, 0.0)
    assert (rejected.status, rejected.reward) == (Outcome.SUBMISSION_FAILURE, 0.0)


@pytest.mark.parametrize(
    "answer_format,tool_names,tool_choice",
    [
        (PlainText(), ["lookup"], None),
        (JsonAnswer(), ["lookup"], None),
        (AnswerCall(), ["lookup", "submit_answer"], "required"),
    ],
)
def test_submission_request_preserves_tools_and_keeps_answer_private(
    specification, answer_format, tool_names, tool_choice
):
    task = _revised(
        specification,
        answer_format=answer_format,
        final_tools=(FunctionDefinition(name="lookup", parameters={"type": "object"}),),
    )
    request = chat_request(task)
    assert request["tools"][0] == {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
    assert [tool["function"]["name"] for tool in request["tools"]] == tool_names
    assert request.get("tool_choice") == tool_choice
    if tool_choice == "required":
        assert request["parallel_tool_calls"] is False
    assert "12" not in render_instruction(task)


def test_answer_call_name_collision_cannot_change_source_tool(specification):
    task = _revised(
        specification,
        answer_format=AnswerCall(),
        final_tools=(FunctionDefinition(name="submit_answer", parameters={"type": "object"}),),
    )
    assert not submission_compatibility(task).compatible
    with pytest.raises(ValueError):
        chat_request(task)


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
    task = _revised(
        specification,
        answer_type=AnswerType.JSON,
        answer_format=JsonValueAnswer(),
        grader=verifyit_package(StructuredExactSpec(expected={"value": 16, "nested": [True, None]})).grader,
    )
    result = grade_answer(task, _attempt(task, _reply(content)))
    assert (result.status, result.reward) == (Outcome.SUBMISSION_FAILURE, 0.0)


@pytest.mark.parametrize(
    "content,status,reward",
    [
        ('```json\n{"value": 16, "nested": [true, null]}\n```', Outcome.GRADED, 1.0),
        ('Here it is:\n```json\n{"value": 16, "nested": [true, null]}\n```', Outcome.SUBMISSION_FAILURE, 0.0),
    ],
)
def test_json_value_unwraps_only_a_fence_enclosing_the_reply(specification, content, status, reward):
    task = _revised(
        specification,
        answer_type=AnswerType.JSON,
        answer_format=JsonValueAnswer(),
        grader=verifyit_package(StructuredExactSpec(expected={"value": 16, "nested": [True, None]})).grader,
    )
    result = grade_answer(task, _attempt(task, _reply(content)))
    assert (result.status, result.reward) == (status, reward)


def test_incompatible_answer_format_and_grader_cannot_form_chat_request(specification):
    task = _revised(
        specification,
        answer_type=AnswerType.TEXT,
        grader=verifyit_package(StructuredExactSpec(expected={"value": 12})).grader,
    )
    assert not submission_compatibility(task).compatible
    with pytest.raises(ValueError):
        chat_request(task)


def test_direct_chat_cannot_acquire_state_for_structured_grader(specification):
    task = _revised(
        specification,
        answer_type=AnswerType.STATE,
        answer_format=JsonValueAnswer(),
        grader=verifyit_package(StructuredExactSpec(expected={"value": 12})).grader,
    )
    with pytest.raises(NotImplementedError):
        chat_request(task)
