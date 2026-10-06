# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Chat request formatting and typed evidence preserve semantic grading contracts."""

import pytest
from pydantic import TypeAdapter
from verifyit.json_comparison import NumericTypePolicy

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
)
from taskcompendium.submission import (
    AnswerCall,
    AnswerFormat,
    JsonAnswer,
    JsonValueAnswer,
    PlainText,
    SubmissionConvention,
    chat_request,
    render_instruction,
    submission_compatibility,
)


def _attempt(specification, response):
    trace = ConversationTrace(events=(*specification.context.events, response))
    return GradingAttempt(ConversationTrace.model_validate_json(trace.model_dump_json()))


def _answer_convention(answer_format: AnswerFormat) -> SubmissionConvention:
    if answer_format == AnswerFormat.PLAIN:
        return PlainText(id="plain")
    if answer_format == AnswerFormat.JSON:
        return JsonAnswer(id="json")
    if answer_format == AnswerFormat.ANSWER_CALL:
        return AnswerCall(id="answer_call")
    raise ValueError(f"Unsupported test answer format: {answer_format}")


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
        verifier=numeric_answer("12.0", tolerance_abs="0.0", tolerance_rel="0.0"),
        source=Source(dataset="hand-authored", revision="2026-09-16", row="arithmetic-7-plus-5", importer_revision="1"),
    )


@pytest.mark.parametrize(
    "response,reward",
    [("12.05", 1.0), ("12.2", 0.0)],
)
def test_numeric_answer_uses_explicit_tolerance(specification, response, reward):
    specification = specification.model_copy(
        update={"verifier": numeric_answer("12.0", tolerance_abs="0.1", tolerance_rel="0.0")}
    )
    convention = PlainText(id="plain")

    result = grade_answer(
        specification,
        convention,
        GradingAttempt(
            conversation=ConversationTrace(
                events=(*specification.context.events, TextMessage(role="assistant", content=response))
            ),
        ),
    )

    assert (result.status, result.reward) == ("graded", reward)


@pytest.mark.parametrize(
    "answer_format,response,status,reward",
    [
        (AnswerFormat.PLAIN, "12", Outcome.GRADED, 1.0),
        (AnswerFormat.PLAIN, "12.0", Outcome.GRADED, 1.0),
        (AnswerFormat.PLAIN, "13", Outcome.GRADED, 0.0),
        (AnswerFormat.PLAIN, "not a number", Outcome.SUBMISSION_FAILURE, 0.0),
        (AnswerFormat.PLAIN, r"\boxed{12}", Outcome.GRADED, 1.0),
        (AnswerFormat.JSON, '{"answer":"12"}', Outcome.GRADED, 1.0),
        (AnswerFormat.JSON, '{"answer":"13"}', Outcome.GRADED, 0.0),
        (AnswerFormat.JSON, '{"answer":"12"', Outcome.SUBMISSION_FAILURE, 0.0),
    ],
)
def test_chat_answer_distinguishes_wrong_and_malformed_submissions(
    specification, answer_format, response, status, reward
):
    specification = TaskSpec.model_validate_json(specification.model_dump_json())
    convention = TypeAdapter(SubmissionConvention).validate_json(_answer_convention(answer_format).model_dump_json())
    assert "12" not in render_instruction(specification, convention)
    result = grade_answer(
        specification, convention, _attempt(specification, TextMessage(role="assistant", content=response))
    )
    assert (result.status, result.reward) == (status, reward)


def test_chat_exact_comparison_uses_unicode_and_whitespace_normalization(specification):
    task = specification.model_copy(update={"verifier": exact_answer("Straße Park"), "answer_type": AnswerType.TEXT})
    convention = PlainText(id="plain")
    result = grade_answer(task, convention, _attempt(task, TextMessage(role="assistant", content="STRASSE   PARK")))
    assert (result.status, result.reward) == (Outcome.GRADED, 1.0)


def test_text_convention_retains_but_rejects_tool_call_evidence(specification):
    convention = PlainText(id="plain")
    attempt = _attempt(specification, _answer_action("12"))
    result = grade_answer(specification, convention, attempt)
    assert (result.status, result.reward) == (Outcome.SUBMISSION_FAILURE, 0.0)
    assert attempt.conversation.events[-1].calls[0].arguments == {"answer": "12"}


@pytest.mark.parametrize(
    "answer_type,verifier,response",
    [
        (AnswerType.TEXT, exact_answer("12"), "12"),
        (AnswerType.NUMBER, numeric_answer("12", tolerance_abs="0", tolerance_rel="0"), "12.0"),
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


@pytest.mark.parametrize("answer_format", [AnswerFormat.PLAIN, AnswerFormat.JSON, AnswerFormat.ANSWER_CALL])
def test_chat_request_preserves_advertised_tools(specification, answer_format):
    task = specification.model_copy(
        update={"final_tools": (FunctionDefinition(name="lookup", parameters={"type": "object"}),)}
    )
    request = chat_request(task, _answer_convention(answer_format))
    assert request["tools"][0] == {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
    assert [tool["function"]["name"] for tool in request["tools"]] == (
        ["lookup", "submit_answer"] if answer_format == AnswerFormat.ANSWER_CALL else ["lookup"]
    )
    assert request.get("tool_choice") == ("required" if answer_format == AnswerFormat.ANSWER_CALL else None)
    if answer_format == AnswerFormat.ANSWER_CALL:
        assert request["parallel_tool_calls"] is False


def test_answer_call_name_collision_cannot_change_source_tool(specification):
    task = specification.model_copy(
        update={"final_tools": (FunctionDefinition(name="submit_answer", parameters={"type": "object"}),)}
    )
    convention = AnswerCall(id="answer-call")
    assert not submission_compatibility(task, convention).compatible
    with pytest.raises(ValueError):
        chat_request(task, convention)


@pytest.mark.parametrize(
    "content,status,reward",
    [
        ('{"value":16.0,"nested":[true,null]}', Outcome.GRADED, 1.0),
        ('{"value":17,"nested":[true,null]}', Outcome.GRADED, 0.0),
        ('{"value":16,"value":17}', Outcome.SUBMISSION_FAILURE, 0.0),
        ('{"value":16,"nested":{"x":1,"\\u0078":2}}', Outcome.SUBMISSION_FAILURE, 0.0),
        ('{"value":NaN}', Outcome.SUBMISSION_FAILURE, 0.0),
        ('{"value":1e400}', Outcome.SUBMISSION_FAILURE, 0.0),
        ('{"value":', Outcome.SUBMISSION_FAILURE, 0.0),
    ],
)
def test_json_value_chat_grading_rejects_ambiguous_and_nonfinite_values(specification, content, status, reward):
    task = specification.model_copy(
        update={"answer_type": AnswerType.JSON, "verifier": structured_exact({"value": 16, "nested": [True, None]})}
    )
    task = TaskSpec.model_validate_json(task.model_dump_json())
    convention = JsonValueAnswer(id="json-value")
    result = grade_answer(task, convention, _attempt(task, TextMessage(role="assistant", content=content)))
    assert (result.status, result.reward) == (status, reward)


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


def test_json_chat_strict_numeric_policy_survives_task_roundtrip(specification):
    task = specification.model_copy(
        update={
            "answer_type": AnswerType.JSON,
            "verifier": structured_exact({"value": 16}, numeric_types=NumericTypePolicy.STRICT),
        }
    )
    task = TaskSpec.model_validate_json(task.model_dump_json())
    convention = JsonValueAnswer(id="json-value")
    result = grade_answer(task, convention, _attempt(task, TextMessage(role="assistant", content='{"value":16.0}')))
    assert (result.status, result.reward) == (Outcome.GRADED, 0.0)
