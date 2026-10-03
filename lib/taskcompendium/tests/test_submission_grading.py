# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed submissions and private grading without an execution provider."""

import json
from typing import Literal

import pytest
from pydantic import JsonValue, PrivateAttr, TypeAdapter
from tasktrove_verify.spec import Mode

from taskcompendium.ejection import enable_ejection, mark_unsolvable
from taskcompendium.examples.ejection import ejection_examples
from taskcompendium.grading import Outcome, exact_answer, grade_answer, numeric_answer, structured_exact
from taskcompendium.lowering import HarborEnvironmentConfig, compatible_lowerings, lower_to_harbor
from taskcompendium.models import (
    SCHEMA_VERSION,
    AnswerType,
    AssistantToolCalls,
    ConversationInput,
    ConversationToolCall,
    ConversationTrace,
    EnvironmentRequirements,
    FunctionCall,
    FunctionDefinition,
    Source,
    TaskSpec,
    TextMessage,
    ToolResult,
    VerifierSpec,
)
from taskcompendium.submission import (
    AnswerCall,
    AnswerFormat,
    Convention,
    FinalAction,
    GradingAttempt,
    JsonAnswer,
    PlainText,
    StateSubmission,
    SubmissionConvention,
    TextSubmission,
    chat_request,
    eject_button,
    render_instruction,
)
from taskcompendium.verifiers.multiple_choice import multiple_choice_answer
from taskcompendium.verifiers.predicted_action import predicted_action_verifier


def _task(verifier, answer_type=AnswerType.TEXT, final_tools=()):
    return TaskSpec(
        id="submission-grading",
        context=ConversationInput(events=(TextMessage(role="user", content="Complete the task."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=answer_type,
        final_tools=final_tools,
        verifier=verifier,
        source=Source(dataset="test", revision="1", row="0", importer_revision="1"),
    )


def _attempt(task, content):
    return GradingAttempt(
        conversation=ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=content))),
        workspace=object(),
    )


class StateAnswer(Convention):
    answer_format: Literal[AnswerFormat.JSON] = AnswerFormat.JSON
    value: JsonValue

    async def extract(self, _attempt: GradingAttempt) -> StateSubmission:
        return StateSubmission(self.value)


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
    result = await grade_answer(task, StateAnswer(id="state", value=actual), _attempt(task, "Done."))
    assert (result.status, result.reward) == (Outcome.GRADED, reward)


class ChangingText(PlainText):
    answer_format: Literal[AnswerFormat.PLAIN] = AnswerFormat.PLAIN
    _submitted: bool = PrivateAttr(default=False)

    async def extract(self, _attempt: GradingAttempt) -> TextSubmission:
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
    task = _task(VerifierSpec(kind=Mode.EXACT, parameters_json="{}"))
    with pytest.raises(ValueError):
        await grade_answer(task, JsonAnswer(id="json"), _attempt(task, '{"answer":'))


@pytest.mark.parametrize(
    "wire_kind,verifier,answer_type,correct,wrong",
    [
        ("exact", exact_answer("yes"), AnswerType.TEXT, "yes", "no"),
        ("numeric", numeric_answer(12.0, tolerance_abs=0.0, tolerance_rel=0.0), AnswerType.NUMBER, "12", "13"),
        ("mcq", multiple_choice_answer("B", 3), AnswerType.TEXT, "B", "A"),
    ],
)
async def test_canonical_verifier_kinds_serialize_load_and_grade(wire_kind, verifier, answer_type, correct, wrong):
    serialized = json.loads(_task(verifier, answer_type).model_dump_json())
    assert serialized["verifier"]["kind"] == wire_kind
    assert serialized["schema_version"] == SCHEMA_VERSION
    loaded = TaskSpec.model_validate_json(json.dumps(serialized))
    convention = PlainText(id="plain")
    correct_result = await grade_answer(loaded, convention, _attempt(loaded, correct))
    wrong_result = await grade_answer(loaded, convention, _attempt(loaded, wrong))
    assert (correct_result.status, correct_result.reward) == (Outcome.GRADED, 1.0)
    assert (wrong_result.status, wrong_result.reward) == (Outcome.GRADED, 0.0)


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
async def test_structured_exact_rejects_nonfinite_gold_before_serialization_but_preserves_null(number):
    with pytest.raises(ValueError):
        structured_exact({"nested": [number]})
    task = _task(structured_exact(None), AnswerType.STATE)
    result = await grade_answer(task, StateAnswer(id="state", value=None), _attempt(task, "Done."))
    assert (result.status, result.reward) == (Outcome.GRADED, 1.0)


@pytest.mark.parametrize("normal", [PlainText(id="plain"), JsonAnswer(id="json"), AnswerCall(id="call")])
async def test_ejection_and_normal_answers_have_one_private_reward(normal):
    examples = ejection_examples(1, normal)
    solvable, impossible = examples[:2]
    for example, eject_reward, answer_reward in ((solvable, 0.0, 1.0), (impossible, 1.0, 0.0)):
        task, convention = example.specification, example.convention
        ejection = AssistantToolCalls(
            calls=(
                ConversationToolCall(
                    call_id="eject", name="eject_button", arguments={"reason": "Required information is unavailable"}
                ),
            )
        )
        result = await grade_answer(task, convention, _message_attempt(task, ejection))
        assert (result.status, result.reward) == (Outcome.GRADED, eject_reward)
        for answer, expected_reward in (("5", answer_reward), ("6", 0.0)):
            if isinstance(normal, AnswerCall):
                final = AssistantToolCalls(
                    calls=(ConversationToolCall(call_id="answer", name="submit_answer", arguments={"answer": answer}),)
                )
            else:
                content = json.dumps({"answer": answer}) if isinstance(normal, JsonAnswer) else answer
                final = TextMessage(role="assistant", content=content)
            result = await grade_answer(task, convention, _message_attempt(task, final))
            assert (result.status, result.reward) == (Outcome.GRADED, expected_reward)
        restored = TypeAdapter(SubmissionConvention).validate_json(convention.model_dump_json())
        restored_result = await grade_answer(task, restored, _message_attempt(task, ejection))
        assert (restored_result.status, restored_result.reward) == (Outcome.GRADED, eject_reward)


def _message_attempt(task, final):
    return GradingAttempt(ConversationTrace(events=(*task.context.events, final)), object())


@pytest.mark.parametrize("normal", [PlainText(id="plain"), JsonAnswer(id="json"), AnswerCall(id="call")])
def test_ejection_availability_and_instructions_do_not_disclose_private_labels(normal):
    for solvable, impossible in zip(ejection_examples(2, normal)[::2], ejection_examples(2, normal)[1::2], strict=True):
        # Hold source conversation fixed to detect any leak from private metadata.
        hidden = impossible.specification.model_copy(update={"context": solvable.specification.context})
        assert chat_request(solvable.specification, solvable.convention) == chat_request(hidden, impossible.convention)
        assert render_instruction(solvable.specification, solvable.convention) == render_instruction(
            hidden, impossible.convention
        )
        assert impossible.convention.answer_format == AnswerFormat.FINAL_ACTION
        assert solvable.convention.answer_format == normal.answer_format


@pytest.mark.parametrize("arguments", [{"reason": ""}, {"reason": 7}, {"reason": "why", "answer": "5"}])
async def test_malformed_ejection_is_submission_failure(arguments):
    example = ejection_examples(1, PlainText(id="plain"))[1]
    final = AssistantToolCalls(calls=(ConversationToolCall(call_id="eject", name="eject_button", arguments=arguments),))
    result = await grade_answer(
        example.specification, example.convention, _message_attempt(example.specification, final)
    )
    assert (result.status, result.reward) == (Outcome.SUBMISSION_FAILURE, 0.0)


async def test_ejection_rejects_mixed_terminal_actions_and_continuation():
    example = ejection_examples(1, AnswerCall(id="answer"))[1]
    task, convention = example.specification, example.convention
    call = ConversationToolCall(call_id="eject", name="eject_button", arguments={"reason": "Missing information"})
    mixed = AssistantToolCalls(
        calls=(call, ConversationToolCall(call_id="answer", name="submit_answer", arguments={"answer": "5"}))
    )
    result = await grade_answer(task, convention, _message_attempt(task, mixed))
    assert (result.status, result.reward) == (Outcome.SUBMISSION_FAILURE, 0.0)
    continued = GradingAttempt(
        ConversationTrace(
            events=(
                *task.context.events,
                AssistantToolCalls(calls=(call,)),
                ToolResult(call_id="eject", content="ignored"),
                TextMessage(role="assistant", content="5"),
            )
        ),
        object(),
    )
    result = await grade_answer(task, convention, continued)
    assert (result.status, result.reward) == (Outcome.SUBMISSION_FAILURE, 0.0)


async def test_ejection_is_optional_and_cannot_bypass_invalid_private_verifier():
    task = _task(exact_answer("5"))
    final = AssistantToolCalls(
        calls=(ConversationToolCall(call_id="eject", name="eject_button", arguments={"reason": "why"}),)
    )
    result = await grade_answer(task, PlainText(id="plain"), _message_attempt(task, final))
    assert (result.status, result.reward) == (Outcome.SUBMISSION_FAILURE, 0.0)
    invalid = task.model_copy(update={"verifier": VerifierSpec(kind=Mode.EXACT, parameters_json="{}")})
    with pytest.raises(ValueError, match="Invalid 'exact'"):
        await grade_answer(invalid, PlainText(id="plain"), _message_attempt(invalid, final))


async def test_native_action_tasks_can_eject_without_changing_their_normal_verifier():
    expected = FunctionCall(name="choose", arguments={"option": "A"})
    task = _task(
        predicted_action_verifier((expected,)),
        AnswerType.NATIVE_ACTION,
        (FunctionDefinition(name="choose", parameters={"type": "object"}), eject_button()),
    )
    convention = FinalAction(id="native", require_call=True, max_calls=1)
    for name, arguments, reward in (
        ("choose", {"option": "A"}, 1.0),
        ("choose", {"option": "B"}, 0.0),
        ("eject_button", {"reason": "Cannot choose"}, 0.0),
    ):
        final = AssistantToolCalls(calls=(ConversationToolCall(call_id="choice", name=name, arguments=arguments),))
        result = await grade_answer(task, convention, _message_attempt(task, final))
        assert (result.status, result.reward) == (Outcome.GRADED, reward)


def test_unsolvable_tasks_reject_normal_only_lowerings(tmp_path):
    example = ejection_examples(1, AnswerCall(id="answer"))[1]
    for convention in (PlainText(id="plain"), AnswerCall(id="answer")):
        assert compatible_lowerings(example.specification, (convention,), (HarborEnvironmentConfig(),)) == ()
        with pytest.raises(ValueError, match="incompatible"):
            lower_to_harbor(example.specification, convention, HarborEnvironmentConfig(), tmp_path / convention.id)


async def test_historical_ejection_does_not_invalidate_a_normal_answer():
    task = _task(exact_answer("5"), final_tools=(eject_button(),))
    call = ConversationToolCall(call_id="historical", name="eject_button", arguments={"reason": "Previous task"})
    task = task.model_copy(
        update={
            "context": ConversationInput(
                events=(
                    *task.context.events,
                    AssistantToolCalls(calls=(call,)),
                    ToolResult(call_id="historical", content="Recorded"),
                    TextMessage(role="user", content="Now answer this separate question: what is 2 + 3?"),
                )
            )
        }
    )
    result = await grade_answer(task, PlainText(id="plain"), _attempt(task, "5"))
    assert (result.status, result.reward) == (Outcome.GRADED, 1.0)


async def test_native_multicall_answers_remain_available_with_ejection():
    expected = (
        FunctionCall(name="choose", arguments={"option": "A"}),
        FunctionCall(name="choose", arguments={"option": "B"}),
    )
    task = _task(
        predicted_action_verifier(expected),
        AnswerType.NATIVE_ACTION,
        (FunctionDefinition(name="choose", parameters={"type": "object"}), eject_button()),
    )
    convention = FinalAction(id="native", require_call=True, max_calls=2)
    request = chat_request(task, convention)
    assert request.get("parallel_tool_calls") is not False
    final = AssistantToolCalls(
        calls=tuple(
            ConversationToolCall(call_id=str(index), name=call.name, arguments=call.arguments)
            for index, call in enumerate(expected)
        )
    )
    result = await grade_answer(task, convention, _message_attempt(task, final))
    assert (result.status, result.reward) == (Outcome.GRADED, 1.0)
    mixed = AssistantToolCalls(
        calls=(*final.calls, ConversationToolCall(call_id="eject", name="eject_button", arguments={"reason": "why"}))
    )
    result = await grade_answer(task, convention, _message_attempt(task, mixed))
    assert (result.status, result.reward) == (Outcome.SUBMISSION_FAILURE, 0.0)


async def test_source_ejection_helpers_preserve_native_presentation_and_private_rewards():
    expected = (
        FunctionCall(name="choose", arguments={"option": "A"}),
        FunctionCall(name="choose", arguments={"option": "B"}),
    )
    original = _task(
        predicted_action_verifier(expected),
        AnswerType.NATIVE_ACTION,
        (FunctionDefinition(name="choose", parameters={"type": "object"}),),
    )
    normal = FinalAction(id="native", require_call=True, max_calls=2)
    solvable = enable_ejection(original)
    broken, convention = mark_unsolvable(original, normal)
    assert broken.context == original.context
    assert broken.answer_type == original.answer_type
    assert broken.source == original.source
    assert chat_request(solvable, normal) == chat_request(broken, convention)
    assert render_instruction(solvable, normal) == render_instruction(broken, convention)
    answer = AssistantToolCalls(
        calls=tuple(
            ConversationToolCall(call_id=str(index), name=call.name, arguments=call.arguments)
            for index, call in enumerate(expected)
        )
    )
    ejection = AssistantToolCalls(
        calls=(ConversationToolCall(call_id="eject", name="eject_button", arguments={"reason": "Contradiction"}),)
    )
    for task, policy, final, reward in (
        (solvable, normal, answer, 1.0),
        (solvable, normal, ejection, 0.0),
        (broken, convention, answer, 0.0),
        (broken, convention, ejection, 1.0),
    ):
        result = await grade_answer(task, policy, _message_attempt(task, final))
        assert (result.status, result.reward) == (Outcome.GRADED, reward)


async def test_confirmed_broken_task_can_replace_invalid_private_gold():
    original = _task(VerifierSpec(kind=Mode.EXACT, parameters_json="{}"))
    broken, convention = mark_unsolvable(original, AnswerCall(id="answer"))
    ejection = AssistantToolCalls(
        calls=(ConversationToolCall(call_id="eject", name="eject_button", arguments={"reason": "Invalid gold"}),)
    )
    result = await grade_answer(broken, convention, _message_attempt(broken, ejection))
    assert (result.status, result.reward) == (Outcome.GRADED, 1.0)


def test_source_ejection_helper_does_not_reclassify_unsupported_runtime(tmp_path):
    original = _task(exact_answer("5"))
    unsupported = original.model_copy(
        update={"environment_requirements": EnvironmentRequirements(capabilities=("shell",))}
    )
    with pytest.raises(NotImplementedError):
        mark_unsolvable(unsupported, PlainText(id="plain"))
    assert unsupported.verifier == original.verifier
    assert (
        compatible_lowerings(enable_ejection(unsupported), (PlainText(id="plain"),), (HarborEnvironmentConfig(),)) == ()
    )
    with pytest.raises(NotImplementedError):
        lower_to_harbor(
            enable_ejection(unsupported), PlainText(id="plain"), HarborEnvironmentConfig(), tmp_path / "task"
        )
    assert not (tmp_path / "task").exists()
