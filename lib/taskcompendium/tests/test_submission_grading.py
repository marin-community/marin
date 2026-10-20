# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed submissions and in-process grading without an execution provider."""

import json

import pytest
from verifyit.closure_scoring import ClosureFormat
from verifyit.json_comparison import NumericTypePolicy
from verifyit.modes.grade_judge import JudgeConnection
from verifyit.spec import (
    RUBRIC_LABELS,
    EmptyOutputPolicy,
    ExactSpec,
    JudgeSpec,
    McqSpec,
    NumericSpec,
    ScriptSpec,
    Spec,
    StructureClosureSpec,
    StructuredExactSpec,
)

from taskcompendium.grader import verifyit_package
from taskcompendium.grading import grade_answer, grade_remote_judge_answer
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    CommandSemantics,
    ConversationInput,
    ConversationToolCall,
    ConversationTrace,
    EnvironmentRequirements,
    FinalAction,
    FunctionDefinition,
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


@pytest.mark.parametrize("stop_reason,trailing", [("stop", False), ("length", True), (None, True)])
def test_structure_closure_strips_one_terminal_marker_only_after_normal_stop(stop_reason, trailing):
    spec = StructureClosureSpec(
        prefix='{"a":',
        expected='{"a":1}',
        format=ClosureFormat.JSON,
        terminal_markers=("<|eot_id|>", "<|end_of_text|>"),
    )
    task = _task(_grader(spec))
    result = grade_answer(task, GradingAttempt(_conversation(task, "1}<|eot_id|>"), stop_reason=stop_reason))
    assert (result.status, result.reward) == (Outcome.GRADED, 1.0)
    assert result.detail["trailing_content"] is trailing
    assert result.detail["whole_response_success"] is not trailing
    repeated = grade_answer(task, GradingAttempt(_conversation(task, "1}<|eot_id|><|eot_id|>"), stop_reason=stop_reason))
    assert repeated.reward == 1.0
    assert repeated.detail["trailing_content"]
    assert grade_answer(task, _attempt(task, "")).reward == 0.0
    assert grade_answer(task, _attempt(task, "}")).reward == 0.0


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


def test_plain_text_passes_blank_candidates_to_a_verifier_that_grades_them():
    task = _task(_grader(ExactSpec(expected=("",), empty_output=EmptyOutputPolicy.GRADE)))
    result = grade_answer(task, _attempt(task, " \n"))
    assert (result.status, result.reward) == (Outcome.GRADED, 1.0)


def test_captured_null_state_is_a_submission_but_missing_state_is_not():
    task = _task(_grader(StructuredExactSpec(expected=None)), AnswerType.STATE)
    missing = grade_answer(task, _attempt(task, "Done."))
    captured = grade_answer(task, _attempt(task, "Done.", StateSubmission(None)))
    assert (missing.status, missing.reward) == (Outcome.SUBMISSION_FAILURE, 0.0)
    assert (captured.status, captured.reward) == (Outcome.GRADED, 1.0)


def test_runtime_grades_a_typed_native_action_through_the_supplied_judge_endpoint(judge_server):
    response = AssistantToolCalls(
        calls=(ConversationToolCall(call_id="transport-id", name="lookup", arguments={"query": "weather"}),)
    )
    package = verifyit_package(
        JudgeSpec(
            references=("Public conversation and tool schema",),
            rubric=RUBRIC_LABELS,
            system_prompt="Grade the proposed action.",
            prompt_template="Reference: {reference}\nCandidate: {candidate}",
            label_scores={"PASS": 1.0, "FAIL": 0.0},
            model="fixture-judge",
        ),
        environment=GRADING_ENVIRONMENT,
    )
    task = TaskSpec(
        id="judged-native-action",
        context=ConversationInput(events=(TextMessage(role="user", content="Look up the weather."),)),
        environment_requirements=EnvironmentRequirements(),
        final_tools=(
            FunctionDefinition(
                name="lookup",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
        ),
        answer_type=AnswerType.NATIVE_ACTION,
        answer_format=FinalAction(require_call=True, max_calls=1),
        grader=package.grader,
        source=Source(dataset="test", revision="1", row="0", importer_revision="1"),
    )
    attempt = GradingAttempt(ConversationTrace(events=(*task.context.events, response)))

    result = grade_task(
        task,
        attempt,
        judge_connection=JudgeConnection(f"http://127.0.0.1:{judge_server.server_port}/v1", "secret"),
    )

    assert (result.status, result.reward) == (Outcome.GRADED, 1.0)
    assert result.detail["verdict"] == "PASS"
    assert len(judge_server.requests) == 1
    prompt = judge_server.requests[0]["messages"][-1]["content"]
    assert json.loads(prompt.split("\nCandidate: ", 1)[1]) == {
        "type": "function_call",
        "name": "lookup",
        "arguments": {"query": "weather"},
    }


def test_runtime_requires_an_explicit_connection_for_judge_grading():
    task = _task(verifyit_package(JudgeSpec(references=("answer",)), environment=GRADING_ENVIRONMENT).grader)
    result = grade_task(task, _attempt(task, "candidate"))
    assert (result.status, result.reward) == (Outcome.INFRA_ERROR, None)


def test_remote_judge_rejects_an_empty_candidate_without_calling_the_endpoint():
    package = verifyit_package(
        JudgeSpec(
            references=("Complete the task.",),
            rubric=RUBRIC_LABELS,
            system_prompt="Grade the response.",
            prompt_template="Reference: {reference}\nCandidate: {candidate}",
            label_scores={"PASS": 1.0, "FAIL": 0.0},
        ),
        environment=GRADING_ENVIRONMENT,
    )
    task = TaskSpec(
        id="empty-judged-native-action",
        context=ConversationInput(events=(TextMessage(role="user", content="Look up the weather."),)),
        environment_requirements=EnvironmentRequirements(),
        final_tools=(
            FunctionDefinition(
                name="lookup",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
        ),
        answer_type=AnswerType.NATIVE_ACTION,
        answer_format=FinalAction(max_calls=1),
        grader=package.grader,
        source=Source(dataset="test", revision="1", row="0", importer_revision="1"),
    )

    result = grade_remote_judge_answer(
        task,
        _attempt(task, " \n"),
        JudgeConnection("http://unused.invalid/v1", "secret"),
    )

    assert (result.status, result.reward, result.detail) == (
        Outcome.GRADED,
        0.0,
        {"reason": "empty_candidate"},
    )


def test_remote_judge_grades_blank_plain_text_when_its_policy_requires_it(judge_server):
    judge_server.reply = "C"
    package = verifyit_package(
        JudgeSpec(
            references=("A reference answer.",),
            rubric=RUBRIC_LABELS,
            system_prompt="Grade the response.",
            prompt_template="Reference: {reference}\nCandidate: {candidate}",
            label_scores={"A": 1.0, "B": 0.0, "C": 0.5},
            empty_output=EmptyOutputPolicy.GRADE,
            model="fixture-judge",
        ),
        environment=GRADING_ENVIRONMENT,
    )
    task = _task(package.grader)

    result = grade_remote_judge_answer(
        task,
        _attempt(task, " \n"),
        JudgeConnection(f"http://127.0.0.1:{judge_server.server_port}/v1", "secret"),
    )

    assert (result.status, result.reward) == (Outcome.GRADED, 0.5)
    assert len(judge_server.requests) == 1


def test_remote_judge_provider_error_is_retryable_infrastructure_failure(judge_server):
    judge_server.status_code = 400
    package = verifyit_package(
        JudgeSpec(
            references=("A reference answer.",),
            rubric=RUBRIC_LABELS,
            system_prompt="Grade the response.",
            prompt_template="Reference: {reference}\nCandidate: {candidate}",
            label_scores={"PASS": 1.0, "FAIL": 0.0},
            model="fixture-judge",
        ),
        environment=GRADING_ENVIRONMENT,
    )
    task = _task(package.grader)

    result = grade_remote_judge_answer(
        task,
        _attempt(task, "candidate"),
        JudgeConnection(f"http://127.0.0.1:{judge_server.server_port}/v1", "secret"),
    )

    assert (result.status, result.reward) == (Outcome.INFRA_ERROR, None)
    assert len(judge_server.requests) == 1


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
