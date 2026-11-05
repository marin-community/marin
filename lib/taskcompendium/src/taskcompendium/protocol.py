# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade how an attempt followed the agent protocols its task implies, separately from its answer.

Every graded attempt is assessed for each protocol the task's own fields imply: a file answer,
tools bound to the shell, declared fetch pages, and a request that turned thinking off. The
assessments go in ``GradeResult.diagnostics["protocol"]``, keyed by check, and never change the
reward. Each reports protocol compliance, submission completeness and answer correctness
separately, so passing a protocol never implies a correct answer. A result the check does not
assess is ``None``, and the whole assessment is not evaluable when the evidence cannot decide it.
"""

import json
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any

from pydantic import JsonValue
from shellbox.machine import ExitReason
from verifyit.reasoning_markup import REASONING_DELIMITERS
from verifyit.tool_call_markup import TOOL_OPEN

from taskcompendium.fetch import FETCH_ACTION_INTERFACE, FETCH_TOOL, FetchState
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    CHAT_TEMPLATE_KWARGS,
    ENABLE_THINKING,
    AnswerType,
    AssistantToolCalls,
    ConversationEvent,
    GradingAttempt,
    TaskSpec,
    TextMessage,
    ToolResult,
)

PROTOCOL_DIAGNOSTICS_KEY = "protocol"


class ProtocolCheck(StrEnum):
    REQUIRED_FILE = "required_file"
    TOOL_EXECUTION = "tool_execution"
    THINKING_DISABLED = "thinking_disabled"
    URL_LOADING = "url_loading"


class ProtocolLabel(StrEnum):
    SUBMITTED = "submitted"
    MISSING_FILE = "missing_file"
    EMPTY_FILE = "empty_file"
    EXECUTED = "executed"
    COMMAND_FAILED = "command_failed"
    NO_INVOCATION = "no_invocation"
    MARKUP_IN_TEXT = "markup_in_text"
    INVALID_INVOCATION = "invalid_invocation"
    NOT_EXECUTED = "not_executed"
    EXECUTOR_TIMEOUT = "executor_timeout"
    EXECUTOR_FAILURE = "executor_failure"
    RESPECTED = "respected"
    REASONING_EMITTED = "reasoning_emitted"
    RETRIEVED = "retrieved"
    HTTP_ERROR = "http_error"
    NETWORK_ERROR = "network_error"
    NOT_ATTEMPTED = "not_attempted"


@dataclass(frozen=True)
class ProtocolAssessment:
    label: ProtocolLabel | None
    not_evaluable: str | None = None
    protocol_compliance: bool | None = None
    submission_completeness: bool | None = None
    answer_correctness: bool | None = None

    def diagnostics(self) -> dict[str, Any]:
        """The assessment as JSON-compatible grade diagnostics, omitting unassessed results."""
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class _ToolCall:
    name: str
    arguments: JsonValue
    result: dict[str, Any] | None
    """The decoded tool result, or ``None`` when the call has none."""


def _attempt_events(task: TaskSpec, attempt: GradingAttempt) -> tuple[ConversationEvent, ...]:
    """The events the agent produced after the task's own context."""
    return attempt.conversation.events[len(task.context.events) :]


def _decoded(content: str) -> dict[str, Any]:
    """A tool result's JSON fields; a result that is not a JSON object has none."""
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _tool_calls(events: tuple[ConversationEvent, ...]) -> list[_ToolCall]:
    results = {event.call_id: _decoded(event.content) for event in events if isinstance(event, ToolResult)}
    return [
        _ToolCall(call.name, call.arguments, results.get(call.call_id))
        for event in events
        if isinstance(event, AssistantToolCalls)
        for call in event.calls
    ]


def _assistant_text(events: tuple[ConversationEvent, ...]) -> list[str]:
    texts = []
    for event in events:
        if isinstance(event, TextMessage) and event.role == "assistant":
            texts.append(event.content)
        elif isinstance(event, AssistantToolCalls) and event.content:
            texts.append(event.content)
    return texts


def declared_pages(task: TaskSpec) -> frozenset[str]:
    """The URLs the task's fetch providers serve."""
    requirements = task.environment_requirements.tool_providers.values()
    return frozenset(
        url
        for requirement in requirements
        if requirement.action_interface == FETCH_ACTION_INTERFACE
        for url in FetchState.model_validate(requirement.initial_state).pages
    )


def thinking_disabled(attempt: GradingAttempt) -> bool:
    """Whether the attempt's requests turned thinking off."""
    kwargs = attempt.request_settings.get(CHAT_TEMPLATE_KWARGS)
    return isinstance(kwargs, dict) and kwargs.get(ENABLE_THINKING) is False


def _required_file(task: TaskSpec, attempt: GradingAttempt) -> ProtocolLabel:
    """Every declared output path, a file or a directory, must hold a nonempty captured file."""
    captured = [
        [content for path, content in attempt.files.items() if PurePosixPath(path).is_relative_to(root)]
        for root in task.output_paths
    ]
    if not all(captured):
        return ProtocolLabel.MISSING_FILE
    if not all(any(content.strip() for content in contents) for contents in captured):
        return ProtocolLabel.EMPTY_FILE
    return ProtocolLabel.SUBMITTED


def _tool_execution(task: TaskSpec, events: tuple[ConversationEvent, ...]) -> ProtocolLabel:
    """Label the best outcome of the attempt's calls to the task's shell-bound tools."""
    calls = _tool_calls(events)
    if not calls:
        markup = any(TOOL_OPEN in text for text in _assistant_text(events))
        return ProtocolLabel.MARKUP_IN_TEXT if markup else ProtocolLabel.NO_INVOCATION
    calls = [call for call in calls if call.name in task.tool_bindings]
    results = [call.result for call in calls if call.result is not None]
    exited = [result for result in results if result.get("reason") == ExitReason.EXITED]
    if any(result.get("exit_code") == 0 for result in exited):
        return ProtocolLabel.EXECUTED
    if exited:
        return ProtocolLabel.COMMAND_FAILED
    if any(result.get("reason") == ExitReason.TIMED_OUT for result in results):
        return ProtocolLabel.EXECUTOR_TIMEOUT
    if calls and not results:
        # A call without a result ended the attempt before it ran, for example at the turn limit.
        return ProtocolLabel.NOT_EXECUTED
    # The shell tool runs every well-formed call, so the remaining results are argument errors.
    return ProtocolLabel.INVALID_INVOCATION


def _url_loading(pages: frozenset[str], events: tuple[ConversationEvent, ...]) -> ProtocolLabel:
    attempts = [
        call.result
        for call in _tool_calls(events)
        if call.name == FETCH_TOOL.name
        and isinstance(call.arguments, dict)
        and call.arguments.get("url") in pages
        and call.result is not None
    ]
    if not attempts:
        return ProtocolLabel.NOT_ATTEMPTED
    statuses = [result.get("status") for result in attempts]
    if any(isinstance(status, int) and 200 <= status < 300 for status in statuses):
        return ProtocolLabel.RETRIEVED
    if any(isinstance(status, int) for status in statuses):
        return ProtocolLabel.HTTP_ERROR
    return ProtocolLabel.NETWORK_ERROR


def _thinking_disabled(attempt: GradingAttempt) -> ProtocolLabel:
    emitted = any(marker in raw for raw in attempt.raw_responses for pair in REASONING_DELIMITERS for marker in pair)
    return ProtocolLabel.REASONING_EMITTED if emitted else ProtocolLabel.RESPECTED


def _passed(grade: GradeResult) -> bool:
    """Whether the answer passed; in-process verifiers report only a reward."""
    return grade.passed if grade.passed is not None else grade.reward == grade.score_max


def _checks(task: TaskSpec, attempt: GradingAttempt) -> list[ProtocolCheck]:
    checks = []
    if task.answer_type == AnswerType.FILE and task.output_paths:
        checks.append(ProtocolCheck.REQUIRED_FILE)
    if task.tool_bindings:
        checks.append(ProtocolCheck.TOOL_EXECUTION)
    if thinking_disabled(attempt):
        checks.append(ProtocolCheck.THINKING_DISABLED)
    if declared_pages(task):
        checks.append(ProtocolCheck.URL_LOADING)
    return checks


def _assess(check: ProtocolCheck, task: TaskSpec, attempt: GradingAttempt, grade: GradeResult) -> ProtocolAssessment:
    if grade.status == Outcome.INFRA_ERROR and check == ProtocolCheck.TOOL_EXECUTION:
        return ProtocolAssessment(ProtocolLabel.EXECUTOR_FAILURE, f"grade {grade.status}: {grade.error}")
    if grade.status not in (Outcome.GRADED, Outcome.SUBMISSION_FAILURE):
        return ProtocolAssessment(None, f"grade {grade.status}: {grade.error}")
    correct = grade.status == Outcome.GRADED and _passed(grade)
    if check == ProtocolCheck.REQUIRED_FILE:
        # Writing the file is the protocol under test, so it is reported as submission completeness.
        label = _required_file(task, attempt)
        submitted = label == ProtocolLabel.SUBMITTED
        return ProtocolAssessment(label, None, None, submitted, submitted and correct)
    submitted = grade.status == Outcome.GRADED
    events = _attempt_events(task, attempt)
    if check == ProtocolCheck.THINKING_DISABLED:
        if not attempt.raw_responses:
            return ProtocolAssessment(None, "the runtime recorded no raw responses")
        label = _thinking_disabled(attempt)
        return ProtocolAssessment(label, None, label == ProtocolLabel.RESPECTED, submitted, correct)
    if check == ProtocolCheck.TOOL_EXECUTION:
        label = _tool_execution(task, events)
        return ProtocolAssessment(label, None, label == ProtocolLabel.EXECUTED, submitted, correct)
    label = _url_loading(declared_pages(task), events)
    # A failed retrieval is still an attempt; ``label`` records the failure without charging the model.
    return ProtocolAssessment(label, None, label != ProtocolLabel.NOT_ATTEMPTED, submitted, correct)


def assess_protocol(
    task: TaskSpec, attempt: GradingAttempt, grade: GradeResult
) -> dict[ProtocolCheck, ProtocolAssessment]:
    """Assess one graded attempt against every protocol its task implies; empty when it implies none."""
    return {check: _assess(check, task, attempt, grade) for check in _checks(task, attempt)}


def with_protocol_assessment(task: TaskSpec, attempt: GradingAttempt, grade: GradeResult) -> GradeResult:
    """Add the attempt's protocol assessments to its grade diagnostics; the grade is unchanged otherwise."""
    assessments = assess_protocol(task, attempt, grade)
    if not assessments:
        return grade
    protocol = {check.value: assessment.diagnostics() for check, assessment in assessments.items()}
    return replace(grade, diagnostics={**grade.diagnostics, PROTOCOL_DIAGNOSTICS_KEY: protocol})
