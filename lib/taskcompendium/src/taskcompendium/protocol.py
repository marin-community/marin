# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade how an attempt followed a task's protocol, separately from whether its answer is right.

A task with a ``protocol_check`` gets an assessment in ``GradeResult.diagnostics["protocol"]``. It
reports protocol compliance, submission completeness and answer correctness as separate results, so
passing the protocol never implies a correct answer. A result is ``None`` when the check does not
assess it, and the whole assessment is not evaluable when the evidence cannot decide it.
"""

import json
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Any

from pydantic import JsonValue
from shellbox.machine import ExitReason
from verifyit.tool_call_markup import TOOL_OPEN

from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    AssistantToolCalls,
    ConversationEvent,
    GradingAttempt,
    ReasoningApplied,
    RequiredFileCheck,
    TaskSpec,
    TextMessage,
    ThinkingDisabledCheck,
    ToolExecutionCheck,
    ToolResult,
    UrlLoadingCheck,
)

PROTOCOL_DIAGNOSTICS_KEY = "protocol"


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


class ProtocolAxis(StrEnum):
    PROTOCOL_COMPLIANCE = "protocol_compliance"
    SUBMISSION_COMPLETENESS = "submission_completeness"
    ANSWER_CORRECTNESS = "answer_correctness"


@dataclass(frozen=True)
class ProtocolAssessment:
    check: str
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


def _required_file(task: TaskSpec, attempt: GradingAttempt) -> ProtocolLabel:
    contents = [attempt.files.get(path) for path in task.output_paths]
    if any(content is None for content in contents):
        return ProtocolLabel.MISSING_FILE
    if any(not content.strip() for content in contents if content is not None):
        return ProtocolLabel.EMPTY_FILE
    return ProtocolLabel.SUBMITTED


def _tool_execution(check: ToolExecutionCheck, events: tuple[ConversationEvent, ...]) -> ProtocolLabel:
    """Label the best outcome of the attempt's calls to ``check.tool``."""
    calls = _tool_calls(events)
    if not calls:
        markup = any(TOOL_OPEN in text for text in _assistant_text(events))
        return ProtocolLabel.MARKUP_IN_TEXT if markup else ProtocolLabel.NO_INVOCATION
    calls = [call for call in calls if call.name == check.tool]
    results = [call.result for call in calls if call.result is not None]
    exited = [result for result in results if result.get("reason") == ExitReason.EXITED]
    if any(result.get("exit_code") == 0 for result in exited):
        return ProtocolLabel.EXECUTED
    if exited:
        return ProtocolLabel.COMMAND_FAILED
    if any(result.get("reason") == ExitReason.TIMED_OUT for result in results):
        return ProtocolLabel.EXECUTOR_TIMEOUT
    if calls and not results:
        # A call without a result ended the attempt before it ran, for example at the turn budget.
        return ProtocolLabel.NOT_EXECUTED
    # The shell tool runs every well-formed call, so the remaining results are argument errors.
    return ProtocolLabel.INVALID_INVOCATION


def _url_loading(check: UrlLoadingCheck, events: tuple[ConversationEvent, ...]) -> ProtocolLabel:
    attempts = [
        call.result
        for call in _tool_calls(events)
        if call.name == check.tool
        and isinstance(call.arguments, dict)
        and call.arguments.get("url") == check.url
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


def _thinking_disabled(attempt: GradingAttempt) -> ProtocolLabel | None:
    """The label, or ``None`` when the serving evidence cannot show that thinking was off."""
    turns = attempt.turns
    if not turns or any(turn.reasoning_applied != ReasoningApplied.OFF or not turn.reasoning_markers for turn in turns):
        return None
    emitted = any(marker in turn.raw_response for turn in turns for marker in turn.reasoning_markers)
    return ProtocolLabel.REASONING_EMITTED if emitted else ProtocolLabel.RESPECTED


def _passed(grade: GradeResult) -> bool:
    """Whether the answer passed; in-process verifiers report only a reward."""
    return grade.passed if grade.passed is not None else grade.reward == grade.score_max


def assess_protocol(task: TaskSpec, attempt: GradingAttempt, grade: GradeResult) -> ProtocolAssessment | None:
    """Assess one graded attempt against the task's protocol check; ``None`` when the task has none."""
    check = task.protocol_check
    if check is None:
        return None
    if grade.status == Outcome.INFRA_ERROR and isinstance(check, ToolExecutionCheck):
        return ProtocolAssessment(check.kind, ProtocolLabel.EXECUTOR_FAILURE, f"grade {grade.status}: {grade.error}")
    if grade.status not in (Outcome.GRADED, Outcome.SUBMISSION_FAILURE):
        return ProtocolAssessment(check.kind, None, f"grade {grade.status}: {grade.error}")
    correct = grade.status == Outcome.GRADED and _passed(grade)
    if isinstance(check, RequiredFileCheck):
        # Writing the file is the protocol under test, so it is reported as submission completeness.
        label = _required_file(task, attempt)
        submitted = label == ProtocolLabel.SUBMITTED
        return ProtocolAssessment(check.kind, label, None, None, submitted, submitted and correct)
    submitted = grade.status == Outcome.GRADED
    events = _attempt_events(task, attempt)
    if isinstance(check, ThinkingDisabledCheck):
        label = _thinking_disabled(attempt)
        if label is None:
            return ProtocolAssessment(check.kind, None, "serving evidence does not show reasoning was off")
        return ProtocolAssessment(check.kind, label, None, label == ProtocolLabel.RESPECTED, submitted, correct)
    if isinstance(check, ToolExecutionCheck):
        label = _tool_execution(check, events)
        return ProtocolAssessment(check.kind, label, None, label == ProtocolLabel.EXECUTED, submitted, correct)
    label = _url_loading(check, events)
    # A failed retrieval is still an attempt; ``label`` records the failure without charging the model.
    return ProtocolAssessment(check.kind, label, None, label != ProtocolLabel.NOT_ATTEMPTED, submitted, correct)


def with_protocol_assessment(task: TaskSpec, attempt: GradingAttempt, grade: GradeResult) -> GradeResult:
    """Add the task's protocol assessment to its grade diagnostics; the grade is unchanged otherwise."""
    assessment = assess_protocol(task, attempt, grade)
    if assessment is None:
        return grade
    return replace(grade, diagnostics={**grade.diagnostics, PROTOCOL_DIAGNOSTICS_KEY: assessment.diagnostics()})
