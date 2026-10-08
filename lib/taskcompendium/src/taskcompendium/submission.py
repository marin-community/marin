# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Answer-format instructions, compatibility checks, and chat requests.

An answer is the task's semantic result, identified by ``TaskSpec.answer_type``. A submission is
the typed value the task's answer format extracts from a completed attempt for grading. Plain
text, ``{"answer": "12"}``, and ``submit_answer(answer="12")`` all yield ``TextSubmission("12")``.
"""

import json
from dataclasses import dataclass
from typing import Any

from verifyit.spec import ExactSpec, PredictedActionSpec, Spec, StructuredExactSpec

from taskcompendium.direct_chat import unsupported_direct_chat_features
from taskcompendium.models import (
    ANSWER_CALL_NAME,
    ANSWER_FIELD,
    ActionSubmission,
    AnswerCall,
    AssistantToolCalls,
    BaseAnswerFormat,
    Boxed,
    ConversationEvent,
    FinalAction,
    JsonAnswer,
    JsonSubmission,
    JsonValueAnswer,
    PlainText,
    StateSubmission,
    Submission,
    TaskSpec,
    TextMessage,
    TextSubmission,
    VerifyitGrader,
    format_conversation,
    verifyit_spec,
)


def answer_call_tool() -> dict[str, object]:
    """Return the function definition advertised by the answer-call format."""
    return {
        "type": "function",
        "function": {
            "name": ANSWER_CALL_NAME,
            "description": "Submit the final answer to the task.",
            "parameters": {
                "type": "object",
                "properties": {ANSWER_FIELD: {"type": "string"}},
                "required": [ANSWER_FIELD],
                "additionalProperties": False,
            },
        },
    }


def accepted_submission_types(verifier: Spec) -> tuple[type[Submission], ...]:
    """Declare the submissions a verifyit mode can grade."""
    if isinstance(verifier, StructuredExactSpec):
        return (JsonSubmission, StateSubmission)
    if isinstance(verifier, PredictedActionSpec):
        return (ActionSubmission,)
    if isinstance(verifier, ExactSpec):
        return (TextSubmission, JsonSubmission, StateSubmission)
    return (TextSubmission,)


@dataclass(frozen=True)
class SubmissionCompatibility:
    """Whether the answer format preserves the task's result contract, with reasons when it does not."""

    reasons: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        return not self.reasons


def submission_compatibility(task: TaskSpec) -> SubmissionCompatibility:
    """Explain which parts of the task its answer format cannot carry."""
    answer_format = task.answer_format
    if not answer_format.supports(task.answer_type):
        return SubmissionCompatibility((f"{type(answer_format).__name__} cannot carry {task.answer_type.value}",))
    if isinstance(task.grader, VerifyitGrader):
        accepted = accepted_submission_types(verifyit_spec(task.grader))
        if not any(produced in accepted for produced in answer_format.submission_types):
            return SubmissionCompatibility(("Submission envelope is not accepted by the selected verifier",))
    if isinstance(answer_format, FinalAction) and not task.final_tools:
        return SubmissionCompatibility(("final action requires at least one final tool",))
    if isinstance(answer_format, AnswerCall) and any(function.name == ANSWER_CALL_NAME for function in task.final_tools):
        return SubmissionCompatibility(("final tool name collides with submit_answer",))
    return SubmissionCompatibility(())


def submission_instruction(answer_format: BaseAnswerFormat) -> str:
    """Return the instruction added after a conversation prefix."""
    match answer_format:
        case PlainText():
            return "Give your answer as plain text."
        case Boxed():
            return "Put your final answer within \\boxed{}."
        case JsonAnswer():
            return f'Give your answer as a JSON object with an "{ANSWER_FIELD}" field.'
        case JsonValueAnswer():
            return "Give your final answer as one JSON value, without Markdown fences."
        case AnswerCall():
            return f'Call {ANSWER_CALL_NAME} with your final answer as the "{ANSWER_FIELD}" string.'
        case FinalAction():
            return ""
        case _:
            raise ValueError(f"Unsupported answer format: {type(answer_format).__name__}")


def render_instruction(task: TaskSpec) -> str:
    """Return readable instruction text for the task's answer format."""
    compatibility = submission_compatibility(task)
    if not compatibility.compatible:
        raise ValueError(
            f"Answer format {task.answer_format.kind!r} is incompatible: {'; '.join(compatibility.reasons)}"
        )
    if isinstance(task.answer_format, FinalAction):
        return format_conversation(task.context.events)
    return f"{format_conversation(task.context.events)}\n\n{submission_instruction(task.answer_format)}\n"


def conversation_messages(events: tuple[ConversationEvent, ...]) -> list[dict[str, Any]]:
    """Convert conversation events to OpenAI-compatible chat messages."""
    messages: list[dict[str, Any]] = []
    for event in events:
        if isinstance(event, TextMessage):
            messages.append({"role": event.role, "content": event.content})
        elif isinstance(event, AssistantToolCalls):
            messages.append(
                {
                    "role": "assistant",
                    "content": event.content,
                    "tool_calls": [
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, separators=(",", ":"), ensure_ascii=False),
                            },
                        }
                        for call in event.calls
                    ],
                }
            )
        else:
            messages.append({"role": "tool", "tool_call_id": event.call_id, "content": event.content})
    return messages


def chat_request(task: TaskSpec) -> dict[str, Any]:
    """Prepare the conversation and tools for the task's answer format."""
    unsupported = unsupported_direct_chat_features(task)
    if unsupported:
        raise NotImplementedError(f"Direct chat cannot satisfy requirements: {', '.join(unsupported)}")
    compatibility = submission_compatibility(task)
    if not compatibility.compatible:
        raise ValueError(f"Answer format is incompatible: {'; '.join(compatibility.reasons)}")
    answer_format = task.answer_format
    messages = conversation_messages(task.context.events)
    instruction = submission_instruction(answer_format)
    if instruction:
        messages.append({"role": "user", "content": instruction})
    request: dict[str, Any] = {"messages": messages}
    tools: list[dict[str, object]] = [
        {"type": "function", "function": function.model_dump(exclude_none=True)} for function in task.final_tools
    ]
    if isinstance(answer_format, AnswerCall):
        tools.append(answer_call_tool())
        request.update(tool_choice="required", parallel_tool_calls=False)
    if tools:
        request["tools"] = tools
    if isinstance(answer_format, FinalAction):
        if answer_format.require_call:
            request["tool_choice"] = "required"
        if answer_format.max_calls == 1:
            request["parallel_tool_calls"] = False
    return request
