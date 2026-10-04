# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submission conventions for semantic answer tasks."""

import json
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from taskcompendium.direct_chat import unsupported_direct_chat_features
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    ConversationEvent,
    ConversationInput,
    TaskSpec,
    TextMessage,
    format_conversation,
)

ANSWER_CALL_NAME = "submit_answer"
ANSWER_FIELD = "answer"


def answer_call_tool() -> dict[str, object]:
    """Return the function definition advertised by the answer-call convention."""
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


class AnswerFormat(StrEnum):
    """The envelope used to deliver a result."""

    PLAIN = "plain"
    JSON = "json"
    ANSWER_CALL = "answer_call"
    FINAL_ACTION = "final_action"


class _SubmissionConvention(BaseModel):
    """How a result is requested, delivered, and extracted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    answer_format: AnswerFormat

    @model_validator(mode="after")
    def validate_convention(self) -> "_SubmissionConvention":
        if not self.id:
            raise ValueError("A submission convention id is required")
        return self

    def supports(self, answer_type: AnswerType) -> bool:
        """Whether this convention can carry the semantic result."""
        if self.answer_format == AnswerFormat.FINAL_ACTION:
            return answer_type == AnswerType.NATIVE_ACTION
        return answer_type in (AnswerType.TEXT, AnswerType.NUMBER)


class SubmissionConvention(_SubmissionConvention):
    """Deliver a text or numeric answer through a selected chat envelope."""

    answer_format: Literal[AnswerFormat.PLAIN, AnswerFormat.JSON, AnswerFormat.ANSWER_CALL]


class FinalAction(_SubmissionConvention):
    """Capture the final assistant turn with explicit function-call limits."""

    answer_format: Literal[AnswerFormat.FINAL_ACTION] = AnswerFormat.FINAL_ACTION
    require_call: bool = False
    max_calls: int | None = Field(default=None, gt=0)

    def validate_final_message(self, response: ConversationEvent) -> None:
        """Require the assistant's final message to honor the call contract."""
        if not isinstance(response, (TextMessage, AssistantToolCalls)) or (
            isinstance(response, TextMessage) and response.role != "assistant"
        ):
            raise ValueError("Final action requires an assistant message")
        if self.require_call and not isinstance(response, AssistantToolCalls):
            raise ValueError("Final action requires a function call")
        if (
            isinstance(response, AssistantToolCalls)
            and self.max_calls is not None
            and len(response.calls) > self.max_calls
        ):
            raise ValueError(f"Final action permits at most {self.max_calls} function calls")


type Submission = SubmissionConvention | FinalAction


def submission_compatible(specification: TaskSpec, convention: Submission) -> bool:
    if not convention.supports(specification.answer_type):
        return False
    if convention.answer_format == AnswerFormat.FINAL_ACTION:
        return bool(specification.final_tools)
    if convention.answer_format == AnswerFormat.ANSWER_CALL:
        return all(function.name != ANSWER_CALL_NAME for function in specification.final_tools)
    return True


def submission_instruction(convention: Submission) -> str:
    """Return the instruction added after a conversation prefix."""
    if convention.answer_format == AnswerFormat.PLAIN:
        return "Give your answer as plain text."
    if convention.answer_format == AnswerFormat.JSON:
        return f'Give your answer as a JSON object with an "{ANSWER_FIELD}" field.'
    if convention.answer_format == AnswerFormat.ANSWER_CALL:
        return f'Call {ANSWER_CALL_NAME} with your final answer as the "{ANSWER_FIELD}" string.'
    if convention.answer_format == AnswerFormat.FINAL_ACTION:
        return ""
    raise ValueError(f"Unsupported answer format: {convention.answer_format}")


def render_instruction(specification: TaskSpec, convention: Submission) -> str:
    """Return Harbor instruction text for the selected convention."""
    context = specification.context
    if not submission_compatible(specification, convention):
        raise ValueError(
            f"Submission convention {convention.id!r} cannot carry {specification.answer_type.value!r} in this context"
        )
    if convention.answer_format == AnswerFormat.FINAL_ACTION:
        return format_conversation(context.events)
    return f"{format_conversation(context.events)}\n\n{submission_instruction(convention)}\n"


def conversation_messages(context: ConversationInput) -> list[dict[str, Any]]:
    """Convert the model-visible prefix to OpenAI-compatible chat messages."""
    messages: list[dict[str, Any]] = []
    for event in context.events:
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


def chat_request(specification: TaskSpec, convention: Submission) -> dict[str, Any]:
    """Prepare a direct-chat request for the selected submission convention."""
    unsupported = unsupported_direct_chat_features(specification)
    if unsupported:
        raise NotImplementedError(f"Direct chat cannot satisfy requirements: {', '.join(unsupported)}")
    return submission_request(specification, convention)


def submission_request(specification: TaskSpec, convention: Submission) -> dict[str, Any]:
    """Prepare the model-visible conversation and final submission tools."""
    if not submission_compatible(specification, convention):
        raise ValueError("Submission convention is incompatible with the task")
    messages = conversation_messages(specification.context)
    instruction = submission_instruction(convention)
    if instruction:
        messages.append({"role": "user", "content": instruction})
    request: dict[str, Any] = {"messages": messages}
    tools: list[dict[str, object]] = [
        {"type": "function", "function": function.model_dump(exclude_none=True)}
        for function in specification.final_tools
    ]
    if convention.answer_format == AnswerFormat.ANSWER_CALL:
        tools.append(answer_call_tool())
        if not specification.final_tools:
            request.update(tool_choice="required", parallel_tool_calls=False)
    if tools:
        request["tools"] = tools
    if isinstance(convention, FinalAction):
        if convention.require_call:
            request["tool_choice"] = "required"
        if convention.max_calls == 1:
            request["parallel_tool_calls"] = False
    return request


def extract_answer(response: ConversationEvent, convention: Submission) -> str:
    """Extract semantic answer content from a typed assistant turn."""
    if convention.answer_format == AnswerFormat.ANSWER_CALL:
        if (
            not isinstance(response, AssistantToolCalls)
            or len(response.calls) != 1
            or response.calls[0].name != ANSWER_CALL_NAME
        ):
            raise ValueError(f"Answer call requires one {ANSWER_CALL_NAME} function call")
        arguments = response.calls[0].arguments
        if (
            set(arguments) != {ANSWER_FIELD}
            or not isinstance(arguments[ANSWER_FIELD], str)
            or not arguments[ANSWER_FIELD].strip()
        ):
            raise ValueError("Answer call requires a nonempty string answer")
        return arguments[ANSWER_FIELD]
    if not isinstance(response, TextMessage) or response.role != "assistant" or not response.content.strip():
        raise ValueError("Text submission requires nonempty assistant content without tool calls")
    if convention.answer_format == AnswerFormat.PLAIN:
        return response.content
    if convention.answer_format == AnswerFormat.JSON:
        value = json.loads(response.content)
        if (
            not isinstance(value, dict)
            or not isinstance(value.get(ANSWER_FIELD), str)
            or not value[ANSWER_FIELD].strip()
        ):
            raise ValueError("JSON submission requires a nonempty string answer")
        return value[ANSWER_FIELD]
    raise ValueError(f"Unsupported answer format: {convention.answer_format}")
