# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submission conventions for semantic answer tasks."""

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

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
    STATE = "state"
    ANSWER_CALL = "answer_call"
    FINAL_ACTION = "final_action"


class SubmissionConvention(BaseModel):
    """How a result is requested, delivered, and extracted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    answer_format: AnswerFormat

    @model_validator(mode="after")
    def validate_convention(self) -> "SubmissionConvention":
        if not self.id:
            raise ValueError("A submission convention id is required")
        return self

    def supports(self, answer_type: AnswerType) -> bool:
        """Whether this convention can carry the semantic result."""
        if self.answer_format == AnswerFormat.STATE:
            return answer_type == AnswerType.STATE
        if self.answer_format == AnswerFormat.FINAL_ACTION:
            return answer_type == AnswerType.NATIVE_ACTION
        return answer_type in (AnswerType.TEXT, AnswerType.NUMBER)


@dataclass(frozen=True)
class SubmissionCompatibility:
    """Whether a convention preserves the task's result contract, with reasons when it does not."""

    reasons: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        return not self.reasons


def submission_compatible(specification: TaskSpec, convention: SubmissionConvention) -> SubmissionCompatibility:
    """Explain which parts of the task a submission convention cannot carry."""
    if not convention.supports(specification.answer_type):
        return SubmissionCompatibility(
            (f"{convention.answer_format.value} cannot carry {specification.answer_type.value}",)
        )
    if convention.answer_format == AnswerFormat.FINAL_ACTION:
        reasons = []
        if not specification.final_tools.functions:
            reasons.append("final action requires at least one final tool")
        if specification.final_tools.tool_choice == "none":
            reasons.append("final action conflicts with tool_choice=none")
        return SubmissionCompatibility(tuple(reasons))
    if convention.answer_format == AnswerFormat.ANSWER_CALL:
        reasons = []
        if specification.final_tools.tool_choice == "none":
            reasons.append("answer call conflicts with tool_choice=none")
        if any(function.name == ANSWER_CALL_NAME for function in specification.final_tools.functions):
            reasons.append("final tool name collides with submit_answer")
        return SubmissionCompatibility(tuple(reasons))
    if convention.answer_format == AnswerFormat.STATE:
        reasons = []
        if specification.final_tools.functions:
            reasons.append("state submission does not carry final tools")
        if specification.final_tools.tool_choice is not None:
            reasons.append("state submission does not carry final tool choice")
        if specification.final_tools.parallel_tool_calls is not None:
            reasons.append("state submission does not carry final tool parallel policy")
        return SubmissionCompatibility(tuple(reasons))
    if specification.final_tools.tool_choice == "required":
        return SubmissionCompatibility(("text submission conflicts with tool_choice=required",))
    return SubmissionCompatibility(())


def submission_instruction(convention: SubmissionConvention) -> str:
    """Return the instruction added after a conversation prefix."""
    if convention.answer_format == AnswerFormat.PLAIN:
        return "Give your answer as plain text."
    if convention.answer_format == AnswerFormat.JSON:
        return f'Give your answer as a JSON object with an "{ANSWER_FIELD}" field.'
    if convention.answer_format == AnswerFormat.STATE:
        return (
            "Use the available tools to complete the task. "
            "Your final message ends the interaction; the result is graded from the environment state."
        )
    if convention.answer_format == AnswerFormat.ANSWER_CALL:
        return f'Call {ANSWER_CALL_NAME} with your final answer as the "{ANSWER_FIELD}" string.'
    if convention.answer_format == AnswerFormat.FINAL_ACTION:
        return ""
    raise ValueError(f"Unsupported answer format: {convention.answer_format}")


def render_instruction(specification: TaskSpec, convention: SubmissionConvention) -> str:
    """Return Harbor instruction text for the selected convention."""
    context = specification.context
    compatibility = submission_compatible(specification, convention)
    if not compatibility.compatible:
        raise ValueError(f"Submission convention {convention.id!r} is incompatible: {'; '.join(compatibility.reasons)}")
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


def chat_request(specification: TaskSpec, convention: SubmissionConvention) -> dict[str, Any]:
    """Prepare the conversation and tools for the selected submission convention."""
    compatibility = submission_compatible(specification, convention)
    if not compatibility.compatible:
        raise ValueError(f"Submission convention is incompatible: {'; '.join(compatibility.reasons)}")
    messages = conversation_messages(specification.context)
    instruction = submission_instruction(convention)
    if instruction:
        messages.append({"role": "user", "content": instruction})
    request: dict[str, Any] = {"messages": messages}
    tools: list[dict[str, object]] = [
        {"type": "function", "function": function.model_dump(exclude_none=True)}
        for function in specification.final_tools.functions
    ]
    if convention.answer_format == AnswerFormat.ANSWER_CALL:
        tools.append(answer_call_tool())
        if not specification.final_tools.functions:
            request.update(tool_choice="required", parallel_tool_calls=False)
    if tools:
        request["tools"] = tools
    if specification.final_tools.tool_choice is not None:
        request["tool_choice"] = specification.final_tools.tool_choice
    if specification.final_tools.parallel_tool_calls is not None:
        request["parallel_tool_calls"] = specification.final_tools.parallel_tool_calls
    return request


def extract_answer(response: ConversationEvent, convention: SubmissionConvention) -> str:
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
