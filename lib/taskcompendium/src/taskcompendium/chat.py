# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize OpenAI chat messages into semantic submission evidence."""

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter
from verifyit.json_objects import unique_object

from taskcompendium.grading_contract import SubmissionFailure
from taskcompendium.models import (
    AssistantToolCalls,
    ConversationEvent,
    ConversationToolCall,
    ConversationTrace,
    TextMessage,
    ToolResult,
)

ARGUMENT_OBJECT = TypeAdapter(dict[str, JsonValue], config=ConfigDict(strict=True, allow_inf_nan=False))


class ChatFunction(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    name: str = Field(min_length=1)
    arguments: str


class ChatToolCall(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str = Field(min_length=1)
    type: Literal["function"]
    function: ChatFunction


class ChatAssistantMessage(BaseModel):
    model_config = ConfigDict(strict=True)

    role: Literal["assistant"]
    content: str | None = None
    tool_calls: list[ChatToolCall] | None = None


def assistant_message(message: dict[str, Any]) -> TextMessage | AssistantToolCalls:
    """Decode assistant output, raising SubmissionFailure for malformed wire data.

    Callers may report structural failures with reward zero. Valid text replies
    and decoded calls remain available for semantic grading.
    """
    try:
        validated = ChatAssistantMessage.model_validate(message)
        if validated.tool_calls:
            return AssistantToolCalls(
                calls=tuple(
                    ConversationToolCall(
                        call_id=call.id,
                        name=call.function.name,
                        arguments=ARGUMENT_OBJECT.validate_python(
                            json.loads(call.function.arguments, object_pairs_hook=unique_object)
                        ),
                    )
                    for call in validated.tool_calls
                ),
                content=validated.content,
            )
        if validated.content is None:
            raise ValueError("Chat response requires assistant content or function calls")
        return TextMessage(role="assistant", content=validated.content)
    except ValueError as error:
        raise SubmissionFailure("Assistant output is structurally invalid") from error


def chat_conversation(messages: list[dict[str, Any]]) -> ConversationTrace:
    """Normalize a complete chat transcript for any TaskCompendium verifier."""
    events: list[ConversationEvent] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            events.append(assistant_message(message))
        elif role == "tool":
            events.append(ToolResult(call_id=message["tool_call_id"], content=message["content"]))
        else:
            events.append(TextMessage.model_validate(message))
    return ConversationTrace(events=tuple(events))
