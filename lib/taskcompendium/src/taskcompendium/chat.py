# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert OpenAI chat messages to typed conversation evidence."""

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator
from verifyit.json_objects import unique_object

from taskcompendium.models import (
    AssistantToolCalls,
    ConversationEvent,
    ConversationInput,
    ConversationToolCall,
    ConversationTrace,
    TextMessage,
    ToolResult,
)


class ChatFunction(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    name: str = Field(min_length=1)
    arguments: dict[str, JsonValue]

    @field_validator("arguments", mode="before")
    @classmethod
    def decode_arguments(cls, value: str) -> dict[str, JsonValue]:
        if not isinstance(value, str):
            raise ValueError("Provider tool arguments must be a JSON string")
        return json.loads(value, object_pairs_hook=unique_object)


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
    """Validate chat wire data and return protocol-independent submission evidence.

    Malformed protocol data raises at this harness boundary. A valid text reply,
    wrong function name, or wrong arguments remain available for grading.
    """
    validated = ChatAssistantMessage.model_validate(message)
    if validated.tool_calls:
        return AssistantToolCalls(
            calls=tuple(
                ConversationToolCall(call_id=call.id, name=call.function.name, arguments=call.function.arguments)
                for call in validated.tool_calls
            ),
            content=validated.content,
        )
    if validated.content is None:
        raise ValueError("Chat response requires assistant content or function calls")
    return TextMessage(role="assistant", content=validated.content)


def _conversation_events(messages: list[dict[str, Any]]) -> tuple[ConversationEvent, ...]:
    events: list[ConversationEvent] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            events.append(assistant_message(message))
        elif role == "tool":
            events.append(ToolResult(call_id=message["tool_call_id"], content=message["content"]))
        else:
            events.append(TextMessage.model_validate(message))
    return tuple(events)


def chat_input(messages: list[dict[str, Any]]) -> ConversationInput:
    """Normalize a public task prefix before model inference."""
    return ConversationInput(events=_conversation_events(messages))


def chat_conversation(messages: list[dict[str, Any]]) -> ConversationTrace:
    """Normalize a complete chat transcript for a TaskCompendium verifier."""
    return ConversationTrace(events=_conversation_events(messages))
