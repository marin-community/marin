# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize OpenAI chat messages at the Harbor harness boundary."""

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter
from verifyit.json_objects import unique_object

from taskcompendium.models import (
    AssistantToolCalls,
    ConversationEvent,
    ConversationToolCall,
    ConversationTrace,
    RawAssistantToolCalls,
    RawToolCall,
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


def assistant_message(message: dict[str, Any]) -> TextMessage | RawAssistantToolCalls:
    """Validate chat wire data and return protocol-independent submission evidence.

    Malformed protocol data raises at this harness boundary. A valid text reply,
    wrong function name, or wrong arguments remain available for grading.
    """
    validated = ChatAssistantMessage.model_validate(message)
    if validated.tool_calls:
        return RawAssistantToolCalls(
            calls=tuple(
                RawToolCall(call_id=call.id, name=call.function.name, arguments_json=call.function.arguments)
                for call in validated.tool_calls
            ),
            content=validated.content,
        )
    if validated.content is None:
        raise ValueError("Chat response requires assistant content or function calls")
    return TextMessage(role="assistant", content=validated.content)


def chat_conversation(messages: list[dict[str, Any]]) -> ConversationTrace:
    """Normalize a complete chat transcript for any TaskCompendium verifier."""
    events: list[ConversationEvent | RawAssistantToolCalls] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant":
            response = assistant_message(message)
            if isinstance(response, RawAssistantToolCalls) and index != len(messages) - 1:
                response = AssistantToolCalls(
                    calls=tuple(
                        ConversationToolCall(
                            call_id=call.call_id,
                            name=call.name,
                            arguments=ARGUMENT_OBJECT.validate_python(
                                json.loads(call.arguments_json, object_pairs_hook=unique_object)
                            ),
                        )
                        for call in response.calls
                    ),
                    content=response.content,
                )
            events.append(response)
        elif role == "tool":
            events.append(ToolResult(call_id=message["tool_call_id"], content=message["content"]))
        else:
            events.append(TextMessage.model_validate(message))
    return ConversationTrace(events=tuple(events))
