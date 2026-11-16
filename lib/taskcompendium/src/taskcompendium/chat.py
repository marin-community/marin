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
    ReplyStatus,
    TextMessage,
    ToolResult,
)

TEXT_BLOCKS = frozenset({"input_text", "output_text", "text"})
TOOL_EVENT_KEYS = frozenset({"tool_calls", "function_call", "annotations"})
TOOL_EVENT_TYPES = frozenset({"tool_use", "tool_result", "function_call", "web_search_call"})
CONTROL_REPLY_STATUS = ReplyStatus(
    origin="control",
    finish_reason="control",
    truncated=False,
    provider_refusal=False,
    tool_events=(),
)


def tool_event_paths(value: Any, path: str = "response") -> tuple[str, ...]:
    """Locate tool calls and tool-derived annotations throughout a provider reply."""
    events: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in TOOL_EVENT_KEYS and item:
                events.append(f"{path}.{key}")
            if key == "type" and isinstance(item, str) and item in TOOL_EVENT_TYPES:
                events.append(f"{path}.type:{item}")
            events.extend(tool_event_paths(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            events.extend(tool_event_paths(item, f"{path}[{index}]"))
    return tuple(events)


def chat_reply_status(response: dict[str, Any]) -> ReplyStatus:
    """Capture one chat completion's status before discarding provider-only fields."""
    (choice,) = response["choices"]
    finish_reason = choice.get("finish_reason")
    return ReplyStatus(
        origin="provider",
        finish_reason=finish_reason,
        truncated=finish_reason == "length",
        provider_refusal=bool(choice["message"].get("refusal")),
        tool_events=tool_event_paths(response),
    )


def message_text(content: Any) -> str:
    """Join text blocks, rejecting media rather than silently dropping their context."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("Message content must be text or text blocks")
    if any(block.get("type") not in TEXT_BLOCKS for block in content):
        raise ValueError("Unsupported nontext message content")
    return "".join(block["text"] for block in content)


class ChatFunction(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    name: str = Field(min_length=1)
    arguments: JsonValue

    @field_validator("arguments", mode="before")
    @classmethod
    def decode_arguments(cls, value: str) -> JsonValue:
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

    Invalid JSON raises at this ingestion boundary. JSON arguments of the wrong
    shape remain evidence; execution and submission validation reject them.
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
    # Reasoning-only provider responses have no answer, even when finish_reason is stop.
    # An empty final submission is valid evidence; the task grader decides its score.
    return TextMessage(role="assistant", content=validated.content or "")


def _conversation_events(messages: list[dict[str, Any]]) -> tuple[ConversationEvent, ...]:
    events: list[ConversationEvent] = []
    for message in messages:
        if isinstance(message.get("content"), list):
            message = {**message, "content": message_text(message["content"])}
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
