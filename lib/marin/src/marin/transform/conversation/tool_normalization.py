# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared normalization helpers for OpenAI-style tool messages."""

import json
import re
from typing import Any

from marin.core.conversation import OpenAIChatMessage

_TOOL_RESPONSE_RE = re.compile(
    r"^\s*<tool_response(?P<attrs>[^>]*)>\s*(?P<body>.*?)\s*</tool_response>\s*$",
    re.DOTALL,
)
_TOOL_RESPONSE_ATTR_RE = re.compile(r"""(?P<key>name|id)\s*=\s*(?P<quote>["'])(?P<value>.*?)(?P=quote)""")


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def normalize_tool_call_arguments(message: dict[str, Any]) -> dict[str, Any]:
    """Parse JSON-string function arguments in OpenAI-style ``tool_calls``."""
    tool_calls = message.get("tool_calls")
    if not tool_calls:
        return message

    normalized_calls: list[dict[str, Any]] = []
    for call in tool_calls:
        call_dict = dict(call)
        function = call_dict.get("function")
        if isinstance(function, dict):
            function_dict = dict(function)
            arguments = function_dict.get("arguments")
            if isinstance(arguments, str):
                try:
                    function_dict["arguments"] = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            call_dict["function"] = function_dict
        normalized_calls.append(call_dict)

    return {**message, "tool_calls": normalized_calls}


def _parse_tool_response_attrs(attrs: str) -> tuple[str | None, str | None]:
    name: str | None = None
    tool_call_id: str | None = None

    for match in _TOOL_RESPONSE_ATTR_RE.finditer(attrs):
        key = match.group("key")
        value = match.group("value")
        if key == "name":
            name = value
        elif key == "id":
            tool_call_id = value

    return name, tool_call_id


def _parse_tool_response_body(
    body: str,
    *,
    name: str | None,
    tool_call_id: str | None,
) -> tuple[str | None, str | None, Any] | None:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    normalized_name = name or _string_or_none(payload.get("name"))
    normalized_tool_call_id = tool_call_id or _string_or_none(payload.get("tool_call_id"))
    if "content" in payload:
        return normalized_name, normalized_tool_call_id, payload["content"]

    return normalized_name, normalized_tool_call_id, payload


def normalize_wrapped_tool_response_message(message: OpenAIChatMessage) -> OpenAIChatMessage:
    """Unwrap a tool message whose content is a JSON ``<tool_response>`` block.

    Marin's chat template renders structured tool messages inside
    ``<tool_response>`` tags. Some trace datasets already store the tool result
    in that rendered form, so unwrapping avoids nesting one response block
    inside another while preserving malformed source content unchanged.
    """
    if message.role != "tool" or not isinstance(message.content, str):
        return message

    match = _TOOL_RESPONSE_RE.fullmatch(message.content)
    if match is None:
        return message

    name, tool_call_id = _parse_tool_response_attrs(match.group("attrs"))
    parsed_body = _parse_tool_response_body(
        match.group("body").strip(),
        name=name,
        tool_call_id=tool_call_id,
    )
    if parsed_body is None:
        return message

    normalized_name, normalized_tool_call_id, normalized_content = parsed_body
    return message.model_copy(
        update={
            "content": normalized_content,
            "name": normalized_name,
            "tool_call_id": normalized_tool_call_id,
        }
    )


def normalize_wrapped_tool_response_messages(
    messages: list[OpenAIChatMessage],
    row: dict[str, Any],
) -> list[OpenAIChatMessage]:
    """Unwrap rendered tool responses in a conversation message list."""
    del row
    return [normalize_wrapped_tool_response_message(message) for message in messages]
