# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalization helpers for trace-like conversation datasets."""

import hashlib
import json
import re
from typing import Any

from marin.core.conversation import OpenAIChatMessage

_HERMES_TOOL_RESPONSE_RE = re.compile(
    r"^\s*<tool_response(?P<attrs>[^>]*)>\s*(?P<body>.*?)\s*</tool_response>\s*$",
    re.DOTALL,
)
_HERMES_TOOL_RESPONSE_ATTR_RE = re.compile(r"""(?P<key>name|id)\s*=\s*(?P<quote>["'])(?P<value>.*?)(?P=quote)""")


def _hash_messages(messages: list[dict[str, Any]]) -> str:
    return hashlib.sha256(str(messages).encode()).hexdigest()


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _parse_tool_response_attrs(attrs: str) -> tuple[str | None, str | None]:
    name: str | None = None
    tool_call_id: str | None = None

    for match in _HERMES_TOOL_RESPONSE_ATTR_RE.finditer(attrs):
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


def _normalize_hermes_tool_response_message(message: OpenAIChatMessage) -> OpenAIChatMessage:
    if message.role != "tool" or not isinstance(message.content, str):
        return message

    match = _HERMES_TOOL_RESPONSE_RE.fullmatch(message.content)
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


def normalize_hermes_trace_messages(
    messages: list[OpenAIChatMessage],
    row: dict[str, Any],
) -> list[OpenAIChatMessage]:
    """Normalize Hermes trace messages for Marin's conversation pipeline.

    Hermes assistant turns already contain the desired `<think>` and `<tool_call>` blocks, so we
    leave them untouched. Tool turns arrive wrapped in `<tool_response>` tags; Marin's chat
    template would wrap those again, so we strip only the outer wrapper when the payload parses
    cleanly. If wrapper parsing fails, we preserve the raw source content unchanged.
    """

    return [_normalize_hermes_tool_response_message(message) for message in messages]


def hermes_trace_row_id(row: dict[str, Any], messages: list[dict[str, Any]]) -> str:
    """Return the source trace ID when available, otherwise fall back to the message hash."""

    source_id = row.get("id")
    if isinstance(source_id, str) and source_id:
        return source_id
    return _hash_messages(messages)
