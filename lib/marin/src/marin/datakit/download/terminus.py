# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parse the Terminus JSON-command conversation protocol."""

import json

from marin.datakit.download.rollout_transforms import (
    REASONING_END,
    REASONING_START,
    TOOL_WRAPPER,
    ReasoningFormatError,
    normalize_reasoning_tokens,
)


def _json_command_payload_and_prefix(content: str) -> tuple[dict, str] | None:
    decoder = json.JSONDecoder()
    search_start = 0
    if content.lstrip().lower().startswith("<think>"):
        end = content.lower().find("</think>")
        if end != -1:
            search_start = end + len("</think>")
    elif content.lstrip().startswith(REASONING_START):
        end = content.find(REASONING_END)
        if end != -1:
            search_start = end + len(REASONING_END)
    for index in range(search_start, len(content)):
        char = content[index]
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("commands"), list):
            return payload, content[:index].strip()
    return None


def terminus_protocol_messages(conversations: list[dict]) -> list[dict] | None:
    """Retain Terminus JSON responses and terminal observations as chat turns."""
    messages: list[dict] = []
    pending_observation = False
    for message in conversations:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            return None
        if role == "user":
            if not content.strip() or (pending_observation and TOOL_WRAPPER.search(content)):
                return None
            messages.append({"role": "user", "content": content})
            pending_observation = False
            continue
        if role != "assistant":
            messages.append(dict(message))
            continue

        parsed = _json_command_payload_and_prefix(content)
        if parsed is None:
            return None
        payload, prefix = parsed
        response = {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)}
        if prefix.lower().startswith(("<think>", REASONING_START)):
            try:
                normalized_prefix = normalize_reasoning_tokens(prefix)
            except ReasoningFormatError:
                return None
            reasoning, separator, _ = normalized_prefix.removeprefix(REASONING_START).partition(REASONING_END)
            if not normalized_prefix.startswith(REASONING_START) or not separator or not reasoning.strip():
                return None
            response["reasoning_content"] = reasoning.strip()
        messages.append(response)
        pending_observation = bool(payload["commands"]) or not payload.get("task_complete")
    if not messages or messages[-1]["role"] != "assistant":
        return None
    return messages
