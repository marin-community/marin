# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parse the Terminus JSON-command conversation protocol."""

import json

from marin.datakit.download.rollout_transforms import TOOL_WRAPPER


def _json_command_payload(content: str) -> dict | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(content):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("commands"), list):
            return payload
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

        payload = _json_command_payload(content)
        if payload is None:
            return None
        messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)})
        pending_observation = bool(payload["commands"]) or not payload.get("task_complete")
    if not messages or messages[-1]["role"] != "assistant":
        return None
    return messages
