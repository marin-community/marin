# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parse the Terminus JSON-command conversation protocol."""

import json


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


def terminus_protocol_messages(conversations: list[dict]) -> tuple[list[dict], dict] | None:
    """Retain Terminus JSON responses and terminal observations as chat turns."""
    messages: list[dict] = []
    for message in conversations:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            return None
        if role == "user":
            if not content.strip():
                return None
            messages.append({"role": "user", "content": content})
            continue
        if role != "assistant":
            messages.append(dict(message))
            continue

        payload = _json_command_payload(content)
        if payload is None:
            return None
        messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)})
    if not messages or messages[-1]["role"] != "assistant":
        return None
    return messages, {}
