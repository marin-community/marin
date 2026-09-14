# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parse the Terminus JSON-command conversation protocol."""

import json
import re

TASK_DESCRIPTION_MARKER = "Task Description:"
TERMINAL_TOOL = {
    "type": "function",
    "name": "terminal",
    "description": "Send one or more commands or keystroke sequences to the task terminal.",
    "parameters": {
        "type": "object",
        "properties": {
            "commands": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "keystrokes": {"type": "string"},
                        "duration": {"type": "number"},
                    },
                    "required": ["keystrokes"],
                },
            }
        },
        "required": ["commands"],
    },
}


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


def _reasoning_content(content: str, payload: dict) -> str:
    match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if match is not None:
        reasoning = match.group(1).strip()
    else:
        reasoning = "\n\n".join(
            value.strip()
            for key in ("analysis", "plan")
            if isinstance((value := payload.get(key)), str) and value.strip()
        )
    reasoning = re.sub(r"</?think>|<\|(start|end)_think\|>", "", reasoning).strip()
    return f"<think>{reasoning}</think>" if reasoning else ""


def terminus_protocol_messages(conversations: list[dict]) -> tuple[list[dict], dict] | None:
    """Parse Terminus JSON command batches into source turns and tool definitions."""
    messages: list[dict] = []
    pending_call: tuple[str, str] | None = None
    for index, message in enumerate(conversations):
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            return None
        if role == "user":
            if pending_call is not None:
                call_id, tool_name = pending_call
                messages.append({"role": "tool", "content": content, "name": tool_name, "tool_call_id": call_id})
                pending_call = None
                continue
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
        reasoning = _reasoning_content(content, payload)
        commands = payload["commands"]
        if not commands and payload.get("task_complete"):
            messages.append({"role": "assistant", "content": f"{reasoning}\n\nTask complete.".strip()})
            continue
        call_id = f"call_terminal_{index}"
        messages.append(
            {
                "role": "assistant",
                "content": reasoning,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "terminal", "arguments": {"commands": commands}},
                    }
                ],
            }
        )
        pending_call = (call_id, "terminal")
    if not messages or messages[-1]["role"] != "assistant":
        return None
    return messages, {"chat_template_kwargs": {"tools": [TERMINAL_TOOL]}}
