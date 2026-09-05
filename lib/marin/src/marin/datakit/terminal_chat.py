# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adapters for terminal-agent conversations that use JSON command batches."""

import json
import re

TASK_DESCRIPTION_MARKER = "Task Description:"
INLINE_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
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
    return f"<think>{reasoning}</think>" if reasoning else ""


def terminal_protocol_messages(conversations: list[dict]) -> tuple[list[dict], dict] | None:
    """Convert a Terminus JSON-command transcript to canonical tool messages."""
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
            if not messages and TASK_DESCRIPTION_MARKER in content:
                content = content[content.index(TASK_DESCRIPTION_MARKER) :]
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
        if not commands:
            final = "Task complete." if payload.get("task_complete") else "No terminal action is needed."
            messages.append({"role": "assistant", "content": f"{reasoning}\n\n{final}".strip()})
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


def _tool_schema(name: str, arguments: dict) -> dict:
    properties = {}
    for key, value in arguments.items():
        if isinstance(value, bool):
            json_type = "boolean"
        elif isinstance(value, (int, float)):
            json_type = "number"
        elif isinstance(value, list):
            json_type = "array"
        elif isinstance(value, dict):
            json_type = "object"
        else:
            json_type = "string"
        properties[key] = {"type": json_type}
    return {
        "type": "function",
        "name": name,
        "description": f"Execute the {name} tool.",
        "parameters": {"type": "object", "properties": properties, "required": list(arguments)},
    }


def opencode_protocol_messages(
    conversations: list[dict], initial_user_content: str | None = None
) -> tuple[list[dict], dict] | None:
    """Convert inline OpenCode tool tags and observations to canonical messages."""
    messages: list[dict] = []
    tools: dict[str, dict] = {}
    pending_calls: list[tuple[str, str]] = []
    for index, message in enumerate(conversations):
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            return None
        if role == "user":
            if pending_calls:
                call_id, tool_name = pending_calls.pop(0)
                messages.append({"role": "tool", "content": content, "name": tool_name, "tool_call_id": call_id})
                continue
            if not messages and not content.strip() and initial_user_content is not None:
                content = initial_user_content
            if not content.strip():
                return None
            messages.append({"role": "user", "content": content})
            continue
        if role != "assistant":
            messages.append(dict(message))
            continue

        encoded_calls = INLINE_TOOL_CALL.findall(content)
        if not encoded_calls:
            messages.append({"role": "assistant", "content": content})
            continue
        tool_calls = []
        for call_index, encoded_call in enumerate(encoded_calls):
            try:
                call = json.loads(encoded_call)
            except json.JSONDecodeError:
                return None
            if (
                not isinstance(call, dict)
                or not isinstance(call.get("name"), str)
                or not isinstance(call.get("arguments"), dict)
            ):
                return None
            call_id = f"call_{call['name']}_{index}_{call_index}"
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]},
                }
            )
            tools[call["name"]] = _tool_schema(call["name"], call["arguments"])
            pending_calls.append((call_id, call["name"]))
        assistant_content = INLINE_TOOL_CALL.sub("", content).strip()
        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
        )
    if not messages or messages[-1]["role"] != "assistant":
        return None
    return messages, {"chat_template_kwargs": {"tools": list(tools.values())}}


def agent_protocol_messages(conversations: list[dict]) -> tuple[list[dict], dict] | None:
    """Dispatch a terminal-agent transcript to its provider-format adapter."""
    if any(INLINE_TOOL_CALL.search(message.get("content") or "") for message in conversations):
        return opencode_protocol_messages(conversations)
    return terminal_protocol_messages(conversations)
