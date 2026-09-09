# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parse OpenCode inline tool calls and observations."""

import json
import re

INLINE_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _tool_schema(name: str, invocations: list[dict]) -> dict:
    properties = {}
    required = set(invocations[0])
    observed_types: dict[str, set[str]] = {}
    for arguments in invocations:
        required.intersection_update(arguments)
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
            observed_types.setdefault(key, set()).add(json_type)
    for key, json_types in observed_types.items():
        properties[key] = {"type": sorted(json_types) if len(json_types) > 1 else next(iter(json_types))}
    return {
        "type": "function",
        "name": name,
        "description": f"Execute the {name} tool.",
        "parameters": {"type": "object", "properties": properties, "required": sorted(required)},
    }


def opencode_protocol_messages(conversations: list[dict]) -> tuple[list[dict], dict] | None:
    """Parse OpenCode tool tags into source turns and tool definitions."""
    messages: list[dict] = []
    tool_invocations: dict[str, list[dict]] = {}
    pending_calls: list[tuple[str, str]] = []
    for index, message in enumerate(conversations):
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            return None
        if role in {"tool", "user"}:
            if pending_calls:
                consecutive_observations = 0
                for following in conversations[index:]:
                    if following.get("role") not in {"tool", "user"}:
                        break
                    consecutive_observations += 1
                calls_for_observation = (
                    pending_calls[:1] if consecutive_observations >= len(pending_calls) else pending_calls[:]
                )
                for call_id, tool_name in calls_for_observation:
                    messages.append({"role": "tool", "content": content, "name": tool_name, "tool_call_id": call_id})
                del pending_calls[: len(calls_for_observation)]
                continue
            if role == "tool":
                return None
            if not content.strip():
                return None
            messages.append({"role": "user", "content": content})
            continue
        if role != "assistant":
            messages.append(dict(message))
            continue

        if pending_calls:
            return None

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
            tool_invocations.setdefault(call["name"], []).append(call["arguments"])
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
    tools = [_tool_schema(name, invocations) for name, invocations in tool_invocations.items()]
    return messages, {"chat_template_kwargs": {"tools": tools}}
