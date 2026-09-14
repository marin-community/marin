# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parse OpenCode inline tool calls and observations."""

import json
import re

from zephyr import counters

INLINE_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_USE_PLACEHOLDER = "(tool use)"


def prompt_tool_definitions(prompt: str) -> list[dict] | None:
    """Read the definitions actually presented in a Qwen tool prompt."""
    match = re.search(r"<tools>\s*(.*?)\s*</tools>", prompt, re.DOTALL)
    if match is None:
        return None
    tools = [json.loads(line) for line in match.group(1).splitlines() if line.strip()]
    if not tools or not all(isinstance(tool, dict) for tool in tools):
        raise ValueError("OpenCode prompt must contain function definitions")
    return tools


def opencode_conversation(conversations: list[dict], prompt: str) -> tuple[list[dict], list[dict]]:
    """Recover the initial turns and tool definitions from the served literal prompt."""
    tools = prompt_tool_definitions(prompt)
    if tools is None:
        raise ValueError("OpenCode source is missing explicit tool definitions")
    leading = re.findall(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", prompt, re.DOTALL)
    if len(leading) != 2 or [role for role, _ in leading] != ["system", "user"]:
        raise ValueError("OpenCode first prompt must contain system and user turns")
    system = leading[0][1]
    _, separator, system_instructions = system.partition("</IMPORTANT>")
    if not separator or not system_instructions.strip():
        raise ValueError("OpenCode prompt is missing the harness system instructions after its tool preamble")
    if not conversations or conversations[0].get("role") != "user":
        raise ValueError("OpenCode conversation is missing its initial user turn")
    return [
        {"role": "system", "content": system_instructions.lstrip()},
        {"role": "user", "content": leading[1][1]},
        *conversations[1:],
    ], tools


def opencode_protocol_messages(conversations: list[dict], tools: list[dict]) -> tuple[list[dict], dict] | None:
    """Parse OpenCode tool tags into source turns and tool definitions."""
    messages: list[dict] = []
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
                if consecutive_observations < len(pending_calls):
                    return None
                call_id, tool_name = pending_calls.pop(0)
                messages.append({"role": "tool", "content": content, "name": tool_name, "tool_call_id": call_id})
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
            # Exports can split a placeholder from the assistant's actual continuation.
            if (
                content.rstrip().endswith(TOOL_USE_PLACEHOLDER)
                and index + 1 < len(conversations)
                and conversations[index + 1].get("role") == "assistant"
            ):
                content = content.rstrip().removesuffix(TOOL_USE_PLACEHOLDER).rstrip()
                counters.pipeline.update_counter("opencode/continuation_placeholder_removed", 1)
                if not content:
                    continue
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
    return messages, {"chat_template_kwargs": {"tools": tools}}
