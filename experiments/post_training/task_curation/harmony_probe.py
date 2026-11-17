# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harmony (GPT-OSS) request rendering for the difficulty serving probe.

The rendering helpers are derived from vLLM's
``vllm/entrypoints/openai/parser/harmony_utils.py`` (Apache-2.0,
marin-community/vllm @ 01911be34fac) so the probe computes the same prompt
token count the serving engine builds for gpt-oss chat requests: system
preamble (default identity, reasoning effort, today's conversation start
date), optional developer message with function-tool descriptions, then the
request messages rendered for an assistant completion.

vLLM reads two environment knobs here; both default to the serving
configuration this pipeline runs (system instructions off, unpinned start
date): ``VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS`` and
``VLLM_SYSTEM_START_DATE``.
"""

import datetime as dt
import os
from collections.abc import Sequence
from typing import Any

from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    RenderConversationConfig,
    Role,
    SystemContent,
    TextContent,
    ToolDescription,
    load_harmony_encoding,
)

# vLLM envs.VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS defaults to False; keep the
# same "1" truthiness convention so an operator pinning it on the serving job can
# pin it here identically.
_HARMONY_SYSTEM_INSTRUCTIONS = os.environ.get("VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS", "") == "1"

REASONING_EFFORT: dict[str, ReasoningEffort] = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}

_ENCODING = None


def _get_encoding():
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _ENCODING


def flatten_input_text_content(content: Any) -> str | None:
    """Extract text parts from a content field and flatten them into one string."""
    if content is None or isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None

    texts: list[str] = []
    for item in content:
        if isinstance(item, str):
            texts.append(item)
            continue
        if isinstance(item, dict):
            text = item.get("text")
            if text is not None:
                texts.append(text)
    return "".join(texts) if texts else None


def get_system_message(
    model_identity: str | None = None,
    reasoning_effort: str | None = None,
    start_date: str | None = None,
    browser_description: str | None = None,
    python_description: str | None = None,
    container_description: str | None = None,
    instructions: str | None = None,
) -> Message:
    sys_msg_content = SystemContent.new()
    if model_identity is not None:
        sys_msg_content = sys_msg_content.with_model_identity(model_identity)
    if instructions is not None and _HARMONY_SYSTEM_INSTRUCTIONS:
        current_identity = sys_msg_content.model_identity
        new_identity = f"{current_identity}\n{instructions}" if current_identity else instructions
        sys_msg_content = sys_msg_content.with_model_identity(new_identity)
    if reasoning_effort is not None:
        if reasoning_effort not in REASONING_EFFORT:
            supported = ", ".join(REASONING_EFFORT)
            raise ValueError(
                f"reasoning_effort={reasoning_effort!r} is not supported by "
                f"Harmony. Supported values are: {supported}."
            )
        sys_msg_content = sys_msg_content.with_reasoning_effort(REASONING_EFFORT[reasoning_effort])
    if start_date is None:
        # NOTE(vLLM): this brings non-determinism; VLLM_SYSTEM_START_DATE pins it.
        start_date = os.environ.get("VLLM_SYSTEM_START_DATE") or dt.datetime.now().strftime("%Y-%m-%d")
    sys_msg_content = sys_msg_content.with_conversation_start_date(start_date)
    if browser_description is not None:
        sys_msg_content = sys_msg_content.with_tools(browser_description)
    if python_description is not None:
        sys_msg_content = sys_msg_content.with_tools(python_description)
    if container_description is not None:
        sys_msg_content = sys_msg_content.with_tools(container_description)
    return Message.from_role_and_content(Role.SYSTEM, sys_msg_content)


def create_tool_definition(tool: dict[str, Any]):
    function = tool.get("function") or tool
    return ToolDescription.new(
        name=function["name"],
        description=function.get("description") or "",
        parameters=function.get("parameters"),
    )


def get_developer_message(
    instructions: str | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> Message:
    dev_msg_content = DeveloperContent.new()
    if instructions is not None and not _HARMONY_SYSTEM_INSTRUCTIONS:
        dev_msg_content = dev_msg_content.with_instructions(instructions)
    if tools is not None:
        function_tools: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") in ("web_search_preview", "code_interpreter", "container"):
                continue
            if tool.get("type") == "function":
                function_tools.append(tool)
            else:
                raise ValueError(f"tool type {tool.get('type')!r} is not supported.")
        if function_tools:
            dev_msg_content = dev_msg_content.with_function_tools(
                [create_tool_definition(tool) for tool in function_tools]
            )
    return Message.from_role_and_content(Role.DEVELOPER, dev_msg_content)


def build_harmony_preamble(
    *,
    instructions: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    reasoning_effort: str | None = None,
) -> list[Message]:
    """Build the standard Harmony system/developer prefix for a request."""
    developer_instructions = system_instructions = None
    if _HARMONY_SYSTEM_INSTRUCTIONS:
        system_instructions = instructions
    else:
        developer_instructions = instructions

    messages = [get_system_message(reasoning_effort=reasoning_effort, instructions=system_instructions)]
    if developer_instructions or tools:
        messages.append(get_developer_message(instructions=developer_instructions, tools=tools))
    return messages


def extract_instructions_from_messages(
    messages: Sequence[Any],
) -> tuple[str | None, list[Any]]:
    """Peel a leading system/developer message and flatten its instruction text."""
    remaining_messages = list(messages)
    if not remaining_messages:
        return None, remaining_messages

    first_message = remaining_messages[0]
    if not isinstance(first_message, dict):
        if hasattr(first_message, "to_dict"):
            first_message = first_message.to_dict()
        elif hasattr(first_message, "model_dump"):
            first_message = first_message.model_dump(exclude_none=True)
        else:
            raise ValueError(f"Unknown message type: {type(first_message)}")

    if first_message.get("role") not in ("system", "developer"):
        return None, remaining_messages

    instructions = flatten_input_text_content(first_message.get("content"))
    return instructions, remaining_messages[1:]


def parse_chat_input_to_harmony_message(chat_msg, tool_id_names: dict[str, str] | None = None) -> list[Message]:
    """Parse one Chat Completion message into Harmony messages."""
    tool_id_names = tool_id_names or {}

    if not isinstance(chat_msg, dict):
        chat_msg = chat_msg.model_dump(exclude_none=True)

    role = chat_msg.get("role")
    msgs: list[Message] = []

    tool_calls = chat_msg.get("tool_calls", [])
    if role == "assistant" and tool_calls:
        content = flatten_input_text_content(chat_msg.get("content"))
        if content:
            commentary_msg = Message.from_role_and_content(Role.ASSISTANT, content)
            commentary_msg = commentary_msg.with_channel("commentary")
            msgs.append(commentary_msg)

        reasoning = chat_msg.get("reasoning")
        if reasoning:
            analysis_msg = Message.from_role_and_content(Role.ASSISTANT, reasoning)
            analysis_msg = analysis_msg.with_channel("analysis")
            msgs.append(analysis_msg)

        for call in tool_calls:
            func = call.get("function", {})
            name = func.get("name", "")
            arguments = func.get("arguments", "") or ""
            msg = Message.from_role_and_content(Role.ASSISTANT, arguments)
            msg = msg.with_channel("commentary")
            msg = msg.with_recipient(f"functions.{name}")
            msg = msg.with_content_type("json")
            msgs.append(msg)
        return msgs

    if role == "tool":
        tool_call_id = chat_msg.get("tool_call_id", "")
        name = tool_id_names.get(tool_call_id, "")
        content = flatten_input_text_content(chat_msg.get("content")) or ""
        msg = (
            Message.from_author_and_content(Author.new(Role.TOOL, f"functions.{name}"), content)
            .with_channel("commentary")
            .with_recipient("assistant")
        )
        return [msg]

    reasoning = chat_msg.get("reasoning")
    if role == "assistant" and reasoning:
        analysis_msg = Message.from_role_and_content(Role.ASSISTANT, reasoning)
        analysis_msg = analysis_msg.with_channel("analysis")
        msgs.append(analysis_msg)

    content = chat_msg.get("content") or ""
    if content is None:
        content = ""
    if isinstance(content, str):
        contents = [TextContent(text=content)]
    else:
        contents = [TextContent(text=c.get("text", "")) for c in content]

    if role == "assistant" and contents and contents[0].text:
        msg = Message.from_role_and_contents(role, contents)
        msg = msg.with_channel("final")
        msgs.append(msg)
    elif role in ("system", "developer"):
        instructions = flatten_input_text_content(chat_msg.get("content"))
        if instructions is not None:
            if role == "system" and _HARMONY_SYSTEM_INSTRUCTIONS:
                msgs.append(get_system_message(instructions=instructions))
            else:
                msgs.append(get_developer_message(instructions=instructions))
    elif role != "assistant":
        msg = Message.from_role_and_contents(role, contents)
        msgs.append(msg)

    return msgs


def parse_chat_inputs_to_harmony_messages(chat_msgs: list) -> list[Message]:
    """Parse request.messages into Harmony messages."""
    msgs: list[Message] = []
    tool_id_names: dict[str, str] = {}
    for chat_msg in chat_msgs:
        for tool_call in chat_msg.get("tool_calls", []) or []:
            tool_id_names[tool_call.get("id")] = tool_call.get("function", {}).get("name")
    for chat_msg in chat_msgs:
        msgs.extend(parse_chat_input_to_harmony_message(chat_msg, tool_id_names))
    return msgs


def auto_drop_analysis_messages(msgs: list[Message]) -> list[Message]:
    """Drop analysis messages that precede the last assistant final-channel message."""
    last_assistant_final_index = -1
    for i in range(len(msgs) - 1, -1, -1):
        msg = msgs[i]
        if msg.author.role == "assistant" and msg.channel == "final":
            last_assistant_final_index = i
            break

    cleaned_msgs: list[Message] = []
    for i, msg in enumerate(msgs):
        if i < last_assistant_final_index and msg.channel == "analysis":
            continue
        cleaned_msgs.append(msg)
    return cleaned_msgs


def render_for_completion(messages: list[Message]) -> list[int]:
    """Render prompt token ids for an assistant completion, as the engine does."""
    messages = auto_drop_analysis_messages(messages)
    conversation = Conversation.from_messages(messages)
    return _get_encoding().render_conversation_for_completion(
        conversation,
        Role.ASSISTANT,
        config=RenderConversationConfig(auto_drop_analysis=False),
    )


def harmony_prompt_tokens(request: dict[str, Any]) -> int:
    """Prompt token count the serving engine builds for this chat request.

    Mirrors the gpt-oss serving path: peel leading instructions, build the
    system/developer preamble from the request's tools and reasoning effort,
    parse the remaining messages, render for an assistant completion.
    """
    instructions, chat_messages = extract_instructions_from_messages(request["messages"])
    messages = build_harmony_preamble(
        instructions=instructions,
        tools=request.get("tools"),
        reasoning_effort=request.get("reasoning_effort"),
    )
    messages.extend(parse_chat_inputs_to_harmony_messages(chat_messages))
    return len(render_for_completion(messages))
