# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Discover model-output conventions from tokenizer chat templates."""

from dataclasses import dataclass
from enum import StrEnum

# The browser repeats these as a fallback when /info metadata is absent or stale;
# Python and TypeScript cannot share this runtime constant.
_THINKING_DELIMITERS = (
    ("<|start_think|>", "<|end_think|>"),
    ("<think>", "</think>"),
    ("<THINK>", "</THINK>"),
)
_TOOL_CALL_DELIMITERS = (
    ("<|tool_call|>", "<|tool_call_end|>"),
    ("<tool_call>", "</tool_call>"),
)
_JSON_TOOL_CALL_MARKER = (
    'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.'
)


class ToolCallFormat(StrEnum):
    DELIMITED = "delimited"
    JSON = "json"


@dataclass(frozen=True)
class ChatTemplateProtocol:
    """Reasoning and tool-call output conventions used by a chat template."""

    thinking_start: str | None = None
    thinking_end: str | None = None
    tool_call_start: str | None = None
    tool_call_end: str | None = None
    tool_call_format: ToolCallFormat | None = None


def _template_delimiters(template: str | None, candidates: tuple[tuple[str, str], ...]) -> tuple[str | None, str | None]:
    if template is None:
        return None, None
    for start, end in candidates:
        if start in template and end in template:
            return start, end
    return None, None


def protocol_for_chat_template(template: str | None) -> ChatTemplateProtocol:
    """Describe the reasoning and tool-call syntax emitted by a chat template."""
    thinking_start, thinking_end = _template_delimiters(template, _THINKING_DELIMITERS)
    tool_call_start, tool_call_end = _template_delimiters(template, _TOOL_CALL_DELIMITERS)
    if tool_call_start is not None:
        tool_call_format = ToolCallFormat.DELIMITED
    elif template is not None and _JSON_TOOL_CALL_MARKER in template:
        tool_call_format = ToolCallFormat.JSON
    else:
        tool_call_format = None
    return ChatTemplateProtocol(
        thinking_start=thinking_start,
        thinking_end=thinking_end,
        tool_call_start=tool_call_start,
        tool_call_end=tool_call_end,
        tool_call_format=tool_call_format,
    )
