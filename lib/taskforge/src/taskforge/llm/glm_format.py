# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3's sampled output format, parsed client-side for the token-id route (pure, no I/O).

The token-id route returns ids and their bytes only, so the reasoning, the answer and the tool
calls that chat completions would report come from this parser. The served generation prompt ends
in ``<think>``, so a reply is the reasoning, ``</think>``, the answer, then zero or more
``<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>`` blocks, then the stop
token (``<|observation|>`` after tool calls, ``<|user|>`` after a reply). Parsing follows vLLM's
GLM tool parser: names and argument keys and values are stripped of surrounding whitespace, a value
is kept as a string when the tool's parameter schema types it ``string`` and is otherwise
``json.loads``-ed, and a value that does not parse stays a string. A reply cut off inside its
reasoning has no ``</think>``; all of it is reasoning. A tool call cut off before ``</tool_call>``
is not reported.
"""

import codecs
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from taskforge.llm.client import ToolCall

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
TOOL_CALL_OPEN, TOOL_CALL_CLOSE = "<tool_call>", "</tool_call>"
STOP_TEXTS = ("<|observation|>", "<|user|>", "<|endoftext|>")
"""What a stop token's bytes decode to when the route reports them as text."""

_TOOL_CALL = re.compile(re.escape(TOOL_CALL_OPEN) + r"(.*?)" + re.escape(TOOL_CALL_CLOSE), re.DOTALL)
_ARGUMENT = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)


@dataclass(frozen=True)
class ParsedTurn:
    reasoning: str
    content: str
    tool_calls: tuple[ToolCall, ...]
    """Complete calls only; ``arguments`` is a JSON object string."""
    reasoning_tokens: int
    """Tokens up to and including the one that completes ``</think>``; every token when there is none."""


def decode(token_bytes: Sequence[bytes]) -> str:
    """The UTF-8 text of the joined bytes; an incomplete trailing sequence (a length cut) is dropped.

    Raises:
        UnicodeDecodeError: invalid UTF-8 before the end.
    """
    return codecs.getincrementaldecoder("utf-8")().decode(b"".join(token_bytes), final=False)


def _reasoning_tokens(token_bytes: Sequence[bytes]) -> int:
    joined = b"".join(token_bytes)
    end = joined.find(THINK_CLOSE.encode())
    if end < 0:
        return len(token_bytes)
    end += len(THINK_CLOSE)
    consumed = 0
    for count, data in enumerate(token_bytes, start=1):
        consumed += len(data)
        if consumed >= end:
            return count
    raise AssertionError("</think> lies inside the joined bytes")


def _string_parameters(tools: Sequence[Mapping[str, object]], name: str) -> frozenset[str]:
    for tool in tools:
        function = tool["function"]
        assert isinstance(function, Mapping)
        if function.get("name") != name:
            continue
        parameters = function.get("parameters") or {}
        assert isinstance(parameters, Mapping)
        properties = parameters.get("properties") or {}
        assert isinstance(properties, Mapping)
        return frozenset(key for key, schema in properties.items() if schema.get("type") == "string")
    return frozenset()


def _argument_value(text: str, is_string: bool) -> object:
    value = text.strip()
    if is_string:
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _tool_call(body: str, tools: Sequence[Mapping[str, object]], call_id: str) -> ToolCall:
    split = body.find("<arg_key>")
    name = (body if split < 0 else body[:split]).strip()
    strings = _string_parameters(tools, name)
    arguments = {
        key.strip(): _argument_value(value, key.strip() in strings)
        for key, value in _ARGUMENT.findall("" if split < 0 else body[split:])
    }
    return ToolCall(id=call_id, name=name, arguments=json.dumps(arguments, ensure_ascii=False))


def parse_turn(token_bytes: Sequence[bytes], tools: Sequence[Mapping[str, object]], call_id_prefix: str) -> ParsedTurn:
    """Split one sampled reply into reasoning, answer and tool calls.

    Args:
        token_bytes: Each sampled token's bytes, in order, including the stop token.
        tools: The request's tool definitions, which decide which argument values stay strings.
        call_id_prefix: Tool call ids are ``f"{call_id_prefix}-{index}"``.
    """
    text = decode(token_bytes)
    for stop in STOP_TEXTS:
        if text.endswith(stop):
            text = text.removesuffix(stop)
            break
    reasoning_tokens = _reasoning_tokens(token_bytes)
    reasoning, closed, rest = text.partition(THINK_CLOSE)
    reasoning = reasoning.removeprefix(THINK_OPEN)
    if not closed:
        return ParsedTurn(reasoning=reasoning, content="", tool_calls=(), reasoning_tokens=reasoning_tokens)
    start = rest.find(TOOL_CALL_OPEN)
    content = rest if start < 0 else rest[:start]
    calls = tuple(
        _tool_call(body, tools, f"{call_id_prefix}-{index}")
        for index, body in enumerate(_TOOL_CALL.findall("" if start < 0 else rest[start:]))
    )
    return ParsedTurn(reasoning=reasoning, content=content, tool_calls=calls, reasoning_tokens=reasoning_tokens)
