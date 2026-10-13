# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from taskforge.llm.glm_format import decode, parse_turn

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "text": {"type": "string"}, "mode": {"type": "integer"}},
            },
        },
    },
    {"type": "function", "function": {"name": "shell", "parameters": {"properties": {"command": {"type": "string"}}}}},
]


def arguments(call) -> dict:
    return json.loads(call.arguments)


def test_plain_reply_splits_reasoning_and_answer():
    parsed = parse_turn([b"The user ", b"greets.", b"</think>", b"Hello", b"!", b"<|user|>"], TOOLS, "call-1")
    assert (parsed.reasoning, parsed.content, parsed.tool_calls) == ("The user greets.", "Hello!", ())
    assert parsed.reasoning_tokens == 3


def test_reply_cut_off_while_reasoning_is_all_reasoning():
    parsed = parse_turn([b"Let me ", b"think about"], TOOLS, "call-1")
    assert (parsed.reasoning, parsed.content, parsed.tool_calls) == ("Let me think about", "", ())
    assert parsed.reasoning_tokens == 2


def test_two_tool_calls_with_typed_values_and_ids():
    sampled = [
        b"two steps</think>",
        b"<tool_call>write<arg_key>path</arg_key><arg_value>a.txt</arg_value>",
        b"<arg_key>mode</arg_key><arg_value>644</arg_value></tool_call>",
        b"<tool_call>shell\n<arg_key>command</arg_key>\n<arg_value>cat a.txt</arg_value>\n</tool_call>",
        b"<|observation|>",
    ]
    parsed = parse_turn(sampled, TOOLS, "call-3")
    assert [(call.id, call.name) for call in parsed.tool_calls] == [("call-3-0", "write"), ("call-3-1", "shell")]
    assert arguments(parsed.tool_calls[0]) == {"path": "a.txt", "mode": 644}
    assert arguments(parsed.tool_calls[1]) == {"command": "cat a.txt"}
    assert parsed.content == ""


def test_multi_line_string_value_keeps_its_inner_lines():
    text = 'def f():\n    return "x"\n'
    sampled = [f"</think><tool_call>write<arg_key>text</arg_key><arg_value>{text}</arg_value></tool_call>".encode()]
    [call] = parse_turn(sampled, TOOLS, "c").tool_calls
    assert arguments(call) == {"text": text.strip()}


def test_value_that_does_not_parse_stays_a_string():
    sampled = [b"</think><tool_call>write<arg_key>mode</arg_key><arg_value>rw-r</arg_value></tool_call>"]
    [call] = parse_turn(sampled, TOOLS, "c").tool_calls
    assert arguments(call) == {"mode": "rw-r"}


def test_tool_call_cut_before_its_close_is_not_reported():
    sampled = [b"</think>first", b"<tool_call>shell<arg_key>command</arg_key><arg_value>ls"]
    parsed = parse_turn(sampled, TOOLS, "c")
    assert (parsed.content, parsed.tool_calls) == ("first", ())


def test_utf8_character_split_across_tokens_decodes_once_joined():
    snowman = "\N{SNOWMAN}".encode()
    assert decode([b"a", snowman[:1], snowman[1:], b"b"]) == "a\N{SNOWMAN}b"
    assert decode([b"a", snowman[:2]]) == "a"
