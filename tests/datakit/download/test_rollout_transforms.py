# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
from marin.datakit.download.rollout_transforms import (
    canonical_chat_messages,
    chat_document,
    render_tool_call,
    render_tool_message,
)


def test_canonical_chat_messages_normalizes_role_aliases_and_preserves_tools():
    tool_calls = [{"id": "call_1", "function": {"name": "bash", "arguments": {"cmd": "pwd"}}}]
    messages = canonical_chat_messages(
        [
            {"from": "human", "value": "Inspect the repository."},
            {"from": "gpt", "value": None, "tool_calls": tool_calls},
            {"role": "function", "content": "/workspace", "tool_call_id": "call_1"},
        ]
    )

    assert messages == [
        {"role": "user", "content": "Inspect the repository."},
        {
            "tool_calls": [{"id": "call_1", "function": {"name": "bash", "arguments": '{"cmd":"pwd"}'}}],
            "role": "assistant",
            "content": None,
        },
        {"role": "tool", "content": "/workspace", "tool_call_id": "call_1", "name": "bash"},
    ]


def test_chat_document_tool_arguments_have_stable_arrow_schema():
    documents = [
        chat_document(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"function": {"name": "edit", "arguments": arguments}}],
                }
            ],
            "test",
        )
        for arguments in ({"path": "a.py", "lines": [1, 2]}, {"path": ["a.py"], "lines": "all"})
    ]

    table = pa.Table.from_pylist(documents)

    assert table.num_rows == 2
    assert documents[0]["messages"][0]["tool_calls"][0]["function"]["arguments"] == ('{"lines":[1,2],"path":"a.py"}')


def test_canonical_chat_messages_serializes_parallel_tool_calls():
    messages = canonical_chat_messages(
        [
            {
                "role": "assistant",
                "content": "I will inspect both files.",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "read", "arguments": {"path": "a.py"}}},
                    {"id": "call_2", "function": {"name": "read", "arguments": {"path": "b.py"}}},
                ],
            },
            {"role": "tool", "content": "a", "tool_call_id": "call_1"},
            {"role": "tool", "content": "b", "tool_call_id": "call_2"},
        ]
    )

    assert messages == [
        {
            "role": "assistant",
            "content": "I will inspect both files.",
            "tool_calls": [
                {"id": "call_1", "function": {"name": "read", "arguments": '{"path":"a.py"}'}},
                {"id": "call_2", "function": {"name": "read", "arguments": '{"path":"b.py"}'}},
            ],
        },
        {"role": "tool", "content": "a", "tool_call_id": "call_1", "name": "read"},
        {"role": "tool", "content": "b", "tool_call_id": "call_2", "name": "read"},
    ]


def test_canonical_chat_messages_converts_legacy_function_call_and_drops_unknown_fields():
    messages = canonical_chat_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "function_call": '{"name":"search","arguments":{"query":"marin"}}',
                "provider_metadata": {"shape": "varies"},
            }
        ]
    )

    assert messages == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_0_0", "function": {"name": "search", "arguments": '{"query":"marin"}'}}],
        }
    ]


def test_canonical_chat_messages_rejects_unknown_roles():
    with pytest.raises(ValueError, match="Unsupported chat role"):
        canonical_chat_messages([{"role": "critic", "content": "No."}])


def test_canonical_chat_messages_uses_atomic_reasoning_tokens_for_assistant():
    messages = canonical_chat_messages([{"role": "assistant", "content": "<think>plan</think>answer"}])

    assert messages[0]["content"] == "<|start_think|>plan<|end_think|>answer"


def test_canonical_chat_messages_normalizes_case_and_outer_whitespace_in_reasoning():
    messages = canonical_chat_messages([{"role": "assistant", "content": "  \n<THINK>plan</Think>answer\n  "}])

    assert messages[0]["content"] == "<|start_think|>plan<|end_think|>answer"


def test_canonical_chat_messages_rejects_unbalanced_reasoning_tokens():
    with pytest.raises(ValueError, match="reasoning delimiters must be balanced"):
        canonical_chat_messages([{"role": "assistant", "content": "<think>unfinished"}])


def test_render_tool_call_dict_arguments():
    tool_call = {"function": {"name": "bash", "arguments": {"cmd": "ls", "dir": "/tmp"}}}
    assert render_tool_call(tool_call) == "<tool_call:bash>\n  cmd: ls\n  dir: /tmp\n</tool_call:bash>"


def test_render_tool_call_json_string_arguments():
    tool_call = {"function": {"name": "edit", "arguments": '{"path": "a.py"}'}}
    assert render_tool_call(tool_call) == "<tool_call:edit>\n  path: a.py\n</tool_call:edit>"


def test_render_tool_call_malformed_json_kept_as_raw_string():
    # A tool call whose arguments are an unparseable string must not abort the transform.
    tool_call = {"function": {"name": "run", "arguments": "not json"}}
    assert render_tool_call(tool_call) == "<tool_call:run>\n  not json\n</tool_call:run>"


def test_render_tool_message_includes_content_and_tool_calls():
    message = {
        "role": "assistant",
        "content": "checking",
        "tool_calls": [{"function": {"name": "ls", "arguments": {"dir": "/tmp"}}}],
    }
    assert render_tool_message(message) == (
        "<assistant>\nchecking\n<tool_call:ls>\n  dir: /tmp\n</tool_call:ls>\n</assistant>"
    )


def test_render_tool_message_omits_empty_content_line():
    assert render_tool_message({"role": "user", "content": ""}) == "<user>\n</user>"
