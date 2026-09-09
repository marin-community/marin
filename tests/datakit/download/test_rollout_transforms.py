# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from marin.datakit.download.rollout_transforms import (
    openai_chat_messages,
    render_tool_call,
    render_tool_message,
)
from openai_harmony import Role


def test_source_aliases_and_tool_results_become_harmony_messages():
    messages = openai_chat_messages(
        [
            {"from": "human", "value": "Inspect the repository."},
            {
                "from": "gpt",
                "value": None,
                "tool_calls": [{"id": "call_1", "function": {"name": "bash", "arguments": {"cmd": "pwd"}}}],
            },
            {"role": "function", "content": "/workspace", "tool_call_id": "call_1"},
        ]
    )
    assert [message.author.role for message in messages] == [Role.USER, Role.ASSISTANT, Role.TOOL]
    assert messages[1].recipient == "functions.bash"
    assert json.loads(messages[1].content[0].to_dict()["text"]) == {"cmd": "pwd"}
    assert messages[2].author.name == "functions.bash"
    assert messages[2].content[0].to_dict()["text"] == "/workspace"


def test_legacy_function_call_becomes_harmony_recipient_and_arguments():
    [message] = openai_chat_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "function_call": '{"name":"search","arguments":{"query":"marin"}}',
                "provider_metadata": {"shape": "varies"},
            }
        ]
    )
    assert message.to_dict() == {
        "role": "assistant",
        "name": None,
        "channel": "commentary",
        "recipient": "functions.search",
        "content": [{"type": "text", "text": '{"query":"marin"}'}],
    }


@pytest.mark.parametrize("content", ["<think>unfinished", "<think>a</think>answer<think>b</think>"])
def test_source_adapter_rejects_malformed_reasoning(content):
    with pytest.raises(ValueError):
        openai_chat_messages([{"role": "assistant", "content": content}])


def test_source_adapter_rejects_repeated_call_ids_before_discarding_them():
    with pytest.raises(ValueError, match="unique"):
        openai_chat_messages(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "same", "function": {"name": "read", "arguments": {"path": "a"}}},
                        {"id": "same", "function": {"name": "read", "arguments": {"path": "b"}}},
                    ],
                }
            ]
        )


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
