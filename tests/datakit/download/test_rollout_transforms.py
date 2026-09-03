# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.datakit.download.rollout_transforms import canonical_chat_messages, render_tool_call, render_tool_message


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
        {"tool_calls": tool_calls, "role": "assistant", "content": None},
        {"role": "tool", "content": "/workspace", "tool_call_id": "call_1"},
    ]


def test_canonical_chat_messages_rejects_unknown_roles():
    with pytest.raises(ValueError, match="Unsupported chat role"):
        canonical_chat_messages([{"role": "critic", "content": "No."}])


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
