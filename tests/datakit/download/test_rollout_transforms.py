# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.datakit.download.rollout_transforms import render_tool_call, render_tool_message


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
