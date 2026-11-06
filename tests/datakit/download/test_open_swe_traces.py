# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from marin.datakit.chat_normalize import validate_chat_messages, validate_tool_definitions
from marin.datakit.chat_render import render_chat_record
from marin.datakit.download.open_swe_traces import row_to_chat_doc
from openai_harmony import Message


@pytest.fixture
def open_swe_row():
    tool = {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    }
    row = {
        "trajectory_id": "trajectory-123",
        "resolved": 1,
        "tools": [json.dumps(tool)],
        "messages": [
            {"role": "system", "content": "Use the bash tool."},
            {"role": "user", "content": "Fix the failing test in /testbed."},
            {
                "role": "assistant",
                "content": "I will inspect both files.",
                "reasoning_content": "Check the failing assertion first.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command":"cat test.py"}'},
                    },
                    {
                        "id": "call-2",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command":"cat src.py"}'},
                    },
                ],
            },
            {"role": "tool", "content": "TEST_OUTPUT_SENTINEL"},
            {"role": "tool", "content": "SOURCE_OUTPUT_SENTINEL"},
            {"role": "assistant", "content": "I fixed the assertion."},
        ],
    }

    return row


def test_open_swe_traces_renders_prompt_reasoning_and_parallel_tool_calls(open_swe_row):
    [doc] = row_to_chat_doc(open_swe_row)
    messages = [Message.from_dict(message) for message in doc["messages"]]
    validate_chat_messages(messages)
    validate_tool_definitions(json.loads(doc["chat_template_kwargs"])["tools"], messages)
    rendered = render_chat_record(doc)["text"]

    assert doc["source_id"] == "trajectory-123"
    assert doc["resolved"] == 1
    assert "Fix the failing test in /testbed." in rendered
    assert "<|start_think|>Check the failing assertion first.<|end_think|>" in rendered
    assert rendered.index("TEST_OUTPUT_SENTINEL") < rendered.index("SOURCE_OUTPUT_SENTINEL")
    assert '<tool_call>\n{"name": "bash"' in rendered
    assert "I fixed the assertion." in rendered


@pytest.mark.parametrize("resolved", [0, -1, None])
def test_open_swe_traces_excludes_failed_and_unknown_trajectories(open_swe_row, resolved):
    open_swe_row["resolved"] = resolved
    assert row_to_chat_doc(open_swe_row) == []


def test_open_swe_traces_respects_explicit_ids_for_reversed_observations(open_swe_row):
    messages = open_swe_row["messages"]
    messages[3]["tool_call_id"] = "call-1"
    messages[4]["tool_call_id"] = "call-2"
    messages[3], messages[4] = messages[4], messages[3]
    [doc] = row_to_chat_doc(open_swe_row)
    rendered = render_chat_record(doc)["text"]
    # Conversion emits results in call order, even if their arrival order differs.
    assert rendered.index("TEST_OUTPUT_SENTINEL") < rendered.index("SOURCE_OUTPUT_SENTINEL")
