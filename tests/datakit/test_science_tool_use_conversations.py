# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from marin.datakit.chat_normalize import validate_chat_messages, validate_tool_definitions
from marin.datakit.chat_render import render_chat_record
from marin.datakit.download.science_tool_use_conversations import row_to_chat_doc
from openai_harmony import Message


def test_science_tool_use_keeps_tool_transcript_and_drops_unanswered_prompt():
    tool = {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Run a command in a persistent shell",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    }
    row = {
        "idx": 7,
        "messages": [{"role": "user", "content": "An abridged transcript without tools"}],
        "full_log": [
            {"role": "system", "content": "Use the shell when calculation helps."},
            {"role": "user", "content": "What is 2 + 2?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "execute_bash", "arguments": '{"command":"echo $((2+2))"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "4"},
            {"role": "assistant", "content": "It is 4."},
            {"role": "user", "content": "What about 3 + 3?"},
        ],
        "tools": [tool],
    }

    [document] = row_to_chat_doc(row)
    messages = [Message.from_dict(message) for message in document["messages"]]
    validate_chat_messages(messages)
    validate_tool_definitions([tool], messages)
    rendered = render_chat_record(document)["text"]

    assert document["source_id"] == "7"
    assert json.loads(document["chat_template_kwargs"])["tools"] == [tool]
    assert messages[-1].channel == "final"
    assert "echo $((2+2))" in rendered
    assert "<tool_response" in rendered
    assert "It is 4." in rendered
    assert "What about 3 + 3?" not in rendered
    assert "An abridged transcript" not in rendered
