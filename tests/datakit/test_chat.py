# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from marin.datakit.download.rollout_transforms import openai_chat_document
from openai_harmony import Conversation, Message


def test_harmony_separates_reasoning_answer_and_tool_handoff():
    record = {
        "messages": [
            {"role": "developer", "content": "Be concise."},
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "<think>Check Paris.</think>Let me check.",
                "tool_calls": [{"id": "call_1", "function": {"name": "weather", "arguments": {"city": "Paris"}}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
            {"role": "assistant", "content": "It is sunny."},
        ]
    }
    normalized = openai_chat_document(**record, source="test")
    assert normalized["messages"] == [
        {"role": "developer", "name": None, "content": [{"type": "text", "text": "Be concise."}]},
        {"role": "user", "name": None, "content": [{"type": "text", "text": "Weather?"}]},
        {
            "role": "assistant",
            "name": None,
            "channel": "analysis",
            "content": [{"type": "text", "text": "Check Paris."}],
        },
        {
            "role": "assistant",
            "name": None,
            "channel": "commentary",
            "content": [{"type": "text", "text": "Let me check."}],
        },
        {
            "role": "assistant",
            "name": None,
            "channel": "commentary",
            "recipient": "functions.weather",
            "content": [{"type": "text", "text": '{"city":"Paris"}'}],
        },
        {
            "role": "tool",
            "name": "functions.weather",
            "channel": "commentary",
            "recipient": "assistant",
            "content": [{"type": "text", "text": "Sunny"}],
        },
        {"role": "assistant", "name": None, "channel": "final", "content": [{"type": "text", "text": "It is sunny."}]},
    ]
    conversation = Conversation.from_json(json.dumps({"messages": normalized["messages"]}))
    assert conversation.to_dict()["messages"] == normalized["messages"]


def test_harmony_preserves_parallel_call_association_for_repeated_tool():
    record = {
        "messages": [
            {"role": "user", "content": "Read both files."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "a", "function": {"name": "read", "arguments": {"path": "a.py"}}},
                    {"id": "b", "function": {"name": "read", "arguments": {"path": "b.py"}}},
                ],
            },
            {"role": "tool", "tool_call_id": "b", "content": "B"},
            {"role": "tool", "tool_call_id": "a", "content": "A"},
            {"role": "assistant", "content": "Done."},
        ]
    }
    normalized = openai_chat_document(**record, source="test")
    messages = [Message.from_dict(message) for message in normalized["messages"]]
    assert [message.content[0].to_dict()["text"] for message in messages[1:5]] == [
        '{"path":"a.py"}',
        '{"path":"b.py"}',
        "A",
        "B",
    ]
    assert [message.recipient for message in messages[1:3]] == ["functions.read", "functions.read"]


@pytest.mark.parametrize(
    "reasoning", ["<think>Plan.</think>", "<THINK>Plan.</THINK>", "<|start_think|>Plan.<|end_think|>"]
)
def test_harmony_normalizes_reasoning_spellings_to_same_identity(reasoning):
    messages = [{"role": "user", "content": "Question"}, {"role": "assistant", "content": reasoning + "Answer"}]
    tagged = openai_chat_document(**{"messages": messages}, source="test")
    messages[1] = {"role": "assistant", "reasoning_content": "Plan.", "content": "Answer"}
    separated = openai_chat_document(**{"messages": messages}, source="test")
    assert tagged["id"] == separated["id"]
    assert tagged["messages"][1]["content"] == [{"type": "text", "text": "Plan."}]


def test_json_answer_keys_do_not_create_reasoning_or_tool_calls():
    answer = '{"analysis":"a report", "commands":["help"]}'
    messages = openai_chat_document(
        **{
            "messages": [
                {"role": "user", "content": "Return a JSON report."},
                {"role": "assistant", "content": answer},
            ]
        },
        source="test",
    )["messages"]
    assert len(messages) == 2
    assert messages[-1]["channel"] == "final"
    assert messages[-1]["content"] == [{"type": "text", "text": answer}]


@pytest.mark.parametrize("content", [None, ""])
def test_separate_reasoning_without_answer_matches_inline_reasoning(content):
    user = {"role": "user", "content": "Think about it."}
    inline = openai_chat_document(
        **{"messages": [user, {"role": "assistant", "content": "<think>Plan.</think>"}]}, source="test"
    )
    separate = openai_chat_document(
        **{"messages": [user, {"role": "assistant", "content": content, "reasoning_content": "Plan."}]}, source="test"
    )
    assert separate["id"] == inline["id"]
    assert separate["messages"][-1]["channel"] == "analysis"
