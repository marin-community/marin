# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from marin.datakit.chat_normalize import _normalize_chat_record
from marin.datakit.download.glm_kernelgym_rollouts import chat_conversation_messages
from marin.datakit.download.rollout_transforms import openai_chat_document
from marin.datakit.download.swe_zero_12m import row_to_chat_doc as swe_zero_row
from openai_harmony import Conversation, Message
from zephyr.writers import write_parquet_file


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


def test_swe_zero_thought_becomes_harmony_analysis():
    [source] = swe_zero_row(
        {
            "messages": [
                {"role": "user", "content": "Fix the bug."},
                {"role": "assistant", "content": "THOUGHT: Inspect files.\n```bash\nls\n```"},
                {"role": "user", "content": "Observation: a.py"},
                {
                    "role": "assistant",
                    "content": "THOUGHT: Finished.\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```",
                },
            ]
        }
    )
    messages = _normalize_chat_record(source, "messages", "id")["messages"]
    assert [(m["channel"], m["content"][0]["text"]) for m in messages if m.get("channel") == "analysis"] == [
        ("analysis", "Inspect files."),
        ("analysis", "Finished."),
    ]
    assert messages[-1]["channel"] == "final"


def test_harmony_keeps_final_tool_call_without_observation():
    record = {
        "messages": [
            {"role": "user", "content": "Run it."},
            {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "run", "arguments": {}}}]},
        ]
    }
    messages = openai_chat_document(**record, source="test")["messages"]
    assert messages[-1]["recipient"] == "functions.run"
    assert messages[-1]["channel"] == "commentary"


def test_glm_reasoning_with_missing_opener_becomes_harmony_analysis():
    response = "Inspect the kernel.</think>Use shared memory."
    source_messages = chat_conversation_messages(
        [{"role": "user", "content": "Optimize it."}, {"role": "assistant", "content": response}],
        [{"response": response}],
    )
    messages = openai_chat_document(**{"messages": source_messages}, source="test")["messages"]
    assert [(m["channel"], m["content"][0]["text"]) for m in messages[1:]] == [
        ("analysis", "Inspect the kernel."),
        ("final", "Use shared memory."),
    ]


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


def test_source_harmony_parquet_is_consumed_without_a_second_conversion(tmp_path: Path):
    record = openai_chat_document(
        [
            {"role": "user", "content": "Inspect the file."},
            {
                "role": "assistant",
                "content": "<think>Read it.</think>",
                "tool_calls": [
                    {"id": "read", "function": {"name": "read", "arguments": {"path": "a.py"}}},
                ],
            },
            {"role": "tool", "tool_call_id": "read", "content": "contents"},
            {"role": "assistant", "content": "Done."},
        ],
        "test",
    )
    path = tmp_path / "processed.parquet"
    write_parquet_file([record], str(path))
    [persisted] = pq.read_table(path).to_pylist()
    normalized = _normalize_chat_record(persisted, "messages", "id")
    assert normalized["messages"] == record["messages"]
    assert normalized["messages"][1]["channel"] == "analysis"
    assert normalized["messages"][2]["recipient"] == "functions.read"
    assert normalized["messages"][3]["name"] == "functions.read"
    assert normalized["messages"][-1]["channel"] == "final"
