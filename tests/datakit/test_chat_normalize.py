# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.datakit.chat_normalize import _normalize_chat_record, normalize_chat_to_parquet, validate_chat_messages


@pytest.fixture(autouse=True)
def flow_backend_ctx():
    with set_current_client(LocalClient()):
        yield


def test_normalize_chat_record_canonicalizes_reasoning_and_tools():
    record = {
        "id": "source-row",
        "messages": [
            {"role": "user", "content": "Check the weather."},
            {
                "role": "assistant",
                "content": "<think>Need the forecast.</think>",
                "tool_calls": [{"function": {"name": "weather", "arguments": {"city": "Paris"}}}],
            },
            {"role": "tool", "content": "Sunny"},
            {"role": "assistant", "content": "It is sunny."},
        ],
    }

    normalized = _normalize_chat_record(record, "messages", "id")

    call = normalized["messages"][1]["tool_calls"][0]
    assert normalized["messages"][1]["content"] == "<|start_think|>Need the forecast.<|end_think|>"
    assert call["id"] == "call_1_0"
    assert normalized["messages"][2]["tool_call_id"] == call["id"]
    assert normalized["messages"][2]["name"] == "weather"
    assert json.loads(normalized["chat_template_kwargs"])["tools"][0]["name"] == "weather"


def test_chat_identity_includes_tool_definitions():
    base = {
        "messages": [
            {"role": "user", "content": "Run it."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call", "function": {"name": "run", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "done", "tool_call_id": "call"},
            {"role": "assistant", "content": "Done."},
        ]
    }
    first = _normalize_chat_record(
        {**base, "chat_template_kwargs": {"tools": [{"name": "run", "description": "First"}]}},
        "messages",
        "id",
    )
    second = _normalize_chat_record(
        {**base, "chat_template_kwargs": {"tools": [{"name": "run", "description": "Second"}]}},
        "messages",
        "id",
    )

    assert first["id"] != second["id"]


def test_normalize_chat_to_parquet_keeps_varying_tool_schemas_arrow_stable(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    records = []
    for name, arguments in (("read", {"path": "a.py"}), ("search", {"query": "Marin", "limit": 3})):
        records.append(
            {
                "messages": [
                    {"role": "user", "content": f"Use {name}."},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
                    },
                ]
            }
        )
    (input_dir / "data.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))

    normalize_chat_to_parquet(input_path=str(input_dir), output_path=str(output_dir))

    normalized = [
        row for path in (output_dir / "outputs" / "main").glob("*.parquet") for row in pq.read_table(path).to_pylist()
    ]
    assert len(normalized) == 2
    assert all(isinstance(record["chat_template_kwargs"], str) for record in normalized)
    assert {json.loads(record["chat_template_kwargs"])["tools"][0]["name"] for record in normalized} == {
        "read",
        "search",
    }


@pytest.mark.parametrize(
    "messages,error",
    [
        (
            [
                {"role": "user", "content": "one"},
                {"role": "user", "content": "two"},
                {"role": "assistant", "content": "answer"},
            ],
            "Consecutive user",
        ),
        (
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "first"},
                {"role": "assistant", "content": "second"},
            ],
            "Consecutive assistant",
        ),
        (
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": '<tool_call>{"name":"run"}</tool_call>'},
            ],
            "Inline tool-call",
        ),
        (
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "<|start_think|><|end_think|>answer"},
            ],
            "non-empty prefix",
        ),
    ],
)
def test_validate_chat_messages_rejects_invalid_conversations(messages, error):
    with pytest.raises(ValueError, match=error):
        validate_chat_messages(messages, [])
