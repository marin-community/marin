# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import msgspec
import pytest
from marin.datakit.download.nemotron_agentic import (
    NemotronAgenticSubset,
    load_nemotron_agentic_jsonl,
    row_to_doc,
)


def tool_calling_row() -> dict:
    return {
        "model": "deepseek/DeepSeek-V3.2",
        "messages": [
            {"role": "system", "content": "Use tools when the request needs account data."},
            {"role": "user", "content": "Can you check order A-100?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "I need to look up the order before answering.",
                "tool_calls": [
                    {
                        "id": "call_order",
                        "type": "function",
                        "function": {
                            "name": "get_order",
                            "arguments": '{"order_id":"A-100","include_items":true}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_order",
                "name": "get_order",
                "content": '{"status":"shipped"}',
            },
            {"role": "assistant", "content": "Order A-100 has shipped."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_order",
                    "description": "Fetch an order by id.",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"],
                    },
                    "strict": True,
                },
            }
        ],
        "parallel_tool_calls": False,
        "domain": "retail customer-service",
        "temperature": 1.0,
        "chat_template_kwargs": {"thinking": True},
        "processing_info": {"stages": ["tool_validation", "token_counting"]},
        "metadata": {
            "uuid": "trajectory-123",
            "alt_id": "alt-123",
            "turn_token_count": [{"reasoning": 8, "content": 12}],
            "all_turns_token_count": {"reasoning": 8, "content": 42, "all": 50},
        },
    }


def test_tool_calling_row_preserves_tool_calls_and_metadata():
    (record,) = row_to_doc(tool_calling_row(), NemotronAgenticSubset.TOOL_CALLING)

    assert record["source"] == "nvidia/Nemotron-SFT-Agentic-v2"
    assert record["source_subset"] == "tool_calling"
    assert record["id"] == "trajectory-123"
    assert record["trajectory_id"] == "trajectory-123"
    assert record["domain"] == "retail customer-service"
    assert record["parallel_tool_calls"] is False
    assert record["tool_call_count"] == 1
    assert record["turn_count"] == 5
    assert record["reasoning_token_count"] == 8
    assert record["content_token_count"] == 42
    assert record["all_token_count"] == 50

    messages = json.loads(record["messages_json"])
    tools = json.loads(record["tools_json"])
    tool_names = json.loads(record["tool_names_json"])

    assert messages[2]["thinking"] == "I need to look up the order before answering."
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == '{"include_items":true,"order_id":"A-100"}'
    assert tools[0]["function"]["strict"] is True
    assert tool_names == ["get_order"]

    assert "<tools>" in record["text"]
    assert "<thinking>" in record["text"]
    assert '<tool_call id="call_order" name="get_order">' in record["text"]
    assert '{"include_items":true,"order_id":"A-100"}' in record["text"]
    assert '<tool tool_call_id="call_order" name="get_order">' in record["text"]


def test_missing_messages_are_dropped():
    assert row_to_doc({"messages": []}, NemotronAgenticSubset.TOOL_CALLING) == []


def test_invalid_tool_call_arguments_fail_with_source_id():
    row = tool_calling_row()
    row["messages"][2]["tool_calls"][0]["function"]["arguments"] = "{not-json"

    with pytest.raises(ValueError, match="trajectory-123"):
        row_to_doc(row, NemotronAgenticSubset.TOOL_CALLING)


def test_non_object_tool_call_fails_with_source_id():
    row = tool_calling_row()
    row["messages"][2]["tool_calls"] = ["not-a-tool-call"]

    with pytest.raises(ValueError, match="Tool call 0 in trajectory-123"):
        row_to_doc(row, NemotronAgenticSubset.TOOL_CALLING)


def test_known_malformed_tool_calling_source_line_is_skipped(tmp_path):
    source = tmp_path / "tool_calling.jsonl"
    lines = ['{"ok":1}', *[""] * 1093, "{not-json", '{"ok":2}']
    source.write_text("\n".join(lines), encoding="utf-8")

    rows = list(load_nemotron_agentic_jsonl(str(source), NemotronAgenticSubset.TOOL_CALLING))

    assert rows == [{"ok": 1}, {"ok": 2}]


def test_unexpected_malformed_source_line_fails(tmp_path):
    source = tmp_path / "tool_calling.jsonl"
    source.write_text("{not-json\n", encoding="utf-8")

    with pytest.raises(msgspec.DecodeError):
        list(load_nemotron_agentic_jsonl(str(source), NemotronAgenticSubset.TOOL_CALLING))
