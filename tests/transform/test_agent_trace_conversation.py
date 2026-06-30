# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for agent-trace conversation normalization."""

import json
from pathlib import Path
from typing import Any

import pytest
from marin.core.conversation import OpenAIChatMessage
from marin.transform.conversation.tool_normalization import (
    normalize_tool_call_arguments,
    normalize_wrapped_tool_response_messages,
)
from marin.transform.conversation.transform_conversation import (
    TransformSFTDatasetConfig,
    source_id_or_message_hash,
    transform_row,
)

from experiments.posttrain.instruction_datasets import INSTRUCTION_DATASET_NAME_TO_CONFIG

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "agent_traces"


def _load_json_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("dataset_name", "fixture_name", "expected_tool_name", "expected_tool_call_id", "expected_tool_content"),
    [
        (
            "lambda/hermes-agent-reasoning-traces/glm-5.1",
            "hermes_glm_sample.json",
            "write_file",
            "glm-tool-call-001",
            {"bytes_written": 15, "dirs_created": False},
        ),
        (
            "lambda/hermes-agent-reasoning-traces/kimi",
            "hermes_kimi_sample.json",
            "terminal",
            "kimi-tool-call-001",
            {"output": "/workspace/project", "exit_code": 0, "error": None},
        ),
    ],
)
def test_registered_hermes_dataset_transforms(
    dataset_name: str,
    fixture_name: str,
    expected_tool_name: str,
    expected_tool_call_id: str,
    expected_tool_content: dict,
):
    row = _load_json_fixture(fixture_name)
    dataset_cfg = INSTRUCTION_DATASET_NAME_TO_CONFIG[dataset_name]
    cfg = TransformSFTDatasetConfig(
        source=dataset_cfg.hf_dataset_id,
        revision=dataset_cfg.revision,
        output_path="/tmp/output",
        metadata_columns=dataset_cfg.metadata_columns,
        adapter=dataset_cfg.adapter,
        subsets=dataset_cfg.subsets,
        splits=dataset_cfg.splits,
    )

    result = transform_row(row, cfg, dataset_cfg.adapter)

    assert result is not None
    assert result.id == row["id"]
    assert result.metadata == {
        "category": row["category"],
        "subcategory": row["subcategory"],
        "task": row["task"],
    }

    roles = [message.role for message in result.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]

    assistant_tool_call = result.messages[2]
    assert "<|start_think|>" in assistant_tool_call.content
    assert "<tool_call>" in assistant_tool_call.content

    tool_message = result.messages[3]
    assert tool_message.name == expected_tool_name
    assert tool_message.tool_call_id == expected_tool_call_id
    assert tool_message.content == expected_tool_content
    assert "<tool_response>" not in json.dumps(tool_message.content)


@pytest.mark.parametrize(
    "content",
    [
        '<tool_response>\n{"name": "terminal"}',
        '<tool_response>\n{"name": "terminal",}\n</tool_response>',
    ],
)
def test_wrapped_tool_response_normalization_preserves_raw_content_when_wrapper_cannot_parse(content: str):
    original = OpenAIChatMessage(role="tool", content=content)

    normalized = normalize_wrapped_tool_response_messages([original], row={})

    assert normalized == [original]


def test_source_id_or_message_hash_uses_source_id_or_deterministic_message_hash():
    messages = [{"role": "assistant", "content": "hello"}]

    assert source_id_or_message_hash({"id": "trace-from-source"}, messages) == "trace-from-source"
    first_fallback = source_id_or_message_hash({}, messages)
    second_fallback = source_id_or_message_hash({}, messages)
    changed_fallback = source_id_or_message_hash({}, [{"role": "assistant", "content": "goodbye"}])

    assert first_fallback == second_fallback
    assert first_fallback != changed_fallback


def test_tool_call_argument_normalization_parses_json_strings_without_mutating_source_message():
    original: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "write_file", "arguments": '{"path": "notes.txt", "content": "hi"}'},
            },
            {
                "type": "function",
                "function": {"name": "terminal", "arguments": "{not-json"},
            },
        ],
    }

    normalized = normalize_tool_call_arguments(original)

    assert normalized["tool_calls"][0]["function"]["arguments"] == {"path": "notes.txt", "content": "hi"}
    assert normalized["tool_calls"][1]["function"]["arguments"] == "{not-json"
    assert original["tool_calls"][0]["function"]["arguments"] == '{"path": "notes.txt", "content": "hi"}'
