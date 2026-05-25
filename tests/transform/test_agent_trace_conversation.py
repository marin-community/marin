# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for agent-trace conversation normalization."""

import json
from pathlib import Path

import pytest
from marin.core.conversation import OpenAIChatMessage
from marin.transform.conversation.trace_normalization import (
    hermes_trace_row_id,
    normalize_hermes_trace_messages,
)
from marin.transform.conversation.transform_conversation import TransformSFTDatasetConfig, transform_row

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
def test_hermes_tool_response_normalization_preserves_raw_content_when_wrapper_cannot_parse(content: str):
    original = OpenAIChatMessage(role="tool", content=content)

    normalized = normalize_hermes_trace_messages([original], row={})

    assert normalized == [original]


def test_hermes_trace_row_id_uses_source_id_or_deterministic_message_hash():
    messages = [{"role": "assistant", "content": "hello"}]

    assert hermes_trace_row_id({"id": "trace-from-source"}, messages) == "trace-from-source"
    first_fallback = hermes_trace_row_id({}, messages)
    second_fallback = hermes_trace_row_id({}, messages)
    changed_fallback = hermes_trace_row_id({}, [{"role": "assistant", "content": "goodbye"}])

    assert first_fallback == second_fallback
    assert first_fallback != changed_fallback
