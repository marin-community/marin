# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for agent-trace conversation normalization."""

import json
from pathlib import Path

import pytest

from experiments.posttrain.instruction_datasets import INSTRUCTION_DATASET_NAME_TO_CONFIG
from marin.core.conversation import OpenAIChatMessage
from marin.transform.conversation.trace_normalization import (
    hermes_trace_row_id,
    normalize_hermes_trace_messages,
)
from marin.transform.conversation.transform_conversation import TransformSFTDatasetConfig, transform_row

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "agent_traces"
EXPECTED_HERMES_FEATURES = ["id", "conversations", "tools", "category", "subcategory", "task"]


def _load_json_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("fixture_name", ["hermes_glm_schema.json", "hermes_kimi_schema.json"])
def test_hermes_schema_fixtures_match_expected_fields(fixture_name: str):
    schema = _load_json_fixture(fixture_name)
    assert [feature["name"] for feature in schema["features"]] == EXPECTED_HERMES_FEATURES


@pytest.mark.parametrize(
    ("dataset_name", "fixture_name", "expected_tool_name", "expected_tool_call_id"),
    [
        (
            "lambda/hermes-agent-reasoning-traces/glm-5.1",
            "hermes_glm_sample.json",
            "write_file",
            "glm-tool-call-001",
        ),
        (
            "lambda/hermes-agent-reasoning-traces/kimi",
            "hermes_kimi_sample.json",
            "terminal",
            "kimi-tool-call-001",
        ),
    ],
)
def test_registered_hermes_dataset_transforms(
    dataset_name: str,
    fixture_name: str,
    expected_tool_name: str,
    expected_tool_call_id: str,
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
    assert isinstance(tool_message.content, dict)
    assert "<tool_response>" not in json.dumps(tool_message.content)


def test_hermes_tool_response_normalization_preserves_raw_content_on_malformed_wrapper():
    original = OpenAIChatMessage(role="tool", content='<tool_response>\n{"name": "terminal"}')

    normalized = normalize_hermes_trace_messages([original], row={})

    assert normalized == [original]


def test_hermes_tool_response_normalization_preserves_wrapped_content_on_invalid_json():
    original = OpenAIChatMessage(role="tool", content='<tool_response>\n{"name": "terminal",}\n</tool_response>')

    normalized = normalize_hermes_trace_messages([original], row={})

    assert normalized == [original]


def test_hermes_trace_row_id_falls_back_to_message_hash_without_source_id():
    messages = [{"role": "assistant", "content": "hello"}]

    row_id = hermes_trace_row_id({}, messages)

    assert isinstance(row_id, str)
    assert len(row_id) == 64
