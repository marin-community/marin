# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow as pa
import pyarrow.parquet as pq
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.datakit.chat_normalize import CHAT_SCHEMA
from marin.datakit.normalize import generate_id
from marin.datakit.sft_sources import DatakitChatSource
from marin.execution.step_spec import StepSpec
from openai_harmony import Message, Role


def test_sft_source_renders_then_normalizes_and_deduplicates(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    messages = [
        Message.from_role_and_content(Role.USER, "What is 2 + 2?"),
        Message.from_role_and_content(Role.ASSISTANT, "Add two and two.").with_channel("analysis"),
        Message.from_role_and_content(Role.ASSISTANT, "4").with_channel("final"),
    ]
    records = [
        {
            "id": source_id,
            "messages": [message.to_dict() for message in messages],
            "chat_template_kwargs": json.dumps({"enable_thinking": True}),
        }
        for source_id in ["first-source-id", "second-source-id"]
    ]
    input_path = tmp_path / "chat" / "outputs" / "main"
    input_path.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(records, schema=CHAT_SCHEMA), input_path / "part-00000.parquet")
    source = DatakitChatSource(
        name="fixture",
        chat_steps=(StepSpec(name="fixture-chat", override_output_path=str(tmp_path / "chat")),),
        rough_token_count_b=0,
    )

    with set_current_client(LocalClient()):
        for step in source.normalize_steps[1:]:
            assert step.fn is not None
            step.fn(step.output_path)

    table = pq.read_table(source.rendered.output_path)
    assert table.column_names == ["id", "text"]
    expected_text = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>Reasoning: /think<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\nWhat is 2 + 2?<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        "<|start_think|>Add two and two.<|end_think|>4<|eot_id|>"
    )
    assert sorted(table.to_pylist(), key=lambda row: row["id"]) == [
        {"id": "first-source-id", "text": expected_text},
        {"id": "second-source-id", "text": expected_text},
    ]

    normalized = pq.read_table(f"{source.normalized.output_path}/outputs/main").to_pylist()
    duplicates = pq.read_table(f"{source.normalized.output_path}/outputs/dups").to_pylist()
    assert len(normalized) == len(duplicates) == 1
    assert normalized[0]["id"] == generate_id(expected_text)
    assert normalized[0]["text"] == expected_text
    assert {normalized[0]["source_id"], duplicates[0]["source_id"]} == {"first-source-id", "second-source-id"}
    assert (
        pq.read_table(f"{source.chat_normalized.output_path}/outputs/main").to_pylist()
        == pa.Table.from_pylist(records, schema=CHAT_SCHEMA).to_pylist()
    )
