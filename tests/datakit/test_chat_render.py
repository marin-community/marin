# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow as pa
import pyarrow.parquet as pq
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.datakit.chat_normalize import CHAT_SCHEMA
from marin.datakit.chat_render import render_chat_to_parquet
from openai_harmony import Message, Role


def test_rendered_parquet_preserves_ids_and_duplicates(tmp_path):
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
    input_path = tmp_path / "input"
    input_path.mkdir()
    pq.write_table(pa.Table.from_pylist(records, schema=CHAT_SCHEMA), input_path / "part-00000.parquet")
    output_path = tmp_path / "rendered"

    with set_current_client(LocalClient()):
        render_chat_to_parquet(input_path=str(input_path), output_path=str(output_path), max_workers=1)

    table = pq.read_table(output_path)
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
