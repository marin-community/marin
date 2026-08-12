# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit.download.rollout_transforms import (
    render_tool_call,
    render_tool_message,
    text_document,
    write_document_shards,
)


def test_render_tool_call_dict_arguments():
    tool_call = {"function": {"name": "bash", "arguments": {"cmd": "ls", "dir": "/tmp"}}}
    assert render_tool_call(tool_call) == "<tool_call:bash>\n  cmd: ls\n  dir: /tmp\n</tool_call:bash>"


def test_render_tool_call_json_string_arguments():
    tool_call = {"function": {"name": "edit", "arguments": '{"path": "a.py"}'}}
    assert render_tool_call(tool_call) == "<tool_call:edit>\n  path: a.py\n</tool_call:edit>"


def test_render_tool_call_malformed_json_kept_as_raw_string():
    # A tool call whose arguments are an unparseable string must not abort the transform.
    tool_call = {"function": {"name": "run", "arguments": "not json"}}
    assert render_tool_call(tool_call) == "<tool_call:run>\n  not json\n</tool_call:run>"


def test_render_tool_message_includes_content_and_tool_calls():
    message = {
        "role": "assistant",
        "content": "checking",
        "tool_calls": [{"function": {"name": "ls", "arguments": {"dir": "/tmp"}}}],
    }
    assert render_tool_message(message) == (
        "<assistant>\nchecking\n<tool_call:ls>\n  dir: /tmp\n</tool_call:ls>\n</assistant>"
    )


def test_render_tool_message_omits_empty_content_line():
    assert render_tool_message({"role": "user", "content": ""}) == "<user>\n</user>"


def _doc_from_row(row: dict) -> list[dict]:
    if not row["body"]:
        return []
    return [text_document(row["body"], "test-source")]


def test_write_document_shards_reshards_and_drops_empty_rows(tmp_path: Path):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    for index in range(4):
        pq.write_table(
            pa.Table.from_pylist([{"body": f"doc {index}"}, {"body": ""}]),
            input_dir / f"part-{index}.parquet",
        )

    output_dir = tmp_path / "processed"
    write_document_shards(
        f"{input_dir}/*.parquet",
        str(output_dir),
        name="rollout-transforms-test",
        row_to_doc=_doc_from_row,
        resources=ResourceConfig(cpu=1, ram="1g"),
        num_shards=2,
    )

    shards = sorted(output_dir.glob("*.parquet"))
    assert [shard.name for shard in shards] == [
        "data-00000-of-00002.parquet",
        "data-00001-of-00002.parquet",
    ]
    rows = [row for shard in shards for row in pq.read_table(shard).to_pylist()]
    assert sorted(row["text"] for row in rows) == ["doc 0", "doc 1", "doc 2", "doc 3"]
