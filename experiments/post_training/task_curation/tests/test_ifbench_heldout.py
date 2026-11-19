# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check that the held-out reference preserves the evaluated prompts and keys."""

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.post_training.task_curation.ifbench_heldout import ReferenceConfig, prepare_reference


def test_complete_reference_preserves_prompt_bytes_and_excludes_private_fields(tmp_path):
    source = tmp_path / "source" / "data" / "train-00000-of-00001.parquet"
    source.parent.mkdir(parents=True)
    rows = [
        {
            "key": i,
            "prompt": f"  Question {i}: use the word café.\n\nKeep this spacing.  ",
            "answer": "PRIVATE_GOLD_SENTINEL",
            "kwargs": '{"private": true}',
        }
        for i in range(300)
    ]
    pq.write_table(pa.Table.from_pylist(rows), source)
    output = tmp_path / "reference"
    prepare_reference(ReferenceConfig(str(source.parent.parent), str(output)))
    lines = [json.loads(line) for line in (output / "prompts.jsonl").read_text().splitlines()]
    assert [line["text"] for line in lines] == [row["prompt"] for row in rows]
    assert {line["id"] for line in lines} == {f"ifbench-test:{row['key']}" for row in rows}
    assert "PRIVATE_GOLD_SENTINEL" not in (output / "prompts.jsonl").read_text()
    manifest = json.loads((output / "reference-manifest.json").read_text())
    assert manifest["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["prompts_sha256"] == hashlib.sha256((output / "prompts.jsonl").read_bytes()).hexdigest()
