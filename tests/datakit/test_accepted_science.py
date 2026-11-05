# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
from pathlib import Path

import pyarrow.parquet as pq
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.datakit.sft_sources import all_sft_sources


def test_accepted_science_preserves_duplicate_rows_and_serialized_text(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    all_sft_sources.cache_clear()
    source = all_sft_sources()["nemotron_sft_v3/science_v2/rqa"]
    text = (
        "<|start_header_id|>user<|end_header_id|>Keep this spacing:"
        + " " * 512
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        '<|start_think|>Check it.<|end_think|><tool_call>{"name":"evaluate_code",'
        '"arguments":{"source_code":"print(2+2)"}}</tool_call><|eot_id|>'
    )
    directory = Path(source.normalize_steps[0].output_path) / "data/rqa"
    directory.mkdir(parents=True)
    with gzip.open(directory / "part-00000.jsonl.gz", "wt") as stream:
        for identity in ("original-a", "original-b"):
            stream.write(json.dumps({"id": identity, "text": text}) + "\n")
    with set_current_client(LocalClient()):
        source.normalized.fn(source.normalized.output_path)
    rows = pq.read_table(f"{source.normalized.output_path}/outputs/main").to_pylist()
    assert sorted((row["source_id"], row["text"]) for row in rows) == [
        ("original-a", text),
        ("original-b", text),
    ]
    all_sft_sources.cache_clear()
