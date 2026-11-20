# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
from pathlib import Path

import pyarrow.parquet as pq
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.datakit.sft_sources import all_sft_sources

from experiments.sft_new_mix.sources import candidate_sft_sources_v1


def test_accepted_science_preserves_duplicate_rows_and_serialized_text(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    all_sft_sources.cache_clear()
    original = all_sft_sources()
    candidate = candidate_sft_sources_v1()
    changed = {key for key in original if candidate[key] is not original[key]}
    assert changed == {f"nemotron_sft_v3/science_v2/{part}" for part in ("rqa", "so", "syn_mcq", "vendor")}
    assert candidate.keys() == original.keys()
    assert all(candidate[key].rough_token_count_b == original[key].rough_token_count_b for key in original)
    source = candidate["nemotron_sft_v3/science_v2/rqa"]
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
