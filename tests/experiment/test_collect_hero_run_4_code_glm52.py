# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import itertools
import json

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from rigging.filesystem import StoragePath

from experiments.rollout_data import collect_hero_run_4_code_glm52 as collect


def _response(payload):
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(payload).encode()
    return response


def test_partition_specs_cover_every_dataset_row_once():
    partitions = collect.partition_specs()

    assert len(partitions) == 1000
    assert partitions[0].row_start == 0
    assert partitions[-1].row_start + partitions[-1].num_rows == 959216
    assert all(
        previous.row_start + previous.num_rows == current.row_start
        for previous, current in itertools.pairwise(partitions)
    )
    shards = [collect.partition_slices(partitions, shard, 85, None) for shard in range(85)]
    assert {item.partition.ordinal for shard in shards for item in shard} == set(range(1000))
    assert sum(item.num_rows for shard in shards for item in shard) == 959216


def test_run_collection_writes_chunks_and_resumes(tmp_path, monkeypatch):
    table = pa.table(
        {
            "id": ["a", "b", "c", "d", "e"],
            "instruction_seed": ["one", "two", "three", "four", "five"],
            "__original_row_idx": [0, 1, 2, 3, 4],
        }
    )
    parquet_path = tmp_path / "input.parquet"
    pq.write_table(table, parquet_path, row_group_size=3)
    partitions = [
        collect.PartitionSpec(0, 0, 0, 0, 3),
        collect.PartitionSpec(1, 0, 1, 3, 2),
    ]
    monkeypatch.setattr(collect, "partition_specs", lambda: partitions)
    monkeypatch.setattr(collect.PartitionSpec, "url", property(lambda _self: str(parquet_path)))
    calls = []

    def post(_url, **kwargs):
        calls.append(kwargs["json"])
        return _response(
            {
                "id": f"response-{len(calls)}",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"answer-{len(calls)}",
                            "reasoning_content": "reasoning",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            }
        )

    monkeypatch.setattr(requests, "post", post)
    output_path = StoragePath(str(tmp_path / "output"))
    collection = collect.CollectionConfig("test", output_path, 0, 1, None, 2, 2)
    server = collect.ServerConfig(65536, 12)
    sampling = collect.SamplingConfig(1.0, 0.95, 16384)

    collect._run_collection("http://vllm", collection, server, sampling)
    parquet_path.unlink()
    collect._run_collection("http://vllm", collection, server, sampling)

    assert len(calls) == 5
    chunks = sorted((tmp_path / "output" / "responses").rglob("*.jsonl.gz"))
    assert len(chunks) == 3
    with gzip.open(chunks[0], "rt") as handle:
        first_chunk = [json.loads(line) for line in handle]
    assert [record["instruction_seed"] for record in first_chunk] == ["one", "two"]
    assert all(record["response"]["reasoning_content"] == "reasoning" for record in first_chunk)
    progress = json.loads((tmp_path / "output" / "progress" / "shard-000.json").read_text())
    assert progress["state"] == "complete"
    assert progress["complete_records"] == 5
    assert progress["skipped_records_this_attempt"] == 5
