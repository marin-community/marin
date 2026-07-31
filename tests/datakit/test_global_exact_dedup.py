# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the Datakit global exact-dedup step."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData

from experiments.datakit.global_exact_dedup import global_exact_deduplicate


@pytest.fixture(autouse=True)
def flow_backend_ctx():
    with set_current_client(LocalClient()):
        yield


def _normalized_source(root: Path, shards: list[list[dict]]) -> NormalizedData:
    main = root / "outputs" / "main"
    main.mkdir(parents=True)
    for shard_index, records in enumerate(shards):
        pq.write_table(
            pa.Table.from_pylist(records),
            main / f"part-{shard_index:05d}-of-{len(shards):05d}.parquet",
        )
    return NormalizedData(main_output_dir=str(main), dup_output_dir=str(root / "outputs" / "dups"), counters={})


def _shards(source: NormalizedData) -> list[list[dict]]:
    return [pq.read_table(path).to_pylist() for path in sorted(Path(source.main_output_dir).rglob("*.parquet"))]


def test_global_exact_deduplicate_keeps_one_record_for_each_id(tmp_path: Path):
    source_a = _normalized_source(
        tmp_path / "input-a",
        [
            [
                {"id": "a-only", "text": "A only", "a_metadata": 1},
                {"id": "shared", "text": "Canonical text", "a_metadata": 2},
            ],
        ],
    )
    source_b = _normalized_source(
        tmp_path / "input-b",
        [
            [{"id": "b-only", "text": "B only", "b_metadata": "x"}],
            [
                {"id": "shared", "text": "Different text with the same record ID", "b_metadata": "y"},
                {"id": "a-only", "text": "Another duplicate ID", "b_metadata": "z"},
            ],
        ],
    )
    source_c = _normalized_source(
        tmp_path / "input-c",
        [[{"id": "c-only", "text": "C only", "c_metadata": True}]],
    )

    result = global_exact_deduplicate(
        sources={"c": source_c, "b": source_b, "a": source_a},
        output_path=str(tmp_path / "output"),
        worker_resources=ResourceConfig(cpu=1, ram="1g"),
        max_workers=2,
    )

    assert _shards(result.sources["a"]) == [
        [
            {"id": "a-only", "text": "A only", "a_metadata": 1},
            {"id": "shared", "text": "Canonical text", "a_metadata": 2},
        ]
    ]
    assert _shards(result.sources["b"]) == [
        [{"id": "b-only", "text": "B only", "b_metadata": "x"}],
        [],
    ]
    assert _shards(result.sources["c"]) == [[{"id": "c-only", "text": "C only", "c_metadata": True}]]
    assert result.counters["global_exact_dedup/records_in"] == 6
    assert result.counters["global_exact_dedup/records_out"] == 4
    assert result.counters["global_exact_dedup/duplicate_records"] == 2
    assert result.sources["b"].counters == {
        "global_exact_dedup/records_in": 3,
        "global_exact_dedup/records_out": 1,
        "global_exact_dedup/duplicate_records": 2,
    }
