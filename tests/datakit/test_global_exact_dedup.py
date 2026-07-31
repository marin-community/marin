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
from marin.datakit.source_key import datakit_source_key

from experiments.datakit.global_exact_dedup import GlobalExactDedupData, global_exact_deduplicate


@pytest.fixture(autouse=True)
def flow_backend_ctx():
    with set_current_client(LocalClient()):
        yield


def _normalized_source(root: Path, shards: list[list[dict]]) -> NormalizedData:
    main = root / "outputs" / "main"
    main.mkdir(parents=True)
    for shard_index, records in enumerate(shards):
        table = (
            pa.Table.from_pylist(records)
            if records
            else pa.table({"id": pa.array([], type=pa.string()), "text": pa.array([], type=pa.string())})
        )
        pq.write_table(
            table,
            main / f"part-{shard_index:05d}-of-{len(shards):05d}.parquet",
        )
    return NormalizedData(main_output_dir=str(main), dup_output_dir=str(root / "outputs" / "dups"), counters={})


def _attribute_shards(result: GlobalExactDedupData, source: NormalizedData) -> list[Path]:
    return sorted(Path(result.sources[datakit_source_key(source.main_output_dir)].attr_dir).glob("*.parquet"))


def test_global_exact_deduplicate_writes_sparse_copartitioned_attributes(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
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
        [[{"id": "c-only", "text": "C only", "c_metadata": True}], []],
    )

    result = global_exact_deduplicate(
        sources={"c": source_c, "b": source_b, "a": source_a},
        output_path=str(tmp_path / "output"),
        worker_resources=ResourceConfig(cpu=1, ram="1g"),
        max_workers=2,
    )

    a_shards = _attribute_shards(result, source_a)
    b_shards = _attribute_shards(result, source_b)
    c_shards = _attribute_shards(result, source_c)
    assert a_shards == []
    assert [path.name for path in b_shards] == ["part-00001-of-00002.parquet"]
    assert c_shards == []

    assert pq.read_table(b_shards[0]).to_pylist() == [
        {"id": "a-only", "dup_doc": True},
        {"id": "shared", "dup_doc": True},
    ]
    assert pq.read_schema(b_shards[0]).names == ["id", "dup_doc"]

    assert result.counters["global_exact_dedup/records_in"] == 6
    assert result.counters["global_exact_dedup/duplicate_records"] == 2


def test_global_exact_deduplicate_uses_shard_order_within_source(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    source = _normalized_source(
        tmp_path / "input",
        [
            [{"id": "shared", "text": "first shard"}],
            [{"id": "shared", "text": "second shard"}],
        ],
    )

    result = global_exact_deduplicate(
        sources={"source": source},
        output_path=str(tmp_path / "output"),
        worker_resources=ResourceConfig(cpu=1, ram="1g"),
        max_workers=1,
    )

    shards = _attribute_shards(result, source)
    assert [path.name for path in shards] == ["part-00001-of-00002.parquet"]
    assert pq.read_table(shards[0]).to_pylist() == [
        {"id": "shared", "dup_doc": True},
    ]


def test_global_exact_deduplicate_rejects_duplicate_source_directories(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    source = _normalized_source(tmp_path / "input", [[{"id": "a", "text": "A"}]])

    with pytest.raises(ValueError, match="Multiple sources use source_key"):
        global_exact_deduplicate(
            sources={"a": source, "alias": source},
            output_path=str(tmp_path / "output"),
            worker_resources=ResourceConfig(cpu=1, ram="1g"),
            max_workers=1,
        )
