# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.types import ResourceConfig
from rigging.timing import Duration, ExponentialBackoff
from zephyr.context import IntraShardExecution, ZephyrContext
from zephyr.dataset import Dataset
from zephyr.runners import SubprocessRunner


@pytest.fixture(params=[2])
def context(local_client, tmp_path, request):
    ctx = ZephyrContext(
        client=local_client,
        max_workers=request.param,
        resources=ResourceConfig(cpu=1, ram="512m"),
        intra_shard_execution=IntraShardExecution.AUTO,
        stage_runner_factory=SubprocessRunner,
        chunk_storage_prefix=str(tmp_path / "chunks"),
    )
    yield ctx
    ctx.shutdown()


@pytest.mark.parametrize("explicit_schema", [False, True])
def test_parallel_parquet_map_preserves_output_shard(context, tmp_path, explicit_schema):
    source = tmp_path / "input.parquet"
    pq.write_table(pa.table({"id": range(13)}), source, row_group_size=13)

    def transform(row):
        # The later range must finish its mapping before the first can proceed.
        # A single task cannot pass this barrier.
        if row["id"] == 12:
            (tmp_path / "later-range").touch()
        if row["id"] == 0:
            assert ExponentialBackoff(initial=0.01, maximum=0.1).wait_until(
                lambda: (tmp_path / "later-range").exists(), timeout=Duration.from_seconds(15)
            )
        return {
            "id": row["id"],
            "pid": os.getpid(),
            "value": None if row["id"] < 7 else row["id"],
            "nested": [{"value": str(row["id"])}],
        }

    ds = Dataset.from_files(str(source)).load_parquet().map(transform)
    schema = (
        pa.schema(
            [
                ("id", pa.int64()),
                ("pid", pa.int64()),
                ("value", pa.int64()),
                ("nested", pa.list_(pa.struct([("value", pa.string())]))),
            ]
        )
        if explicit_schema
        else None
    )
    paths = context.execute(ds.write_parquet(str(tmp_path / "part-{shard}-of-{total}.parquet"), schema=schema)).results
    assert paths == [str(tmp_path / "part-0-of-1.parquet")]
    records = pq.read_table(paths[0]).to_pylist()
    assert [row["id"] for row in records] == list(range(13))
    assert [row["value"] for row in records] == [None] * 7 + list(range(7, 13))
    assert [row["nested"] for row in records] == [[{"value": str(i)}] for i in range(13)]
    assert len({row["pid"] for row in records}) == 2
    assert all(row["pid"] != os.getpid() for row in records)


@pytest.mark.parametrize("batch_mode", [False, True])
@pytest.mark.parametrize("context", [4], indirect=True)
def test_concat_preserves_empty_shards_limits_and_windows(context, tmp_path, batch_mode):
    sources = []
    for index, ids in enumerate([[], list(range(13)), [0, 1]]):
        path = str(tmp_path / f"input-{index}.parquet")
        pq.write_table(pa.table({"id": pa.array(ids, type=pa.int64())}), path, row_group_size=13)
        sources.append(path)
    ds = Dataset.from_list(sources).load_parquet(batch_mode=batch_mode)
    if batch_mode:
        ds = ds.flat_map(lambda batch: batch.to_pylist())
    ds = ds.filter(lambda row: row["id"] >= 3).map(lambda row: row["id"])
    ds = (
        ds.take_per_shard(9)
        .window(4)
        .map_shard(lambda batches, shard: [(shard.shard_idx, shard.total_shards, list(batches))])
    )
    assert context.execute(ds).results == [
        (0, 3, []),
        (1, 3, [[3, 4, 5, 6], [7, 8, 9, 10], [11]]),
        (2, 3, []),
    ]


def test_concat_preserves_sparse_join_alignment(context, tmp_path):
    left = tmp_path / "left.parquet"
    right = tmp_path / "right.parquet"
    pq.write_table(pa.table({"id": range(13)}), left)
    pq.write_table(pa.table({"id": [3, 4, 10, 12]}), right)
    joined = (
        Dataset.from_files(str(left))
        .load_parquet()
        .sorted_merge_join(
            Dataset.from_files(str(right)).load_parquet(),
            left_key=lambda row: row["id"],
            right_key=lambda row: row["id"],
            combiner=lambda row, attr: (row["id"], attr is not None),
            how="left",
        )
    )
    assert context.execute(joined).results == [(i, i in {3, 4, 10, 12}) for i in range(13)]


def test_intermediate_chunks_run_concurrently_and_concat(context, tmp_path):
    def transform(value):
        if value == 100_000:
            (tmp_path / "second-chunk").touch()
        if value == 0:
            assert ExponentialBackoff(initial=0.01, maximum=0.1).wait_until(
                lambda: (tmp_path / "second-chunk").exists(), timeout=Duration.from_seconds(15)
            )
        return value

    ds = Dataset.from_list([100_003]).flat_map(range).reshard(1).map(transform)
    ds = ds.map_shard(lambda rows, shard: [(shard.shard_idx, shard.total_shards, list(rows))])
    assert context.execute(ds).results == [(0, 1, list(range(100_003)))]


def test_batch_maps_keep_original_record_batches(context, tmp_path):
    source = tmp_path / "batch.parquet"
    pq.write_table(pa.table({"id": range(13)}), source, row_group_size=13)
    ds = Dataset.from_files(str(source)).load_parquet(batch_mode=True).map(lambda batch: batch.num_rows)
    assert context.execute(ds).results == [13]


def test_empty_parquet_fragments_preserve_output_schema(context, tmp_path):
    schema = pa.schema([("id", pa.int64())])
    source = tmp_path / "input.parquet"
    pq.write_table(pa.table({"id": range(13)}), source)
    output = tmp_path / "empty-0-of-1.parquet"
    ds = Dataset.from_files(str(source)).load_parquet().filter(lambda row: row["id"] < 0)
    result = context.execute(ds.write_parquet(str(tmp_path / "empty-{shard}-of-{total}.parquet"), schema=schema))
    assert result.results == [str(output)]
    table = pq.read_table(output)
    assert table.num_rows == 0
    assert table.schema == schema
