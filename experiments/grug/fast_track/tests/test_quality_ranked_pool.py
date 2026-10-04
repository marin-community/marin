# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.grug.fast_track.ranked_pool import ranked_pool, take_token_prefix


@pytest.fixture
def zephyr_context(tmp_path):
    client = LocalClient()
    context = ZephyrContext(
        client=client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name="quality-pool-test",
    )
    yield context
    context.shutdown()
    client.shutdown(wait=True)


@pytest.fixture
def pool_parquets(tmp_path):
    rows = [
        {"id": "a", "rank": "00-a", "range": 0, "token_count": 4, "input_ids": [1, 2, 3, 4]},
        {"id": "b", "rank": "00-b", "range": 0, "token_count": 3, "input_ids": [5, 6, 7]},
        {"id": "c", "rank": "01-c", "range": 1, "token_count": 5, "input_ids": [8, 9, 10, 11, 12]},
        {"id": "d", "rank": "02-d", "range": 2, "token_count": 2, "input_ids": [13, 14]},
        {"id": "e", "rank": "03-e", "range": 3, "token_count": 6, "input_ids": [15, 16, 17, 18, 19, 20]},
    ]
    first = tmp_path / "raw-a.parquet"
    second = tmp_path / "raw-b.parquet"
    pq.write_table(pa.Table.from_pylist(rows[::2]), first)
    pq.write_table(pa.Table.from_pylist(rows[1::2]), second)
    return rows, (first, second)


def _rank(dataset, context, output_path: Path, ranges: int):
    return ranked_pool(
        dataset,
        ctx=context,
        output_path=str(output_path),
        rank_key=lambda row: (row["rank"], row["id"]),
        token_count=lambda row: row["token_count"],
        range_key=lambda row: row["range"],
        num_ranges=ranges,
    )


def test_ranked_pool_orders_shards_and_prefix_independently_of_input_partitions(pool_parquets, zephyr_context, tmp_path):
    rows, paths = pool_parquets
    first = _rank(Dataset.from_files(str(paths[0])).load_parquet(), zephyr_context, tmp_path / "first", 4)
    second = _rank(Dataset.from_files(str(paths[1])).load_parquet(), zephyr_context, tmp_path / "second", 4)
    combined = _rank(
        Dataset.from_files(str(tmp_path / "raw-*.parquet")).load_parquet(), zephyr_context, tmp_path / "combined", 4
    )

    assert [item.range_key for item in combined.range_totals] == sorted({row["range"] for row in rows})
    assert combined.total_tokens == sum(row["token_count"] for row in rows)
    assert [path.rsplit("/", 1)[-1] for path in combined.shards] == [f"range-{key:05d}.parquet" for key in range(4)]
    assert first.total_documents + second.total_documents == combined.total_documents

    prefix = take_token_prefix(combined, ctx=zephyr_context, output_path=str(tmp_path / "prefix"), token_budget=10)
    selected = [row["id"] for path in prefix.shards for row in pq.read_table(path).to_pylist()]

    assert selected == ["a", "b", "c"]
    assert prefix.total_tokens == 12
    assert prefix.usable_tokens == 10
    assert prefix.overshoot_tokens == 2


def test_ranked_pool_rejects_budget_above_available_token_mass(pool_parquets, zephyr_context, tmp_path):
    _, _paths = pool_parquets
    pool = _rank(
        Dataset.from_files(str(tmp_path / "raw-*.parquet")).load_parquet(), zephyr_context, tmp_path / "ranked", 4
    )

    with pytest.raises(ValueError, match="below the 21 token budget"):
        take_token_prefix(pool, ctx=zephyr_context, output_path=str(tmp_path / "prefix"), token_budget=21)
