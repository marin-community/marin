# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import polars as pl
import pytest
from fray.local_backend import LocalClient
from marin.datakit.download.stack_v2 import (
    BLOB_PREFIX_PARTITION_COUNT,
    StackV2ParquetTask,
    build_stack_v2_pipeline,
    partition_stack_v2_parquet,
)
from zephyr.context import ZephyrContext
from zephyr.plan import compute_plan


def test_partition_stack_v2_parquet_streams_directly_to_blob_prefixes(tmp_path: Path):
    source_path = tmp_path / "source.parquet"
    source = pl.DataFrame(
        {
            "blob_id": ["00abcdef", "Fe012345", "ff987654", "00fedcba"],
            "path": ["a.py", "b.rs", "c.go", "d.java"],
        }
    )
    source.write_parquet(source_path)

    output_path = tmp_path / "partitioned"
    result = partition_stack_v2_parquet(
        StackV2ParquetTask(
            source_url=str(source_path),
            relative_source_path="data/Python/train-00000.parquet",
            output_path=str(output_path),
        )
    )

    assert result["partition_count"] == 256
    parquet_paths = sorted(output_path.glob("blob_prefix=*/*.parquet"))
    assert {path.parent.name for path in parquet_paths} == {
        "blob_prefix=00",
        "blob_prefix=fe",
        "blob_prefix=ff",
    }
    written = pl.concat(pl.read_parquet(path) for path in parquet_paths).sort("blob_id")
    assert written.columns == source.columns
    assert written.to_dicts() == source.sort("blob_id").to_dicts()


def test_partition_stack_v2_parquet_rejects_non_hex_blob_prefix(tmp_path: Path):
    source_path = tmp_path / "source.parquet"
    pl.DataFrame({"blob_id": ["zz-not-a-hash"], "path": ["bad.py"]}).write_parquet(source_path)

    with pytest.raises(ValueError, match="blob_id must start with two hexadecimal characters"):
        partition_stack_v2_parquet(
            StackV2ParquetTask(
                source_url=str(source_path),
                relative_source_path="data/Python/train-00000.parquet",
                output_path=str(tmp_path / "partitioned"),
            )
        )


def test_stack_v2_pipeline_assigns_one_zephyr_shard_per_parquet_file(tmp_path: Path):
    tasks = [
        StackV2ParquetTask(
            source_url=str(tmp_path / f"source-{index}.parquet"),
            relative_source_path=f"data/Python/train-{index:05d}.parquet",
            output_path=str(tmp_path / "partitioned"),
        )
        for index in range(3)
    ]

    plan = compute_plan(build_stack_v2_pipeline(tasks, str(tmp_path / "output")))

    assert BLOB_PREFIX_PARTITION_COUNT == 256
    assert plan.num_shards == len(tasks)
    assert [item.data for item in plan.source_items] == tasks


def test_stack_v2_pipeline_executes_partitioning_and_metrics(tmp_path: Path):
    tasks = []
    for index, prefix in enumerate(("00", "ff")):
        source_path = tmp_path / f"source-{index}.parquet"
        pl.DataFrame({"blob_id": [f"{prefix}abcdef"], "path": [f"{index}.py"]}).write_parquet(source_path)
        tasks.append(
            StackV2ParquetTask(
                source_url=str(source_path),
                relative_source_path=f"data/Python/train-{index:05d}.parquet",
                output_path=str(tmp_path / "output" / "data"),
            )
        )

    context = ZephyrContext(
        client=LocalClient(),
        max_workers=2,
        chunk_storage_prefix=str(tmp_path / "zephyr"),
    )
    context.execute(build_stack_v2_pipeline(tasks, str(tmp_path / "output")))

    assert len(list((tmp_path / "output" / "data").glob("blob_prefix=*/*.parquet"))) == 2
    assert len(list((tmp_path / "output" / ".metrics").glob("success-part-*.jsonl"))) == 2
