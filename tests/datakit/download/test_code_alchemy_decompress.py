# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
from pathlib import Path

import polars as pl
import pytest

from marin.datakit.download.code_alchemy_decompress import (
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    CodeAlchemyDecompressConfig,
    DecompressPrefixTask,
    DecompressionMetrics,
    decompress_gzip_batch,
    decompress_prefix_task,
)


def test_decompress_gzip_batch_is_deterministic_and_counts_lossy_utf8():
    frame = pl.DataFrame(
        {
            "blob_id": ["aa00", "aa01", "aa02", "aa03", "aa04", "aa05"],
            "source_gzip": [
                gzip.compress(b"first", mtime=0),
                b"not gzip",
                gzip.compress(b"\xff\xfe", mtime=0),
                gzip.compress("snowman: ☃".encode(), mtime=0),
                gzip.compress(b"", mtime=0),
                gzip.compress(b"truncated", mtime=0)[:-2],
            ],
        },
        schema={"blob_id": pl.String, "source_gzip": pl.Binary},
    )

    single_threaded, single_metrics = decompress_gzip_batch(frame, thread_count=1)
    threaded, threaded_metrics = decompress_gzip_batch(frame, thread_count=4)

    expected = pl.DataFrame(
        {
            "blob_id": ["aa00", "aa02", "aa03", "aa04"],
            "source": ["first", "", "snowman: ☃", ""],
        },
        schema={"blob_id": pl.String, "source": pl.String},
    )
    assert single_threaded.equals(expected)
    assert threaded.equals(expected)
    assert single_metrics == threaded_metrics == DecompressionMetrics(
        input_rows=6,
        decoded_rows=4,
        corrupt_gzip_rows=2,
        lossy_utf8_rows=1,
    )


def test_decompress_gzip_batch_rejects_duplicate_ids_and_noncanonical_types():
    duplicate_ids = pl.DataFrame(
        {
            "blob_id": ["aa00", "aa00"],
            "source_gzip": [gzip.compress(b"one", mtime=0), gzip.compress(b"two", mtime=0)],
        },
        schema={"blob_id": pl.String, "source_gzip": pl.Binary},
    )
    with pytest.raises(ValueError, match="duplicate blob_id"):
        decompress_gzip_batch(duplicate_ids, thread_count=2)

    utf8_payload_column = pl.DataFrame({"blob_id": ["aa00"], "source_gzip": ["not binary"]})
    with pytest.raises(TypeError, match="non-canonical types"):
        decompress_gzip_batch(utf8_payload_column, thread_count=1)


def test_decompress_prefix_task_writes_canonical_partition_and_metrics(tmp_path: Path):
    input_path = tmp_path / "downloaded.parquet"
    output_path = tmp_path / "data" / "blob_prefix=ab" / "part-00000.parquet"
    pl.DataFrame(
        {
            "blob_id": ["ab00", "AB01"],
            "source_gzip": [
                gzip.compress(b"alpha", mtime=0),
                gzip.compress("☃".encode(), mtime=0),
            ],
        },
        schema={"blob_id": pl.String, "source_gzip": pl.Binary},
    ).write_parquet(input_path)

    result = decompress_prefix_task(
        DecompressPrefixTask(
            prefix="ab",
            input_paths=(str(input_path),),
            output_path=str(output_path),
            thread_count=4,
        )
    )

    output = pl.read_parquet(output_path)
    assert output.schema == pl.Schema({"blob_id": pl.String, "source": pl.String})
    assert output.to_dicts() == [
        {"blob_id": "ab00", "source": "alpha"},
        {"blob_id": "AB01", "source": "☃"},
    ]
    assert result.input_rows == 2
    assert result.decoded_rows == 2
    assert result.corrupt_gzip_rows == 0
    assert result.lossy_utf8_rows == 0


def test_decompress_prefix_task_normalizes_invalid_utf8(tmp_path: Path):
    input_path = tmp_path / "downloaded.parquet"
    output_path = tmp_path / "output.parquet"
    pl.DataFrame(
        {
            "blob_id": ["ab02"],
            "source_gzip": [gzip.compress(b"\xff", mtime=0)],
        },
        schema={"blob_id": pl.String, "source_gzip": pl.Binary},
    ).write_parquet(input_path)

    result = decompress_prefix_task(
        DecompressPrefixTask(
            prefix="ab",
            input_paths=(str(input_path),),
            output_path=str(output_path),
            thread_count=1,
        )
    )

    assert pl.read_parquet(output_path).to_dicts() == [{"blob_id": "ab02", "source": ""}]
    assert result.decoded_rows == 1
    assert result.lossy_utf8_rows == 1


def test_full_node_resource_defaults_and_contracted_paths():
    cfg = CodeAlchemyDecompressConfig()

    assert cfg.input_path == DEFAULT_INPUT_PATH
    assert cfg.output_path == DEFAULT_OUTPUT_PATH
    assert cfg.max_workers == 1
    assert cfg.thread_count == 192
    assert cfg.worker_resources.cpu == 180
    assert cfg.worker_resources.ram == "1200g"
    assert cfg.task_resources.cpu == 180
    assert cfg.worker_resources.disk == "1000g"
    assert cfg.task_resources.disk == "1000g"
    assert cfg.task_resources.ram == "1200g"
