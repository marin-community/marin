# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.ops.storage.recompress_parquet import RewriteDisposition, RewriteOptions, recompress_parquet, run_migration


def _compressions(path: Path) -> set[str]:
    metadata = pq.ParquetFile(path).metadata
    return {
        metadata.row_group(row_group).column(column).compression
        for row_group in range(metadata.num_row_groups)
        for column in range(metadata.num_columns)
    }


def _has_page_indexes(path: Path) -> bool:
    metadata = pq.ParquetFile(path).metadata
    return all(
        metadata.row_group(row_group).column(column).has_column_index
        and metadata.row_group(row_group).column(column).has_offset_index
        for row_group in range(metadata.num_row_groups)
        for column in range(metadata.num_columns)
    )


def test_recompress_parquet_replaces_snappy_with_smaller_zstd(tmp_path):
    path = tmp_path / "records.parquet"
    expected = pa.table({"id": range(10_000), "text": ["compressible record payload"] * 10_000})
    pq.write_table(expected, path, compression="snappy", row_group_size=1_000)
    input_bytes = path.stat().st_size

    result = recompress_parquet(str(path), RewriteOptions(apply=True, batch_rows=1_000))

    assert result.disposition is RewriteDisposition.REWRITTEN
    assert result.input_bytes == input_bytes
    assert result.output_bytes == path.stat().st_size
    assert result.output_bytes < result.input_bytes
    assert _compressions(path) == {"ZSTD"}
    assert _has_page_indexes(path)
    assert pq.read_table(path).equals(expected)


def test_recompress_parquet_dry_run_preserves_source(tmp_path):
    path = tmp_path / "records.parquet"
    pq.write_table(pa.table({"value": ["same"] * 1_000}), path, compression="snappy")
    original = path.read_bytes()

    result = recompress_parquet(str(path))

    assert result.disposition is RewriteDisposition.DRY_RUN
    assert result.output_bytes is None
    assert path.read_bytes() == original
    assert _compressions(path) == {"SNAPPY"}


def test_recompress_parquet_skips_existing_zstd(tmp_path):
    path = tmp_path / "records.parquet"
    pq.write_table(pa.table({"value": ["same"] * 1_000}), path, compression="zstd", write_page_index=True)
    original = path.read_bytes()

    result = recompress_parquet(str(path), RewriteOptions(apply=True))

    assert result.disposition is RewriteDisposition.ALREADY_ZSTD
    assert result.output_bytes is None
    assert path.read_bytes() == original


def test_recompress_parquet_adds_page_indexes_to_existing_zstd(tmp_path):
    path = tmp_path / "records.parquet"
    expected = pa.table({"value": [f"record {index}" for index in range(10_000)]})
    pq.write_table(expected, path, compression="zstd")

    result = recompress_parquet(str(path), RewriteOptions(apply=True))

    assert result.disposition is RewriteDisposition.REWRITTEN
    assert _compressions(path) == {"ZSTD"}
    assert _has_page_indexes(path)
    assert pq.read_table(path).equals(expected)


def test_recompress_parquet_preserves_source_when_zstd_is_larger(tmp_path):
    path = tmp_path / "records.parquet"
    pq.write_table(
        pa.table({"value": ["highly compressible payload"] * 10_000}),
        path,
        compression="brotli",
        compression_level=11,
    )
    original = path.read_bytes()

    result = recompress_parquet(str(path), RewriteOptions(apply=True))

    assert result.disposition is RewriteDisposition.NOT_SMALLER
    assert result.output_bytes is not None
    assert result.output_bytes >= result.input_bytes
    assert path.read_bytes() == original
    assert _compressions(path) == {"BROTLI"}


def test_recompress_parquet_rejects_lifecycle_prefix(tmp_path):
    path = tmp_path / "tmp" / "ttl=7d" / "records.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"value": [1]}), path, compression="snappy")

    with pytest.raises(ValueError, match="refusing to reset lifecycle retention"):
        recompress_parquet(str(path), RewriteOptions(apply=True))


def test_run_migration_rewrites_files_with_bounded_worker_pool(tmp_path):
    paths = [tmp_path / f"records-{index}.parquet" for index in range(3)]
    values = [f"record {index}: {'same payload ' * 20}" for index in range(10_000)]
    for path in paths:
        pq.write_table(pa.table({"value": values}), path, compression="snappy")

    run_migration(
        str(tmp_path / "*.parquet"),
        workers=2,
        worker_cpu=1,
        worker_ram="1g",
        options=RewriteOptions(apply=True),
    )

    assert [_compressions(path) for path in paths] == [{"ZSTD"}, {"ZSTD"}, {"ZSTD"}]
    assert all(_has_page_indexes(path) for path in paths)
