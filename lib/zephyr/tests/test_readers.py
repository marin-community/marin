# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for parquet reader (load_parquet)."""


import itertools
import sys

import msgspec
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zstandard as zstd
from zephyr.expr import ColumnExpr, CompareExpr, LiteralExpr
from zephyr.readers import (
    SUPPORTED_EXTENSIONS,
    InputFileSpec,
    compute_parquet_splits,
    load_jsonl,
    load_parquet,
    load_parquet_batch,
)


def _write_test_parquet(path: str, records: list[dict], row_group_size: int = 2) -> None:
    """Write a parquet file with small row groups for testing."""
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path, row_group_size=row_group_size)


RECORDS = [{"id": i, "name": f"row{i}", "score": float(i * 10)} for i in range(10)]


def test_load_parquet_plain(tmp_path):
    path = str(tmp_path / "data.parquet")
    _write_test_parquet(path, RECORDS)

    result = list(load_parquet(path))
    assert result == RECORDS


def test_load_parquet_columns(tmp_path):
    path = str(tmp_path / "data.parquet")
    _write_test_parquet(path, RECORDS)

    spec = InputFileSpec(path=path, columns=["id", "name"])
    result = list(load_parquet(spec))
    assert result == [{"id": r["id"], "name": r["name"]} for r in RECORDS]


def test_load_parquet_row_range(tmp_path):
    path = str(tmp_path / "data.parquet")
    _write_test_parquet(path, RECORDS, row_group_size=3)

    spec = InputFileSpec(path=path, row_start=2, row_end=7)
    result = list(load_parquet(spec))
    assert [r["id"] for r in result] == [2, 3, 4, 5, 6]


def test_load_parquet_filter(tmp_path):
    path = str(tmp_path / "data.parquet")
    _write_test_parquet(path, RECORDS)

    spec = InputFileSpec(
        path=path,
        filter_expr=CompareExpr(op="ge", left=ColumnExpr(name="score"), right=LiteralExpr(value=50.0)),
    )
    result = list(load_parquet(spec))
    assert all(r["score"] >= 50.0 for r in result)
    assert [r["id"] for r in result] == [5, 6, 7, 8, 9]


def test_load_parquet_filter_and_row_range(tmp_path):
    path = str(tmp_path / "data.parquet")
    _write_test_parquet(path, RECORDS, row_group_size=3)

    spec = InputFileSpec(
        path=path,
        row_start=1,
        row_end=8,
        filter_expr=CompareExpr(op="ge", left=ColumnExpr(name="score"), right=LiteralExpr(value=50.0)),
    )
    result = list(load_parquet(spec))
    # rows 1-7, then filtered to score >= 50 → ids 5, 6, 7
    assert [r["id"] for r in result] == [5, 6, 7]


def test_load_parquet_filter_on_unprojected_column(tmp_path):
    """Filter can reference columns not in the projection."""
    path = str(tmp_path / "data.parquet")
    _write_test_parquet(path, RECORDS)

    spec = InputFileSpec(
        path=path,
        columns=["id", "name"],
        filter_expr=CompareExpr(op="ge", left=ColumnExpr(name="score"), right=LiteralExpr(value=50.0)),
    )
    result = list(load_parquet(spec))
    assert [r["id"] for r in result] == [5, 6, 7, 8, 9]
    assert all(set(r.keys()) == {"id", "name"} for r in result)


def test_load_parquet_empty(tmp_path):
    path = str(tmp_path / "empty.parquet")
    table = pa.Table.from_pylist([], schema=pa.schema([("id", pa.int64())]))
    pq.write_table(table, path)

    result = list(load_parquet(path))
    assert result == []


def test_compute_parquet_splits_single(tmp_path):
    """File smaller than approx_shard_bytes returns one split covering all rows."""
    path = str(tmp_path / "data.parquet")
    _write_test_parquet(path, RECORDS, row_group_size=5)

    splits = compute_parquet_splits(path, approx_shard_bytes=256 * 1024 * 1024)
    assert splits == [(0, len(RECORDS))]


def test_compute_parquet_splits_multiple(tmp_path):
    """File whose row groups exceed approx_shard_bytes returns multiple splits."""
    path = str(tmp_path / "data.parquet")
    # Write 10 row groups of 1 row each with a large-ish payload so we can
    # force splits at a small byte threshold.
    records = [{"id": i, "payload": "x" * 1000} for i in range(10)]
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path, row_group_size=1)

    pf = pq.ParquetFile(path)
    single_rg_bytes = pf.metadata.row_group(0).total_byte_size
    # Threshold just above one row group forces a split after every row group.
    threshold = single_rg_bytes + 1

    splits = compute_parquet_splits(path, approx_shard_bytes=threshold)
    assert len(splits) > 1
    # Splits must be contiguous and cover all rows.
    assert splits[0][0] == 0
    assert splits[-1][1] == 10
    for (_, end), (start, _) in itertools.pairwise(splits):
        assert end == start


def test_compute_parquet_splits_row_ranges_are_readable(tmp_path):
    """Each split returned by compute_parquet_splits can be read back via load_parquet."""
    path = str(tmp_path / "data.parquet")
    records = [{"id": i} for i in range(20)]
    pq.write_table(pa.Table.from_pylist(records), path, row_group_size=2)

    pf = pq.ParquetFile(path)
    single_rg_bytes = pf.metadata.row_group(0).total_byte_size
    splits = compute_parquet_splits(path, approx_shard_bytes=single_rg_bytes * 3 + 1)

    all_ids = []
    for row_start, row_end in splits:
        spec = InputFileSpec(path=path, row_start=row_start, row_end=row_end)
        all_ids.extend(r["id"] for r in load_parquet(spec))

    assert sorted(all_ids) == list(range(20))


def test_load_parquet_no_dataset_api(tmp_path, monkeypatch):
    """Verify that load_parquet does NOT import pyarrow.dataset."""

    path = str(tmp_path / "data.parquet")
    _write_test_parquet(path, RECORDS)

    # Remove pyarrow.dataset from sys.modules and block re-import
    sys.modules.pop("pyarrow.dataset", None)
    monkeypatch.setitem(sys.modules, "pyarrow.dataset", None)

    # Should succeed without pyarrow.dataset
    result = list(load_parquet(path))
    assert len(result) == len(RECORDS)


def test_load_parquet_batch_returns_record_batches(tmp_path):
    path = str(tmp_path / "data.parquet")
    _write_test_parquet(path, RECORDS, row_group_size=4)

    batches = list(load_parquet_batch(path))
    assert all(isinstance(b, pa.RecordBatch) for b in batches)
    # All rows present across batches
    all_rows = [row for b in batches for row in b.to_pylist()]
    assert sorted(all_rows, key=lambda r: r["id"]) == RECORDS


def test_load_parquet_batch_consistent_with_load_parquet(tmp_path):
    """load_parquet_batch + to_pylist must equal load_parquet."""
    path = str(tmp_path / "data.parquet")
    _write_test_parquet(path, RECORDS, row_group_size=3)

    spec = InputFileSpec(path=path, row_start=2, row_end=8)
    via_batch = [row for b in load_parquet_batch(spec) for row in b.to_pylist()]
    via_dict = list(load_parquet(spec))
    assert via_batch == via_dict


@pytest.mark.parametrize("ext", [".jsonl.zst", ".jsonl.zstd"])
def test_load_jsonl_decompresses_zstd_extensions(tmp_path, ext):
    """Both the ``.zst`` and ``.zstd`` extensions must be zstd-decompressed.

    ``.zstd`` is advertised in SUPPORTED_EXTENSIONS; a genuinely-compressed
    file written under it must decode, not be read back as raw bytes.
    """
    path = str(tmp_path / f"data{ext}")
    encoder = msgspec.json.Encoder()
    raw = b"".join(encoder.encode(r) + b"\n" for r in RECORDS)
    with open(path, "wb") as f:
        f.write(zstd.ZstdCompressor().compress(raw))

    assert path.endswith(SUPPORTED_EXTENSIONS)
    assert list(load_jsonl(path)) == RECORDS
