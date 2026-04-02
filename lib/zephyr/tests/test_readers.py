# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for parquet reader (load_parquet)."""


import pyarrow as pa
import pyarrow.parquet as pq

from zephyr.expr import ColumnExpr, CompareExpr, LiteralExpr
from zephyr.readers import InputFileSpec, load_parquet


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


def test_load_parquet_empty(tmp_path):
    path = str(tmp_path / "empty.parquet")
    table = pa.Table.from_pylist([], schema=pa.schema([("id", pa.int64())]))
    pq.write_table(table, path)

    result = list(load_parquet(path))
    assert result == []


def test_load_parquet_no_dataset_api(tmp_path, monkeypatch):
    """Verify that load_parquet does NOT import pyarrow.dataset."""
    import sys

    path = str(tmp_path / "data.parquet")
    _write_test_parquet(path, RECORDS)

    # Remove pyarrow.dataset from sys.modules and block re-import
    sys.modules.pop("pyarrow.dataset", None)
    monkeypatch.setitem(sys.modules, "pyarrow.dataset", None)

    # Should succeed without pyarrow.dataset
    result = list(load_parquet(path))
    assert len(result) == len(RECORDS)
