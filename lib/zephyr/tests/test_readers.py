# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for parquet reader (load_parquet)."""


import pyarrow as pa
import pyarrow.parquet as pq
from zephyr.expr import ColumnExpr, CompareExpr, LiteralExpr
from zephyr.readers import InputFileSpec, iter_parquet_row_groups, load_parquet


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


def test_iter_parquet_row_groups_equality_predicates(tmp_path):
    """Statistics-based row group skipping via equality_predicates."""
    path = str(tmp_path / "scatter.parquet")
    # Write a scatter-like file: each row group has one shard_idx value
    writer = None
    for shard_idx in range(4):
        rows = [{"shard_idx": shard_idx, "chunk_idx": 0, "item": f"s{shard_idx}-{i}"} for i in range(5)]
        table = pa.Table.from_pylist(rows)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema)
        writer.write_table(table)
    writer.close()

    # Read only shard_idx == 2
    tables = list(iter_parquet_row_groups(path, columns=["item"], equality_predicates={"shard_idx": 2}))
    assert len(tables) == 1
    items = tables[0].column("item").to_pylist()
    assert items == [f"s2-{i}" for i in range(5)]


def test_iter_parquet_row_groups_multi_predicate(tmp_path):
    """Statistics-based skipping with multiple equality predicates."""
    path = str(tmp_path / "scatter.parquet")
    writer = None
    for shard_idx in range(3):
        for chunk_idx in range(2):
            rows = [{"shard_idx": shard_idx, "chunk_idx": chunk_idx, "data": f"{shard_idx}-{chunk_idx}"}]
            table = pa.Table.from_pylist(rows)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
    writer.close()

    tables = list(
        iter_parquet_row_groups(
            path,
            columns=["data"],
            equality_predicates={"shard_idx": 1, "chunk_idx": 1},
        )
    )
    assert len(tables) == 1
    assert tables[0].column("data").to_pylist() == ["1-1"]


def test_iter_parquet_row_groups_skips_all_non_matching(tmp_path):
    """No row groups match → empty result, no data read."""
    path = str(tmp_path / "data.parquet")
    _write_test_parquet(path, RECORDS, row_group_size=5)

    tables = list(iter_parquet_row_groups(path, equality_predicates={"id": 999}))
    assert tables == []


def test_iter_parquet_row_groups_filters_within_row_group(tmp_path):
    """Row-level filtering when a row group contains multiple predicate values."""
    path = str(tmp_path / "mixed.parquet")
    # Single row group with mixed shard values
    rows = [{"shard": s, "val": f"s{s}-{i}"} for s in range(3) for i in range(2)]
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=100)

    tables = list(iter_parquet_row_groups(path, equality_predicates={"shard": 1}))
    assert len(tables) == 1
    assert tables[0].column("val").to_pylist() == ["s1-0", "s1-1"]


def test_iter_parquet_row_groups_predicate_columns_dropped(tmp_path):
    """Predicate columns not in requested columns are read for filtering then dropped."""
    path = str(tmp_path / "drop.parquet")
    rows = [{"shard": s, "data": f"d{s}"} for s in range(3)]
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=100)

    tables = list(iter_parquet_row_groups(path, columns=["data"], equality_predicates={"shard": 2}))
    assert len(tables) == 1
    assert tables[0].column_names == ["data"]
    assert tables[0].column("data").to_pylist() == ["d2"]


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
