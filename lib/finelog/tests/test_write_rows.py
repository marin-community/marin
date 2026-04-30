# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the WriteRows RPC: append round-trip, error cases,
dict-encoded column decode, nested-type rejection, and size/row caps.
All tests operate directly on ``DuckDBLogStore`` without going through
the ASGI layer.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finelog.store.duckdb_store import DuckDBLogStore
from finelog.store.schema import (
    MAX_WRITE_ROWS_BYTES,
    Column,
    ColumnType,
    NamespaceNotFoundError,
    Schema,
    SchemaValidationError,
)

from tests.conftest import _ipc_bytes, _worker_schema

# ---------------------------------------------------------------------------
# WriteRows: happy path + validation
# ---------------------------------------------------------------------------


def test_write_rows_append_round_trip(store: DuckDBLogStore):
    schema = _worker_schema()
    store.register_table("iris.worker", schema)
    batch = pa.RecordBatch.from_pydict(
        {
            "worker_id": ["w-1", "w-2"],
            "mem_bytes": [100, 200],
            "timestamp_ms": [1, 2],
        },
        schema=pa.schema(
            [
                pa.field("worker_id", pa.string(), nullable=False),
                pa.field("mem_bytes", pa.int64(), nullable=False),
                pa.field("timestamp_ms", pa.int64(), nullable=False),
            ]
        ),
    )
    n = store.write_rows("iris.worker", _ipc_bytes(batch))
    assert n == 2

    # Parquet on disk has the registered schema.
    store._namespaces["iris.worker"]._flush_step()
    seg_dir = store._data_dir / "iris.worker"
    parquet_files = sorted(seg_dir.glob("*.parquet"))
    assert len(parquet_files) >= 1
    table = pq.read_table(parquet_files[-1])
    assert sorted(table.column_names) == ["mem_bytes", "timestamp_ms", "worker_id"]


def test_write_rows_unknown_namespace_raises(store: DuckDBLogStore):
    batch = pa.RecordBatch.from_pydict({"x": [1]}, schema=pa.schema([pa.field("x", pa.int64())]))
    with pytest.raises(NamespaceNotFoundError):
        store.write_rows("not.registered", _ipc_bytes(batch))


def test_write_rows_missing_nullable_column_filled_with_null(store: DuckDBLogStore):
    schema = Schema(
        columns=(
            Column(name="worker_id", type=ColumnType.STRING),
            Column(name="note", type=ColumnType.STRING, nullable=True),
            Column(name="timestamp_ms", type=ColumnType.INT64),
        ),
    )
    store.register_table("iris.worker", schema)
    # Batch omits the nullable "note" column entirely.
    batch = pa.RecordBatch.from_pydict(
        {"worker_id": ["w-1"], "timestamp_ms": [1]},
        schema=pa.schema(
            [
                pa.field("worker_id", pa.string(), nullable=False),
                pa.field("timestamp_ms", pa.int64(), nullable=False),
            ]
        ),
    )
    store.write_rows("iris.worker", _ipc_bytes(batch))
    store._namespaces["iris.worker"]._flush_step()
    seg_dir = store._data_dir / "iris.worker"
    table = pq.read_table(sorted(seg_dir.glob("*.parquet"))[-1])
    assert "note" in table.column_names
    note_col = table.column("note")
    assert note_col[0].as_py() is None


def test_write_rows_missing_non_nullable_column_rejected(store: DuckDBLogStore):
    store.register_table("iris.worker", _worker_schema())
    batch = pa.RecordBatch.from_pydict(
        {"worker_id": ["w-1"]},
        schema=pa.schema([pa.field("worker_id", pa.string(), nullable=False)]),
    )
    with pytest.raises(SchemaValidationError):
        store.write_rows("iris.worker", _ipc_bytes(batch))


def test_write_rows_unknown_column_rejected(store: DuckDBLogStore):
    store.register_table("iris.worker", _worker_schema())
    batch = pa.RecordBatch.from_pydict(
        {
            "worker_id": ["w-1"],
            "mem_bytes": [100],
            "timestamp_ms": [1],
            "rogue": ["x"],
        }
    )
    with pytest.raises(SchemaValidationError):
        store.write_rows("iris.worker", _ipc_bytes(batch))


def test_write_rows_type_mismatch_rejected(store: DuckDBLogStore):
    store.register_table("iris.worker", _worker_schema())
    batch = pa.RecordBatch.from_pydict(
        {
            "worker_id": ["w-1"],
            "mem_bytes": [1.5],  # float, not int64
            "timestamp_ms": [1],
        }
    )
    with pytest.raises(SchemaValidationError):
        store.write_rows("iris.worker", _ipc_bytes(batch))


def test_write_rows_dictionary_encoded_column_decoded(store: DuckDBLogStore):
    store.register_table("iris.worker", _worker_schema())
    # Dictionary-encoded string column.
    dict_arr = pa.DictionaryArray.from_arrays(pa.array([0, 1, 0]), pa.array(["w-1", "w-2"]))
    batch = pa.RecordBatch.from_arrays(
        [
            dict_arr,
            pa.array([100, 200, 300], type=pa.int64()),
            pa.array([1, 2, 3], type=pa.int64()),
        ],
        schema=pa.schema(
            [
                pa.field("worker_id", dict_arr.type, nullable=False),
                pa.field("mem_bytes", pa.int64(), nullable=False),
                pa.field("timestamp_ms", pa.int64(), nullable=False),
            ]
        ),
    )
    n = store.write_rows("iris.worker", _ipc_bytes(batch))
    assert n == 3
    store._namespaces["iris.worker"]._flush_step()
    parquet_files = sorted((store._data_dir / "iris.worker").glob("*.parquet"))
    table = pq.read_table(parquet_files[-1])
    # Decoded to plain string on disk.
    assert table.schema.field("worker_id").type == pa.string()
    assert table.column("worker_id").to_pylist() == ["w-1", "w-2", "w-1"]


def test_write_rows_nested_type_rejected(store: DuckDBLogStore):
    schema = Schema(
        columns=(
            Column(name="worker_id", type=ColumnType.STRING),
            Column(name="timestamp_ms", type=ColumnType.INT64),
        ),
    )
    store.register_table("iris.worker", schema)
    # Even if "tags" isn't in the registered schema, its type triggers the
    # nested-rejection rule before the unknown-column check.
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array(["w-1"]),
            pa.array([1], type=pa.int64()),
            pa.array([["a", "b"]]),
        ],
        names=["worker_id", "timestamp_ms", "tags"],
    )
    with pytest.raises(SchemaValidationError):
        store.write_rows("iris.worker", _ipc_bytes(batch))


def test_write_rows_oversize_request_rejected(store: DuckDBLogStore):
    store.register_table("iris.worker", _worker_schema())
    blob = b"\x00" * (MAX_WRITE_ROWS_BYTES + 1)
    with pytest.raises(SchemaValidationError):
        store.write_rows("iris.worker", blob)


def test_write_rows_too_many_rows_rejected(store: DuckDBLogStore):
    # We don't actually want to allocate 1M rows; instead write a small batch
    # whose row count we'll fudge by repeating. Use a moderately-large batch
    # that fits in 16 MiB but exceeds the 1M row cap when combined.
    schema = Schema(columns=(Column(name="timestamp_ms", type=ColumnType.INT64),))
    store.register_table("ns.bulk", schema)
    n = 1_000_001
    arr = pa.array([0] * n, type=pa.int64())
    batch = pa.RecordBatch.from_arrays([arr], names=["timestamp_ms"])
    with pytest.raises(SchemaValidationError):
        store.write_rows("ns.bulk", _ipc_bytes(batch))
