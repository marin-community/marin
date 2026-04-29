# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stats service tests: schema registry + Arrow-IPC WriteRows.

Covers the schema-registry side (register, evolve, conflict detection) and
the Arrow-IPC WriteRows path (validation, alignment, dictionary decode,
nested-type rejection, oversize rejection). Also tests:

- Compaction across an additive evolution: register (a, b) → write → evolve
  to (a, b, c) → write more → trigger compaction → result has the full
  registered column set in registered order with NULLs for missing-c rows.
- Rehydration of namespaces from the sidecar registry across store restarts.
- Log namespace parity: the existing log RPC path still works alongside
  the stats RPCs.
"""

from __future__ import annotations

import io
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as paipc
import pyarrow.parquet as pq
import pytest

from starlette.testclient import TestClient

from finelog.rpc import logging_pb2
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.server.asgi import build_log_server_asgi
from finelog.server.service import LogServiceImpl
from finelog.server.stats_service import StatsServiceImpl
from finelog.store.duckdb_store import DuckDBLogStore
from finelog.store.schema import (
    MAX_WRITE_ROWS_BYTES,
    Column,
    ColumnType,
    InvalidNamespaceError,
    NamespaceNotFoundError,
    Schema,
    SchemaConflictError,
    SchemaValidationError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ipc_bytes(batch: pa.RecordBatch) -> bytes:
    """Serialize a single RecordBatch as an Arrow IPC stream."""
    sink = io.BytesIO()
    with paipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue()


def _worker_schema() -> Schema:
    return Schema(
        columns=(
            Column(name="worker_id", type=ColumnType.STRING, nullable=False),
            Column(name="mem_bytes", type=ColumnType.INT64, nullable=False),
            Column(name="timestamp_ms", type=ColumnType.INT64, nullable=False),
        ),
        key_column="",
    )


def _make_store(tmp_path: Path, **kwargs) -> DuckDBLogStore:
    return DuckDBLogStore(log_dir=tmp_path / "data", **kwargs)


@pytest.fixture()
def store(tmp_path: Path):
    s = _make_store(tmp_path)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# RegisterTable: name validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "iris.worker",
        "iris.worker.v2",
        "a",
        "a-b",
        "abc.def_ghi",
        "x" * 64,
    ],
)
def test_register_accepts_valid_names(store: DuckDBLogStore, name: str):
    schema = _worker_schema()
    effective = store.register_table(name, schema)
    assert effective == schema


@pytest.mark.parametrize(
    "name",
    [
        "",
        "Iris.Worker",  # uppercase
        ".starts-dot",
        "1starts-digit",
        "x" * 65,
        "has space",
        "has/slash",
        "..",
    ],
)
def test_register_rejects_invalid_names(store: DuckDBLogStore, name: str):
    with pytest.raises(InvalidNamespaceError):
        store.register_table(name, _worker_schema())


def test_register_rejects_path_traversal(store: DuckDBLogStore):
    # The regex check filters these too, but we confirm path containment is
    # exercised by names containing dots.
    with pytest.raises(InvalidNamespaceError):
        store.register_table("../escape", _worker_schema())


# ---------------------------------------------------------------------------
# RegisterTable: schema validation (ordering key)
# ---------------------------------------------------------------------------


def test_register_rejects_schema_without_ordering_key(store: DuckDBLogStore):
    # No key_column and no implicit timestamp_ms column.
    schema = Schema(
        columns=(
            Column(name="worker_id", type=ColumnType.STRING),
            Column(name="mem_bytes", type=ColumnType.INT64),
        ),
        key_column="",
    )
    with pytest.raises(SchemaValidationError):
        store.register_table("iris.worker", schema)


def test_register_accepts_implicit_timestamp_ms_int64(store: DuckDBLogStore):
    schema = Schema(
        columns=(
            Column(name="worker_id", type=ColumnType.STRING),
            Column(name="timestamp_ms", type=ColumnType.INT64),
        ),
        key_column="",
    )
    store.register_table("iris.worker", schema)


def test_register_accepts_implicit_timestamp_ms_timestamp(store: DuckDBLogStore):
    schema = Schema(
        columns=(
            Column(name="worker_id", type=ColumnType.STRING),
            Column(name="timestamp_ms", type=ColumnType.TIMESTAMP_MS),
        ),
        key_column="",
    )
    store.register_table("iris.worker", schema)


def test_register_accepts_explicit_key_column(store: DuckDBLogStore):
    schema = Schema(
        columns=(
            Column(name="worker_id", type=ColumnType.STRING),
            Column(name="ts", type=ColumnType.TIMESTAMP_MS),
        ),
        key_column="ts",
    )
    store.register_table("iris.worker", schema)


def test_register_rejects_explicit_key_missing_from_columns(store: DuckDBLogStore):
    schema = Schema(
        columns=(Column(name="worker_id", type=ColumnType.STRING),),
        key_column="ts",  # not in columns
    )
    with pytest.raises(SchemaValidationError):
        store.register_table("iris.worker", schema)


def test_register_rejects_explicit_key_wrong_type(store: DuckDBLogStore):
    schema = Schema(
        columns=(
            Column(name="worker_id", type=ColumnType.STRING),
            Column(name="ts", type=ColumnType.STRING),  # wrong type for key
        ),
        key_column="ts",
    )
    with pytest.raises(SchemaValidationError):
        store.register_table("iris.worker", schema)


# ---------------------------------------------------------------------------
# RegisterTable: evolve-by-default
# ---------------------------------------------------------------------------


def test_register_idempotent_returns_existing_schema(store: DuckDBLogStore):
    schema = _worker_schema()
    first = store.register_table("iris.worker", schema)
    second = store.register_table("iris.worker", schema)
    assert first == second == schema


def test_register_subset_returns_full_registered_schema(store: DuckDBLogStore):
    full = Schema(
        columns=(
            Column(name="worker_id", type=ColumnType.STRING),
            Column(name="mem_bytes", type=ColumnType.INT64),
            Column(name="cpu_pct", type=ColumnType.FLOAT64, nullable=True),
            Column(name="timestamp_ms", type=ColumnType.INT64),
        ),
    )
    store.register_table("iris.worker", full)
    subset = Schema(
        columns=(
            Column(name="worker_id", type=ColumnType.STRING),
            Column(name="timestamp_ms", type=ColumnType.INT64),
        ),
    )
    effective = store.register_table("iris.worker", subset)
    assert effective == full


def test_register_additive_nullable_extension_merges(store: DuckDBLogStore):
    base = _worker_schema()
    store.register_table("iris.worker", base)
    extended = Schema(
        columns=(*base.columns, Column(name="note", type=ColumnType.STRING, nullable=True)),
    )
    effective = store.register_table("iris.worker", extended)
    assert effective.column_names() == ("worker_id", "mem_bytes", "timestamp_ms", "note")
    # Re-registering the base schema after evolution returns the merged schema.
    again = store.register_table("iris.worker", base)
    assert again == effective


def test_register_non_additive_new_non_nullable_rejects(store: DuckDBLogStore):
    base = _worker_schema()
    store.register_table("iris.worker", base)
    bad = Schema(
        columns=(*base.columns, Column(name="cpu_pct", type=ColumnType.FLOAT64, nullable=False)),
    )
    with pytest.raises(SchemaConflictError):
        store.register_table("iris.worker", bad)


def test_register_type_change_rejects(store: DuckDBLogStore):
    base = _worker_schema()
    store.register_table("iris.worker", base)
    bad = Schema(
        columns=(
            Column(name="worker_id", type=ColumnType.STRING),
            Column(name="mem_bytes", type=ColumnType.FLOAT64),  # was INT64
            Column(name="timestamp_ms", type=ColumnType.INT64),
        ),
    )
    with pytest.raises(SchemaConflictError):
        store.register_table("iris.worker", bad)


def test_register_key_column_change_rejects(store: DuckDBLogStore):
    base = _worker_schema()
    store.register_table("iris.worker", base)
    bad = Schema(columns=base.columns, key_column="timestamp_ms")  # was empty
    with pytest.raises(SchemaConflictError):
        store.register_table("iris.worker", bad)


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


# ---------------------------------------------------------------------------
# Compaction across additive evolution
# ---------------------------------------------------------------------------


def test_compaction_across_additive_evolution(tmp_path: Path):
    # Start with (a, b), write a row, evolve to (a, b, c), write another row,
    # trigger compaction. Result has all three columns in registered order
    # with NULL for the column that wasn't yet declared at first write.
    store = _make_store(tmp_path)
    try:
        s1 = Schema(
            columns=(
                Column(name="a", type=ColumnType.STRING),
                Column(name="b", type=ColumnType.INT64),
                Column(name="timestamp_ms", type=ColumnType.INT64),
            ),
        )
        store.register_table("ns.evolve", s1)
        batch1 = pa.RecordBatch.from_pydict(
            {"a": ["x"], "b": [1], "timestamp_ms": [10]},
            schema=pa.schema(
                [
                    pa.field("a", pa.string(), nullable=False),
                    pa.field("b", pa.int64(), nullable=False),
                    pa.field("timestamp_ms", pa.int64(), nullable=False),
                ]
            ),
        )
        store.write_rows("ns.evolve", _ipc_bytes(batch1))
        ns = store._namespaces["ns.evolve"]
        ns._flush_step()  # write tmp_001 with (a, b, timestamp_ms)

        # Evolve schema to add nullable c.
        s2 = Schema(
            columns=(*s1.columns, Column(name="c", type=ColumnType.FLOAT64, nullable=True)),
        )
        store.register_table("ns.evolve", s2)
        batch2 = pa.RecordBatch.from_pydict(
            {"a": ["y"], "b": [2], "c": [2.5], "timestamp_ms": [20]},
            schema=pa.schema(
                [
                    pa.field("a", pa.string(), nullable=False),
                    pa.field("b", pa.int64(), nullable=False),
                    pa.field("c", pa.float64(), nullable=True),
                    pa.field("timestamp_ms", pa.int64(), nullable=False),
                ]
            ),
        )
        store.write_rows("ns.evolve", _ipc_bytes(batch2))
        ns._flush_step()  # write tmp_002 with all four columns

        # Trigger compaction.
        ns._compaction_step()
        seg_dir = tmp_path / "data" / "ns.evolve"
        assert sorted(p.name for p in seg_dir.glob("tmp_*.parquet")) == []
        log_files = sorted(seg_dir.glob("logs_*.parquet"))
        assert len(log_files) == 1
        table = pq.read_table(log_files[0])
        # Registered order: a, b, timestamp_ms, c.
        assert table.column_names == ["a", "b", "timestamp_ms", "c"]
        # First row (from pre-evolution segment) has c = NULL.
        rows = table.to_pylist()
        rows.sort(key=lambda r: r["timestamp_ms"])
        assert rows[0] == {"a": "x", "b": 1, "timestamp_ms": 10, "c": None}
        assert rows[1] == {"a": "y", "b": 2, "timestamp_ms": 20, "c": 2.5}
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Registry persistence and rehydration across restarts
# ---------------------------------------------------------------------------


def test_registry_survives_restart(tmp_path: Path):
    s1 = _make_store(tmp_path)
    schema = _worker_schema()
    s1.register_table("iris.worker", schema)
    batch = pa.RecordBatch.from_pydict(
        {"worker_id": ["w-1"], "mem_bytes": [100], "timestamp_ms": [1]},
        schema=pa.schema(
            [
                pa.field("worker_id", pa.string(), nullable=False),
                pa.field("mem_bytes", pa.int64(), nullable=False),
                pa.field("timestamp_ms", pa.int64(), nullable=False),
            ]
        ),
    )
    s1.write_rows("iris.worker", _ipc_bytes(batch))
    s1.close()

    s2 = _make_store(tmp_path)
    try:
        # Re-register with the same schema is idempotent. Returns the
        # registered schema, which means rehydration worked.
        effective = s2.register_table("iris.worker", schema)
        assert effective == schema
        # And writes continue to work without re-registering.
        s2.write_rows("iris.worker", _ipc_bytes(batch))
    finally:
        s2.close()


# ---------------------------------------------------------------------------
# Log namespace parity
# ---------------------------------------------------------------------------


def _entry(data: str, epoch_ms: int) -> logging_pb2.LogEntry:
    e = logging_pb2.LogEntry(source="stdout", data=data)
    e.timestamp.epoch_ms = epoch_ms
    return e


def test_log_namespace_round_trip_after_stage2(tmp_path: Path):
    store = _make_store(tmp_path)
    try:
        store.append("/job/test/0:0", [_entry(f"line-{i}", epoch_ms=i) for i in range(5)])
        # Drain pending to chunks so the read sees the data without waiting
        # on the bg flush thread (parity with the existing log-store tests).
        store._compact_step()
        result = store.get_logs("/job/test/0:0")
        assert [e.data for e in result.entries] == [f"line-{i}" for i in range(5)]
    finally:
        store.close()


def test_log_namespace_eagerly_registered(tmp_path: Path):
    store = _make_store(tmp_path)
    try:
        # The log namespace exists before any explicit register_table call.
        assert "log" in store._namespaces
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Connect/RPC end-to-end via the StatsService ASGI app
# ---------------------------------------------------------------------------


def test_stats_service_register_and_write_via_asgi(tmp_path: Path):
    """End-to-end: register a table and write rows via the Connect RPCs."""
    log_service = LogServiceImpl(log_dir=tmp_path / "data")
    stats_service = StatsServiceImpl(log_store=log_service.log_store)
    app = build_log_server_asgi(log_service, stats_service=stats_service)

    try:
        with TestClient(app) as client:
            schema_msg = stats_pb2.Schema(
                columns=[
                    stats_pb2.Column(
                        name="worker_id",
                        type=stats_pb2.COLUMN_TYPE_STRING,
                        nullable=False,
                    ),
                    stats_pb2.Column(
                        name="mem_bytes",
                        type=stats_pb2.COLUMN_TYPE_INT64,
                        nullable=False,
                    ),
                    stats_pb2.Column(
                        name="timestamp_ms",
                        type=stats_pb2.COLUMN_TYPE_INT64,
                        nullable=False,
                    ),
                ],
            )
            req = stats_pb2.RegisterTableRequest(namespace="iris.worker", schema=schema_msg)
            resp = client.post(
                "/finelog.stats.StatsService/RegisterTable",
                content=req.SerializeToString(),
                headers={"Content-Type": "application/proto"},
            )
            assert resp.status_code == 200, resp.text
            register_resp = stats_pb2.RegisterTableResponse.FromString(resp.content)
            assert [c.name for c in register_resp.effective_schema.columns] == [
                "worker_id",
                "mem_bytes",
                "timestamp_ms",
            ]

            # Write a batch.
            batch = pa.RecordBatch.from_pydict(
                {"worker_id": ["w-1"], "mem_bytes": [100], "timestamp_ms": [1]},
                schema=pa.schema(
                    [
                        pa.field("worker_id", pa.string(), nullable=False),
                        pa.field("mem_bytes", pa.int64(), nullable=False),
                        pa.field("timestamp_ms", pa.int64(), nullable=False),
                    ]
                ),
            )
            write_req = stats_pb2.WriteRowsRequest(namespace="iris.worker", arrow_ipc=_ipc_bytes(batch))
            resp = client.post(
                "/finelog.stats.StatsService/WriteRows",
                content=write_req.SerializeToString(),
                headers={"Content-Type": "application/proto"},
            )
            assert resp.status_code == 200, resp.text
            write_resp = stats_pb2.WriteRowsResponse.FromString(resp.content)
            assert write_resp.rows_written == 1
    finally:
        log_service.close()
