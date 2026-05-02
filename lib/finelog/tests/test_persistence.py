# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for persistence and rehydration: compaction across additive schema
evolution (NULL backfill for pre-evolution segments), registry survival across
store restarts, log-namespace round-trip after stage-2 compaction, and eager
registration of the log namespace on store open.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from finelog.rpc import logging_pb2
from finelog.store.duckdb_store import DuckDBLogStore
from finelog.store.schema import Column, ColumnType, Schema

from tests.conftest import _ipc_bytes, _worker_schema

# ---------------------------------------------------------------------------
# Compaction across additive evolution
# ---------------------------------------------------------------------------


def test_compaction_across_additive_evolution(tmp_path: Path):
    # Start with (a, b), write a row, evolve to (a, b, c), write another row,
    # trigger compaction. Result has all three columns in registered order
    # with NULL for the column that wasn't yet declared at first write.
    store = DuckDBLogStore(log_dir=tmp_path / "data")
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
    s1 = DuckDBLogStore(log_dir=tmp_path / "data")
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

    s2 = DuckDBLogStore(log_dir=tmp_path / "data")
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
    store = DuckDBLogStore(log_dir=tmp_path / "data")
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
    store = DuckDBLogStore(log_dir=tmp_path / "data")
    try:
        # The log namespace exists before any explicit register_table call.
        assert "log" in store._namespaces
    finally:
        store.close()
