# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.rpc import logging_pb2
from finelog.store.duckdb_store import DuckDBLogStore
from finelog.store.schema import Column, Schema, with_implicit_seq

from tests.conftest import _ipc_bytes, _worker_schema


def test_compaction_across_additive_evolution(tmp_path: Path):
    """Evolve a schema between flushes; compaction backfills NULL for the
    new column on pre-evolution rows."""
    store = DuckDBLogStore(log_dir=tmp_path / "data")
    try:
        s1 = Schema(
            columns=(
                Column(name="a", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
                Column(name="b", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
                Column(name="timestamp_ms", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
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
        ns._flush_step()

        s2 = Schema(
            columns=(*s1.columns, Column(name="c", type=stats_pb2.COLUMN_TYPE_FLOAT64, nullable=True)),
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
        ns._flush_step()

        ns._force_compact_l0()
        seg_dir = tmp_path / "data" / "ns.evolve"
        assert sorted(p.name for p in seg_dir.glob("seg_L0_*.parquet")) == []
        l1_files = sorted(seg_dir.glob("seg_L1_*.parquet"))
        assert len(l1_files) == 1
        table = pq.read_table(l1_files[0])
        # Implicit ``seq`` first, then registered columns, additive ``c`` last.
        assert table.column_names == ["seq", "a", "b", "timestamp_ms", "c"]
        rows = table.to_pylist()
        rows.sort(key=lambda r: r["timestamp_ms"])
        assert rows[0] == {"seq": 1, "a": "x", "b": 1, "timestamp_ms": 10, "c": None}
        assert rows[1] == {"seq": 2, "a": "y", "b": 2, "timestamp_ms": 20, "c": 2.5}
    finally:
        store.close()


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
        effective = s2.register_table("iris.worker", schema)
        assert effective == with_implicit_seq(schema)
        s2.write_rows("iris.worker", _ipc_bytes(batch))
    finally:
        s2.close()


def _entry(data: str, epoch_ms: int) -> logging_pb2.LogEntry:
    e = logging_pb2.LogEntry(source="stdout", data=data)
    e.timestamp.epoch_ms = epoch_ms
    return e


def test_log_namespace_round_trip_after_stage2(tmp_path: Path):
    store = DuckDBLogStore(log_dir=tmp_path / "data")
    try:
        store.append("/job/test/0:0", [_entry(f"line-{i}", epoch_ms=i) for i in range(5)])
        result = store.get_logs("/job/test/0:0")
        assert [e.data for e in result.entries] == [f"line-{i}" for i in range(5)]
    finally:
        store.close()


def test_log_namespace_eagerly_registered(tmp_path: Path):
    store = DuckDBLogStore(log_dir=tmp_path / "data")
    try:
        assert "log" in store._namespaces
    finally:
        store.close()
