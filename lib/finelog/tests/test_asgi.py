# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Connect/ASGI tests for the StatsService: register+write,
query+drop, and the rejection/error cases for DropTable (log namespace
protected, unknown namespace returns 404). All tests drive the full HTTP
stack via Starlette's ``TestClient`` without mocking the store layer.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.server.asgi import build_log_server_asgi
from finelog.server.service import LogServiceImpl
from finelog.server.stats_service import StatsServiceImpl
from starlette.testclient import TestClient

from tests.conftest import _ipc_bytes, _ipc_to_table, _worker_batch

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


def test_query_and_drop_via_asgi(tmp_path: Path):
    """Full Connect round-trip for Query and DropTable."""
    log_service = LogServiceImpl(log_dir=tmp_path / "data")
    stats_service = StatsServiceImpl(log_store=log_service.log_store)
    app = build_log_server_asgi(log_service, stats_service=stats_service)

    try:
        with TestClient(app) as client:
            # Register + write enough data that a query returns rows.
            schema_msg = stats_pb2.Schema(
                columns=[
                    stats_pb2.Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
                    stats_pb2.Column(name="mem_bytes", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
                    stats_pb2.Column(name="timestamp_ms", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
                ],
            )
            resp = client.post(
                "/finelog.stats.StatsService/RegisterTable",
                content=stats_pb2.RegisterTableRequest(namespace="iris.worker", schema=schema_msg).SerializeToString(),
                headers={"Content-Type": "application/proto"},
            )
            assert resp.status_code == 200, resp.text

            batch = _worker_batch(["w-1", "w-2"], [100, 200], [1, 2])
            resp = client.post(
                "/finelog.stats.StatsService/WriteRows",
                content=stats_pb2.WriteRowsRequest(
                    namespace="iris.worker", arrow_ipc=_ipc_bytes(batch)
                ).SerializeToString(),
                headers={"Content-Type": "application/proto"},
            )
            assert resp.status_code == 200, resp.text

            # Force a flush + compaction so the data is in a sealed segment
            # (queries see only logs_*.parquet).
            ns = log_service.log_store._namespaces["iris.worker"]
            ns._flush_step()
            ns._compaction_step(compact_single=True)

            # Query.
            resp = client.post(
                "/finelog.stats.StatsService/Query",
                content=stats_pb2.QueryRequest(
                    sql='SELECT worker_id, mem_bytes FROM "iris.worker" ORDER BY worker_id'
                ).SerializeToString(),
                headers={"Content-Type": "application/proto"},
            )
            assert resp.status_code == 200, resp.text
            query_resp = stats_pb2.QueryResponse.FromString(resp.content)
            assert query_resp.row_count == 2
            result_table = _ipc_to_table(query_resp.arrow_ipc)
            assert result_table.column("worker_id").to_pylist() == ["w-1", "w-2"]
            assert result_table.column("mem_bytes").to_pylist() == [100, 200]

            # DropTable.
            resp = client.post(
                "/finelog.stats.StatsService/DropTable",
                content=stats_pb2.DropTableRequest(namespace="iris.worker").SerializeToString(),
                headers={"Content-Type": "application/proto"},
            )
            assert resp.status_code == 200, resp.text

            # Subsequent query against the dropped namespace fails.
            resp = client.post(
                "/finelog.stats.StatsService/Query",
                content=stats_pb2.QueryRequest(sql='SELECT * FROM "iris.worker"').SerializeToString(),
                headers={"Content-Type": "application/proto"},
            )
            assert resp.status_code == 400, resp.text
    finally:
        log_service.close()


def test_drop_table_log_namespace_rejected_via_asgi(tmp_path: Path):
    log_service = LogServiceImpl(log_dir=tmp_path / "data")
    stats_service = StatsServiceImpl(log_store=log_service.log_store)
    app = build_log_server_asgi(log_service, stats_service=stats_service)

    try:
        with TestClient(app) as client:
            resp = client.post(
                "/finelog.stats.StatsService/DropTable",
                content=stats_pb2.DropTableRequest(namespace="log").SerializeToString(),
                headers={"Content-Type": "application/proto"},
            )
            assert resp.status_code == 400, resp.text
    finally:
        log_service.close()


def test_drop_table_unknown_namespace_via_asgi(tmp_path: Path):
    log_service = LogServiceImpl(log_dir=tmp_path / "data")
    stats_service = StatsServiceImpl(log_store=log_service.log_store)
    app = build_log_server_asgi(log_service, stats_service=stats_service)

    try:
        with TestClient(app) as client:
            resp = client.post(
                "/finelog.stats.StatsService/DropTable",
                content=stats_pb2.DropTableRequest(namespace="nope.unknown").SerializeToString(),
                headers={"Content-Type": "application/proto"},
            )
            assert resp.status_code == 404, resp.text
    finally:
        log_service.close()
