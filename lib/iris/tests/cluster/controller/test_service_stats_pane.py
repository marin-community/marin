# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the stats-service worker pane.

``ControllerServiceImpl.list_workers`` reads unconditionally from the
``iris.worker`` finelog stats namespace. Stats are observation-only, so a
transport-level outage soft-fails to an empty roster; schema-mapping bugs
propagate.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pyarrow as pa
import pytest
from iris.cluster.bundle import BundleStore
from iris.cluster.controller.service import ControllerServiceImpl
from iris.rpc import controller_pb2
from rigging.timing import Timestamp

from tests.cluster.controller.conftest import (
    MockController,
    make_controller_state,
    make_worker_metadata,
)


class _FakeStatsTable:
    """Stand-in for ``finelog.client.Table`` exposing only ``query``."""

    def __init__(self, *, query_result=None, query_error: BaseException | None = None) -> None:
        self.query_result = query_result if query_result is not None else pa.table({})
        self.query_error = query_error
        self.queries: list[str] = []

    def query(self, sql: str, *, max_rows: int = 100_000):
        self.queries.append(sql)
        if self.query_error is not None:
            raise self.query_error
        return self.query_result


class _FakeLogClient:
    def __init__(self, table: _FakeStatsTable) -> None:
        self._table = table
        self.calls: list[tuple[str, type]] = []

    def get_table(self, namespace: str, schema):
        self.calls.append((namespace, schema))
        return self._table


def _build_service(*, stats_log_client=None, tmp_path):
    state_ctx = make_controller_state()
    state = state_ctx.__enter__()
    bundle = BundleStore(storage_dir=str(tmp_path / "bundles"))
    service = ControllerServiceImpl(
        state,
        state._store,
        controller=MockController(),
        bundle_store=bundle,
        log_service=None,
        stats_log_client=stats_log_client,
    )
    return service, state, state_ctx


def _stats_table_with_one_worker(worker_id: str = "w-1"):
    return pa.table(
        {
            "worker_id": [worker_id],
            "ts": [datetime.now(tz=timezone.utc)],
            "status": ["RUNNING"],
            "address": [f"{worker_id}:8080"],
            "healthy": [True],
            "cpu_pct": [12.5],
            "mem_bytes": [1024],
            "mem_total_bytes": [4096],
            "disk_used_bytes": [10],
            "disk_total_bytes": [100],
            "running_task_count": [2],
            "total_process_count": [10],
            "device_type": ["cpu"],
            "device_variant": ["cpu"],
            "cpu_count": [8],
            "memory_bytes": [16 * 1024**3],
            "tpu_name": [""],
            "gce_instance_name": ["instance-1"],
            "zone": ["us-central1-a"],
        }
    )


def _register_one(state, worker_id: str = "w-1"):
    from iris.cluster.types import WorkerId

    with state._store.transaction() as cur:
        state.register_or_refresh_worker(
            cur,
            worker_id=WorkerId(worker_id),
            address=f"{worker_id}:8080",
            metadata=make_worker_metadata(),
            ts=Timestamp.now(),
        )


def test_reads_from_stats_namespace(tmp_path):
    table = _FakeStatsTable(query_result=_stats_table_with_one_worker("w-from-stats"))
    log_client = _FakeLogClient(table)
    service, state, ctx = _build_service(stats_log_client=log_client, tmp_path=tmp_path)
    try:
        _register_one(state, "w-from-sqlite")
        resp = service.list_workers(controller_pb2.Controller.ListWorkersRequest(), None)
        ids = [w.worker_id for w in resp.workers]
        assert ids == ["w-from-stats"]
        assert table.queries, "stats query must have run"
        assert len(resp.workers[0].running_job_ids) == 2
        assert resp.workers[0].metadata.cpu_count == 8
        # Healthy worker: status_message stays empty (the dashboard banner
        # reads off this; RUNNING / IDLE must NOT bleed into status_message).
        assert resp.workers[0].status_message == ""
    finally:
        ctx.__exit__(None, None, None)


def test_status_message_for_unhealthy_worker(tmp_path):
    """Unhealthy stats row produces ``Unhealthy (last seen Ns ago)`` — exact
    mirror of the previous sqlite path so the dashboard banner reads the same.
    """
    now_naive = datetime.utcnow().replace(microsecond=0)
    five_s_ago = now_naive.fromtimestamp(now_naive.timestamp() - 5)
    pa_table = _stats_table_with_one_worker("w-bad")
    pa_table = pa_table.set_column(pa_table.schema.get_field_index("healthy"), "healthy", pa.array([False]))
    pa_table = pa_table.set_column(
        pa_table.schema.get_field_index("ts"), "ts", pa.array([five_s_ago], type=pa.timestamp("ms"))
    )
    table = _FakeStatsTable(query_result=pa_table)
    log_client = _FakeLogClient(table)
    service, _state, ctx = _build_service(stats_log_client=log_client, tmp_path=tmp_path)
    try:
        resp = service.list_workers(controller_pb2.Controller.ListWorkersRequest(), None)
        assert len(resp.workers) == 1
        msg = resp.workers[0].status_message
        assert msg.startswith("Unhealthy (last seen ")
        assert msg.endswith("s ago)")
    finally:
        ctx.__exit__(None, None, None)


def test_returns_empty_when_stats_query_raises(tmp_path):
    table = _FakeStatsTable(query_error=ConnectionError("finelog unreachable"))
    log_client = _FakeLogClient(table)
    service, state, ctx = _build_service(stats_log_client=log_client, tmp_path=tmp_path)
    try:
        _register_one(state, "still-here")
        resp = service.list_workers(controller_pb2.Controller.ListWorkersRequest(), None)
        assert resp.workers == []
    finally:
        ctx.__exit__(None, None, None)


def test_returns_empty_when_query_too_large(tmp_path, caplog):
    """``QueryResultTooLargeError`` is a ``StatsError`` subclass — soft-fail
    catches it rather than letting it crash the dashboard.
    """
    from finelog.client import QueryResultTooLargeError

    table = _FakeStatsTable(query_error=QueryResultTooLargeError("got 200_000 rows, max 100_000"))
    log_client = _FakeLogClient(table)
    service, state, ctx = _build_service(stats_log_client=log_client, tmp_path=tmp_path)
    try:
        _register_one(state, "alive")
        with caplog.at_level(logging.WARNING, logger="iris.cluster.controller.service"):
            resp = service.list_workers(controller_pb2.Controller.ListWorkersRequest(), None)
        assert resp.workers == []
        assert any("stats pane query failed" in rec.message for rec in caplog.records)
    finally:
        ctx.__exit__(None, None, None)


def test_propagates_schema_mapping_bug(tmp_path):
    """A KeyError out of the response-mapping path is a schema-mapping bug
    and must propagate so operators see it; the soft-fail catch must not
    swallow programming errors.
    """
    table = _FakeStatsTable(query_error=KeyError("missing column 'worker_id'"))
    log_client = _FakeLogClient(table)
    service, state, ctx = _build_service(stats_log_client=log_client, tmp_path=tmp_path)
    try:
        _register_one(state, "alive")
        with pytest.raises(KeyError):
            service.list_workers(controller_pb2.Controller.ListWorkersRequest(), None)
    finally:
        ctx.__exit__(None, None, None)


def test_no_log_client_returns_empty(tmp_path):
    """No stats client wired (e.g. a misconfigured controller) ⇒ empty
    roster. The dashboard's empty-state is the right rendering."""
    service, state, ctx = _build_service(stats_log_client=None, tmp_path=tmp_path)
    try:
        _register_one(state, "lonely")
        resp = service.list_workers(controller_pb2.Controller.ListWorkersRequest(), None)
        assert resp.workers == []
    finally:
        ctx.__exit__(None, None, None)
