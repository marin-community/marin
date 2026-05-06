# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from finelog.server.asgi import build_log_server_asgi
from finelog.server.service import LogServiceImpl
from finelog.server.stats_service import StatsServiceImpl
from finelog.store.duckdb_store import DuckDBLogStore
from starlette.testclient import TestClient


@pytest.fixture
def service(tmp_path: Path):
    svc = LogServiceImpl(log_dir=tmp_path, remote_log_dir="")
    try:
        yield svc
    finally:
        svc.close()


def test_fetch_logs_concurrency_cap_enforced_by_interceptor(service: LogServiceImpl):
    limit = 2
    release = threading.Event()
    in_flight = 0
    peak = 0
    lock = threading.Lock()

    original_fetch = service.fetch_logs

    def slow_fetch(request, ctx):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        try:
            assert release.wait(timeout=5.0), "handler never released"
            return original_fetch(request, ctx)
        finally:
            with lock:
                in_flight -= 1

    service.fetch_logs = slow_fetch  # type: ignore[method-assign]

    app = build_log_server_asgi(service, max_concurrent_fetch_logs=limit)
    num_callers = limit + 3

    with TestClient(app) as client:

        def call():
            return client.post(
                "/finelog.logging.LogService/FetchLogs",
                json={"source": "/does/not/matter"},
                headers={"Content-Type": "application/json"},
            )

        with ThreadPoolExecutor(max_workers=num_callers) as pool:
            futures = [pool.submit(call) for _ in range(num_callers)]

            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                with lock:
                    if in_flight >= limit:
                        break

            with lock:
                assert in_flight == limit, f"saturation never reached, in_flight={in_flight}"

            release.set()
            responses = [f.result(timeout=5.0) for f in futures]

    assert all(r.status_code == 200 for r in responses)
    assert peak == limit


def test_push_then_fetch_round_trip(tmp_path: Path):
    svc = LogServiceImpl(log_store=DuckDBLogStore(log_dir=tmp_path / "data"))
    try:
        app = build_log_server_asgi(svc)
        with TestClient(app) as client:
            push_resp = client.post(
                "/finelog.logging.LogService/PushLogs",
                json={
                    "key": "/job/test/0:0",
                    "entries": [
                        {"source": "stdout", "data": "hello", "timestamp": {"epoch_ms": 1}},
                        {"source": "stdout", "data": "world", "timestamp": {"epoch_ms": 2}},
                    ],
                },
                headers={"Content-Type": "application/json"},
            )
            assert push_resp.status_code == 200

            fetch_resp = client.post(
                "/finelog.logging.LogService/FetchLogs",
                json={"source": "/job/test/0:0"},
                headers={"Content-Type": "application/json"},
            )
        assert fetch_resp.status_code == 200
        body = fetch_resp.json()
        entries = body.get("entries", [])
        assert [e["data"] for e in entries] == ["hello", "world"]
    finally:
        svc.close()


def test_query_concurrency_cap_enforced_by_interceptor(tmp_path: Path):
    log_service = LogServiceImpl(log_store=DuckDBLogStore(log_dir=tmp_path / "data"))
    stats_service = StatsServiceImpl(log_store=log_service.log_store)
    limit = 2
    release = threading.Event()
    in_flight = 0
    peak = 0
    lock = threading.Lock()

    original_query = stats_service.query

    def slow_query(request, ctx):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        try:
            assert release.wait(timeout=5.0), "handler never released"
            return original_query(request, ctx)
        finally:
            with lock:
                in_flight -= 1

    stats_service.query = slow_query  # type: ignore[method-assign]
    try:
        app = build_log_server_asgi(
            log_service,
            stats_service=stats_service,
            max_concurrent_query=limit,
        )
        num_callers = limit + 3
        with TestClient(app) as client:

            def call():
                return client.post(
                    "/finelog.stats.StatsService/Query",
                    json={"sql": "SELECT 1 AS one"},
                    headers={"Content-Type": "application/json"},
                )

            with ThreadPoolExecutor(max_workers=num_callers) as pool:
                futures = [pool.submit(call) for _ in range(num_callers)]

                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    with lock:
                        if in_flight >= limit:
                            break

                with lock:
                    assert in_flight == limit, f"saturation never reached, in_flight={in_flight}"

                release.set()
                responses = [f.result(timeout=5.0) for f in futures]

        assert all(r.status_code == 200 for r in responses)
        assert peak == limit
    finally:
        log_service.close()


def test_legacy_iris_logging_path_compat():
    """Pre-#5212 workers send to /iris.logging.LogService/*; verify the
    server-side path rewrite still routes them. Removable once those
    workers have rotated out."""
    svc = LogServiceImpl(log_store=DuckDBLogStore())
    try:
        app = build_log_server_asgi(svc)
        with TestClient(app) as client:
            push_resp = client.post(
                "/iris.logging.LogService/PushLogs",
                json={
                    "key": "/legacy/probe",
                    "entries": [{"source": "stdout", "data": "old-worker", "timestamp": {"epoch_ms": 1}}],
                },
                headers={"Content-Type": "application/json"},
            )
            assert push_resp.status_code == 200

            fetch_resp = client.post(
                "/iris.logging.LogService/FetchLogs",
                json={"source": "/legacy/probe"},
                headers={"Content-Type": "application/json"},
            )
            assert fetch_resp.status_code == 200
            assert [e["data"] for e in fetch_resp.json().get("entries", [])] == ["old-worker"]
    finally:
        svc.close()
