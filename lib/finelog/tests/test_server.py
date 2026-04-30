# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the finelog server ASGI wiring."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from finelog.server.asgi import build_log_server_asgi
from finelog.server.service import LogServiceImpl


@pytest.fixture
def service(tmp_path: Path):
    svc = LogServiceImpl(log_dir=tmp_path, remote_log_dir="")
    try:
        yield svc
    finally:
        svc.close()


def test_fetch_logs_concurrency_cap_enforced_by_interceptor(service: LogServiceImpl):
    """Parallel FetchLogs calls never exceed the configured concurrency cap."""
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
    """End-to-end: push entries via RPC, then fetch them back.

    Uses an in-memory MemStore directly so the round trip is synchronous —
    the LogStore default factory may resolve to DuckDB if the env probe
    happens before PYTEST_CURRENT_TEST is set (xdist worker startup).
    """
    from finelog.store.mem_store import MemStore

    svc = LogServiceImpl(log_store=MemStore())
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


def test_legacy_iris_logging_path_compat():
    """Pre-#5212 workers send to /iris.logging.LogService/* (the package was
    renamed when finelog was extracted from iris). The wire format is
    identical between iris.logging and finelog.logging — same field numbers
    on PushLogsRequest/LogEntry/etc — so a server-side path rewrite restores
    delivery without any worker-side change. Removable once those workers
    have rotated out.
    """
    from finelog.store.mem_store import MemStore

    svc = LogServiceImpl(log_store=MemStore())
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
