# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the log server ASGI wiring.

These tests exercise the real interceptor chain installed by
``build_log_server_asgi``, so they catch regressions in either the
interceptor itself or how it is wired into the FetchLogs endpoint.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from iris.log_server.main import build_log_server_asgi
from iris.log_server.server import LogServiceImpl


@pytest.fixture
def service(tmp_path: Path):
    svc = LogServiceImpl(log_dir=tmp_path, remote_log_dir="")
    try:
        yield svc
    finally:
        svc.close()


def test_fetch_logs_concurrency_cap_enforced_by_interceptor(service: LogServiceImpl):
    """Parallel FetchLogs calls never exceed the configured concurrency cap.

    Drives the real ASGI app so the ConcurrencyLimitInterceptor must be
    correctly installed on the FetchLogs endpoint for the cap to take hold.
    """
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
                "/iris.logging.LogService/FetchLogs",
                json={"source": "/does/not/matter"},
                headers={"Content-Type": "application/json"},
            )

        with ThreadPoolExecutor(max_workers=num_callers) as pool:
            futures = [pool.submit(call) for _ in range(num_callers)]

            # Wait for the handler to saturate.
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
