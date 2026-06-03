# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase-5 hardened-server parity tests.

Exercise the hardened server stack (raised limits, zstd/gzip, SlowRpc +
Concurrency interceptors, legacy-path transport rewrite, graceful shutdown) over
real HTTP/RPC against BOTH backends. The seam is the RPC socket; assertions are
on structured RPC responses / process exit codes / returned entry payloads —
never on log strings (the slow-RPC WARN and the diagnostics line are diagnostic
aids covered only by Rust unit tests, per the AGENTS.md anti-slop rule).

The exact concurrency-cap "peak in-flight == cap" assertion is a CARGO unit test
(``server::interceptors::concurrency_caps_in_flight``, run_chain + a parked
terminal, no wall-clock sleep). Asserting an exact peak over a real socket would
need a server-side cap-lowering + handler-parking knob that neither backend
exposes without a behavior change; the RPC-level test here instead verifies the
backend-agnostic invariant the semaphore guarantees — N > cap concurrent reads
all complete (the cap queues, never drops or deadlocks).
"""

from __future__ import annotations

import io
import signal
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.ipc as paipc
import pytest
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.rpc import logging_pb2
from finelog.rpc.finelog_stats_connect import StatsServiceClientSync
from finelog.rpc.logging_connect import LogServiceClientSync

from tests.parity.conftest import Backend

pytestmark = pytest.mark.timeout(60)


# ---------------------------------------------------------------------------
# Wire helpers.
# ---------------------------------------------------------------------------


def _log_client(url: str) -> LogServiceClientSync:
    return LogServiceClientSync(address=url)


def _stats_client(url: str) -> StatsServiceClientSync:
    return StatsServiceClientSync(address=url)


def _entry(data: str, source: str = "stdout", epoch_ms: int = 0) -> logging_pb2.LogEntry:
    return logging_pb2.LogEntry(
        source=source,
        data=data,
        timestamp=logging_pb2.Timestamp(epoch_ms=epoch_ms),
        level=logging_pb2.LOG_LEVEL_INFO,
    )


# ---------------------------------------------------------------------------
# 5a: raised message/body limits.
# ---------------------------------------------------------------------------


def test_large_write_rows_within_limit(client, server_backend: Backend) -> None:
    """A WriteRows whose arrow_ipc is >4MB but <16MB succeeds, proving build_app
    raised the connect limit above the 4MB default (a >4MB write would
    ResourceExhaust at the transport layer without 5a's Limits)."""
    stats = _stats_client(client._server_url)
    schema = pa.schema(
        [
            pa.field("worker_id", pa.string(), nullable=False),
            pa.field("payload", pa.string(), nullable=False),
            pa.field("timestamp_ms", pa.int64(), nullable=False),
        ]
    )
    stats.register_table(
        stats_pb2.RegisterTableRequest(
            namespace="iris.big",
            schema=stats_pb2.Schema(
                columns=[
                    stats_pb2.Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
                    stats_pb2.Column(name="payload", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
                    stats_pb2.Column(name="timestamp_ms", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
                ],
            ),
        )
    )
    # ~6 MB of payload: 6000 rows x ~1KB string each.
    n = 6000
    blob = "x" * 1024
    batch = pa.RecordBatch.from_pydict(
        {
            "worker_id": [f"w{i}" for i in range(n)],
            "payload": [blob] * n,
            "timestamp_ms": [1000 + i for i in range(n)],
        },
        schema=schema,
    )
    sink = io.BytesIO()
    with paipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    ipc = sink.getvalue()
    assert 4 * 1024 * 1024 < len(ipc) < 16 * 1024 * 1024, f"ipc size {len(ipc)} out of test range"

    resp = stats.write_rows(stats_pb2.WriteRowsRequest(namespace="iris.big", arrow_ipc=ipc))
    assert resp.rows_written == n


# ---------------------------------------------------------------------------
# 5c: concurrency cap (backend-agnostic; the exact-cap proof is a cargo test).
# ---------------------------------------------------------------------------


def test_fetch_logs_concurrency_does_not_drop_requests(finelog_url: str, server_backend: Backend) -> None:
    """N > the FetchLogs cap (4) concurrent reads all complete: the semaphore
    queues callers, never drops or deadlocks them."""
    client = _log_client(finelog_url)
    # Seed one log line so FetchLogs has something to scan.
    client.push_logs(logging_pb2.PushLogsRequest(key="/c/probe", entries=[_entry("hi")]))

    num_callers = 4 + 6  # comfortably over the cap of 4

    def call() -> int:
        resp = client.fetch_logs(logging_pb2.FetchLogsRequest(source="/c/probe"))
        return len(resp.entries)

    with ThreadPoolExecutor(max_workers=num_callers) as pool:
        results = [f.result(timeout=30.0) for f in [pool.submit(call) for _ in range(num_callers)]]
    assert all(r == 1 for r in results), results


def test_query_concurrency_does_not_drop_requests(client, server_backend: Backend) -> None:
    """N > the Query cap (4) concurrent queries all complete."""
    stats = _stats_client(client._server_url)
    num_callers = 4 + 6

    def call() -> int:
        resp = stats.query(stats_pb2.QueryRequest(sql="SELECT 1 AS n"))
        return resp.row_count

    with ThreadPoolExecutor(max_workers=num_callers) as pool:
        results = [f.result(timeout=30.0) for f in [pool.submit(call) for _ in range(num_callers)]]
    assert all(r == 1 for r in results), results


# ---------------------------------------------------------------------------
# Push -> Fetch round-trip + UNSPECIFIED->REGEX through the hardened stack.
# ---------------------------------------------------------------------------


def test_push_then_fetch_round_trip(finelog_url: str, server_backend: Backend) -> None:
    client = _log_client(finelog_url)
    client.push_logs(
        logging_pb2.PushLogsRequest(
            key="/job/test/0:0",
            entries=[_entry("hello", epoch_ms=1), _entry("world", epoch_ms=2)],
        )
    )
    resp = client.fetch_logs(
        logging_pb2.FetchLogsRequest(source="/job/test/0:0", match_scope=logging_pb2.MATCH_SCOPE_EXACT)
    )
    assert [e.data for e in resp.entries] == ["hello", "world"]


def test_fetch_logs_unspecified_reads_as_regex(finelog_url: str, server_backend: Backend) -> None:
    client = _log_client(finelog_url)
    for attempt in range(2):
        client.push_logs(
            logging_pb2.PushLogsRequest(
                key=f"/job/test/0:{attempt}", entries=[_entry(f"a{attempt}", epoch_ms=attempt + 1)]
            )
        )
    # match_scope unset (UNSPECIFIED) falls back to REGEX, so the regex source
    # pattern matches both keys.
    resp = client.fetch_logs(logging_pb2.FetchLogsRequest(source=r"/job/test/0:\d+"))
    assert sorted(e.data for e in resp.entries) == ["a0", "a1"]


# ---------------------------------------------------------------------------
# 5e: legacy /iris.logging.LogService/* path rewrite.
# ---------------------------------------------------------------------------


def test_legacy_iris_logging_path_compat(finelog_url: str, server_backend: Backend) -> None:
    """A push+fetch over the legacy /iris.logging.LogService/ prefix round-trips
    the same entries, proving the transport-layer path rewrite routes legacy
    clients. Uses raw httpx with the JSON codec so the request path is exactly
    the legacy prefix (the generated client would use the current path)."""
    base = finelog_url
    push = httpx.post(
        f"{base}/iris.logging.LogService/PushLogs",
        json={
            "key": "/legacy/probe",
            "entries": [{"source": "stdout", "data": "old-worker", "timestamp": {"epoch_ms": 1}}],
        },
        headers={"Content-Type": "application/json"},
        timeout=30.0,
    )
    assert push.status_code == 200, push.text

    fetch = httpx.post(
        f"{base}/iris.logging.LogService/FetchLogs",
        json={"source": "/legacy/probe"},
        headers={"Content-Type": "application/json"},
        timeout=30.0,
    )
    assert fetch.status_code == 200, fetch.text
    entries = fetch.json().get("entries", [])
    assert [e["data"] for e in entries] == ["old-worker"]


# ---------------------------------------------------------------------------
# 5f: graceful SIGTERM shutdown after a durable write.
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(base_url: str, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            if httpx.get(f"{base_url}/health", timeout=1.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    raise TimeoutError(f"{base_url}/health did not come up within {timeout}s")


def test_clean_shutdown_after_durable_write(server_backend: Backend, tmp_path: Path) -> None:
    """Write + durably-ack a row, send SIGTERM, assert the process exits 0 within
    the terminate window (no SIGKILL escalation). Proves graceful drain of the
    background tasks; durability is unaffected because writes ack only after L0
    persist."""
    port = _free_port()
    log_dir = tmp_path / "store"
    log_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        server_backend.command(port=port, log_dir=log_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_health(base_url, proc, timeout=20.0)
        # A durable write: PushLogs acks only after the row is on a sealed L0
        # segment.
        client = _log_client(base_url)
        client.push_logs(logging_pb2.PushLogsRequest(key="/shutdown/probe", entries=[_entry("durable", epoch_ms=1)]))
        # Graceful SIGTERM: the server stops accepting, drains, shuts background
        # tasks down, and exits 0.
        proc.send_signal(signal.SIGTERM)
        rc = proc.wait(timeout=10.0)
        assert rc == 0, f"expected clean exit 0 on SIGTERM, got {rc}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)


def test_durable_write_survives_restart(server_backend: Backend, tmp_path: Path) -> None:
    """After a clean SIGTERM shutdown, a second process over the same log_dir
    still sees the durably-acked row — graceful shutdown preserves durability."""
    port = _free_port()
    log_dir = tmp_path / "store"
    log_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        server_backend.command(port=port, log_dir=log_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_health(base_url, proc, timeout=20.0)
        client = _log_client(base_url)
        client.push_logs(logging_pb2.PushLogsRequest(key="/restart/probe", entries=[_entry("persisted", epoch_ms=1)]))
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=10.0) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)

    # Second process over the SAME log_dir; the row survived.
    port2 = _free_port()
    base_url2 = f"http://127.0.0.1:{port2}"
    proc2 = subprocess.Popen(
        server_backend.command(port=port2, log_dir=log_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_health(base_url2, proc2, timeout=20.0)
        client2 = _log_client(base_url2)
        resp = client2.fetch_logs(logging_pb2.FetchLogsRequest(source="/restart/probe"))
        assert [e.data for e in resp.entries] == ["persisted"]
    finally:
        proc2.terminate()
        try:
            proc2.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc2.kill()
            proc2.wait(timeout=5.0)
