# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the in-process native server (``finelog._native.EmbeddedServer``).

Boots the same axum app the ``finelog-server`` binary serves and exercises the
real wire contract end to end via ``LogClient``. Skips when the native extension
is not built (e.g. a pure-Python checkout without a maturin/dev build).

The server runs over an on-disk ``log_dir``: in-memory mode (``log_dir=None``)
spawns no maintenance task, so its RAM buffer never flushes to a readable
segment and reads come back empty. A disk-backed store serves reads.
"""

import pytest
from finelog.client import LogClient
from finelog.embedded import is_available, require_embedded_server
from finelog.rpc import logging_pb2


@pytest.fixture
def embedded_server(tmp_path):
    if not is_available():
        pytest.skip("finelog native extension (finelog._native) not available")
    server = require_embedded_server()(log_dir=str(tmp_path / "log-server"))
    try:
        yield server
    finally:
        server.stop()


def test_embedded_server_log_roundtrip(embedded_server):
    """Push a log batch through the embedded server and read it back."""
    client = LogClient.connect(embedded_server.address)
    try:
        key = "smoke-key-0"
        entries = [
            logging_pb2.LogEntry(
                timestamp=logging_pb2.Timestamp(epoch_ms=1_000 + i),
                source="stdout",
                data=f"line {i}",
            )
            for i in range(5)
        ]
        client.write_batch(key, entries)
        client.flush(timeout=5.0)

        resp = client.fetch_logs(logging_pb2.FetchLogsRequest(source=key, tail=True, max_lines=100))
        assert {e.data for e in resp.entries} == {f"line {i}" for i in range(5)}
        assert all(e.key == key for e in resp.entries)
    finally:
        client.close()
