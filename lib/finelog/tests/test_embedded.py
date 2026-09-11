# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the in-process native server (``finelog_server.EmbeddedServer``).

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
        pytest.skip("finelog native server extension (finelog_server) not available")
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


def test_cursors_bracket_a_window_around_a_named_row(embedded_server):
    """`cursor`/`until_cursor` page either side of a row named by its `seq`."""
    client = LogClient.connect(embedded_server.address)
    try:
        key = "window-key"
        client.write_batch(
            key,
            [
                logging_pb2.LogEntry(
                    timestamp=logging_pb2.Timestamp(epoch_ms=1_000 + i),
                    source="stdout",
                    data=f"line {i}",
                )
                for i in range(20)
            ],
        )
        client.flush(timeout=5.0)

        def fetch(**kwargs) -> list[str]:
            request = logging_pb2.FetchLogsRequest(source=key, match_scope=logging_pb2.MATCH_SCOPE_EXACT, **kwargs)
            return [e.data for e in client.fetch_logs(request).entries]

        everything = client.fetch_logs(
            logging_pb2.FetchLogsRequest(source=key, match_scope=logging_pb2.MATCH_SCOPE_EXACT, max_lines=100)
        ).entries
        seqs = [e.seq for e in everything]
        if not all(seq > 0 for seq in seqs):
            pytest.skip(
                "the finelog_server extension predates LogEntry.seq. It installs from a pre-built "
                "wheel by default; build it from this tree with "
                "`python scripts/rust_mode.py dev && uv sync` to exercise the cursor contract."
            )
        assert seqs == sorted(seqs), "seq ascends in write order"
        assert len(set(seqs)) == len(seqs), "seq uniquely names a row"

        anchor = everything[10].seq
        assert everything[10].data == "line 10"

        # `until_cursor` is exclusive, so tailing it reads the rows *before* the anchor.
        assert fetch(until_cursor=anchor, tail=True, max_lines=3) == ["line 7", "line 8", "line 9"]
        # `cursor` is exclusive too, so `anchor - 1` is the first request that
        # still includes the anchor row itself.
        assert fetch(cursor=anchor - 1, max_lines=3) == ["line 10", "line 11", "line 12"]
        # Together they bracket an open interval.
        assert fetch(cursor=seqs[2], until_cursor=seqs[6], max_lines=100) == ["line 3", "line 4", "line 5"]
        # An unset upper bound reads to the end: seq starts at 1, so 0 excludes nothing.
        assert len(fetch(until_cursor=0, max_lines=100)) == 20
    finally:
        client.close()
