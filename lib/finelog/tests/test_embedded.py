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

import json
from contextlib import contextmanager
from datetime import UTC, datetime

import pytest
from click.testing import CliRunner
from finelog.client import LogClient
from finelog.embedded import is_available, require_embedded_server
from finelog.ops_feed import NAMESPACE, PAGE_SIZE, Event, Severity, Subscription, ack, emit, poll
from finelog.ops_feed_cli import cli
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


def test_ops_feed_replay_filter_and_reader_positions(embedded_server, tmp_path):
    client = LogClient.connect(embedded_server.address)
    state = tmp_path / "reader.sqlite"
    subscriptions = [Subscription("watchdog", Severity.WARNING)]
    try:
        first = emit("watchdog", "failure", Severity.ERROR, "job:opaque", {}, dedup_key="one", client=client)
        emit("watchdog", "failure", Severity.ERROR, "job:opaque", {}, dedup_key="one", client=client)
        emit("watchdog", "progress", Severity.INFO, "job:opaque", {}, client=client)
        emit("other", "failure", Severity.CRITICAL, "job:opaque", {}, client=client)
        assert [e.event_id for e in poll(client, subscriptions, "coordinator", state=state)] == [first]
        # Reload the database through a new connection; filtered rows advance too.
        assert poll(client, [Subscription("*", Severity.INFO)], "coordinator", state=state) == []
        emit("watchdog", "failure", Severity.ERROR, "job:opaque", {}, dedup_key="one", client=client)
        second = emit("watchdog", "failure", Severity.ERROR, "job:opaque", {}, dedup_key="two", client=client)
        assert [e.event_id for e in poll(client, subscriptions, "coordinator", state=state)] == [second]
        assert len(poll(client, subscriptions, "independent", state=state)) == 2
    finally:
        client.close()


def test_ops_feed_ack_closes_subject_kind_and_later_event_reopens(embedded_server, tmp_path):
    client = LogClient.connect(embedded_server.address)
    subscriptions = [Subscription("*", Severity.INFO)]
    state = tmp_path / "reader.sqlite"
    try:
        first = emit("watchdog", "failure", Severity.ERROR, "job:opaque", {}, dedup_key="one", client=client)
        emit("watchdog", "failure", Severity.WARNING, "job:opaque", {}, dedup_key="two", client=client)
        other_kind = emit("watchdog", "stalled", Severity.WARNING, "job:opaque", {}, client=client)
        other_source = emit("other", "failure", Severity.WARNING, "job:opaque", {}, client=client)
        ack(client, first)
        # A delayed at-least-once replay of the acknowledged ID stays closed.
        emit("watchdog", "failure", Severity.ERROR, "job:opaque", {}, dedup_key="one", client=client)
        emit("watchdog", "failure", Severity.WARNING, "job:opaque", {}, dedup_key="two", client=client)
        reopened = emit("watchdog", "failure", Severity.ERROR, "job:opaque", {}, client=client)
        assert {e.event_id for e in poll(client, subscriptions, "fresh", state=state)} == {
            other_kind,
            other_source,
            reopened,
        }
    finally:
        client.close()


def test_ops_feed_filtered_page_advances_to_next_page(embedded_server, tmp_path):
    client = LogClient.connect(embedded_server.address)
    state = tmp_path / "reader.sqlite"
    subscriptions = [Subscription("watchdog", Severity.ERROR)]
    try:
        table = client.get_table(NAMESPACE, Event)
        table.write(
            [Event(str(i), datetime.now(UTC), "watchdog", "progress", "info", "opaque", "{}") for i in range(PAGE_SIZE)]
        )
        assert table.flush(timeout=5) == "succeeded"
        last = emit("watchdog", "failure", Severity.ERROR, "opaque", {}, client=client)
        assert poll(client, subscriptions, "reader", state=state) == []
        assert [e.event_id for e in poll(client, subscriptions, "reader", state=state)] == [last]
        assert poll(client, subscriptions, "reader", state=state) == []
    finally:
        client.close()


def test_ops_feed_cli_lowercase_severity_poll_and_ack(embedded_server, tmp_path, monkeypatch):
    client = LogClient.connect(embedded_server.address)

    @contextmanager
    def connection(*args, **kwargs):
        yield client

    monkeypatch.setattr("finelog.ops_feed_cli.open_client", connection)
    runner = CliRunner()
    prefix = ["--state", str(tmp_path / "state.sqlite")]
    subscriptions = tmp_path / "subscriptions.yaml"
    subscriptions.write_text("subscriptions:\n  - source: cli-probe\n    min_severity: info\n")
    poll_args = ["poll", "--subscriptions", str(subscriptions), "--reader", "first"]
    try:
        emitted = runner.invoke(
            cli,
            [*prefix, "emit", "--source", "cli-probe", "--kind", "test", "--severity", "info", "--subject", "opaque"],
        )
        assert emitted.exit_code == 0, emitted.output
        event_id = emitted.output.strip()
        result = runner.invoke(cli, prefix + poll_args)
        assert result.exit_code == 0, result.output
        event = json.loads(result.output)
        assert event["event_id"] == event_id
        assert event["severity"] == "info"
        assert event["ts"].endswith("+00:00")
        acknowledged = runner.invoke(cli, [*prefix, "ack", event_id])
        assert acknowledged.exit_code == 0, acknowledged.output
        poll_args[-1] = "fresh"
        result = runner.invoke(cli, prefix + poll_args)
        assert result.exit_code == 0, result.output
        assert result.output == ""
    finally:
        client.close()
