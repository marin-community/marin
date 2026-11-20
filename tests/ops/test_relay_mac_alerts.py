# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from finelog.client import LogClient
from finelog.client.log_client import DEFAULT_BATCH_ROWS, DEFAULT_MAX_BUFFER_BYTES
from finelog.embedded import is_available, require_embedded_server
from finelog.ops_feed import NAMESPACE, Severity, Subscription, poll

from scripts.ops.relay_mac_alerts import MacSource, import_alerts


@pytest.fixture
def embedded_server(tmp_path):
    if not is_available():
        pytest.skip("finelog native server extension not available")
    server = require_embedded_server()(log_dir=str(tmp_path / "finelog"))
    try:
        yield server
    finally:
        server.stop()


@pytest.mark.parametrize("source", list(MacSource))
def test_ops_feed_mac_import_replay_rotation_and_partial_write(embedded_server, tmp_path, source):
    client = LogClient.connect(embedded_server.address)
    path = tmp_path / "alerts.jsonl"
    state = tmp_path / "state.sqlite"
    if source == MacSource.JOB_WATCHDOG:
        record = {
            "entry": "private-owner",
            "kind": "JOB_FAILED",
            "severity": "critical",
            "state": "failed",
            "time_utc": "2026-11-20T10:00:00+00:00",
            "message": "private@example.com",
            "evidence": "secret",
        }
    else:
        record = {
            "metric": "ram",
            "kind": "enter",
            "value": 1.0,
            "threshold": 2.0,
            "direction": "below",
            "ts": 1795168800,
            "top_processes": [{"command": "private@example.com", "worktree": "secret"}],
        }
    line = json.dumps(record) + "\n"
    try:
        path.write_text(line + '{"partial":')
        assert import_alerts(client, source, path, destination="test", state=state) == 1
        # Replacing a rolling file with the same retained record must not replay it.
        path.write_text(line)
        assert import_alerts(client, source, path, destination="test", state=state) == 0
        events = poll(client, [Subscription("*", Severity.INFO)], "reader", state=state)
        assert len(events) == 1
        assert "private" not in events[0].payload_json + events[0].subject
        assert "secret" not in events[0].payload_json
        assert events[0].source == source
    finally:
        client.close()


@pytest.mark.parametrize("rows,state_bytes", [(DEFAULT_BATCH_ROWS + 1, 6), (9, DEFAULT_MAX_BUFFER_BYTES // 8)])
def test_mac_import_backlog_exceeding_queue_is_delivered_and_not_replayed(embedded_server, tmp_path, rows, state_bytes):
    client = LogClient.connect(embedded_server.address)
    path = tmp_path / "alerts.jsonl"
    state = tmp_path / "state.sqlite"
    try:
        with path.open("w") as stream:
            for index in range(rows):
                record = {
                    "entry": str(index),
                    "kind": "JOB_FAILED",
                    "severity": "critical",
                    "state": "x" * state_bytes,
                    "time_utc": "2026-11-20T10:00:00+00:00",
                }
                stream.write(json.dumps(record) + "\n")
        assert import_alerts(client, MacSource.JOB_WATCHDOG, path, destination="test", state=state) == rows
        assert import_alerts(client, MacSource.JOB_WATCHDOG, path, destination="test", state=state) == 0
        counts = client.query(
            f'SELECT count(*) AS event_count, count(DISTINCT event_id) AS unique_ids FROM "{NAMESPACE}"'
        ).to_pylist()
        assert counts == [{"event_count": rows, "unique_ids": rows}]
    finally:
        client.close()
