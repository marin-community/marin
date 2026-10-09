# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest
from finelog.client import FlushResult, LogClient
from finelog.embedded import is_available, require_embedded_server
from iris.runtime import telemetry as runtime_telemetry

from taskforge.ledger.finelog import LEDGER_NAMESPACE, CompositeLedger, FinelogLedger, run_ledger
from taskforge.ledger.jsonl import JsonlLedger, read_entries
from taskforge.ledger.records import EntryKind, LedgerEntry, span

ENTRY = LedgerEntry(
    item_id="item-1",
    round=1,
    step="author",
    kind=EntryKind.LLM_CALL,
    started=100.0,
    ended=102.5,
    model="glm-5.3",
    tokens_in=7,
    finish_reason="stop",
    attrs={"tier": "high"},
)


class UnconfirmedTable:
    def __init__(self):
        self.rows = []

    def write(self, rows):
        self.rows.extend(rows)

    def flush(self, timeout=None):
        return FlushResult.DROPPED


class FakeLogClient:
    def __init__(self, table):
        self.table = table
        self.closed = False

    def get_table(self, namespace, schema, *, table_spec=None):
        return self.table

    def close(self):
        self.closed = True


@pytest.fixture
def embedded_server(tmp_path):
    if not is_available():
        pytest.skip("finelog native server extension (finelog_server) not available")
    server = require_embedded_server()(log_dir=str(tmp_path / "finelog"))
    try:
        yield server
    finally:
        server.stop()


def test_rows_round_trip_through_finelog_server(embedded_server):
    ledger = FinelogLedger(LogClient.connect(embedded_server.address), run_id="run-1")
    ledger.record(ENTRY)
    assert ledger.close() is FlushResult.SUCCEEDED

    reader = LogClient.connect(embedded_server.address)
    try:
        (row,) = reader.query(f'SELECT * FROM "{LEDGER_NAMESPACE}"').to_pylist()
    finally:
        reader.close()
    assert row["run_id"] == "run-1" and row["item_id"] == "item-1" and row["kind"] == "llm_call"
    assert (row["started_ms"], row["ended_ms"], row["timestamp_ms"]) == (100_000, 102_500, 102_500)
    assert row["wall_time"] == 2.5
    assert (row["model"], row["tokens_in"], row["tokens_out"]) == ("glm-5.3", 7, None)
    assert dict(row["attrs"]) == {"tier": "high"}


def test_undelivered_rows_never_fail_the_run_and_are_logged_at_close(tmp_path, caplog):
    client = FakeLogClient(UnconfirmedTable())
    remote = FinelogLedger(client, run_id="run-1")
    ledger = CompositeLedger(JsonlLedger(tmp_path), remote)

    with caplog.at_level(logging.WARNING, logger="taskforge.ledger.finelog"):
        for _ in range(3):
            with span(ledger, EntryKind.STEP, item_id="item-1", round=0, step="build"):
                pass
        assert remote.close() is FlushResult.DROPPED

    assert len(list(read_entries(tmp_path / "item-1.jsonl"))) == 3
    assert len(client.table.rows) == 3
    assert len(caplog.records) == 1 and client.closed


def test_run_ledger_off_cluster_writes_jsonl_only(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_telemetry, "resolve", lambda run_id: None)
    with run_ledger(tmp_path, run_id="run-1") as ledger:
        with span(ledger, EntryKind.STAGE, item_id="item-1", round=0, step="propose"):
            pass
    assert isinstance(ledger, JsonlLedger)
    (entry,) = read_entries(tmp_path / "item-1.jsonl")
    assert (entry.kind, entry.step) == (EntryKind.STAGE, "propose")
