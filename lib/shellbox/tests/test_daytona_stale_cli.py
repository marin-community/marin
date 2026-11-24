# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

pytest.importorskip("daytona")

from daytona import DaytonaError
from shellbox.backends.daytona import stale_cli
from shellbox.backends.daytona.connection import DaytonaConnection


class FakeSandbox:
    def __init__(self, sandbox_id, timestamp, deleted, fail=False, state="started"):
        self.id = sandbox_id
        self.last_activity_at = timestamp
        self.updated_at = None
        self.state = SimpleNamespace(value=state)
        self.deleted = deleted
        self.fail = fail

    async def delete(self, timeout, wait):
        if self.fail:
            raise DaytonaError("provider failure")
        self.deleted.add(self.id)


class FakeClient:
    def __init__(self, deleted):
        old = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        self.rows = [
            FakeSandbox("idle", old, deleted),
            FakeSandbox("reactivated", old, deleted),
            FakeSandbox("became-destroying", old, deleted),
            FakeSandbox("destroying", old, deleted, state="destroying"),
            FakeSandbox("destroyed", old, deleted, state="destroyed"),
            FakeSandbox("mismatched", old, deleted),
            FakeSandbox("failed", old, deleted, fail=True),
            FakeSandbox("unknown", None, deleted),
        ]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    def list(self, request_timeout):
        async def iterate():
            for row in self.rows:
                yield row

        return iterate()

    async def get(self, sandbox_id, request_timeout):
        row = next(row for row in self.rows if row.id == sandbox_id)
        if sandbox_id == "mismatched":
            return SimpleNamespace(id="different", last_activity_at=row.last_activity_at, updated_at=None)
        if sandbox_id == "reactivated":
            return SimpleNamespace(
                id=sandbox_id, last_activity_at=datetime.now(UTC).isoformat(), updated_at=None, state=row.state
            )
        if sandbox_id == "became-destroying":
            return SimpleNamespace(
                id=sandbox_id,
                last_activity_at=row.last_activity_at,
                updated_at=None,
                state=SimpleNamespace(value="destroying"),
            )
        return row


@pytest.fixture
def deleted_sandboxes(monkeypatch):
    deleted = set()
    monkeypatch.setattr(DaytonaConnection, "client_factory", lambda self: lambda: FakeClient(deleted))
    monkeypatch.setattr(stale_cli, "DELETE_INTERVAL", 0)
    monkeypatch.setenv("TEST_DAYTONA_KEY", "test-key")
    return deleted


@pytest.mark.parametrize("delete", [False, True])
def test_cli_dry_run_and_rechecked_delete_preserve_active_ids_and_append_ledger(
    monkeypatch, tmp_path, deleted_sandboxes, delete
):
    ledger = tmp_path / "ledger.jsonl"
    previous = '{"outcome": "previous run"}\n'
    ledger.write_text(previous)
    arguments = ["shellbox-daytona-stale", "--api-key-env", "TEST_DAYTONA_KEY", "--target", "us"]
    if delete:
        arguments += ["--delete", "--ledger", str(ledger)]
    monkeypatch.setattr(sys, "argv", arguments)
    if delete:
        with pytest.raises(SystemExit) as error:
            stale_cli.main()
        assert error.value.code == 1
        assert deleted_sandboxes == {"idle"}
        records = [json.loads(line) for line in ledger.read_text().splitlines()]
        assert records[0] == {"outcome": "previous run"}
        assert [(row["id"], row["outcome"]) for row in records[1:]] == [
            ("idle", "delete requested"),
            ("idle", "deleted"),
            ("reactivated", "skipped on recheck"),
            ("became-destroying", "skipped on recheck"),
            ("mismatched", "error: DaytonaError"),
            ("failed", "delete requested"),
            ("failed", "error: DaytonaError"),
        ]
    else:
        stale_cli.main()
        assert deleted_sandboxes == set()
        assert ledger.read_text() == previous


def test_cli_ledger_without_delete_rejects_run_and_preserves_ledger(monkeypatch, tmp_path):
    monkeypatch.delenv("TEST_DAYTONA_KEY", raising=False)
    ledger = tmp_path / "ledger.jsonl"
    previous = '{"outcome": "previous run"}\n'
    ledger.write_text(previous)
    monkeypatch.setattr(
        sys,
        "argv",
        ["shellbox-daytona-stale", "--api-key-env", "TEST_DAYTONA_KEY", "--target", "us", "--ledger", str(ledger)],
    )
    with pytest.raises(SystemExit) as error:
        stale_cli.main()
    assert error.value.code == 2
    assert ledger.read_text() == previous
