# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest
from marin.daytona.config import DaytonaConfig, resolve_daytona_credentials
from marin.daytona.health import run_health_probe
from marin.daytona.sandboxes import audit_sandboxes, delete_audited_sandboxes
from marin.daytona.snapshots import audit_snapshots, delete_audited_snapshots, list_snapshots


@dataclass
class FakeSandbox:
    id: str
    state: str
    created_at: datetime
    last_activity_at: datetime | None = None
    deleted: bool = False

    def delete(self) -> None:
        self.deleted = True


@dataclass
class FakeSnapshot:
    id: str
    name: str
    state: str
    created_at: datetime
    last_used_at: datetime | None = None


@dataclass
class Page:
    items: list[FakeSnapshot]
    total_pages: int


class FakeSnapshotService:
    def __init__(self, pages: list[Page]):
        self.pages = pages
        self.deleted: list[str] = []

    def list(self, *, page: int, limit: int) -> Page:
        assert limit == 100
        return self.pages[page - 1]

    def delete(self, snapshot: FakeSnapshot) -> None:
        self.deleted.append(snapshot.id)


def test_daytona_credentials_read_only_configured_environment_variable(monkeypatch):
    monkeypatch.setenv("MARIN_DAYTONA_KEY", "credential")

    credentials = resolve_daytona_credentials(
        DaytonaConfig(endpoint="https://api", target="us", api_key_env="MARIN_DAYTONA_KEY")
    )

    assert credentials.config.endpoint == "https://api"
    assert credentials.config.target == "us"
    assert credentials.api_key == "credential"


def test_daytona_credentials_fail_without_echoing_a_secret(monkeypatch):
    monkeypatch.delenv("MISSING_DAYTONA_KEY", raising=False)

    with pytest.raises(ValueError, match=r"\$MISSING_DAYTONA_KEY") as error:
        resolve_daytona_credentials(DaytonaConfig(api_key_env="MISSING_DAYTONA_KEY"))

    assert "credential" not in str(error.value)


def test_sandbox_audit_selects_only_stale_or_terminal_and_deletion_requires_confirmation():
    now = datetime(2026, 7, 22, tzinfo=UTC)
    stale = FakeSandbox("job-stale", "started", now - timedelta(hours=2))
    fresh = FakeSandbox("job-fresh", "started", now - timedelta(minutes=5))
    terminal = FakeSandbox("job-terminal", "error", now)

    unscoped_rows = audit_sandboxes([fresh, terminal, stale], stale_after_minutes=60, now=now)
    assert not any(row.delete_eligible for row in unscoped_rows)

    rows = audit_sandboxes([fresh, terminal, stale], stale_after_minutes=60, id_prefix="job-", now=now)

    assert {row.sandbox_id for row in rows if row.delete_eligible} == {"job-stale", "job-terminal"}
    assert delete_audited_sandboxes(rows, confirm=lambda _: False) == []
    assert not stale.deleted and not terminal.deleted

    results = delete_audited_sandboxes(rows, confirm=lambda count: count == 2)

    assert [result.sandbox_id for result in results] == ["job-stale", "job-terminal"]
    assert stale.deleted and terminal.deleted and not fresh.deleted


def test_snapshot_audit_pages_and_never_selects_transitional_or_out_of_scope_snapshots():
    now = datetime(2026, 7, 22, tzinfo=UTC)
    stale = FakeSnapshot("old", "tasks-old", "active", now - timedelta(days=30))
    protected = FakeSnapshot("building", "tasks-building", "building", now - timedelta(days=30))
    other = FakeSnapshot("other", "shared-base", "active", now - timedelta(days=30))
    service = FakeSnapshotService([Page([stale, protected], 2), Page([other], 2)])

    rows = audit_snapshots(list_snapshots(service), stale_after_days=14, name_prefix="tasks-", now=now)

    assert [row.snapshot_id for row in rows if row.delete_eligible] == ["old"]
    assert delete_audited_snapshots(service, rows, confirm=lambda count: count == 1) == ["old"]
    assert service.deleted == ["old"]


def test_snapshot_audit_rejects_an_empty_scope_prefix():
    with pytest.raises(ValueError, match="name_prefix must be non-empty"):
        audit_snapshots([], stale_after_days=14, name_prefix="")


def test_health_probe_cleans_up_owned_sandbox_when_command_fails():
    deleted: list[str] = []

    def raise_command(_sandbox, _command):
        raise RuntimeError("command failed")

    with pytest.raises(RuntimeError, match="command failed"):
        run_health_probe(
            create=lambda: "sandbox",
            command="false",
            execute=raise_command,
            delete=lambda sandbox: deleted.append(sandbox),
        )

    assert deleted == ["sandbox"]
