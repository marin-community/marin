# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise quarantine and exact-ID recovery through a provider boundary fake."""

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("daytona")
from daytona import DaytonaError, DaytonaNotFoundError
from shellbox.backends.daytona.ownership import (
    TrialOwnershipRecord,
    TrialOwnershipStatus,
    lifecycle_ownership,
    resolve_ownership,
    run_quarantined_trials,
)
from shellbox.backends.daytona.provider import read_provider_call


class SandboxStore:
    def __init__(self, ids=(), failures=()):
        self.ids = set(ids)
        self.failures = list(failures)
        self.reads = []
        self.deletes = []

    async def get(self, sandbox_id):
        self.reads.append(sandbox_id)
        if self.failures:
            raise self.failures.pop(0)
        if sandbox_id not in self.ids:
            raise DaytonaNotFoundError("absent")
        return SimpleNamespace(id=sandbox_id)

    async def delete(self, sandbox, wait):
        assert wait
        self.deletes.append(sandbox.id)
        self.ids.remove(sandbox.id)


def ownership(tmp_path, row_id):
    path = tmp_path / f"{row_id}.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "kind": "daytona_sandbox_lifecycle",
                    "schema_version": 1,
                    "event": event,
                    "requested_name": f"owned-{row_id}",
                    "sandbox_id": sandbox_id,
                }
            )
            for event, sandbox_id in [("create_requested", None), ("created", row_id)]
        )
        + "\n"
    )
    return lifecycle_ownership([path])


@pytest.fixture
def no_provider_wait(monkeypatch):
    delays = []

    async def wait(delay):
        delays.append(delay)

    monkeypatch.setattr("shellbox.backends.daytona.provider.sleep", wait)
    return delays


def test_read_retries_502_and_timeout_then_returns_remote_value(no_provider_wait):
    store = SandboxStore(["owned"], [DaytonaError("gateway", status_code=502), TimeoutError("read timeout")])
    result = asyncio.run(read_provider_call("sandbox.get", lambda: store.get("owned")))
    assert result.id == "owned"
    assert store.reads == ["owned"] * 3 and store.deletes == []
    assert len(no_provider_wait) == 2 and all(0 < delay <= 120 for delay in no_provider_wait)


def test_reconciliation_deletes_only_frozen_lifecycle_ids_once(tmp_path):
    evidence = ownership(tmp_path, "owned")
    store = SandboxStore(["owned", "foreign"])
    result = asyncio.run(resolve_ownership(store, evidence, delete_owned=True))
    assert result.resolved
    assert store.ids == {"foreign"} and store.deletes == ["owned"]
    assert set(store.reads) == {"owned"}
    forged = evidence.model_copy(update={"sandbox_ids": ("foreign",)})
    with pytest.raises(ValueError, match="lifecycle evidence changed"):
        asyncio.run(resolve_ownership(store, forged, delete_owned=True))
    assert store.ids == {"foreign"}


def test_reconciliation_never_replays_an_ambiguous_delete(tmp_path):
    class AmbiguousDelete(SandboxStore):
        async def delete(self, sandbox, wait):
            await super().delete(sandbox, wait)
            raise DaytonaError("gateway after deletion", status_code=502)

    store = AmbiguousDelete(["owned", "foreign"])
    result = asyncio.run(resolve_ownership(store, ownership(tmp_path, "owned"), delete_owned=True))
    assert result.resolved and result.sandboxes[0].error
    assert store.deletes == ["owned"] and store.ids == {"foreign"}


def test_read_exhaustion_quarantines_then_other_rows_run_and_one_retry_is_bounded(tmp_path, no_provider_wait):
    evidence = {key: ownership(tmp_path, key) for key in ["a", "b"]}
    store = SandboxStore(failures=[DaytonaError("gateway", status_code=502) for _ in range(5)])
    executions = []
    records = []
    resolutions = []

    def execute(row_id, attempt):
        executions.append((row_id, attempt))
        status = TrialOwnershipStatus.COMPLETED
        if row_id == "a":
            result = asyncio.run(resolve_ownership(store, evidence[row_id], delete_owned=False))
            if attempt == 0:
                assert not result.resolved and len(store.reads) == 5
                status = TrialOwnershipStatus.UNRESOLVED_CLEANUP
        return TrialOwnershipRecord(
            row_id=row_id,
            attempt=attempt,
            status=status,
            ownership=evidence[row_id],
            evidence_uri=f"s3://fixture/{row_id}/{attempt}",
            evidence_sha256=hashlib.sha256(b"archive").hexdigest(),
        )

    result = run_quarantined_trials(
        ["a", "b"],
        execute,
        records.append,
        lambda row: asyncio.run(resolve_ownership(store, row.ownership, delete_owned=True)),
        lambda row, resolution: resolutions.append(resolution),
        physical_slots=2,
    )
    assert executions == [("a", 0), ("b", 0), ("a", 1)]
    assert [row.status for row in records] == [
        TrialOwnershipStatus.UNRESOLVED_CLEANUP,
        TrialOwnershipStatus.COMPLETED,
        TrialOwnershipStatus.COMPLETED,
    ]
    assert len(result.completed) == 2 and not result.quarantined and not result.deferred_rows
    assert resolutions[0].resolved and not store.deletes


def test_quarantine_holds_capacity_and_second_unresolved_attempt_is_not_requeued(tmp_path):
    evidence = ownership(tmp_path, "owned")
    executions = []
    store = SandboxStore()

    def execute(row_id, attempt):
        executions.append((row_id, attempt))
        return TrialOwnershipRecord(
            row_id=row_id,
            attempt=attempt,
            status=TrialOwnershipStatus.UNRESOLVED_CLEANUP,
            ownership=evidence,
            evidence_uri=f"s3://fixture/{attempt}",
            evidence_sha256="a" * 64,
        )

    result = run_quarantined_trials(
        ["a", "b"],
        execute,
        lambda row: None,
        lambda row: asyncio.run(resolve_ownership(store, row.ownership, delete_owned=True)),
        lambda row, resolution: None,
        physical_slots=1,
    )
    assert executions == [("a", 0), ("a", 1)]
    assert result.deferred_rows == ("b",) and len(result.quarantined) == 1
