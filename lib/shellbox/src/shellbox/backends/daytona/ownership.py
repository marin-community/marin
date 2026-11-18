# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quarantine trial evidence and reconcile only lifecycle-bound sandbox identities."""

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Protocol

from daytona import AsyncDaytona, DaytonaError, DaytonaNotFoundError
from pydantic import BaseModel, ConfigDict

from shellbox.backends.daytona.lifecycle import SandboxLifecycle
from shellbox.backends.daytona.provider import read_provider_call


class TrialOwnershipStatus(StrEnum):
    COMPLETED = "completed"
    UNRESOLVED_CLEANUP = "unresolved_cleanup"


class OwnershipEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    sandbox_ids: tuple[str, ...]
    unbound_requests: tuple[str, ...]
    lifecycle_sha256: dict[str, str]

    @property
    def retained_slots(self) -> int:
        return len(self.sandbox_ids) + len(self.unbound_requests)


def lifecycle_ownership(paths: Iterable[Path]) -> OwnershipEvidence:
    """Bind exact IDs to this attempt's own create requests, retaining unknown requests."""
    requested: dict[str, set[str]] = {}
    hashes = {}
    for path in sorted(paths):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        for line in raw.splitlines():
            row = json.loads(line)
            event = SandboxLifecycle(**row)
            if event.kind != "daytona_sandbox_lifecycle" or event.schema_version != 1:
                raise ValueError(f"Unsupported ownership event: {path}")
            if event.event == "create_requested":
                if event.requested_name in requested:
                    raise ValueError(f"Repeated create request: {event.requested_name}")
                requested[event.requested_name] = set()
                continue
            if event.requested_name not in requested:
                raise ValueError(f"Sandbox event lacks an owned create request: {path}")
            # The factory resolves its unique create name before emitting DELETE_REQUESTED,
            # including SDK creation failures that never emit CREATED.
            if event.event in {"created", "delete_requested"}:
                if not event.sandbox_id:
                    raise ValueError(f"Ownership event has no exact ID: {path}")
                requested[event.requested_name].add(event.sandbox_id)
            elif event.event not in {"deleted", "absent"}:
                raise ValueError(f"Unknown lifecycle event: {event.event}")
            elif event.sandbox_id and event.sandbox_id not in requested[event.requested_name]:
                raise ValueError(f"Unbound sandbox ID: {event.sandbox_id}")
    ids = [sandbox_id for bound in requested.values() for sandbox_id in bound]
    if len(ids) != len(set(ids)):
        raise ValueError("A sandbox ID is bound to multiple create requests")
    if not hashes:
        raise ValueError("No lifecycle logs were captured")
    return OwnershipEvidence(
        sandbox_ids=tuple(sorted(ids)),
        unbound_requests=tuple(sorted(name for name, ids in requested.items() if not ids)),
        lifecycle_sha256=hashes,
    )


class SandboxResolution(BaseModel):
    sandbox_id: str
    absent: bool
    delete_requested: bool = False
    error: str | None = None


class OwnershipResolution(BaseModel):
    evidence: OwnershipEvidence
    sandboxes: tuple[SandboxResolution, ...]

    @property
    def resolved(self) -> bool:
        return (
            not self.evidence.unbound_requests
            and {row.sandbox_id for row in self.sandboxes} == set(self.evidence.sandbox_ids)
            and all(row.absent for row in self.sandboxes)
        )


async def resolve_ownership(
    client: AsyncDaytona, evidence: OwnershipEvidence, *, delete_owned: bool
) -> OwnershipResolution:
    """Check exact IDs; optionally delete a proved owned object once, then read again.

    Read exhaustion and ambiguous deletion remain typed unresolved evidence. No list,
    label, status, time-window attribution, or mutation retry is used.
    """
    if evidence.lifecycle_sha256 and lifecycle_ownership(Path(path) for path in evidence.lifecycle_sha256) != evidence:
        raise ValueError("Frozen lifecycle evidence changed before reconciliation")
    if evidence.sandbox_ids and not evidence.lifecycle_sha256:
        raise ValueError("Sandbox IDs require frozen lifecycle evidence")
    records = []
    for sandbox_id in evidence.sandbox_ids:
        deleted = False
        error = None
        try:
            sandbox = await read_provider_call("sandbox.get", partial(client.get, sandbox_id))
        except DaytonaNotFoundError:
            records.append(SandboxResolution(sandbox_id=sandbox_id, absent=True))
            continue
        except (DaytonaError, TimeoutError) as failure:
            records.append(SandboxResolution(sandbox_id=sandbox_id, absent=False, error=str(failure)))
            continue
        if sandbox.id != sandbox_id:
            raise ValueError(f"Provider returned a different sandbox for exact ID {sandbox_id}")
        if delete_owned:
            deleted = True
            try:
                await client.delete(sandbox, wait=True)
            except (DaytonaError, TimeoutError) as failure:
                error = str(failure)
            try:
                await read_provider_call("sandbox.absence", partial(client.get, sandbox_id))
            except DaytonaNotFoundError:
                records.append(SandboxResolution(sandbox_id=sandbox_id, absent=True, delete_requested=True, error=error))
                continue
            except (DaytonaError, TimeoutError) as failure:
                error = str(failure)
        records.append(SandboxResolution(sandbox_id=sandbox_id, absent=False, delete_requested=deleted, error=error))
    return OwnershipResolution(evidence=evidence, sandboxes=tuple(records))


class TrialOwnershipRecord(BaseModel):
    row_id: str
    attempt: int
    status: TrialOwnershipStatus
    ownership: OwnershipEvidence
    evidence_uri: str
    evidence_sha256: str
    error: str | None = None


@dataclass(frozen=True)
class QuarantineBatchResult:
    completed: tuple[TrialOwnershipRecord, ...]
    quarantined: tuple[TrialOwnershipRecord, ...]
    deferred_rows: tuple[str, ...]
    unscored_rows: tuple[TrialOwnershipRecord, ...] = ()


class QuarantineCapacity(Protocol):
    def try_acquire(self, quarantines: tuple[TrialOwnershipRecord, ...]) -> bool: ...
    def track(self, record: TrialOwnershipRecord) -> None: ...
    def reconciled(self, record: TrialOwnershipRecord, resolution: OwnershipResolution) -> None: ...


def run_quarantined_trials(
    row_ids: Iterable[str],
    execute: Callable[[str, int], TrialOwnershipRecord],
    persist: Callable[[TrialOwnershipRecord], None],
    reconcile: Callable[[TrialOwnershipRecord], OwnershipResolution],
    persist_resolution: Callable[[TrialOwnershipRecord, OwnershipResolution], None],
    *,
    physical_slots: int,
    reserve: QuarantineCapacity | None = None,
    wait_for_capacity: Callable[[], None] | None = None,
) -> QuarantineBatchResult:
    """Continue independent rows within a physical slot bound, then reconcile and retry once.

    Execution archives each attempt before returning. Quarantine never constitutes a
    scored result. A retained unknown sandbox consumes capacity until exact-ID absence
    is established. If capacity is exhausted, unstarted rows remain explicitly deferred.
    """
    if physical_slots < 1:
        raise ValueError("A root needs an explicit physical sandbox allocation")
    if reserve is not None and (physical_slots != 1 or wait_for_capacity is None):
        raise ValueError("A shared reserve requires one base slot and an explicit capacity wait")
    completed = []
    pending: dict[tuple[str, int], TrialOwnershipRecord] = {}
    deferred = []
    unscored = []

    def run(row_id: str, attempt: int) -> None:
        record = execute(row_id, attempt)
        if record.row_id != row_id or record.attempt != attempt:
            raise ValueError("Trial returned another assignment or attempt")
        persist(record)
        if record.status is TrialOwnershipStatus.COMPLETED:
            completed.append(record)
        else:
            pending[(record.row_id, record.attempt)] = record
            if reserve is not None:
                reserve.track(record)

    def reconcile_pass() -> None:
        for key, record in tuple(pending.items()):
            resolution = reconcile(record)
            if resolution.evidence != record.ownership:
                raise ValueError("Reconciliation changed frozen ownership evidence")
            persist_resolution(record, resolution)
            if reserve is not None:
                reserve.reconciled(record, resolution)
            if not resolution.resolved:
                continue
            del pending[key]
            if record.attempt == 0:
                run(record.row_id, 1)
            else:
                # A cleared second technical attempt is not a source score or a third trial.
                unscored.append(record)

    for row_id in row_ids:
        admitted = True
        while True:
            occupied = sum(max(1, row.ownership.retained_slots) for row in pending.values())
            if occupied < physical_slots:
                break
            if reserve is None:
                deferred.append(row_id)
                admitted = False
                break
            if occupied < physical_slots + 1 and reserve.try_acquire(tuple(pending.values())):
                break
            # Reconcile at this capacity-induced pass boundary to avoid every root
            # waiting for a reserve held by another root with a second quarantine.
            reconcile_pass()
            if sum(max(1, row.ownership.retained_slots) for row in pending.values()) < physical_slots:
                break
            assert wait_for_capacity is not None
            wait_for_capacity()
        if admitted:
            run(row_id, 0)
    reconcile_pass()
    if reserve is not None:
        while pending:
            assert wait_for_capacity is not None
            wait_for_capacity()
            reconcile_pass()
    return QuarantineBatchResult(tuple(completed), tuple(pending.values()), tuple(deferred), tuple(unscored))
