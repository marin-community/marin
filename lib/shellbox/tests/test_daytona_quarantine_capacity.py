# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Race real conditional files and exercise reserve backpressure at the trial boundary."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("daytona")
from shellbox.backends.daytona.ownership import (
    OwnershipResolution,
    SandboxResolution,
    TrialOwnershipRecord,
    TrialOwnershipStatus,
    lifecycle_ownership,
    run_quarantined_trials,
)
from shellbox.backends.daytona.quarantine_capacity import (
    QuarantineReserveConfig,
    QuarantineReservePool,
    ReserveDecision,
    RootQuarantineReserve,
)


def trial(tmp_path, key, attempt=0, *, completed=False):
    path = tmp_path / f"{key}-{attempt}.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "kind": "daytona_sandbox_lifecycle",
                    "schema_version": 1,
                    "event": event,
                    "requested_name": f"owned-{key}-{attempt}",
                    "sandbox_id": identity,
                }
            )
            for event, identity in [("create_requested", None), ("created", f"{key}-{attempt}")]
        )
        + "\n"
    )
    return TrialOwnershipRecord(
        row_id=key,
        attempt=attempt,
        status=TrialOwnershipStatus.COMPLETED if completed else TrialOwnershipStatus.UNRESOLVED_CLEANUP,
        ownership=lifecycle_ownership([path]),
        evidence_uri=f"s3://archive/{key}/{attempt}",
        evidence_sha256="a" * 64,
    )


def resolution(record, *, absent=True):
    return OwnershipResolution(
        evidence=record.ownership,
        sandboxes=tuple(SandboxResolution(sandbox_id=key, absent=absent) for key in record.ownership.sandbox_ids),
    )


def pool(tmp_path):
    config = QuarantineReserveConfig(
        state_uri=str(tmp_path / "reserve.json"),
        pool_size=6,
        max_extra_slots_per_root=1,
        root_keys=tuple(f"root-{n}" for n in range(12)),
        allocation_uri="s3://allocation/grant.json",
        allocation_sha256="b" * 64,
    )
    result = QuarantineReservePool(config)
    result.initialize()
    return result


def test_concurrent_roots_share_six_slots_and_one_slot_per_root(tmp_path):
    shared = pool(tmp_path)
    records = [trial(tmp_path, f"root-{n}") for n in range(12)]

    def claim(n):
        return shared.claim(f"root-{n}", f"attempt-{n}", (records[n],))

    with ThreadPoolExecutor(max_workers=4) as workers:
        receipts = list(workers.map(claim, range(12)))
    winners = [r for r in receipts if r.decision is ReserveDecision.ACQUIRED]
    losers = [r for r in receipts if r.decision is ReserveDecision.WAIT]
    assert len(winners) == len(losers) == 6
    assert len(shared.snapshot().claims) == 6
    assert max(r.occupied_reserve_slots for r in receipts) <= 6
    first = winners[0]
    repeated = shared.claim(first.root_key, first.owner, first.claim.quarantines)
    assert repeated.claim.claim_id == first.claim.claim_id
    assert len(shared.snapshot().claims) == 6
    replacement = shared.claim(first.root_key, "new-process-after-loss", first.claim.quarantines)
    assert replacement.decision is ReserveDecision.WAIT

    record = first.claim.quarantines[0]
    blocked = shared.reconciled(first.claim, record, resolution(record, absent=False))
    assert blocked.decision is ReserveDecision.TRACKED and len(shared.snapshot().claims) == 6
    released = shared.reconciled(first.claim, record, resolution(record))
    assert released.decision is ReserveDecision.RELEASED
    waiting = losers[0]
    replacement = shared.claim(waiting.root_key, waiting.owner, (records[int(waiting.root_key.split("-")[1])],))
    assert replacement.decision is ReserveDecision.ACQUIRED and len(shared.snapshot().claims) == 6
    assert all(r.config.pool_size == 6 and r.config.max_extra_slots_per_root == 1 for r in receipts)


def test_release_rejects_foreign_evidence_and_missing_pool_never_recreates_capacity(tmp_path):
    shared = pool(tmp_path)
    own = trial(tmp_path, "owned")
    foreign = trial(tmp_path, "foreign")
    claim = shared.claim("root-0", "attempt", (own,)).claim
    with pytest.raises(ValueError, match="another frozen ownership"):
        shared.reconciled(claim, own, resolution(foreign))
    assert shared.snapshot().claims["root-0"].claim_id == claim.claim_id
    (tmp_path / "reserve.json").unlink()
    with pytest.raises(ValueError, match="state is missing"):
        shared.claim("root-1", "another-attempt", (foreign,))


def test_root_waits_for_shared_reserve_then_continues_without_oversubscription(tmp_path):
    shared = pool(tmp_path)
    held = [trial(tmp_path, f"holder-{n}") for n in range(6)]
    claims = [shared.claim(f"root-{n}", f"owner-{n}", (held[n],)).claim for n in range(6)]
    events = []
    capacity = []
    records = []
    own = {key: trial(tmp_path, key) for key in ["a", "b"]}
    resolver_reads = []
    waits = []
    reserve = RootQuarantineReserve(shared, "root-6", "producer", capacity.append)

    def execute(row_id, attempt):
        events.append((row_id, attempt))
        if row_id == "a" and attempt == 0:
            return own["a"]
        return trial(tmp_path, row_id, attempt, completed=True)

    def reconcile(record):
        resolver_reads.append(record.row_id)
        return resolution(record, absent=bool(waits))

    def wait():
        assert events == [("a", 0)]
        assert len(shared.snapshot().claims) == 6
        shared.reconciled(claims[0], held[0], resolution(held[0]))
        waits.append(True)

    result = run_quarantined_trials(
        ["a", "b"],
        execute,
        records.append,
        reconcile,
        lambda row, result: None,
        physical_slots=1,
        reserve=reserve,
        wait_for_capacity=wait,
    )
    assert waits == [True]
    assert events == [("a", 0), ("b", 0), ("a", 1)]
    assert len(result.completed) == 2 and not result.quarantined
    assert [r.decision for r in capacity] == [ReserveDecision.WAIT, ReserveDecision.ACQUIRED, ReserveDecision.RELEASED]
    assert len(shared.snapshot().claims) == 5


def test_second_quarantine_reconciles_before_a_third_sandbox_and_retry_is_once(tmp_path):
    shared = pool(tmp_path)
    receipts = []
    reserve = RootQuarantineReserve(shared, "root-0", "producer", receipts.append)
    live = set()
    events = []

    def execute(row_id, attempt):
        events.append((row_id, attempt))
        assert len(live) < 2
        record = trial(tmp_path, row_id, attempt, completed=row_id not in {"a", "b"})
        if record.status is TrialOwnershipStatus.UNRESOLVED_CLEANUP:
            live.add((row_id, attempt))
        return record

    def reconcile(record):
        live.remove((record.row_id, record.attempt))
        return resolution(record)

    result = run_quarantined_trials(
        ["a", "b", "c"],
        execute,
        lambda row: None,
        reconcile,
        lambda row, result: None,
        physical_slots=1,
        reserve=reserve,
        wait_for_capacity=lambda: None,
    )
    assert events == [("a", 0), ("b", 0), ("a", 1), ("b", 1), ("c", 0)]
    assert not live and not shared.snapshot().claims
    assert [r.row_id for r in result.unscored_rows] == ["a", "b"]
    assert [r.row_id for r in result.completed] == ["c"]


@pytest.mark.parametrize("inherited_first", [False, True])
def test_absent_ids_do_not_block_independent_rows_behind_an_unbound_request(tmp_path, inherited_first):
    shared = pool(tmp_path)
    inherited = trial(tmp_path, "inherited")
    path = next(iter(inherited.ownership.lifecycle_sha256))
    with open(path, "a") as stream:
        stream.write(
            json.dumps(
                {
                    "kind": "daytona_sandbox_lifecycle",
                    "schema_version": 1,
                    "event": "create_requested",
                    "requested_name": "ambiguous-create",
                    "sandbox_id": None,
                }
            )
            + "\n"
        )
    inherited = inherited.model_copy(update={"ownership": lifecycle_ownership([Path(path)])})
    capacity = []
    reserve = RootQuarantineReserve(shared, "root-0", "producer", capacity.append)
    if inherited_first:
        assert reserve.try_acquire((inherited,))
    original = inherited.model_dump_json()
    events = []

    class CheckpointReached(Exception):
        pass

    def execute(row_id, attempt):
        events.append((row_id, attempt))
        if row_id == "inherited":
            return inherited
        return trial(tmp_path, row_id, attempt, completed=True)

    def checkpoint():
        raise CheckpointReached

    with pytest.raises(CheckpointReached):
        run_quarantined_trials(
            ["next-a", "next-b"] if inherited_first else ["inherited", "next-a", "next-b"],
            execute,
            lambda row: None,
            resolution,
            lambda row, result: None,
            physical_slots=1,
            reserve=reserve,
            wait_for_capacity=checkpoint,
            initial_quarantines=(inherited,) if inherited_first else (),
        )
    assert events == ([] if inherited_first else [("inherited", 0)]) + [("next-a", 0), ("next-b", 0)]
    assert inherited.model_dump_json() == original
    held = shared.snapshot().claims["root-0"]
    assert held.quarantines == (inherited,)
    assert held.resolutions["inherited/0"].retained_slots == 1
    assert not held.resolutions["inherited/0"].resolved
    assert len(shared.snapshot().claims) == 1


def test_terminal_root_handoff_preserves_unknown_capacity_and_fences_old_owner(tmp_path):
    shared = pool(tmp_path)
    record = trial(tmp_path, "held")
    old = shared.claim("root-0", "terminal-attempt", (record,)).claim
    moved = shared.handoff(old, "replacement-attempt")
    assert moved.decision is ReserveDecision.TRANSFERRED
    assert moved.claim.quarantines == old.quarantines
    assert moved.claim.resolutions == old.resolutions
    assert moved.claim.owner == "replacement-attempt"
    assert moved.claim.claim_id != old.claim_id
    assert len(shared.snapshot().claims) == 1
    with pytest.raises(ValueError, match="no longer owned"):
        shared.reconciled(old, record, resolution(record))
    assert len(shared.snapshot().claims) == 1
    released = shared.reconciled(moved.claim, record, resolution(record))
    assert released.decision is ReserveDecision.RELEASED
    assert not shared.snapshot().claims
