# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from ops_workflow.source import WarningEvent, WarningSnapshot
from ops_workflow.state import (
    CaseCreation,
    CaseOutcome,
    CaseState,
    SignalState,
    SnapshotConflict,
    SnapshotDisposition,
    WorkflowState,
    apply_snapshot,
    signal_fingerprint,
)

NOW = datetime(2026, 7, 21, 16, 0, tzinfo=UTC)
COOLDOWN = timedelta(minutes=15)


def event(uid: str = "event-1", *, count: int = 1) -> WarningEvent:
    return WarningEvent(
        event_uid=uid,
        resource_version=str(count),
        object_uid="pod-1",
        namespace="kube-system",
        object_kind="Pod",
        object_name="node-local-dns",
        reason="DNSConfigForming",
        message="Nameserver limits were exceeded",
        first_seen_at=NOW - timedelta(hours=1),
        last_seen_at=NOW,
        count=count,
        reporting_controller="kubelet",
    )


def snapshot(observed_at: datetime, *events: WarningEvent) -> WarningSnapshot:
    return WarningSnapshot(cluster="cw-us-east-08a", observed_at=observed_at, events=events)


def apply(
    state: WorkflowState,
    value: WarningSnapshot,
    *,
    delivery_id: str,
    body_sha256: str,
) -> WorkflowState:
    case_ids = iter((f"case-{delivery_id}-1", f"case-{delivery_id}-2"))
    return apply_snapshot(
        state,
        value,
        delivery_id=delivery_id,
        body_sha256=body_sha256,
        now=NOW,
        reopen_cooldown=COOLDOWN,
        case_creation=CaseCreation.CREATE_CASES,
        case_id_factory=lambda: next(case_ids),
    ).state


def test_snapshot_high_water_rejects_stale_previously_unseen_event() -> None:
    fresh = apply_snapshot(
        WorkflowState.empty(),
        snapshot(NOW),
        delivery_id="fresh",
        body_sha256="fresh-body",
        now=NOW,
        reopen_cooldown=COOLDOWN,
        case_creation=CaseCreation.CREATE_CASES,
        case_id_factory=lambda: "unused",
    )

    stale = apply_snapshot(
        fresh.state,
        snapshot(NOW - timedelta(minutes=1), event("unseen")),
        delivery_id="stale",
        body_sha256="stale-body",
        now=NOW,
        reopen_cooldown=COOLDOWN,
        case_creation=CaseCreation.CREATE_CASES,
        case_id_factory=lambda: "must-not-be-created",
    )

    assert stale.disposition == SnapshotDisposition.STALE
    assert stale.state.signals == {}
    assert stale.state.streams["cw-us-east-08a"].delivery_id == "fresh"


def test_snapshot_equal_high_water_with_different_body_is_conflict() -> None:
    current = apply(WorkflowState.empty(), snapshot(NOW, event()), delivery_id="one", body_sha256="body-one")

    with pytest.raises(SnapshotConflict):
        apply_snapshot(
            current,
            snapshot(NOW, event(count=2)),
            delivery_id="two",
            body_sha256="body-two",
            now=NOW,
            reopen_cooldown=COOLDOWN,
            case_creation=CaseCreation.CREATE_CASES,
            case_id_factory=lambda: "case-two",
        )


def test_complete_snapshot_resolves_pending_case_and_reappearance_starts_new_generation() -> None:
    fingerprint = signal_fingerprint("cw-us-east-08a", "event-1")
    firing = apply(WorkflowState.empty(), snapshot(NOW, event()), delivery_id="one", body_sha256="body-one")

    resolved = apply(
        firing,
        snapshot(NOW + timedelta(minutes=1)),
        delivery_id="two",
        body_sha256="body-two",
    )

    assert resolved.signals[fingerprint].state == SignalState.RESOLVED
    closed_case = resolved.cases[(fingerprint, 1)]
    assert closed_case.state == CaseState.INVESTIGATED
    assert closed_case.outcome == CaseOutcome.NO_ACTION

    reappeared = apply(
        resolved,
        snapshot(NOW + timedelta(minutes=2), event(count=1)),
        delivery_id="three",
        body_sha256="body-three",
    )

    assert reappeared.signals[fingerprint].generation == 2
    assert reappeared.cases[(fingerprint, 2)].state == CaseState.PENDING
    assert reappeared.cases[(fingerprint, 2)].id != closed_case.id


def test_new_observation_reopens_investigated_case_after_cooldown() -> None:
    fingerprint = signal_fingerprint("cw-us-east-08a", "event-1")
    initial = apply(WorkflowState.empty(), snapshot(NOW, event()), delivery_id="one", body_sha256="body-one")
    case_key = (fingerprint, 1)
    investigated_at = NOW - timedelta(minutes=5)
    investigated = replace(
        initial.cases[case_key],
        state=CaseState.INVESTIGATED,
        investigated_at=investigated_at,
        outcome=CaseOutcome.NO_ACTION,
    )
    state = replace(initial, cases={case_key: investigated})

    updated = apply(
        state,
        snapshot(NOW + timedelta(minutes=1), event(count=2)),
        delivery_id="two",
        body_sha256="body-two",
    )

    reopened = updated.cases[case_key]
    assert reopened.state == CaseState.PENDING
    assert reopened.next_eligible_at == investigated_at + COOLDOWN


def test_new_observation_does_not_reopen_archived_generation() -> None:
    fingerprint = signal_fingerprint("cw-us-east-08a", "event-1")
    initial = apply(WorkflowState.empty(), snapshot(NOW, event()), delivery_id="one", body_sha256="body-one")
    case_key = (fingerprint, 1)
    archived = replace(initial.cases[case_key], state=CaseState.ARCHIVED, archived_at=NOW)
    state = replace(initial, cases={case_key: archived})

    updated = apply(
        state,
        snapshot(NOW + timedelta(minutes=1), event(count=2)),
        delivery_id="two",
        body_sha256="body-two",
    )

    assert updated.cases[case_key].state == CaseState.ARCHIVED
    assert len(updated.cases) == 1


def test_enabling_case_creation_materializes_case_for_already_firing_signal() -> None:
    baked = apply_snapshot(
        WorkflowState.empty(),
        snapshot(NOW, event()),
        delivery_id="bake",
        body_sha256="bake-body",
        now=NOW,
        reopen_cooldown=COOLDOWN,
        case_creation=CaseCreation.RECORD_SIGNALS_ONLY,
        case_id_factory=lambda: "must-not-be-created",
    ).state
    assert baked.cases == {}

    enabled = apply_snapshot(
        baked,
        snapshot(NOW + timedelta(minutes=1), event(count=2)),
        delivery_id="enabled",
        body_sha256="enabled-body",
        now=NOW + timedelta(minutes=1),
        reopen_cooldown=COOLDOWN,
        case_creation=CaseCreation.CREATE_CASES,
        case_id_factory=lambda: "case-created-after-bake",
    ).state

    fingerprint = signal_fingerprint("cw-us-east-08a", "event-1")
    assert enabled.cases[(fingerprint, 1)].id == "case-created-after-bake"
    assert enabled.cases[(fingerprint, 1)].state == CaseState.PENDING
