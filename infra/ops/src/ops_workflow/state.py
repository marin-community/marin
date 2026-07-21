# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure Kubernetes signal and operator-case state transitions."""

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum

from ops_workflow.source import WarningEvent, WarningSnapshot


class SignalState(StrEnum):
    FIRING = "firing"
    RESOLVED = "resolved"


class CaseState(StrEnum):
    PENDING = "pending"
    INVESTIGATING = "investigating"
    WAITING_HUMAN = "waiting_human"
    INVESTIGATED = "investigated"
    FAILED = "failed"
    ARCHIVED = "archived"


class CaseOutcome(StrEnum):
    NO_ACTION = "no_action"
    ACTION_RECOMMENDED = "action_recommended"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class SnapshotDisposition(StrEnum):
    APPLIED = "applied"
    DUPLICATE = "duplicate"
    STALE = "stale"


class CaseCreation(StrEnum):
    RECORD_SIGNALS_ONLY = "record_signals_only"
    CREATE_CASES = "create_cases"


class SnapshotConflict(ValueError):
    """An equal-time snapshot disagrees with the stream high-water body."""


@dataclass(frozen=True)
class StreamWatermark:
    observed_at: datetime
    body_sha256: str
    delivery_id: str


@dataclass(frozen=True)
class Signal:
    fingerprint: str
    cluster: str
    event_uid: str
    generation: int
    state: SignalState
    resource_version: str
    namespace: str
    object_kind: str
    object_name: str
    reason: str
    message: str
    first_seen_at: datetime
    last_seen_at: datetime
    last_snapshot_at: datetime
    resolved_at: datetime | None
    occurrence_count: int
    latest_delivery_id: str


@dataclass(frozen=True)
class Case:
    id: str
    signal_fingerprint: str
    signal_generation: int
    state: CaseState
    title: str
    opened_at: datetime
    updated_at: datetime
    next_eligible_at: datetime
    investigated_at: datetime | None = None
    archived_at: datetime | None = None
    outcome: CaseOutcome | None = None
    summary: str | None = None


@dataclass(frozen=True)
class WorkflowState:
    streams: Mapping[str, StreamWatermark]
    signals: Mapping[str, Signal]
    cases: Mapping[tuple[str, int], Case]

    @classmethod
    def empty(cls) -> "WorkflowState":
        return cls(streams={}, signals={}, cases={})


@dataclass(frozen=True)
class StateTransition:
    kind: str
    signal_fingerprint: str
    case_id: str | None


@dataclass(frozen=True)
class AppliedSnapshot:
    state: WorkflowState
    disposition: SnapshotDisposition
    transitions: tuple[StateTransition, ...]


@dataclass(frozen=True)
class _CaseMaterialization:
    case: Case | None
    created: bool


@dataclass
class _StateUpdate:
    signals: dict[str, Signal]
    cases: dict[tuple[str, int], Case]
    transitions: list[StateTransition]


@dataclass(frozen=True)
class _SnapshotContext:
    now: datetime
    reopen_cooldown: timedelta
    case_creation: CaseCreation
    case_id_factory: Callable[[], str]


def signal_fingerprint(cluster: str, event_uid: str) -> str:
    return hashlib.sha256(f"{cluster}\n{event_uid}".encode()).hexdigest()


def apply_snapshot(
    state: WorkflowState,
    snapshot: WarningSnapshot,
    *,
    delivery_id: str,
    body_sha256: str,
    now: datetime,
    reopen_cooldown: timedelta,
    case_creation: CaseCreation,
    case_id_factory: Callable[[], str],
) -> AppliedSnapshot:
    """Apply one complete cluster snapshot to immutable workflow state."""

    prior_disposition = _prior_snapshot_disposition(state, snapshot, body_sha256)
    if prior_disposition is not None:
        return AppliedSnapshot(state, prior_disposition, ())

    streams = dict(state.streams)
    update = _StateUpdate(signals=dict(state.signals), cases=dict(state.cases), transitions=[])
    context = _SnapshotContext(
        now=now,
        reopen_cooldown=reopen_cooldown,
        case_creation=case_creation,
        case_id_factory=case_id_factory,
    )

    seen_fingerprints = _apply_observed_events(
        update,
        snapshot,
        delivery_id=delivery_id,
        context=context,
    )

    _resolve_absent_signals(
        update,
        snapshot,
        seen_fingerprints=seen_fingerprints,
        delivery_id=delivery_id,
        now=now,
    )

    streams[snapshot.cluster] = StreamWatermark(
        observed_at=snapshot.observed_at,
        body_sha256=body_sha256,
        delivery_id=delivery_id,
    )
    return AppliedSnapshot(
        state=WorkflowState(streams=streams, signals=update.signals, cases=update.cases),
        disposition=SnapshotDisposition.APPLIED,
        transitions=tuple(update.transitions),
    )


def _prior_snapshot_disposition(
    state: WorkflowState, snapshot: WarningSnapshot, body_sha256: str
) -> SnapshotDisposition | None:
    # A stale snapshot can contain a UID absent from every newer snapshot, so
    # stream freshness must be checked before event identity.
    watermark = state.streams.get(snapshot.cluster)
    if watermark is None or snapshot.observed_at > watermark.observed_at:
        return None
    if snapshot.observed_at < watermark.observed_at:
        return SnapshotDisposition.STALE
    if body_sha256 != watermark.body_sha256:
        raise SnapshotConflict("equal observed_at has a different snapshot body")
    return SnapshotDisposition.DUPLICATE


def _apply_observed_events(
    update: _StateUpdate,
    snapshot: WarningSnapshot,
    *,
    delivery_id: str,
    context: _SnapshotContext,
) -> set[str]:
    seen_fingerprints: set[str] = set()
    for event in snapshot.events:
        transition = _apply_observed_event(
            update,
            snapshot,
            event,
            delivery_id=delivery_id,
            context=context,
        )
        seen_fingerprints.add(transition.signal_fingerprint)
        update.transitions.append(transition)
    return seen_fingerprints


def _apply_observed_event(
    update: _StateUpdate,
    snapshot: WarningSnapshot,
    event: WarningEvent,
    *,
    delivery_id: str,
    context: _SnapshotContext,
) -> StateTransition:
    fingerprint = signal_fingerprint(snapshot.cluster, event.event_uid)
    existing = update.signals.get(fingerprint)
    if existing is None:
        signal = _observed_signal(fingerprint, snapshot, event, delivery_id, existing=None)
        update.signals[fingerprint] = signal
        materialized = _case_for_signal(
            update.cases, signal, context.case_creation, context.now, context.case_id_factory
        )
        return StateTransition("signal_created", fingerprint, materialized.case.id if materialized.case else None)
    if existing.state == SignalState.RESOLVED:
        signal = _reactivated_signal(existing, snapshot, event, delivery_id)
        update.signals[fingerprint] = signal
        materialized = _case_for_signal(
            update.cases, signal, context.case_creation, context.now, context.case_id_factory
        )
        return StateTransition("signal_reactivated", fingerprint, materialized.case.id if materialized.case else None)

    signal = _observed_signal(fingerprint, snapshot, event, delivery_id, existing=existing)
    update.signals[fingerprint] = signal
    return _updated_case_transition(
        update.cases,
        signal,
        context=context,
    )


def _updated_case_transition(
    cases: dict[tuple[str, int], Case],
    signal: Signal,
    *,
    context: _SnapshotContext,
) -> StateTransition:
    case_key = (signal.fingerprint, signal.generation)
    materialized = _case_for_signal(cases, signal, context.case_creation, context.now, context.case_id_factory)
    case = materialized.case
    if materialized.created:
        assert case is not None
        return StateTransition("case_created", signal.fingerprint, case.id)
    if case is None or case.state != CaseState.INVESTIGATED:
        return StateTransition("signal_updated", signal.fingerprint, case.id if case else None)

    eligible_at = max(context.now, (case.investigated_at or context.now) + context.reopen_cooldown)
    reopened = replace(
        case,
        state=CaseState.PENDING,
        updated_at=context.now,
        next_eligible_at=eligible_at,
        outcome=None,
        summary=None,
    )
    cases[case_key] = reopened
    return StateTransition("case_reopened", signal.fingerprint, reopened.id)


def _resolve_absent_signals(
    update: _StateUpdate,
    snapshot: WarningSnapshot,
    *,
    seen_fingerprints: set[str],
    delivery_id: str,
    now: datetime,
) -> None:
    for fingerprint, signal in tuple(update.signals.items()):
        if signal.cluster != snapshot.cluster or signal.state != SignalState.FIRING:
            continue
        if fingerprint in seen_fingerprints:
            continue
        update.signals[fingerprint] = replace(
            signal,
            state=SignalState.RESOLVED,
            last_snapshot_at=snapshot.observed_at,
            resolved_at=snapshot.observed_at,
            latest_delivery_id=delivery_id,
        )
        case_key = (fingerprint, signal.generation)
        case = update.cases.get(case_key)
        if case is not None and case.state == CaseState.PENDING:
            closed = replace(
                case,
                state=CaseState.INVESTIGATED,
                updated_at=now,
                investigated_at=now,
                outcome=CaseOutcome.NO_ACTION,
                summary="Source resolved before investigation.",
            )
            update.cases[case_key] = closed
            update.transitions.append(StateTransition("case_closed_before_investigation", fingerprint, closed.id))
        else:
            update.transitions.append(StateTransition("signal_resolved", fingerprint, case.id if case else None))


def _observed_signal(
    fingerprint: str,
    snapshot: WarningSnapshot,
    event: WarningEvent,
    delivery_id: str,
    *,
    existing: Signal | None,
) -> Signal:
    return Signal(
        fingerprint=fingerprint,
        cluster=snapshot.cluster,
        event_uid=event.event_uid,
        generation=existing.generation if existing else 1,
        state=SignalState.FIRING,
        resource_version=event.resource_version,
        namespace=event.namespace,
        object_kind=event.object_kind,
        object_name=event.object_name,
        reason=event.reason,
        message=event.message,
        first_seen_at=min(existing.first_seen_at, event.first_seen_at) if existing else event.first_seen_at,
        last_seen_at=max(existing.last_seen_at, event.last_seen_at) if existing else event.last_seen_at,
        last_snapshot_at=snapshot.observed_at,
        resolved_at=None,
        occurrence_count=event.count,
        latest_delivery_id=delivery_id,
    )


def _reactivated_signal(
    existing: Signal,
    snapshot: WarningSnapshot,
    event: WarningEvent,
    delivery_id: str,
) -> Signal:
    return replace(
        _observed_signal(existing.fingerprint, snapshot, event, delivery_id, existing=existing),
        generation=existing.generation + 1,
        state=SignalState.FIRING,
        first_seen_at=event.first_seen_at,
        resolved_at=None,
    )


def _new_case(signal: Signal, now: datetime, case_id_factory: Callable[[], str]) -> Case:
    return Case(
        id=case_id_factory(),
        signal_fingerprint=signal.fingerprint,
        signal_generation=signal.generation,
        state=CaseState.PENDING,
        title=f"{signal.reason}: {signal.object_kind}/{signal.object_name}",
        opened_at=now,
        updated_at=now,
        next_eligible_at=now,
    )


def _case_for_signal(
    cases: dict[tuple[str, int], Case],
    signal: Signal,
    case_creation: CaseCreation,
    now: datetime,
    case_id_factory: Callable[[], str],
) -> _CaseMaterialization:
    case_key = (signal.fingerprint, signal.generation)
    case = cases.get(case_key)
    if case is not None or case_creation == CaseCreation.RECORD_SIGNALS_ONLY:
        return _CaseMaterialization(case=case, created=False)
    case = _new_case(signal, now, case_id_factory)
    cases[case_key] = case
    return _CaseMaterialization(case=case, created=True)
