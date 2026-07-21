# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable agent-turn lifecycle transitions."""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum

from ops_workflow.state import CaseOutcome


class TurnKind(StrEnum):
    AUTOMATIC = "automatic"
    QUESTION = "question"
    FOLLOW_UP = "follow_up"


class TurnState(StrEnum):
    QUEUED = "queued"
    LAUNCHING = "launching"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


TERMINAL_TURN_STATES = frozenset((TurnState.SUCCEEDED, TurnState.FAILED, TurnState.INTERRUPTED, TurnState.CANCELLED))
ACTIVE_TURN_STATES = frozenset((TurnState.LAUNCHING, TurnState.RUNNING))


class TurnTransitionError(ValueError):
    """A requested turn transition is not allowed from the current state."""


@dataclass(frozen=True)
class ResultEvidence:
    claim: str
    source: str


@dataclass(frozen=True)
class OpsResult:
    schema_version: int
    case_id: str
    ops_turn_id: str
    outcome: CaseOutcome
    summary: str
    evidence: tuple[ResultEvidence, ...]
    action_taken: str
    recommended_next_step: str


@dataclass(frozen=True)
class AgentTurn:
    id: str
    session_id: str
    kind: TurnKind
    state: TurnState
    priority: int
    requested_by: str
    client_request_id: str
    prompt: str
    available_at: datetime
    created_at: datetime
    attempt: int = 1
    retry_of: str | None = None
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    loom_turn_number: int | None = None
    started_at: datetime | None = None
    deadline_at: datetime | None = None
    completed_at: datetime | None = None
    result: OpsResult | None = None
    error: str | None = None

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_TURN_STATES


def claim_turn(
    turn: AgentTurn,
    *,
    now: datetime,
    lease_owner: str,
    lease_duration: timedelta,
    turn_timeout: timedelta,
) -> AgentTurn:
    """Acquire a queued turn's durable lease and runtime deadline."""

    _require_state(turn, TurnState.QUEUED)
    if turn.available_at > now:
        raise TurnTransitionError("turn is not eligible yet")
    return replace(
        turn,
        state=TurnState.LAUNCHING,
        lease_owner=lease_owner,
        lease_expires_at=now + lease_duration,
        started_at=now,
        deadline_at=now + turn_timeout,
    )


def acknowledge_turn(turn: AgentTurn, *, loom_turn_number: int, now: datetime, lease_duration: timedelta) -> AgentTurn:
    """Record Loom's idempotent acknowledgement for a launched turn."""

    _require_state(turn, TurnState.LAUNCHING)
    if loom_turn_number < 1:
        raise TurnTransitionError("Loom turn number must be positive")
    return replace(
        turn,
        state=TurnState.RUNNING,
        loom_turn_number=loom_turn_number,
        lease_expires_at=now + lease_duration,
    )


def renew_turn_lease(turn: AgentTurn, *, now: datetime, lease_duration: timedelta) -> AgentTurn:
    """Renew the lease without extending the durable runtime deadline."""

    if turn.state not in ACTIVE_TURN_STATES:
        raise TurnTransitionError(f"cannot renew a {turn.state} turn")
    return replace(turn, lease_expires_at=now + lease_duration)


def complete_turn(turn: AgentTurn, *, now: datetime, result: OpsResult) -> AgentTurn:
    """Complete a running turn with a validated structured result."""

    _require_state(turn, TurnState.RUNNING)
    return replace(
        turn,
        state=TurnState.SUCCEEDED,
        lease_owner=None,
        lease_expires_at=None,
        completed_at=now,
        result=result,
        error=None,
    )


def fail_turn(turn: AgentTurn, *, now: datetime, error: str) -> AgentTurn:
    """Finish an active turn with a bounded operational error."""

    return _finish_active_turn(turn, state=TurnState.FAILED, now=now, error=error)


def interrupt_turn(turn: AgentTurn, *, now: datetime, error: str) -> AgentTurn:
    """Record an exact active-turn interruption."""

    return _finish_active_turn(turn, state=TurnState.INTERRUPTED, now=now, error=error)


def _finish_active_turn(turn: AgentTurn, *, state: TurnState, now: datetime, error: str) -> AgentTurn:
    if turn.state not in ACTIVE_TURN_STATES:
        raise TurnTransitionError(f"cannot finish a {turn.state} turn")
    return replace(
        turn,
        state=state,
        lease_owner=None,
        lease_expires_at=None,
        completed_at=now,
        error=error,
    )


def cancel_queued_turn(turn: AgentTurn, *, now: datetime, reason: str) -> AgentTurn:
    """Cancel an automatic turn whose source resolved before launch."""

    _require_state(turn, TurnState.QUEUED)
    if turn.kind != TurnKind.AUTOMATIC:
        raise TurnTransitionError("only an automatic turn can be cancelled by source resolution")
    return replace(turn, state=TurnState.CANCELLED, completed_at=now, error=reason)


def retry_turn(turn: AgentTurn, *, new_id: str, client_request_id: str, now: datetime) -> AgentTurn:
    """Create the next immutable attempt in a failed/interrupted lineage."""

    if turn.state not in (TurnState.FAILED, TurnState.INTERRUPTED):
        raise TurnTransitionError(f"cannot retry a {turn.state} turn")
    if turn.attempt >= 3:
        raise TurnTransitionError("turn retry budget is exhausted")
    return AgentTurn(
        id=new_id,
        session_id=turn.session_id,
        kind=turn.kind,
        state=TurnState.QUEUED,
        priority=turn.priority,
        requested_by=turn.requested_by,
        client_request_id=client_request_id,
        prompt=turn.prompt,
        available_at=now,
        created_at=now,
        attempt=turn.attempt + 1,
        retry_of=turn.id,
    )


def _require_state(turn: AgentTurn, expected: TurnState) -> None:
    if turn.state != expected:
        raise TurnTransitionError(f"expected {expected}, got {turn.state}")
