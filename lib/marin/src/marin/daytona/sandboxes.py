# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Daytona sandbox inventory and confirmation-gated reclamation."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

DEAD_SANDBOX_STATES = frozenset({"error", "build_failed", "buildfailed"})


class DeletableSandbox(Protocol):
    id: str

    def delete(self) -> None: ...


@dataclass(frozen=True)
class SandboxAuditRow:
    """One normalized sandbox lifecycle record."""

    sandbox_id: str
    sandbox_name: str
    state: str
    created_at: datetime | None
    last_activity_at: datetime | None
    age_minutes: float | None
    delete_eligible: bool
    reason: str
    sandbox: DeletableSandbox


@dataclass(frozen=True)
class DeletionResult:
    """Outcome of one requested deletion."""

    sandbox_id: str
    error: str | None = None


def _as_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed
    return None


def _state_name(sandbox: object) -> str:
    state = getattr(sandbox, "state", "unknown")
    return str(getattr(state, "value", state)).lower()


def audit_sandboxes(
    sandboxes: Iterable[DeletableSandbox],
    *,
    stale_after_minutes: float,
    id_prefix: str | None = None,
    now: datetime | None = None,
) -> list[SandboxAuditRow]:
    """Classify stale started and terminal Daytona sandboxes for an audit."""

    if stale_after_minutes < 0:
        raise ValueError("stale_after_minutes must be non-negative")
    if id_prefix == "":
        raise ValueError("id_prefix must be non-empty when supplied")
    current = now or datetime.now(UTC)
    rows: list[SandboxAuditRow] = []
    for sandbox in sandboxes:
        created_at = _as_datetime(getattr(sandbox, "created_at", None))
        activity_at = _as_datetime(getattr(sandbox, "last_activity_at", None) or getattr(sandbox, "updated_at", None))
        reference = activity_at or created_at
        age_minutes = (current - reference).total_seconds() / 60 if reference else None
        state = _state_name(sandbox)
        sandbox_id = str(sandbox.id)
        sandbox_name = str(getattr(sandbox, "name", sandbox_id))
        in_scope = id_prefix is not None and sandbox_id.startswith(id_prefix)
        dead = state in DEAD_SANDBOX_STATES
        stale_started = state == "started" and age_minutes is not None and age_minutes > stale_after_minutes
        if not in_scope:
            reason = "outside explicit deletion scope"
        elif dead:
            reason = "terminal state"
        elif stale_started:
            reason = f"inactive for {age_minutes:.1f} minutes"
        else:
            reason = "active or within inactivity window"
        rows.append(
            SandboxAuditRow(
                sandbox_id=sandbox_id,
                sandbox_name=sandbox_name,
                state=state,
                created_at=created_at,
                last_activity_at=activity_at,
                age_minutes=age_minutes,
                delete_eligible=in_scope and (dead or stale_started),
                reason=reason,
                sandbox=sandbox,
            )
        )
    return sorted(rows, key=lambda row: (not row.delete_eligible, row.sandbox_id))


def delete_audited_sandboxes(
    rows: Iterable[SandboxAuditRow],
    *,
    confirm: Callable[[int], bool],
) -> list[DeletionResult]:
    """Delete selected audit rows only after a caller confirms their exact count."""

    selected = [row for row in rows if row.delete_eligible]
    if not selected or not confirm(len(selected)):
        return []
    results: list[DeletionResult] = []
    for row in selected:
        try:
            row.sandbox.delete()
        except Exception as exc:
            results.append(DeletionResult(row.sandbox_id, f"{type(exc).__name__}: {exc}"))
        else:
            results.append(DeletionResult(row.sandbox_id))
    return results
