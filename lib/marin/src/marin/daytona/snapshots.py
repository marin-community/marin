# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Daytona snapshot audit and confirmation-gated reclamation."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

PROTECTED_SNAPSHOT_STATES = frozenset({"building", "pending", "pulling", "removing"})


class SnapshotService(Protocol):
    def list(self, *, page: int, limit: int) -> Any: ...

    def delete(self, snapshot: object) -> None: ...


@dataclass(frozen=True)
class SnapshotAuditRow:
    """One snapshot's normalized retention verdict."""

    snapshot_id: str
    name: str
    state: str
    idle_days: float | None
    protected: bool
    delete_eligible: bool
    snapshot: object


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


def list_snapshots(service: SnapshotService, page_size: int = 100) -> list[object]:
    """Fetch all snapshot pages from a Daytona snapshot service."""

    if page_size <= 0:
        raise ValueError("page_size must be positive")
    page, total_pages, snapshots = 1, 1, []
    while page <= total_pages:
        result = service.list(page=page, limit=page_size)
        snapshots.extend(result.items)
        total_pages = int(getattr(result, "total_pages", 1) or 1)
        page += 1
    return snapshots


def audit_snapshots(
    snapshots: list[object],
    *,
    stale_after_days: float,
    name_prefix: str | None,
    now: datetime | None = None,
) -> list[SnapshotAuditRow]:
    """Select idle, non-transitional snapshots within an explicit namespace."""

    if stale_after_days < 0:
        raise ValueError("stale_after_days must be non-negative")
    if name_prefix == "":
        raise ValueError("name_prefix must be non-empty when supplied")
    current = now or datetime.now(UTC)
    rows: list[SnapshotAuditRow] = []
    for snapshot in snapshots:
        name = str(snapshot.name)
        raw_state = getattr(snapshot, "state", "unknown")
        state = str(getattr(raw_state, "value", raw_state)).lower()
        last_used = _as_datetime(getattr(snapshot, "last_used_at", None))
        created = _as_datetime(getattr(snapshot, "created_at", None))
        reference = last_used or created
        idle_days = (current - reference).total_seconds() / 86400 if reference else None
        protected = state in PROTECTED_SNAPSHOT_STATES
        in_scope = name_prefix is not None and name.startswith(name_prefix)
        delete_eligible = bool(in_scope and not protected and idle_days is not None and idle_days > stale_after_days)
        rows.append(
            SnapshotAuditRow(
                snapshot_id=str(snapshot.id),
                name=name,
                state=state,
                idle_days=idle_days,
                protected=protected,
                delete_eligible=delete_eligible,
                snapshot=snapshot,
            )
        )
    return sorted(rows, key=lambda row: (not row.delete_eligible, row.name))


def delete_audited_snapshots(
    service: SnapshotService,
    rows: list[SnapshotAuditRow],
    *,
    confirm: Callable[[int], bool],
) -> list[str]:
    """Delete eligible snapshots only after confirmation of the exact count."""

    selected = [row for row in rows if row.delete_eligible]
    if not selected or not confirm(len(selected)):
        return []
    for row in selected:
        service.delete(row.snapshot)
    return [row.snapshot_id for row in selected]
