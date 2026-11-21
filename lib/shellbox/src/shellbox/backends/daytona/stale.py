# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select idle Daytona sandboxes from provider timestamps without I/O."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum


class DeletionState(StrEnum):
    NOT_DELETING = "not_deleting"
    DESTROYING = "destroying"
    DESTROYED = "destroyed"


@dataclass(frozen=True)
class SandboxActivity:
    id: str
    last_activity_at: str | None
    updated_at: str | None
    deletion_state: DeletionState


@dataclass(frozen=True)
class SandboxDecision:
    id: str
    reason: str
    last_used_at: datetime | None


@dataclass(frozen=True)
class IdleSelection:
    candidates: tuple[SandboxDecision, ...]
    skipped: tuple[SandboxDecision, ...]


def select_idle_sandboxes(
    sandboxes: Iterable[SandboxActivity], *, now: datetime, minimum_idle: timedelta
) -> IdleSelection:
    """Select sandboxes idle for at least the threshold; skip unusable timestamps.

    Use the freshest timezone-aware activity or update timestamp. Creation time
    does not establish inactivity. Sandboxes already being deleted or destroyed
    are skipped regardless of their timestamps.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must include a timezone")
    if minimum_idle <= timedelta(0):
        raise ValueError("minimum_idle must be positive")
    candidates = []
    skipped = []
    for sandbox in sandboxes:
        if sandbox.deletion_state != DeletionState.NOT_DELETING:
            skipped.append(SandboxDecision(sandbox.id, f"already {sandbox.deletion_state.value}", None))
            continue
        timestamps = []
        for value in (sandbox.last_activity_at, sandbox.updated_at):
            if not value:
                continue
            try:
                timestamp = datetime.fromisoformat(value)
            except ValueError:
                continue
            if timestamp.tzinfo is not None and timestamp.utcoffset() is not None:
                timestamps.append(timestamp.astimezone(UTC))
        if not timestamps:
            skipped.append(SandboxDecision(sandbox.id, "no usable activity or update timestamp", None))
            continue
        last_used = max(timestamps)
        if now - last_used >= minimum_idle:
            candidates.append(SandboxDecision(sandbox.id, "idle threshold reached", last_used))
        else:
            skipped.append(SandboxDecision(sandbox.id, "recent or future activity/update", last_used))
    return IdleSelection(tuple(candidates), tuple(skipped))
