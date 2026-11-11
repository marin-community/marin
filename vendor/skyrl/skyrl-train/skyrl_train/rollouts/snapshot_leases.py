"""Bound distinct snapshot identities across concurrent rollout workers."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum


class SnapshotPhase(StrEnum):
    BUILDING = "building"
    READY = "ready"
    DELETING = "deleting"
    UNCERTAIN = "uncertain"


@dataclass
class SnapshotEntry:
    phase: SnapshotPhase
    references: set[str] = field(default_factory=set)
    name: str | None = None


class SnapshotLeasePool:
    """Reserve uncertain provider operations until explicit reconciliation.

    One shared actor owns this pool. Callbacks return after provider state is
    confirmed. Release is allowed only after the associated sandbox is deleted.
    """

    def __init__(self, capacity: int, build: Callable[[str], Awaitable[str]], delete: Callable[[str], Awaitable[None]]):
        if capacity < 1:
            raise ValueError("Snapshot capacity must be positive")
        self.capacity = capacity
        self.build = build
        self.delete = delete
        self.entries: dict[str, SnapshotEntry] = {}
        self.leases: dict[str, str] = {}
        self.changed = asyncio.Condition()

    async def acquire(self, identity: str, lease: str) -> str:
        async with self.changed:
            if lease in self.leases:
                raise ValueError("Snapshot lease already acquired")
            await self.changed.wait_for(
                lambda: (identity in self.entries and self.entries[identity].phase in (SnapshotPhase.BUILDING, SnapshotPhase.READY))
                or (identity not in self.entries and len(self.entries) < self.capacity)
            )
            if identity in self.entries:
                entry = self.entries[identity]
                entry.references.add(lease)
                self.leases[lease] = identity
                await self.changed.wait_for(lambda: entry.phase is not SnapshotPhase.BUILDING)
                if entry.phase is not SnapshotPhase.READY:
                    raise RuntimeError("Shared snapshot build requires reconciliation")
                assert entry.name is not None
                return entry.name
            entry = SnapshotEntry(SnapshotPhase.BUILDING, {lease})
            self.entries[identity] = entry
            self.leases[lease] = identity
        try:
            name = await self.build(identity)
        except BaseException:
            async with self.changed:
                entry.phase = SnapshotPhase.UNCERTAIN
                self.changed.notify_all()
            raise
        async with self.changed:
            entry.name = name
            entry.phase = SnapshotPhase.READY
            self.changed.notify_all()
            return name

    async def release(self, lease: str) -> None:
        async with self.changed:
            identity = self.leases[lease]
            entry = self.entries[identity]
            if entry.phase is not SnapshotPhase.READY:
                raise RuntimeError("Uncertain snapshot requires explicit reconciliation")
            entry.references.remove(lease)
            del self.leases[lease]
            if entry.references:
                return
            entry.phase = SnapshotPhase.DELETING
            assert entry.name is not None
            name = entry.name
        try:
            await self.delete(name)
        except BaseException:
            async with self.changed:
                entry.phase = SnapshotPhase.UNCERTAIN
                self.changed.notify_all()
            raise
        async with self.changed:
            del self.entries[identity]
            self.changed.notify_all()

    async def inventory(self) -> dict[str, dict]:
        """Return ownership evidence without provider credentials."""
        async with self.changed:
            return {
                identity: {"name": entry.name, "phase": entry.phase.value, "leases": sorted(entry.references)}
                for identity, entry in self.entries.items()
            }
