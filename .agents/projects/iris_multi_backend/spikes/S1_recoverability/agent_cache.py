# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""S1 spike: a model of the per-cluster agent's recoverable worker-state cache.

This is throwaway spike code. It models the worker-state the design moves out of
the root DB into the agent's volatile cache (design.md governing invariant):

    worker roster, per-worker health, slice inventory, attempt->worker binding,
    allocated host ports, local placement.

The cache reuses the REAL liveness + port classes so the demonstration exercises
production logic, not a re-implementation:

    iris.cluster.controller.worker_health.WorkerHealthTracker   (health)
    iris.cluster.worker.port_allocator.PortAllocator            (ports)

The roster / slice / binding stores are small dataclasses here because in
production they are SQLAlchemy rows + the cloud SliceHandle protocol; their
*content* is what matters for recoverability, not their storage engine.

The claim under test: this whole cache is a pure function of three sources

    (1) worker re-registration   -- workers dial the agent, report identity AND
                                    their currently-running attempts (substrate
                                    discovery: attempt_uid + worker_id [+ ports])
    (2) list_all_slices()        -- cloud ground-truth slice inventory
    (3) the root's Poll desired-set  -- which attempt_uids are still wanted

so wiping the cache and rebuilding from those three reproduces it exactly, with
per-worker health counters reset as the only allowed difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from iris.cluster.backends.types import CloudSliceState
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.types import AttemptUid, WorkerId
from iris.cluster.worker.port_allocator import PortAllocator
from iris.rpc import job_pb2

# ---------------------------------------------------------------------------
# The three recovery sources
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerRegistration:
    """Source (1a): what a worker re-sends when it dials the agent on boot.

    Mirrors controller_pb2.Controller.RegisterRequest fields consumed by
    ops.worker.register (worker_id, address, slice_id, scale_group, metadata).
    """

    worker_id: WorkerId
    address: str
    slice_id: str
    scale_group: str
    metadata: job_pb2.WorkerMetadata


@dataclass(frozen=True)
class DiscoveredAttempt:
    """Source (1b): a running attempt a worker reports via substrate discovery.

    This is the recoverability-relevant projection of runtime.types
    DiscoveredContainer, PLUS the ``ports`` the production type is missing
    (the hole). ``ports`` is the attempt's allocated host ports keyed by name;
    empty dict models today's DiscoveredContainer, which has no port footprint.
    """

    attempt_uid: AttemptUid
    worker_id: WorkerId
    ports: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ListedSliceLite:
    """Source (2): one entry of list_all_slices() -- cloud slice ground truth."""

    slice_id: str
    scale_group: str
    zone: str
    state: CloudSliceState
    worker_ids: frozenset[WorkerId]


@dataclass(frozen=True)
class RecoverySources:
    registrations: list[WorkerRegistration]
    discoveries: list[DiscoveredAttempt]
    slices: list[ListedSliceLite]
    desired_uids: frozenset[AttemptUid]  # source (3): the root's Poll desired-set


# ---------------------------------------------------------------------------
# The cache records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerRecord:
    worker_id: WorkerId
    address: str
    slice_id: str
    scale_group: str
    device_variant: str


@dataclass(frozen=True)
class SliceRecord:
    slice_id: str
    scale_group: str
    zone: str
    state: CloudSliceState
    worker_ids: frozenset[WorkerId]


def _device_variant(md: job_pb2.WorkerMetadata) -> str:
    if md.device.HasField("tpu"):
        return md.device.tpu.variant
    if md.device.HasField("gpu"):
        return md.device.gpu.variant
    return ""


def _reserve(allocator: PortAllocator, ports: list[int]) -> None:
    """Re-reserve specific ports recovered from a substrate stamp.

    PortAllocator only exposes allocate()/release() today; recovering a known
    port set needs an explicit "reserve these exact numbers". This helper stands
    in for a ``PortAllocator.reserve(ports)`` method the fix must add (see
    SPIKE.md spec implications).
    """
    with allocator._lock:  # noqa: SLF001 -- spike stand-in for PortAllocator.reserve()
        allocator._allocated.update(ports)  # noqa: SLF001


class AgentWorkerCache:
    """The agent's in-memory recoverable worker-state cache (design.md)."""

    def __init__(self) -> None:
        self.roster: dict[WorkerId, WorkerRecord] = {}
        self.slices: dict[str, SliceRecord] = {}
        self.bindings: dict[AttemptUid, WorkerId] = {}
        self.attempt_ports: dict[AttemptUid, list[int]] = {}
        self.port_allocators: dict[WorkerId, PortAllocator] = {}
        self.health = WorkerHealthTracker()

    # -- mutation used both to populate and to rebuild ----------------------

    def add_worker(self, reg: WorkerRegistration, *, now_ms: int) -> None:
        self.roster[reg.worker_id] = WorkerRecord(
            worker_id=reg.worker_id,
            address=reg.address,
            slice_id=reg.slice_id,
            scale_group=reg.scale_group,
            device_variant=_device_variant(reg.metadata),
        )
        self.port_allocators.setdefault(reg.worker_id, PortAllocator())
        # Same call the real controller makes in ops.worker.register: seed
        # liveness as healthy with a fresh heartbeat (counters start at zero).
        self.health.register(reg.worker_id, now_ms=now_ms)

    def add_slice(self, listed: ListedSliceLite) -> None:
        self.slices[listed.slice_id] = SliceRecord(
            slice_id=listed.slice_id,
            scale_group=listed.scale_group,
            zone=listed.zone,
            state=listed.state,
            worker_ids=listed.worker_ids,
        )

    def bind_attempt(self, uid: AttemptUid, worker_id: WorkerId, ports: list[int]) -> None:
        """Bind a running attempt to a worker and reserve its ports.

        Used at populate time to allocate fresh ports, and at rebuild time to
        re-reserve the exact ports recovered from the substrate stamp.
        """
        self.bindings[uid] = worker_id
        self.attempt_ports[uid] = list(ports)
        _reserve(self.port_allocators[worker_id], ports)

    # -- equivalence signature (health deliberately excluded) ---------------

    def signature(self, scope_uids: frozenset[AttemptUid]) -> dict:
        """Functional signature for equivalence checks.

        Bindings/ports are scoped to ``scope_uids`` (the desired-set): an
        attempt that is running but no longer desired is fenced (killed) on
        reconcile, so it is excluded from both the original and rebuilt
        signatures. Health counters are intentionally absent -- a reset is the
        only allowed difference.
        """
        return {
            "roster": {wid: rec for wid, rec in sorted(self.roster.items())},
            "slices": {sid: rec for sid, rec in sorted(self.slices.items())},
            "bindings": {uid: self.bindings[uid] for uid in sorted(scope_uids) if uid in self.bindings},
            "attempt_ports": {
                uid: tuple(sorted(self.attempt_ports[uid])) for uid in sorted(scope_uids) if uid in self.attempt_ports
            },
            "reserved_ports": {
                wid: tuple(sorted(alloc._allocated))  # noqa: SLF001 -- read the real allocator's set
                for wid, alloc in sorted(self.port_allocators.items())
            },
        }

    def health_snapshot(self) -> dict[WorkerId, tuple[int, int]]:
        return self.health.snapshot()


# ---------------------------------------------------------------------------
# Rebuild: the pure function of the three sources
# ---------------------------------------------------------------------------


def rebuild_from_sources(sources: RecoverySources, *, now_ms: int, recover_ports: bool) -> AgentWorkerCache:
    """Rebuild the worker-state cache purely from the three recovery sources.

    ``recover_ports`` toggles the fix: when True a discovery's stamped ports are
    re-reserved on adopt (design.md substrate-stamping). When False we model
    today's DiscoveredContainer / TaskAttempt.adopt(), which carry no ports --
    the latent double-allocation hole.
    """
    cache = AgentWorkerCache()

    # (1a) roster + health from worker re-registration.
    for reg in sources.registrations:
        cache.add_worker(reg, now_ms=now_ms)

    # (2) slice inventory from list_all_slices().
    for listed in sources.slices:
        cache.add_slice(listed)

    # (1b) x (3): attempt->worker bindings from substrate discovery, gated by
    # the Poll desired-set. A discovered attempt absent from desired is a zombie
    # -- fence-by-absence kills it, so it is NOT bound and its ports are dropped.
    for disc in sources.discoveries:
        if disc.attempt_uid not in sources.desired_uids:
            continue  # fenced: running but no longer desired
        if disc.worker_id not in cache.port_allocators:
            continue  # worker did not re-register; nothing to bind to
        ports = list(disc.ports.values()) if recover_ports else []
        cache.bind_attempt(disc.attempt_uid, disc.worker_id, ports)

    return cache
