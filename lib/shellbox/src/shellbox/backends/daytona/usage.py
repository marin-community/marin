# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Estimate per-region Daytona headroom from sandbox inventory and supplied quotas."""

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import NotRequired, TypedDict

from daytona import AsyncDaytona, AsyncSandbox
from daytona_api_client_async import SandboxDesiredState, SandboxState

from shellbox.backends.daytona.connection import DaytonaConnection

REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_HEADROOM = 0.2
GB_TO_GIB = 10**9 / 2**30
# Provider quota rules: daytonaio/daytona@b5a5d9e, apps/api/src/organization/constants/.
COMPUTE_CHARGED_STATES = frozenset(
    (
        SandboxState.CREATING,
        SandboxState.RESTORING,
        SandboxState.STARTED,
        SandboxState.STARTING,
        SandboxState.STOPPING,
        SandboxState.PENDING_BUILD,
        SandboxState.BUILDING_SNAPSHOT,
        SandboxState.UNKNOWN,
        SandboxState.PULLING_SNAPSHOT,
        SandboxState.FORKING,
        SandboxState.PAUSING,
        SandboxState.RESUMING,
    )
)
CONDITIONAL_COMPUTE_CHARGED_STATES = frozenset((SandboxState.RESIZING, SandboxState.SNAPSHOTTING))
DISK_CHARGED_STATES = (
    COMPUTE_CHARGED_STATES
    | CONDITIONAL_COMPUTE_CHARGED_STATES
    | frozenset(
        (
            SandboxState.STOPPED,
            SandboxState.PAUSED,
            SandboxState.ARCHIVING,
        )
    )
)
UNCHARGED_STATES = frozenset(
    (
        SandboxState.ARCHIVED,
        SandboxState.BUILD_FAILED,
        SandboxState.DESTROYED,
        SandboxState.DESTROYING,
        SandboxState.ERROR,
    )
)


class CapacityReport(TypedDict):
    quota: float
    used: float
    remaining: float
    available: float


class RegionLimitsReport(TypedDict):
    cpu: int
    memory_gb: int
    disk_gb: int


class RegionCounts(TypedDict):
    total: int
    compute_charged: int
    disk_charged: int


class ConcurrencyReport(TypedDict):
    profile: dict[str, int]
    by_resource: dict[str, int]
    exceeds_per_sandbox_maximum: list[str]
    limit: int


class RegionReport(TypedDict):
    limits: RegionLimitsReport
    resources: dict[str, CapacityReport]
    sandboxes: RegionCounts
    max_concurrency: NotRequired[ConcurrencyReport]


class SandboxCounts(TypedDict):
    total: int
    started: int
    by_state: dict[str, int]


class UsageReport(TypedDict):
    organization_id: str | None
    observed_at: str
    usage_source: str
    headroom_fraction: float
    per_sandbox: dict[str, int]
    compute_charged_states: list[str]
    conditional_compute_charged_states: list[str]
    conditional_compute_desired_state: str
    disk_charged_states: list[str]
    uncharged_states: list[str]
    regions: dict[str, RegionReport]
    sandboxes: SandboxCounts


def validate_headroom(headroom: float) -> None:
    if not 0 <= headroom < 1:
        raise ValueError("Headroom must be in [0, 1)")


@dataclass(frozen=True)
class SandboxProfile:
    """Declared per-sandbox allocations in vCPU and whole GiB."""

    cpu: int
    memory_gib: int
    disk_gib: int

    def __post_init__(self) -> None:
        if min(self.cpu, self.memory_gib, self.disk_gib) <= 0:
            raise ValueError("Sandbox allocations must be positive")


@dataclass(frozen=True)
class RegionLimits:
    """Owner-supplied limits for one region in vCPU and decimal GB."""

    cpu: int
    memory_gb: int
    disk_gb: int


@dataclass(frozen=True)
class AccountLimits:
    """Owner-supplied quota limits: per-region capacity and the per-sandbox maximum."""

    regions: Mapping[str, RegionLimits]
    per_sandbox: SandboxProfile


@dataclass(frozen=True)
class ResourceCapacity:
    quota: float
    used: float

    @property
    def remaining(self) -> float:
        return max(self.quota - self.used, 0.0)

    def available(self, headroom: float) -> float:
        return max(self.quota * (1 - headroom) - self.used, 0.0)


@dataclass(frozen=True)
class RegionUsage:
    """One region's quota-charged inventory usage."""

    region: str
    sandboxes: int
    compute_sandboxes: int
    disk_sandboxes: int
    cpu: ResourceCapacity
    memory_gib: ResourceCapacity
    disk_gib: ResourceCapacity


@dataclass(frozen=True)
class AccountUsage:
    """One full inventory reading; an empty inventory has no organization ID."""

    organization_id: str | None
    limits: AccountLimits
    regions: tuple[RegionUsage, ...]
    sandbox_states: tuple[tuple[str, int], ...]
    observed_at: str


async def _inventory_sandboxes(client: AsyncDaytona) -> list[AsyncSandbox]:
    """Read every sandbox across all SDK result pages and account regions."""
    sandboxes = []
    seen = set()
    async for sandbox in client.list(request_timeout=REQUEST_TIMEOUT_SECONDS):
        if sandbox.id in seen:
            raise ValueError("Repeated sandbox ID in paginated inventory; retry the reading")
        seen.add(sandbox.id)
        sandboxes.append(sandbox)
    return sandboxes


def _region_usage(
    region: str,
    sandboxes: list[AsyncSandbox],
    compute: list[AsyncSandbox],
    disk: list[AsyncSandbox],
    limits: RegionLimits,
) -> RegionUsage:
    in_region = [sandbox for sandbox in sandboxes if sandbox.target == region]
    compute_region = [sandbox for sandbox in compute if sandbox.target == region]
    disk_region = [sandbox for sandbox in disk if sandbox.target == region]
    return RegionUsage(
        region=region,
        sandboxes=len(in_region),
        compute_sandboxes=len(compute_region),
        disk_sandboxes=len(disk_region),
        cpu=ResourceCapacity(limits.cpu, sum(sandbox.cpu for sandbox in compute_region)),
        memory_gib=ResourceCapacity(limits.memory_gb * GB_TO_GIB, sum(sandbox.memory for sandbox in compute_region)),
        disk_gib=ResourceCapacity(limits.disk_gb * GB_TO_GIB, sum(sandbox.disk for sandbox in disk_region)),
    )


async def account_usage_from_client(client: AsyncDaytona, limits: AccountLimits) -> AccountUsage:
    """Aggregate allocations per region using provider compute and disk quota rules.

    The provider enforces each region's quota separately, so usage is grouped by
    the inventory's ``target`` and never summed across regions. A region without
    recorded limits fails the reading instead of reporting free capacity.
    """
    sandboxes = await _inventory_sandboxes(client)
    organization_ids = {sandbox.organization_id for sandbox in sandboxes}
    if len(organization_ids) > 1 or (sandboxes and not all(organization_ids)):
        raise ValueError("Sandbox inventory must identify exactly one organization")
    for sandbox in sandboxes:
        if not sandbox.target:
            raise ValueError(f"Sandbox {sandbox.id} does not identify its region")
        if sandbox.state is None or sandbox.state == SandboxState.UNKNOWN_DEFAULT_OPEN_API:
            raise ValueError(f"Sandbox {sandbox.id} has an unclassified state")
        if sandbox.state in CONDITIONAL_COMPUTE_CHARGED_STATES and sandbox.desired_state in (
            None,
            SandboxDesiredState.UNKNOWN_DEFAULT_OPEN_API,
        ):
            raise ValueError(f"Sandbox {sandbox.id} lacks the desired state needed to account for compute")
    unrecorded = sorted({sandbox.target for sandbox in sandboxes} - set(limits.regions))
    if unrecorded:
        raise ValueError(f"No recorded limits for region(s) {', '.join(unrecorded)}; add them to ACCOUNT_LIMITS")
    compute = [
        sandbox
        for sandbox in sandboxes
        if sandbox.state in COMPUTE_CHARGED_STATES
        or (sandbox.state in CONDITIONAL_COMPUTE_CHARGED_STATES and sandbox.desired_state == SandboxDesiredState.STARTED)
    ]
    disk = [sandbox for sandbox in sandboxes if sandbox.state in DISK_CHARGED_STATES]
    for sandbox in disk:
        for value in (sandbox.cpu, sandbox.memory, sandbox.disk):
            if value is None or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Quota-charged sandbox {sandbox.id} lacks a positive resource allocation")
    regions = tuple(
        _region_usage(region, sandboxes, compute, disk, limits.regions[region]) for region in sorted(limits.regions)
    )
    states = Counter(sandbox.state.value for sandbox in sandboxes)
    return AccountUsage(
        organization_id=next(iter(organization_ids), None),
        limits=limits,
        regions=regions,
        sandbox_states=tuple(sorted(states.items())),
        observed_at=datetime.now(UTC).isoformat(),
    )


async def account_usage(connection: DaytonaConnection, limits: AccountLimits) -> AccountUsage:
    """Aggregate allocations from the selected account's full sandbox inventory."""
    async with connection.client_factory()() as client:
        return await account_usage_from_client(client, limits)


def _region_report(
    usage: RegionUsage,
    limits: RegionLimits,
    per_sandbox: dict[str, int],
    profile: dict[str, int] | None,
    exceeded: list[str] | None,
    headroom: float,
) -> RegionReport:
    resources = {"cpu": usage.cpu, "memory_gib": usage.memory_gib, "disk_gib": usage.disk_gib}
    report: RegionReport = {
        "limits": {"cpu": limits.cpu, "memory_gb": limits.memory_gb, "disk_gb": limits.disk_gb},
        "resources": {
            name: {
                "quota": capacity.quota,
                "used": capacity.used,
                "remaining": capacity.remaining,
                "available": capacity.available(headroom),
            }
            for name, capacity in resources.items()
        },
        "sandboxes": {
            "total": usage.sandboxes,
            "compute_charged": usage.compute_sandboxes,
            "disk_charged": usage.disk_sandboxes,
        },
    }
    if profile is not None:
        per_resource = {
            name: math.floor(capacity.available(headroom) / profile[name]) for name, capacity in resources.items()
        }
        report["max_concurrency"] = {
            "profile": profile,
            "by_resource": per_resource,
            "exceeds_per_sandbox_maximum": exceeded or [],
            "limit": 0 if exceeded else min(per_resource.values()),
        }
    return report


def usage_report(
    usage: AccountUsage, profile: SandboxProfile | None = None, *, headroom: float = DEFAULT_HEADROOM
) -> UsageReport:
    """Return per-region inventory usage and additional capacity below a quota reserve."""
    validate_headroom(headroom)
    requested = asdict(profile) if profile is not None else None
    maxima = asdict(usage.limits.per_sandbox)
    exceeded = [name for name, value in requested.items() if value > maxima[name]] if requested is not None else None
    states = dict(usage.sandbox_states)
    return {
        "organization_id": usage.organization_id,
        "observed_at": usage.observed_at,
        "usage_source": "sandbox_inventory",
        "headroom_fraction": headroom,
        "per_sandbox": maxima,
        "compute_charged_states": sorted(state.value for state in COMPUTE_CHARGED_STATES),
        "conditional_compute_charged_states": sorted(state.value for state in CONDITIONAL_COMPUTE_CHARGED_STATES),
        "conditional_compute_desired_state": SandboxDesiredState.STARTED.value,
        "disk_charged_states": sorted(state.value for state in DISK_CHARGED_STATES),
        "uncharged_states": sorted(state.value for state in UNCHARGED_STATES),
        "regions": {
            region.region: _region_report(
                region, usage.limits.regions[region.region], maxima, requested, exceeded, headroom
            )
            for region in usage.regions
        },
        "sandboxes": {
            "total": sum(states.values()),
            "started": states.get("started", 0),
            "by_state": states,
        },
    }
