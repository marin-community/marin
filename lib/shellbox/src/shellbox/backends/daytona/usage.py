# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read one Daytona account's quota, live usage and sandbox inventory.

Reports what an account can still run, so agents size concurrency from live
numbers instead of static tables: the provider's CPU, memory, disk, snapshot
and volume quotas, current use, and sandbox counts by state.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from daytona import AsyncDaytona, AsyncSandbox
from daytona.common.snapshot import Snapshot
from daytona.common.volume import Volume
from daytona_api_client_async import (
    OrganizationsApi,
    OrganizationUsageOverview,
    RegionUsageOverview,
)

from shellbox.backends.daytona.connection import DaytonaConnection
from shellbox.backends.daytona.snapshot import list_snapshots

REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_HEADROOM = 0.2
DEFAULT_SANDBOX_CLASS = "container"


@dataclass(frozen=True)
class ResourceCapacity:
    """Quota and live use for one resource; remaining is never negative."""

    quota: float
    used: float

    @property
    def remaining(self) -> float:
        return max(self.quota - self.used, 0.0)


@dataclass(frozen=True)
class RegionCapacity:
    """One region's reported quota, use and per-sandbox maxima."""

    region_id: str
    sandbox_class: str
    cpu: ResourceCapacity
    memory_gib: ResourceCapacity
    disk_gib: ResourceCapacity
    gpu: ResourceCapacity
    max_cpu_per_sandbox: float | None
    max_memory_gib_per_sandbox: float | None
    max_disk_gib_per_sandbox: float | None


@dataclass(frozen=True)
class SandboxProfile:
    """The per-sandbox resource request used to size concurrency."""

    cpu: int
    memory_gib: float
    disk_gib: float


@dataclass(frozen=True)
class AccountUsage:
    """One Daytona organization's quota, live usage and sandbox inventory."""

    organization_id: str
    regions: tuple[RegionCapacity, ...]
    snapshots: ResourceCapacity
    volumes: ResourceCapacity
    sandboxes: int
    sandbox_states: tuple[tuple[str, int], ...]

    @property
    def cpu(self) -> ResourceCapacity:
        return _summed(region.cpu for region in self.regions)

    @property
    def memory_gib(self) -> ResourceCapacity:
        return _summed(region.memory_gib for region in self.regions)

    @property
    def disk_gib(self) -> ResourceCapacity:
        return _summed(region.disk_gib for region in self.regions)


def _summed(capacities: Iterable[ResourceCapacity]) -> ResourceCapacity:
    total_quota = 0.0
    total_used = 0.0
    for capacity in capacities:
        total_quota += capacity.quota
        total_used += capacity.used
    return ResourceCapacity(quota=total_quota, used=total_used)


def account_resource_capacities(usage: AccountUsage) -> dict[str, ResourceCapacity]:
    """The account-level resources in report order; the one shared resource list."""
    return {
        "cpu": usage.cpu,
        "memory_gib": usage.memory_gib,
        "disk_gib": usage.disk_gib,
        "snapshots": usage.snapshots,
        "volumes": usage.volumes,
    }


def region_resource_capacities(region: RegionCapacity) -> dict[str, ResourceCapacity]:
    """One region's resources in report order."""
    return {
        "cpu": region.cpu,
        "memory_gib": region.memory_gib,
        "disk_gib": region.disk_gib,
        "gpu": region.gpu,
    }


def _capacity_dict(capacity: ResourceCapacity) -> dict[str, float]:
    return {"quota": capacity.quota, "used": capacity.used, "remaining": capacity.remaining}


def _region(row: RegionUsageOverview) -> RegionCapacity:
    return RegionCapacity(
        region_id=row.region_id,
        sandbox_class=row.sandbox_class.value,
        cpu=ResourceCapacity(quota=row.total_cpu_quota, used=row.current_cpu_usage),
        memory_gib=ResourceCapacity(quota=row.total_memory_quota, used=row.current_memory_usage),
        disk_gib=ResourceCapacity(quota=row.total_disk_quota, used=row.current_disk_usage),
        gpu=ResourceCapacity(quota=row.total_gpu_quota, used=row.current_gpu_usage),
        max_cpu_per_sandbox=row.max_cpu_per_sandbox,
        max_memory_gib_per_sandbox=row.max_memory_per_sandbox,
        max_disk_gib_per_sandbox=row.max_disk_per_sandbox,
    )


def region_for_launch(usage: AccountUsage, *, target: str, sandbox_class: str = DEFAULT_SANDBOX_CLASS) -> RegionCapacity:
    """The quota row that a launch into ``target`` is charged against.

    Daytona enforces each row's quota per region and sandbox class, so sizing
    must use that one row; summing the rows over-reports capacity when the org
    holds quota in several of them.
    """
    for region in usage.regions:
        if region.region_id == target and region.sandbox_class == sandbox_class:
            return region
    rows = ", ".join(f"{region.region_id}/{region.sandbox_class}" for region in usage.regions) or "none"
    raise RuntimeError(f"No {sandbox_class} quota row for region {target}; the org's quota rows: {rows}")


def max_concurrency(region: RegionCapacity, profile: SandboxProfile, *, headroom: float = DEFAULT_HEADROOM) -> int:
    """Suggest how many additional sandboxes one quota row can hold.

    The minimum over the row's CPU, memory and disk remaining capacity, each
    reduced by ``headroom`` (a fraction kept in reserve, default 20%), divided
    by the profile's per-sandbox request and floored. GPU use is not part of
    the minimum; a GPU workload must check the gpu capacity itself.
    """
    if profile.cpu <= 0 or profile.memory_gib <= 0 or profile.disk_gib <= 0:
        raise ValueError(f"Sandbox profile must be positive: {profile}")
    if not 0 <= headroom < 1:
        raise ValueError(f"Headroom must be in [0, 1): {headroom}")
    return max(
        0,
        math.floor(
            min(
                region.cpu.remaining * (1 - headroom) / profile.cpu,
                region.memory_gib.remaining * (1 - headroom) / profile.memory_gib,
                region.disk_gib.remaining * (1 - headroom) / profile.disk_gib,
            )
        ),
    )


async def _inventory_sandboxes(client: AsyncDaytona) -> list[AsyncSandbox]:
    """Read every sandbox in the authenticated account across all pages."""
    return [sandbox async for sandbox in client.list(request_timeout=REQUEST_TIMEOUT_SECONDS)]


async def _fetch_usage_overview(client: AsyncDaytona, organization_id: str) -> OrganizationUsageOverview:
    api = OrganizationsApi(client._api_client)
    return await api.get_organization_usage_overview(organization_id, _request_timeout=REQUEST_TIMEOUT_SECONDS)


async def _organization_id(client: AsyncDaytona, sandboxes: list[AsyncSandbox]) -> str:
    """Resolve the organization an API key belongs to from its own inventory.

    The ``/organizations`` endpoint rejects API-key credentials with 401, so
    the organization is taken from inventory rows. Sandboxes carry it; when the
    organization has no sandboxes, snapshots and then volumes are read instead.
    An inventory spanning several organizations is an account-selection error,
    not a reportable mix, so it raises rather than reporting either one.
    """
    ids = {row.organization_id for row in sandboxes if row.organization_id}
    if not ids:
        snapshots: list[Snapshot] = await list_snapshots(client)
        ids = {snapshot.organization_id for snapshot in snapshots if snapshot.organization_id}
    if not ids:
        volumes: list[Volume] = await client.volume.list()
        ids = {volume.organization_id for volume in volumes if volume.organization_id}
    if len(ids) > 1:
        raise RuntimeError(f"The API key lists rows in {len(ids)} organizations; the account is ambiguous")
    if not ids:
        raise RuntimeError("Cannot resolve the organization: the account has no sandboxes, snapshots or volumes")
    return ids.pop()


async def account_usage_from_client(client: AsyncDaytona) -> AccountUsage:
    """Read one account's quota, usage and sandbox inventory through an SDK client."""
    sandboxes = await _inventory_sandboxes(client)
    organization_id = await _organization_id(client, sandboxes)
    overview = await _fetch_usage_overview(client, organization_id)
    state_counts: dict[str, int] = {}
    for sandbox in sandboxes:
        state = sandbox.state.value if sandbox.state else "unknown"
        state_counts[state] = state_counts.get(state, 0) + 1
    return AccountUsage(
        organization_id=organization_id,
        regions=tuple(_region(row) for row in overview.region_usage),
        snapshots=ResourceCapacity(quota=overview.total_snapshot_quota, used=overview.current_snapshot_usage),
        volumes=ResourceCapacity(quota=overview.total_volume_quota, used=overview.current_volume_usage),
        sandboxes=len(sandboxes),
        sandbox_states=tuple(sorted(state_counts.items())),
    )


async def account_usage(connection: DaytonaConnection) -> AccountUsage:
    """Read one explicitly selected Daytona account's quota and live usage."""
    async with connection.client_factory()() as client:
        return await account_usage_from_client(client)


def usage_report(
    usage: AccountUsage,
    profile: SandboxProfile | None,
    *,
    region: RegionCapacity | None = None,
    headroom: float = DEFAULT_HEADROOM,
) -> dict[str, Any]:
    """Return one account's usage as the CLI JSON interchange.

    The resource block sums CPU, memory and disk across regions and includes
    account-level snapshot and volume quotas. The region block lists each
    quota row. ``max_concurrency`` is sized against ``region``, the row charged
    for a launch into the target region.
    """
    report: dict[str, Any] = {
        "organization_id": usage.organization_id,
        "resources": {name: _capacity_dict(capacity) for name, capacity in account_resource_capacities(usage).items()},
        "regions": [
            {
                "region_id": row.region_id,
                "sandbox_class": row.sandbox_class,
                **{name: _capacity_dict(capacity) for name, capacity in region_resource_capacities(row).items()},
                "max_cpu_per_sandbox": row.max_cpu_per_sandbox,
                "max_memory_gib_per_sandbox": row.max_memory_gib_per_sandbox,
                "max_disk_gib_per_sandbox": row.max_disk_gib_per_sandbox,
            }
            for row in usage.regions
        ],
        "sandboxes": {"total": usage.sandboxes, "by_state": dict(usage.sandbox_states)},
    }
    if profile is not None:
        if region is None:
            raise ValueError("Sizing a profile requires the launch's quota row")
        report["max_concurrency"] = {
            "region_id": region.region_id,
            "sandbox_class": region.sandbox_class,
            "cpu": profile.cpu,
            "memory_gib": profile.memory_gib,
            "disk_gib": profile.disk_gib,
            "headroom": headroom,
            "limit": max_concurrency(region, profile, headroom=headroom),
        }
    return report
