# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Estimate organization headroom from sandbox inventory and supplied quotas."""

import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from daytona import AsyncDaytona, AsyncSandbox
from daytona_api_client_async import SandboxState

from shellbox.backends.daytona.connection import DaytonaConnection

REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_HEADROOM = 0.2
GB_TO_GIB = 10**9 / 2**30
CHARGED_STATES = frozenset((SandboxState.STARTED, SandboxState.STARTING, SandboxState.CREATING))
QUOTA_SOURCE = "DAYTONA-QUOTAS.md, account owner, 2026-11-19"


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
class OrganizationLimits:
    """Owner-supplied organization limits in vCPU and decimal GB."""

    cpu: int
    memory_gb: int
    disk_gb: int
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
class AccountUsage:
    """One full inventory reading; an empty inventory has no organization ID."""

    organization_id: str | None
    limits: OrganizationLimits
    cpu: ResourceCapacity
    memory_gib: ResourceCapacity
    disk_gib: ResourceCapacity
    sandbox_states: tuple[tuple[str, int], ...]
    observed_at: str


def account_resource_capacities(usage: AccountUsage) -> dict[str, ResourceCapacity]:
    return {"cpu": usage.cpu, "memory_gib": usage.memory_gib, "disk_gib": usage.disk_gib}


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


async def account_usage_from_client(client: AsyncDaytona, limits: OrganizationLimits) -> AccountUsage:
    """Sum started, starting and creating allocations without organization API calls."""
    sandboxes = await _inventory_sandboxes(client)
    organization_ids = {sandbox.organization_id for sandbox in sandboxes}
    if len(organization_ids) > 1 or (sandboxes and not all(organization_ids)):
        raise ValueError("Sandbox inventory must identify exactly one organization")
    active = [sandbox for sandbox in sandboxes if sandbox.state in CHARGED_STATES]
    for sandbox in active:
        for value in (sandbox.cpu, sandbox.memory, sandbox.disk):
            if value is None or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Active sandbox {sandbox.id} lacks a positive resource allocation")
    states = Counter(sandbox.state.value if sandbox.state else "unknown" for sandbox in sandboxes)
    return AccountUsage(
        organization_id=next(iter(organization_ids), None),
        limits=limits,
        cpu=ResourceCapacity(limits.cpu, sum(sandbox.cpu for sandbox in active)),
        memory_gib=ResourceCapacity(limits.memory_gb * GB_TO_GIB, sum(sandbox.memory for sandbox in active)),
        disk_gib=ResourceCapacity(limits.disk_gb * GB_TO_GIB, sum(sandbox.disk for sandbox in active)),
        sandbox_states=tuple(sorted(states.items())),
        observed_at=datetime.now(UTC).isoformat(),
    )


async def account_usage(connection: DaytonaConnection, limits: OrganizationLimits) -> AccountUsage:
    """Read the explicitly selected account's complete sandbox inventory."""
    async with connection.client_factory()() as client:
        return await account_usage_from_client(client, limits)


def usage_report(
    usage: AccountUsage, profile: SandboxProfile | None = None, *, headroom: float = DEFAULT_HEADROOM
) -> dict[str, Any]:
    """Return inventory usage and additional capacity below a total-quota reserve."""
    if not math.isfinite(headroom) or not 0 <= headroom < 1:
        raise ValueError("Headroom must be in [0, 1)")
    resources = account_resource_capacities(usage)
    states = dict(usage.sandbox_states)
    report: dict[str, Any] = {
        "organization_id": usage.organization_id,
        "observed_at": usage.observed_at,
        "usage_source": "sandbox_inventory",
        "quota_source": QUOTA_SOURCE,
        "org_limits": asdict(usage.limits),
        "headroom_fraction": headroom,
        "charged_states": sorted(state.value for state in CHARGED_STATES),
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
            "total": sum(states.values()),
            "started": states.get("started", 0),
            "charged": sum(states.get(state.value, 0) for state in CHARGED_STATES),
            "by_state": states,
        },
    }
    if profile is not None:
        requested = asdict(profile)
        maxima = asdict(usage.limits.per_sandbox)
        exceeded = [name for name, value in requested.items() if value > maxima[name]]
        per_resource = {
            name: math.floor(capacity.available(headroom) / requested[name]) for name, capacity in resources.items()
        }
        report["max_concurrency"] = {
            "profile": requested,
            "by_resource": per_resource,
            "exceeds_per_sandbox_maximum": exceeded,
            "limit": 0 if exceeded else min(per_resource.values()),
        }
    return report
