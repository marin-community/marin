# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read one Daytona project's limits, live usage and sandbox inventory.

Reports what a project can still run, so agents size concurrency from live
numbers instead of static tables: the organization's concurrent-sandbox and
storage limits with their current use, the per-sandbox CPU and memory maxima,
and the project's sandbox counts by state. v2 has no CPU, memory or disk quota
to divide; the concurrent-sandbox limit bounds a launch.
"""

import math
from dataclasses import dataclass
from typing import Any

from daytona import AsyncDaytonaV2Client
from daytona.v2 import UsageLimit

from shellbox.backends.daytona.connection import DaytonaConnection
from shellbox.backends.daytona.machine import VCPUS_PER_CORE

DEFAULT_HEADROOM = 0.2


@dataclass(frozen=True)
class Capacity:
    """A limit and its current use; ``limit`` None means unlimited."""

    limit: int | None
    used: int

    @property
    def remaining(self) -> int | None:
        return None if self.limit is None else max(self.limit - self.used, 0)

    @classmethod
    def from_limit(cls, limit: UsageLimit) -> "Capacity":
        return cls(limit=limit.limit, used=limit.usage)


@dataclass(frozen=True)
class SandboxProfile:
    """The per-sandbox request a launch makes, in vCPUs and GiB."""

    cpus: int
    memory_gib: float


@dataclass(frozen=True)
class ProjectUsage:
    """One Daytona project's limits, live usage and sandbox inventory."""

    organization: str
    project: str
    sandboxes: Capacity
    images: Capacity
    volumes: Capacity
    snapshots: Capacity
    max_cores_per_sandbox: float | None
    max_memory_gib_per_sandbox: float | None
    sandbox_states: tuple[tuple[str, int], ...]


def max_concurrency(usage: ProjectUsage, *, headroom: float = DEFAULT_HEADROOM) -> int | None:
    """Suggest how many additional sandboxes the concurrent limit admits, keeping ``headroom`` in reserve.

    Returns None when the organization has no concurrent-sandbox limit.
    """
    if not 0 <= headroom < 1:
        raise ValueError(f"Headroom must be in [0, 1): {headroom}")
    remaining = usage.sandboxes.remaining
    return None if remaining is None else math.floor(remaining * (1 - headroom))


def profile_violations(usage: ProjectUsage, profile: SandboxProfile) -> list[str]:
    """Describe each per-sandbox maximum the profile exceeds; such creates fail."""
    violations = []
    cores = profile.cpus / VCPUS_PER_CORE
    if usage.max_cores_per_sandbox is not None and cores > usage.max_cores_per_sandbox:
        violations.append(f"{profile.cpus} vCPUs exceed the per-sandbox maximum of {usage.max_cores_per_sandbox} cores")
    if usage.max_memory_gib_per_sandbox is not None and profile.memory_gib > usage.max_memory_gib_per_sandbox:
        violations.append(
            f"{profile.memory_gib} GiB exceeds the per-sandbox maximum of {usage.max_memory_gib_per_sandbox} GiB"
        )
    return violations


async def project_usage_from_client(client: AsyncDaytonaV2Client) -> ProjectUsage:
    """Read one project's limits, usage and sandbox inventory through an SDK client."""
    key = await client.current_api_key()
    limits = await client.get_limits()
    state_counts: dict[str, int] = {}
    async for sandbox in client.iter():
        state_counts[sandbox.state] = state_counts.get(sandbox.state, 0) + 1
    resources = limits.sandbox_resources
    return ProjectUsage(
        organization=key.context.org_name,
        project=key.context.project_name,
        sandboxes=Capacity.from_limit(limits.sandboxes.concurrent),
        images=Capacity.from_limit(limits.storage.images),
        volumes=Capacity.from_limit(limits.storage.volumes),
        snapshots=Capacity.from_limit(limits.storage.snapshots),
        max_cores_per_sandbox=None if resources is None else resources.max_cores,
        max_memory_gib_per_sandbox=None if resources is None else resources.max_memory_mib / 1024,
        sandbox_states=tuple(sorted(state_counts.items())),
    )


async def project_usage(connection: DaytonaConnection) -> ProjectUsage:
    """Read one explicitly selected Daytona project's limits and live usage."""
    async with connection.client_factory()() as client:
        return await project_usage_from_client(client)


def usage_report(
    usage: ProjectUsage, profile: SandboxProfile | None, *, headroom: float = DEFAULT_HEADROOM
) -> dict[str, Any]:
    """Return one project's usage as the CLI JSON interchange."""
    report: dict[str, Any] = {
        "organization": usage.organization,
        "project": usage.project,
        "limits": {
            name: {"limit": capacity.limit, "used": capacity.used, "remaining": capacity.remaining}
            for name, capacity in (
                ("concurrent_sandboxes", usage.sandboxes),
                ("images", usage.images),
                ("volumes", usage.volumes),
                ("snapshots", usage.snapshots),
            )
        },
        "max_cores_per_sandbox": usage.max_cores_per_sandbox,
        "max_memory_gib_per_sandbox": usage.max_memory_gib_per_sandbox,
        "sandboxes": {"total": sum(count for _, count in usage.sandbox_states), "by_state": dict(usage.sandbox_states)},
        "max_concurrency": {"headroom": headroom, "limit": max_concurrency(usage, headroom=headroom)},
    }
    if profile is not None:
        report["profile"] = {
            "cpus": profile.cpus,
            "memory_gib": profile.memory_gib,
            "violations": profile_violations(usage, profile),
        }
    return report
