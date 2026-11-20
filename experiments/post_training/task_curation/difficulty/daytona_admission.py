# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Temporary DATA adapter until Shellbox's inventory-headroom CLI lands."""

import argparse
import asyncio
import json
import math
import os
from collections import Counter
from datetime import UTC, datetime

from daytona import AsyncSandbox
from shellbox.backends.daytona.connection import DEFAULT_DAYTONA_API_URL, DaytonaConnection
from shellbox.backends.daytona.usage import SandboxProfile, _inventory_sandboxes

DATA_LIMITS = {"cpu": 2000, "memory_gib": 4000, "disk_gib": 10000}
SANDBOX_LIMITS = {"cpu": 4, "memory_gib": 8, "disk_gib": 10}
CHARGED_STATES = {"started", "starting", "creating"}


def inventory_report(sandboxes: list[AsyncSandbox], profile: SandboxProfile, headroom: float) -> dict:
    """Size one DATA create from all pages, retaining a fraction of total quota."""
    requested = {"cpu": profile.cpu, "memory_gib": profile.memory_gib, "disk_gib": profile.disk_gib}
    if not 0 <= headroom < 1 or any(value <= 0 for value in requested.values()):
        raise ValueError("Headroom must be in [0, 1) and requested resources must be positive")
    counts = Counter(row.state.value if row.state else "unknown" for row in sandboxes)
    active = [row for row in sandboxes if row.state and row.state.value in CHARGED_STATES]
    used = {
        "cpu": sum(row.cpu for row in active),
        "memory_gib": sum(row.memory for row in active),
        "disk_gib": sum(row.disk for row in active),
    }
    limit = max(
        0,
        math.floor(min((DATA_LIMITS[name] * (1 - headroom) - used[name]) / value for name, value in requested.items())),
    )
    if any(value > SANDBOX_LIMITS[name] for name, value in requested.items()):
        limit = 0
    region = {
        "region_id": "us",
        "sandbox_class": "container",
        **{name: {"quota": quota, "used": used[name]} for name, quota in DATA_LIMITS.items()},
        **{f"max_{name}_per_sandbox": value for name, value in SANDBOX_LIMITS.items()},
    }
    return {
        "checked_at": datetime.now(UTC).isoformat(),
        "quota_source": "DAYTONA-QUOTAS.md: DATA, ruling 71",
        "sandboxes": len(sandboxes),
        "sandbox_states": dict(counts),
        "regions": [region],
        "max_concurrency": {"region_id": "us", "sandbox_class": "container", "headroom": headroom, "limit": limit},
    }


async def current_report(profile: SandboxProfile, headroom: float) -> dict:
    """Read DATA inventory without calling the organization-usage endpoint."""
    if os.environ["DAYTONA_TARGET"] != "us":
        raise ValueError("Difficulty DATA admission requires DAYTONA_TARGET=us")
    connection = DaytonaConnection(os.environ["DAYTONA_DATA_API_KEY"], DEFAULT_DAYTONA_API_URL, "us")
    async with connection.client_factory()() as client:
        sandboxes = await _inventory_sandboxes(client)
    return inventory_report(sandboxes, profile, headroom)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--memory-gib", type=float, required=True)
    parser.add_argument("--disk-gib", type=float, required=True)
    parser.add_argument("--headroom", type=float, required=True)
    parser.add_argument("--json", action="store_true", required=True)
    args = parser.parse_args()
    report = asyncio.run(current_report(SandboxProfile(args.cpu, args.memory_gib, args.disk_gib), args.headroom))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
