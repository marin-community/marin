# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Require the standard inventory command to admit each Daytona sandbox create."""

import asyncio
import json
import logging

from shellbox.backends.daytona.snapshot import snapshot_resources
from shellbox.backends.daytona.usage import SandboxProfile
from shellbox.machine import MachineSpec, UnsupportedMachineSpec

LOGGER = logging.getLogger(__name__)
CHECK_TIMEOUT = 60
RECHECK_INTERVAL = 30
MINIMUM_RESOURCE = 1


async def usage_report(command: tuple[str, ...], profile: SandboxProfile) -> dict:
    """Read the admission command's JSON, closing its process on cancellation."""
    process = await asyncio.create_subprocess_exec(
        *command, "--cpu", str(profile.cpu), "--memory-gib", str(profile.memory_gib),
        "--disk-gib", str(profile.disk_gib), "--json",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), CHECK_TIMEOUT)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    if process.returncode:
        raise RuntimeError(f"Daytona inventory admission failed: {stderr.decode()}")
    report = json.loads(stdout)
    LOGGER.info("DAYTONA_INVENTORY_ADMISSION %s", json.dumps(report, sort_keys=True))
    return report


async def admit_create(
    command: tuple[str, ...], spec: MachineSpec, *, recheck_interval: float = RECHECK_INTERVAL
) -> None:
    """Wait for capacity, budgeting omitted resources at the command's reported maxima."""
    resources = snapshot_resources(
        cpus=spec.cpus, memory_mb=spec.memory_mb, storage_mb=spec.storage_mb, gpus=spec.gpus
    )
    profile = SandboxProfile(
        resources.cpu if resources.cpu is not None else MINIMUM_RESOURCE,
        resources.memory if resources.memory is not None else MINIMUM_RESOURCE,
        resources.disk if resources.disk is not None else MINIMUM_RESOURCE,
    )
    while True:
        report = await usage_report(command, profile)
        capacity = report["max_concurrency"]
        maximum = report["org_limits"]["per_sandbox"]
        cpu, memory, disk = (
            maximum["cpu"], maximum["memory_gib"], maximum["disk_gib"]
        )
        if cpu is None or memory is None or disk is None:
            raise UnsupportedMachineSpec("Inventory admission requires bounded per-sandbox resource maxima")
        resolved = SandboxProfile(
            resources.cpu if resources.cpu is not None else int(cpu),
            resources.memory if resources.memory is not None else memory,
            resources.disk if resources.disk is not None else disk,
        )
        # A discovery check cannot admit an omitted resource until a fresh check
        # budgets that resource at the selected account's reported maximum.
        if resolved != profile:
            profile = resolved
            continue
        if profile.cpu > cpu or profile.memory_gib > memory or profile.disk_gib > disk:
            raise UnsupportedMachineSpec("Daytona request exceeds the documented per-sandbox maximum")
        if capacity["limit"] >= 1:
            return
        LOGGER.info("DAYTONA_CAPACITY_WAIT: retaining headroom before sandbox create")
        await asyncio.sleep(recheck_interval)
