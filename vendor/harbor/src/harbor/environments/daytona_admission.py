# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Require the standard inventory command to admit each Daytona sandbox create."""

import asyncio
import json
import logging

from shellbox.backends.daytona.snapshot import snapshot_resources
from shellbox.machine import MachineSpec, UnsupportedMachineSpec

LOGGER = logging.getLogger(__name__)
CHECK_TIMEOUT = 60
RECHECK_INTERVAL = 30
# An omitted request is budgeted at the documented CPU-sandbox maximum; the
# machine's original resource request is passed to Daytona without modification.
UNSPECIFIED_CPU = 4
UNSPECIFIED_MEMORY_GIB = 8
UNSPECIFIED_DISK_GIB = 10


async def admit_create(
    command: tuple[str, ...], spec: MachineSpec, *, recheck_interval: float = RECHECK_INTERVAL
) -> None:
    """Wait until the standard inventory command admits this unchanged machine request."""
    resources = snapshot_resources(
        cpus=spec.cpus, memory_mb=spec.memory_mb, storage_mb=spec.storage_mb, gpus=spec.gpus
    )
    cpu = resources.cpu if resources.cpu is not None else UNSPECIFIED_CPU
    memory = resources.memory if resources.memory is not None else UNSPECIFIED_MEMORY_GIB
    disk = resources.disk if resources.disk is not None else UNSPECIFIED_DISK_GIB
    command = (*command, "--cpu", str(cpu), "--memory-gib", str(memory), "--disk-gib", str(disk), "--json")
    while True:
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
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
        capacity = report["max_concurrency"]
        region = next(
            row for row in report["regions"]
            if row["region_id"] == capacity["region_id"] and row["sandbox_class"] == capacity["sandbox_class"]
        )
        if (cpu > region["max_cpu_per_sandbox"] or memory > region["max_memory_gib_per_sandbox"]
                or disk > region["max_disk_gib_per_sandbox"]):
            raise UnsupportedMachineSpec("Daytona request exceeds the documented per-sandbox maximum")
        if capacity["limit"] >= 1:
            return
        LOGGER.info("DAYTONA_CAPACITY_WAIT: retaining headroom before sandbox create")
        await asyncio.sleep(recheck_interval)
