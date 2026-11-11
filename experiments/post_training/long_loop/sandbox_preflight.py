# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify local Linux sandbox startup before allocating the RL gang."""

import asyncio
import json
import platform
from pathlib import Path

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join
from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.machine import Command, HostImage, MachineSpec


async def check_local_sandbox() -> dict:
    factory = LocalMachineFactory()
    machine = await factory.create(MachineSpec(HostImage(), workdir="/"))
    try:
        result = await machine.run(Command(("/bin/sh", "-c", "printf sandbox-ready"), timeout=15))
        if result.exit_code != 0 or result.stdout != b"sandbox-ready":
            raise RuntimeError(f"Sandbox command failed: {result.exit_code}: {result.stderr!r}")
        return {"namespace_keeper": "started", "command": "passed", "platform": platform.platform()}
    finally:
        await machine.close()


def sandbox_preflight(*, output: str, image: str, container_profile: str) -> None:
    evidence = asyncio.run(check_local_sandbox())
    evidence.update(image=image, container_profile=container_profile)
    evidence["process_security"] = [
        line
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith(("Cap", "Seccomp", "NoNewPrivs", "Uid:"))
    ]
    fs, key = filesystem_for(prefix_join(output, "sandbox.json"))
    with fs.open(key, "w") as destination:
        json.dump(evidence, destination, indent=2)
