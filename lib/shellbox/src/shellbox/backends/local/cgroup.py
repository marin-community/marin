# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-machine resource accounting inside an explicitly delegated cgroup v2 tree."""

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from shellbox.machine import UnsupportedMachineSpec

CGROUP_ROOT = Path("/sys/fs/cgroup")
CPU_PERIOD = 100000
CLOSE_TIMEOUT = 10


def check_cgroup_parent(parent: Path) -> None:
    """Require writable CPU and memory controllers in a delegated hierarchy."""
    if not (CGROUP_ROOT / "cgroup.controllers").is_file() or not parent.is_dir():
        raise UnsupportedMachineSpec("Local resource limits require a delegated cgroup v2 directory")
    if parent.stat().st_dev != CGROUP_ROOT.stat().st_dev:
        raise UnsupportedMachineSpec("The delegated parent must be on the cgroup v2 filesystem")
    controllers = set((parent / "cgroup.controllers").read_text().split())
    if not {"cpu", "memory"} <= controllers:
        raise UnsupportedMachineSpec("The delegated cgroup must provide CPU and memory controllers")
    membership = next(line for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    current = CGROUP_ROOT / membership.removeprefix("0::").lstrip("/")
    common = Path(os.path.commonpath((parent, current)))
    if not all(
        os.access(path, os.W_OK) for path in (parent, parent / "cgroup.subtree_control", common / "cgroup.procs")
    ):
        raise UnsupportedMachineSpec("Run the local worker inside a writable delegated cgroup hierarchy")


@dataclass(frozen=True)
class MachineCgroup:
    path: Path

    @classmethod
    def create(cls, parent: Path, *, cpus: int | None, memory_mb: int | None) -> "MachineCgroup":
        check_cgroup_parent(parent)
        try:
            (parent / "cgroup.subtree_control").write_text("+cpu +memory")
        except OSError as error:
            raise UnsupportedMachineSpec(
                "The delegated cgroup must be empty; place the worker in a supervisor subgroup"
            ) from error
        path = parent / f"shellbox-{uuid4().hex}"
        path.mkdir()
        try:
            if not (path / "cgroup.kill").is_file():
                raise UnsupportedMachineSpec("Local cgroup cleanup requires kernel cgroup.kill support")
            if cpus is not None:
                (path / "cpu.max").write_text(f"{cpus * CPU_PERIOD} {CPU_PERIOD}")
            if memory_mb is not None:
                (path / "memory.max").write_text(str(memory_mb * 1024 * 1024))
                (path / "memory.swap.max").write_text("0")
                (path / "memory.oom.group").write_text("1")
        except BaseException:
            path.rmdir()
            raise
        return cls(path)

    def kill(self) -> None:
        (self.path / "cgroup.kill").write_text("1")

    async def remove(self) -> None:
        async with asyncio.timeout(CLOSE_TIMEOUT):
            while "populated 1" in (self.path / "cgroup.events").read_text():
                await asyncio.sleep(0.01)
        self.path.rmdir()
