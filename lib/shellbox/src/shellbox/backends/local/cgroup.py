# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Machine resource limits and command containment in a delegated cgroup v2 tree."""

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from shellbox.machine import UnsupportedMachineSpec
from shellbox.process_session import await_cleanup, own_cleanup

CGROUP_ROOT = Path("/sys/fs/cgroup")
CPU_PERIOD = 100000
CLOSE_TIMEOUT = 10


def check_cgroup_parent(parent: Path, *, resource_limits: bool = False) -> None:
    """Require a writable delegated hierarchy with recursive process cleanup."""
    if not (CGROUP_ROOT / "cgroup.controllers").is_file() or not parent.is_dir():
        raise UnsupportedMachineSpec("Local process containment requires a delegated cgroup v2 directory")
    if parent.stat().st_dev != CGROUP_ROOT.stat().st_dev:
        raise UnsupportedMachineSpec("The delegated parent must be on the cgroup v2 filesystem")
    controllers = set((parent / "cgroup.controllers").read_text().split())
    if resource_limits and not {"cpu", "memory"} <= controllers:
        raise UnsupportedMachineSpec("The delegated cgroup must provide CPU and memory controllers")
    if not os.access(parent / "cgroup.kill", os.W_OK):
        raise UnsupportedMachineSpec("Local cgroup cleanup requires writable kernel cgroup.kill support")
    membership = next(line for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    current = CGROUP_ROOT / membership.removeprefix("0::").lstrip("/")
    common = Path(os.path.commonpath((parent, current)))
    if not all(
        os.access(path, os.W_OK) for path in (parent, parent / "cgroup.subtree_control", common / "cgroup.procs")
    ):
        raise UnsupportedMachineSpec("Run the local worker inside a writable delegated cgroup hierarchy")


@dataclass(eq=False)
class MachineCgroup:
    path: Path
    _closing: asyncio.Task[None] | None = field(default=None, init=False, repr=False)

    @classmethod
    def create(cls, parent: Path, *, cpus: int | None, memory_mb: int | None) -> "MachineCgroup":
        resource_limits = cpus is not None or memory_mb is not None
        check_cgroup_parent(parent, resource_limits=resource_limits)
        if resource_limits:
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

    def create_child(self) -> "MachineCgroup":
        """Create a leaf inheriting the machine's aggregate limits."""
        path = self.path / f"command-{uuid4().hex}"
        path.mkdir()
        return MachineCgroup(path)

    @property
    def populated(self) -> bool:
        return "populated 1" in (self.path / "cgroup.events").read_text()

    def remove_if_empty(self) -> bool:
        if self._closing is not None or self.populated:
            return False
        self.path.rmdir()
        return True

    async def close(self) -> None:
        if self._closing is None:
            self._closing = own_cleanup(self._close())
        await await_cleanup(self._closing)

    async def _close(self) -> None:
        self.kill()
        await self.remove()

    def kill(self) -> None:
        (self.path / "cgroup.kill").write_text("1")

    async def remove(self) -> None:
        async with asyncio.timeout(CLOSE_TIMEOUT):
            while self.populated:
                await asyncio.sleep(0.01)
        self.path.rmdir()
