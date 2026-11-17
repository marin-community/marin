# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Machine resource limits and command containment in a delegated cgroup v2 tree."""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from shellbox.machine import UnsupportedMachineSpec
from shellbox.process_session import await_cleanup, own_cleanup

CGROUP_ROOT = Path("/sys/fs/cgroup")
CPU_PERIOD = 100000
CLOSE_TIMEOUT = 10
logger = logging.getLogger(__name__)


def _current_cgroup() -> Path:
    membership = next(line for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    return (CGROUP_ROOT / membership.removeprefix("0::").lstrip("/")).resolve()


def check_cgroup_parent(parent: Path) -> None:
    """Require a writable delegated hierarchy with recursive process cleanup."""
    if not (CGROUP_ROOT / "cgroup.controllers").is_file() or not parent.is_dir():
        raise UnsupportedMachineSpec("Local process containment requires a delegated cgroup v2 directory")
    if parent.stat().st_dev != CGROUP_ROOT.stat().st_dev:
        raise UnsupportedMachineSpec("The delegated parent must be on the cgroup v2 filesystem")
    if not os.access(parent / "cgroup.kill", os.W_OK):
        raise UnsupportedMachineSpec("Local cgroup cleanup requires writable kernel cgroup.kill support")
    current = _current_cgroup()
    if not current.is_relative_to(CGROUP_ROOT):
        raise UnsupportedMachineSpec("The worker's cgroup is outside its mounted cgroup hierarchy")
    common = Path(os.path.commonpath((parent, current)))
    if not all(os.access(path, os.W_OK) for path in (parent, common / "cgroup.procs")):
        raise UnsupportedMachineSpec("Run the local worker inside a writable delegated cgroup hierarchy")


def delegated_cgroup_parent() -> Path | None:
    """Find writable delegation within the worker's own cgroup or systemd unit."""
    if not (CGROUP_ROOT / "cgroup.controllers").is_file():
        return None
    current = _current_cgroup()
    if not current.is_relative_to(CGROUP_ROOT):
        return None
    candidates = [current]
    # DelegateSubgroup moves the worker below its unit. Do not cross a unit
    # boundary into a shared systemd slice, even when that slice is writable.
    unit_suffixes = (".service", ".scope")
    if not current.name.endswith(unit_suffixes) and current.parent.name.endswith(unit_suffixes):
        if not (current.parent / "cgroup.procs").read_text().strip():
            candidates.insert(0, current.parent)
    for parent in candidates:
        try:
            check_cgroup_parent(parent)
        except UnsupportedMachineSpec:
            continue
        return parent
    return None


def _apply_resource_limits(parent: Path, path: Path, *, cpus: int | None, memory_mb: int | None) -> None:
    for controller, value in (("cpu", cpus), ("memory", memory_mb)):
        if value is None:
            continue
        try:
            enabled = (parent / "cgroup.subtree_control").read_text().split()
            if controller not in enabled:
                (parent / "cgroup.subtree_control").write_text(f"+{controller}")
            if controller == "cpu":
                (path / "cpu.max").write_text(f"{value * CPU_PERIOD} {CPU_PERIOD}")
            else:
                (path / "memory.max").write_text(str(value * 1024 * 1024))
                (path / "memory.swap.max").write_text("0")
                (path / "memory.oom.group").write_text("1")
        except OSError as error:
            logger.warning("Cannot apply local %s limit %s in %s: %s", controller, value, path, error)


@dataclass(eq=False)
class MachineCgroup:
    path: Path
    _closing: asyncio.Task[None] | None = field(default=None, init=False, repr=False)

    @classmethod
    def create(cls, parent: Path, *, cpus: int | None, memory_mb: int | None) -> "MachineCgroup":
        check_cgroup_parent(parent)
        path = parent / f"shellbox-{uuid4().hex}"
        path.mkdir()
        try:
            if not (path / "cgroup.kill").is_file():
                raise UnsupportedMachineSpec("Local cgroup cleanup requires kernel cgroup.kill support")
            _apply_resource_limits(parent, path, cpus=cpus, memory_mb=memory_mb)
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
