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
UNIT_SUFFIXES = (".service", ".scope")
WORKER_LEAF_NAME = "worker"
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


def cgroup_limits_applied(path: Path, *, cpus: int, memory_mb: int) -> bool:
    """Whether the kernel actually applied the demanded limits to a machine cgroup."""
    try:
        quota, period = (path / "cpu.max").read_text().split()[:2]
        cpu_applied = int(quota) == cpus * CPU_PERIOD and int(period) == CPU_PERIOD
        memory_applied = int((path / "memory.max").read_text()) == memory_mb * 1024 * 1024
        return cpu_applied and memory_applied
    except (OSError, ValueError) as error:
        logger.warning("Cannot read back limits in %s: %s", path, error)
        return False


def probe_cgroup_limits(parent: Path, *, cpus: int = 1, memory_mb: int = 1024) -> bool:
    """Create a throwaway machine cgroup and verify the kernel applied its limits.

    Container scopes cannot enable domain controllers while the worker's own
    process stays inside them (cgroup v2's no-internal-process rule), and some
    runtimes also refuse the memory controller outright. Readback catches
    limit writes that only logged a warning.
    """
    try:
        cgroup = MachineCgroup.create(parent, cpus=cpus, memory_mb=memory_mb)
    except (OSError, UnsupportedMachineSpec) as error:
        logger.warning("Cannot probe cgroup limits under %s: %s", parent, error)
        return False
    try:
        return cgroup_limits_applied(cgroup.path, cpus=cpus, memory_mb=memory_mb)
    finally:
        cgroup.kill()
        cgroup.remove_if_empty()


def _delegation_target(current: Path) -> tuple[Path, Path | None]:
    """Return the delegation parent and the leaf group to move this worker into.

    The parent never lies above the worker's own unit, so sandbox groups stay
    inside the worker's container scope. An empty enclosing unit (the systemd
    ``DelegateSubgroup`` layout) or an existing worker leaf needs no move.
    Otherwise the unit directly contains the worker — for example a container
    scope — and its processes must move into a child leaf before the unit can
    enable domain controllers for children (cgroup v2's no-internal-process
    rule).
    """
    unit = current.parent
    if unit != current and unit.is_relative_to(CGROUP_ROOT):
        if current.name == WORKER_LEAF_NAME or (
            unit.name.endswith(UNIT_SUFFIXES) and not (unit / "cgroup.procs").read_text().strip()
        ):
            return unit, None
    return current, current / WORKER_LEAF_NAME


def _move_worker_processes(current: Path, leaf: Path) -> None:
    """Move every process listed in ``current`` into the child ``leaf`` group.

    Processes that disappear or refuse to migrate only warn: discovery still
    returns the parent for containment, and the limits probe reports whether
    the kernel can enforce limits afterwards.
    """
    try:
        leaf.mkdir(exist_ok=True)
        for pid in (current / "cgroup.procs").read_text().split():
            try:
                (leaf / "cgroup.procs").write_text(pid)
            except OSError as error:
                logger.warning("Cannot move process %s into %s: %s", pid, leaf, error)
    except OSError as error:
        logger.warning("Cannot create the worker leaf group %s: %s", leaf, error)


def delegated_cgroup_parent() -> Path | None:
    """Find a writable delegation inside the worker's own scope to enforce limits in.

    The parent is the worker's enclosing unit: the empty unit above a
    ``DelegateSubgroup`` subgroup, or the worker's own scope after its
    processes move into a child leaf group. Sandbox groups become siblings of
    that leaf, always inside the worker's container scope — never a pod slice
    or other parent above it. If the parent cannot apply CPU and memory
    limits, it is still returned for process containment and unavailable
    limits warn at use.
    """
    if not (CGROUP_ROOT / "cgroup.controllers").is_file():
        return None
    current = _current_cgroup()
    if not current.is_relative_to(CGROUP_ROOT):
        return None
    parent, leaf = _delegation_target(current)
    try:
        check_cgroup_parent(parent)
    except UnsupportedMachineSpec:
        return None
    if leaf is not None:
        _move_worker_processes(current, leaf)
    if probe_cgroup_limits(parent):
        return parent
    logger.warning("Cgroup delegation %s cannot enforce cpu/memory limits; using it for containment only", parent)
    return parent


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
