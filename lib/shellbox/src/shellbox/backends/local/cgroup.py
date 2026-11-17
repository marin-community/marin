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
    runtimes also refuse the memory controller outright; a candidate that
    silently drops limits is worse than no candidate, so this probes and reads
    back instead of trusting the writes.
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


def _fallback_parents(current: Path) -> list[Path]:
    """Parents that may enforce limits when the worker's own cgroup cannot.

    A candidate must have no internal processes (the no-internal-process rule
    otherwise blocks enabling controllers) and must be dedicated to this
    worker: a systemd DelegateSubgroup unit, or the container runtime's
    pod-private slice. Shared host slices are never candidates — enabling
    controllers there would change accounting for unrelated workloads.
    A verifying dedicated parent is preferred over the worker's own cgroup;
    the worker's cgroup is tried next.
    """
    unit_suffixes = (".service", ".scope")
    parent = current.parent
    if parent == current or not parent.is_relative_to(CGROUP_ROOT):
        return []
    if (parent / "cgroup.procs").read_text().strip():
        return []
    if parent.name.endswith(unit_suffixes):
        return [parent]
    if parent.name.startswith("kubepods") and "-pod" in parent.name:
        return [parent]
    return []


def delegated_cgroup_parent() -> Path | None:
    """Find a writable delegation, preferring one that can enforce resource limits.

    Candidates are the worker's own cgroup and any dedicated parent unit
    (:func:`_fallback_parents`). Container runtimes scope the worker into a
    ``cri-containerd`` scope whose subtree cannot enable domain controllers
    until the worker moves out; the wrapping pod slice can. Each candidate is
    verified by applying and reading back limits in a throwaway machine
    cgroup, and a verifying candidate wins. If no candidate can enforce
    limits, the first candidate that still passes ``check_cgroup_parent`` is
    returned so process containment is preserved and the limits degrade to
    the documented warn-and-continue behavior.
    """
    if not (CGROUP_ROOT / "cgroup.controllers").is_file():
        return None
    current = _current_cgroup()
    if not current.is_relative_to(CGROUP_ROOT):
        return None
    candidates = [current]
    # DelegateSubgroup moves the worker below its unit, which then verifies.
    for parent in _fallback_parents(current):
        if parent not in candidates:
            candidates.insert(0, parent)
    contained = []
    for parent in candidates:
        try:
            check_cgroup_parent(parent)
        except UnsupportedMachineSpec:
            continue
        contained.append(parent)
        if probe_cgroup_limits(parent):
            return parent
    if contained:
        logger.warning(
            "No cgroup delegation candidate could enforce cpu/memory limits; using %s for containment only",
            contained[0],
        )
        return contained[0]
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
