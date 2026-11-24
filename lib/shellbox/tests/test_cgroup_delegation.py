# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Delegation target selection and limit readback tests against a fake cgroup v2 filesystem."""

import errno
import logging
import shutil
from pathlib import Path, PosixPath

import pytest
from shellbox.backends.local import cgroup as cgroup_module
from shellbox.backends.local.cgroup import (
    CPU_PERIOD,
    _current_cgroup,
    cgroup_limits_applied,
    delegated_cgroup_parent,
)

ALL_CONTROLLERS = "cpuset cpu io memory pids"
LIMIT_FILES = {
    "cpu.max": "cpu",
    "memory.max": "memory",
    "memory.swap.max": "memory",
    "memory.oom.group": "memory",
}


class KernelCgroupPath(PosixPath):
    """A temp-directory cgroup v2 hierarchy with the kernel rules discovery depends on.

    - A child group's ``cgroup.controllers`` is its parent's ``cgroup.subtree_control``.
    - Writing a pid to ``cgroup.procs`` migrates it: it leaves every other group.
    - No-internal-process rule: a non-root group that holds processes cannot enable
      controllers for its children (EBUSY), and a controller missing from
      ``cgroup.controllers`` cannot be enabled (ENOENT).
    - Limit files exist only while the parent enables their controller.
    - ``cgroup.kill`` removes every process in the subtree; ``rmdir`` refuses a
      populated group.
    """

    fs_root: Path

    def _is_root(self) -> bool:
        return self == self.fs_root

    def _groups(self) -> list[Path]:
        return [self.fs_root, *[path for path in self.fs_root.rglob("*") if path.is_dir()]]

    def _procs(self) -> set[str]:
        return set(PosixPath.read_text(self / "cgroup.procs").split())

    def _set_procs(self, pids: set[str]) -> None:
        PosixPath.write_text(self / "cgroup.procs", " ".join(sorted(pids)) + ("\n" if pids else ""))

    def _subtree(self) -> set[str]:
        return set(PosixPath.read_text(self / "cgroup.subtree_control").split())

    def mkdir(self, mode: int = 0o777, parents: bool = False, exist_ok: bool = False) -> None:
        existed = self.is_dir()
        PosixPath.mkdir(self, mode, parents=parents, exist_ok=exist_ok)
        if existed:
            return
        for group in [*reversed(list(self.parents)), self]:
            if not group.is_relative_to(self.fs_root) or (group / "cgroup.procs").exists():
                continue
            controllers = (
                ALL_CONTROLLERS
                if group == self.fs_root
                else PosixPath.read_text(group.parent / "cgroup.subtree_control")
            )
            for name, text in (
                ("cgroup.controllers", controllers.strip() + "\n"),
                ("cgroup.subtree_control", ""),
                ("cgroup.procs", ""),
                ("cgroup.kill", ""),
                ("cgroup.events", "populated 0\n"),
            ):
                PosixPath.write_text(group / name, text)

    def rmdir(self) -> None:
        if self._procs() or any(path.is_dir() for path in self.iterdir()):
            raise OSError(errno.EBUSY, "Device or resource busy", str(self))
        shutil.rmtree(self)

    def read_text(self, encoding=None, errors=None, newline=None) -> str:
        if self.name == "cgroup.events":
            group = type(self)(self.parent)
            populated = any(group_path._procs() for group_path in [group, *group.rglob("*")] if group_path.is_dir())
            return f"populated {int(populated)}\n"
        return PosixPath.read_text(self, encoding, errors)

    def write_text(self, data, encoding=None, errors=None, newline=None) -> int:
        group = type(self)(self.parent)
        if self.name == "cgroup.procs":
            if not group._is_root() and group._subtree():
                raise OSError(errno.EBUSY, "Device or resource busy", str(self))
            pids = set(data.split())
            for other in group._groups():
                type(self)(other)._set_procs(type(self)(other)._procs() - pids)
            group._set_procs(group._procs() | pids)
            return len(data)
        if self.name == "cgroup.subtree_control":
            enabled = group._subtree()
            available = set(PosixPath.read_text(group / "cgroup.controllers").split())
            for token in data.split():
                controller = token.lstrip("+-")
                if controller not in available:
                    raise OSError(errno.ENOENT, "No such file or directory", str(self))
                if token.startswith("+") and not group._is_root() and group._procs():
                    raise OSError(errno.EBUSY, "Device or resource busy", str(self))
                enabled = enabled | {controller} if token.startswith("+") else enabled - {controller}
            return PosixPath.write_text(self, " ".join(sorted(enabled)) + "\n")
        if self.name == "cgroup.kill":
            for member in [group, *group.rglob("*")]:
                if member.is_dir():
                    type(self)(member)._set_procs(set())
            return len(data)
        if self.name in LIMIT_FILES and LIMIT_FILES[self.name] not in type(self)(group.parent)._subtree():
            raise OSError(errno.ENOENT, "No such file or directory", str(self))
        return PosixPath.write_text(self, data, encoding, errors)


def _cgroup_fs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path_type = type("TestCgroupPath", (KernelCgroupPath,), {})
    root = path_type(tmp_path / "cgroup")
    path_type.fs_root = root
    root.mkdir()
    monkeypatch.setattr(cgroup_module, "CGROUP_ROOT", root)
    return root


def _group(root: Path, relative: str, *, controllers: str = ALL_CONTROLLERS, procs: str = "") -> Path:
    """Create ``relative`` with every ancestor delegating ``controllers`` to its children."""
    path = root
    for part in relative.split("/"):
        (path / "cgroup.subtree_control").write_text(" ".join(f"+{name}" for name in controllers.split()))
        path = path / part
        path.mkdir(exist_ok=True)
    if procs:
        (path / "cgroup.procs").write_text(procs)
    return path


POD_SLICE = "kubepods.slice/kubepods-burstable.slice/kubepods-burstable-podA.slice"


def _child_groups(path: Path) -> list[str]:
    return sorted(entry.name for entry in path.iterdir() if entry.is_dir())


def test_container_scope_moves_the_worker_into_a_leaf_and_can_enforce_limits(tmp_path, monkeypatch):
    """The worker sits directly in its container scope, so discovery empties the scope into a leaf.

    Only an empty scope can enable domain controllers for children, so the
    limits probe succeeds and the scope ends up delegating cpu and memory. The
    pod slice above the container gains no groups, and the probe leaves none
    behind.
    """
    root = _cgroup_fs(tmp_path, monkeypatch)
    scope = _group(root, f"{POD_SLICE}/cri-containerd.scope", procs="1234 5678")
    monkeypatch.setattr(cgroup_module, "_current_cgroup", lambda: scope)

    assert delegated_cgroup_parent() == scope
    assert (scope / "cgroup.procs").read_text().split() == []
    assert (scope / "worker" / "cgroup.procs").read_text().split() == ["1234", "5678"]
    assert {"cpu", "memory"} <= set((scope / "cgroup.subtree_control").read_text().split())
    assert _child_groups(scope) == ["worker"]
    assert _child_groups(scope.parent) == ["cri-containerd.scope"]


def test_scope_without_a_memory_controller_still_contains_the_worker(tmp_path, monkeypatch, caplog):
    """When the runtime withholds memory from the scope, discovery keeps the scope for containment only."""
    root = _cgroup_fs(tmp_path, monkeypatch)
    scope = _group(root, f"{POD_SLICE}/cri-containerd.scope", controllers="cpu pids", procs="1234")
    monkeypatch.setattr(cgroup_module, "_current_cgroup", lambda: scope)

    with caplog.at_level(logging.WARNING, logger=cgroup_module.__name__):
        assert delegated_cgroup_parent() == scope
    assert "containment only" in caplog.text
    assert (scope / "worker" / "cgroup.procs").read_text().split() == ["1234"]
    assert "memory" not in (scope / "cgroup.subtree_control").read_text().split()
    assert _child_groups(scope) == ["worker"]
    assert _child_groups(scope.parent) == ["cri-containerd.scope"]


def test_a_moved_worker_reuses_its_leaf_without_nesting(tmp_path, monkeypatch):
    """Discovery from inside the worker leaf delegates from the scope, not from a nested leaf."""
    root = _cgroup_fs(tmp_path, monkeypatch)
    scope = _group(root, f"{POD_SLICE}/cri-containerd.scope")
    leaf = _group(root, f"{POD_SLICE}/cri-containerd.scope/worker", procs="1234")
    monkeypatch.setattr(cgroup_module, "_current_cgroup", lambda: leaf)

    assert delegated_cgroup_parent() == scope
    assert (leaf / "cgroup.procs").read_text().split() == ["1234"]
    assert _child_groups(scope) == ["worker"]
    assert _child_groups(leaf) == []


def test_an_empty_enclosing_unit_delegates_without_moving_the_worker(tmp_path, monkeypatch):
    """The systemd DelegateSubgroup layout already provides the worker's leaf subgroup."""
    root = _cgroup_fs(tmp_path, monkeypatch)
    unit = _group(root, "system.slice/marin-worker.service")
    delegate = _group(root, "system.slice/marin-worker.service/shellbox-delegate", procs="1234")
    monkeypatch.setattr(cgroup_module, "_current_cgroup", lambda: delegate)

    assert delegated_cgroup_parent() == unit
    assert (delegate / "cgroup.procs").read_text().split() == ["1234"]
    assert _child_groups(unit) == ["shellbox-delegate"]


def test_a_worker_directly_in_its_unit_moves_into_a_leaf_there(tmp_path, monkeypatch):
    """A plain delegated worker unit delegates from itself, never from the shared slice above it."""
    root = _cgroup_fs(tmp_path, monkeypatch)
    unit = _group(root, "system.slice/marin-worker.service", procs="1234 5678")
    monkeypatch.setattr(cgroup_module, "_current_cgroup", lambda: unit)

    assert delegated_cgroup_parent() == unit
    assert (unit / "cgroup.procs").read_text().split() == []
    assert (unit / "worker" / "cgroup.procs").read_text().split() == ["1234", "5678"]
    assert _child_groups(unit.parent) == ["marin-worker.service"]


def test_delegation_refuses_an_unwritable_hierarchy(tmp_path, monkeypatch):
    """Without a writable delegated unit there is no containment parent, and nothing moves."""
    root = _cgroup_fs(tmp_path, monkeypatch)
    unit = _group(root, "system.slice/marin-worker.service", procs="1234")
    (unit / "cgroup.kill").chmod(0o444)
    monkeypatch.setattr(cgroup_module, "_current_cgroup", lambda: unit)

    assert delegated_cgroup_parent() is None
    assert (unit / "cgroup.procs").read_text().split() == ["1234"]
    assert _child_groups(unit) == []


def test_cgroup_limits_applied_reads_back_written_files(tmp_path: Path):
    applied = tmp_path / "applied"
    applied.mkdir()
    (applied / "cpu.max").write_text(f"{CPU_PERIOD} {CPU_PERIOD}\n")
    (applied / "memory.max").write_text(str(1024 * 1024 * 1024) + "\n")
    assert cgroup_limits_applied(applied, cpus=1, memory_mb=1024)

    unbounded = tmp_path / "unbounded"
    unbounded.mkdir()
    (unbounded / "cpu.max").write_text("max 100000\n")
    (unbounded / "memory.max").write_text("max\n")
    assert not cgroup_limits_applied(unbounded, cpus=1, memory_mb=1024)

    wrong = tmp_path / "wrong"
    wrong.mkdir()
    (wrong / "cpu.max").write_text(f"{2 * CPU_PERIOD} {CPU_PERIOD}\n")
    (wrong / "memory.max").write_text(str(512 * 1024 * 1024) + "\n")
    assert not cgroup_limits_applied(wrong, cpus=1, memory_mb=1024)

    missing = tmp_path / "missing"
    missing.mkdir()
    assert not cgroup_limits_applied(missing, cpus=1, memory_mb=1024)


@pytest.mark.manual
def test_delegated_parent_enforces_and_cleans_up_delegation_probes():
    """Discovery stays inside the worker's unit and leaves no throwaway cgroups on a delegated worker."""
    parent = delegated_cgroup_parent()
    if parent is None:
        pytest.skip("Run inside a worker with a delegated cgroup v2 hierarchy")
    boundary = next(
        (path for path in (_current_cgroup(), *_current_cgroup().parents) if path.name.endswith((".service", ".scope"))),
        _current_cgroup(),
    )
    assert parent.is_relative_to(boundary)
    before = {path for path in boundary.rglob("shellbox-*")}
    assert delegated_cgroup_parent() == parent
    after = {path for path in boundary.rglob("shellbox-*")}
    assert before == after == set()
