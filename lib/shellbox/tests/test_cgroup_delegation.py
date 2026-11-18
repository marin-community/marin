# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Delegation target selection and limit readback tests at the cgroup filesystem boundary."""

from pathlib import Path

import pytest
from shellbox.backends.local import cgroup as cgroup_module
from shellbox.backends.local.cgroup import (
    CPU_PERIOD,
    _current_cgroup,
    cgroup_limits_applied,
    delegated_cgroup_parent,
)


def _make_cgroup(root: Path, relative: str, *, procs: str) -> Path:
    path = root / relative
    path.mkdir(parents=True, exist_ok=True)
    (path / "cgroup.procs").write_text(procs)
    (path / "cgroup.kill").write_text("")
    return path


@pytest.fixture(autouse=True)
def kernel_procs_migration(monkeypatch: pytest.MonkeyPatch):
    """Give plain ``cgroup.procs`` files the kernel's migration semantics.

    The kernel moves each written pid into the destination group, whose file
    then lists all of its members. Plain files would only keep the last write,
    so writes accumulate the listed pids instead.
    """
    write_text = Path.write_text

    def write(path: Path, data: str, *args, **kwargs):
        if path.name == "cgroup.procs":
            members = set(path.read_text().split()) if path.exists() else set()
            members.update(data.split())
            return write_text(path, " ".join(sorted(members)) + "\n", *args, **kwargs)
        return write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write)


@pytest.fixture
def pod_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A fake pod hierarchy: pod slice wrapping the worker's container scope."""
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "cgroup.controllers").write_text("cpuset cpu io memory pids\n")
    scope = _make_cgroup(
        root,
        "kubepods.slice/kubepods-burstable.slice/kubepods-burstable-podA.slice/cri-containerd.scope",
        procs="1234 5678\n",
    )
    slice_level = _make_cgroup(root, "kubepods.slice/kubepods-burstable.slice/kubepods-burstable-podA.slice", procs="")
    monkeypatch.setattr(cgroup_module, "CGROUP_ROOT", root)
    monkeypatch.setattr(cgroup_module, "_current_cgroup", lambda: scope)
    return scope, slice_level


@pytest.fixture
def worker_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A fake host hierarchy: shared slice holding the worker's unit."""
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "cgroup.controllers").write_text("cpuset cpu io memory pids\n")
    slice_level = _make_cgroup(root, "system.slice", procs="")
    unit = _make_cgroup(slice_level, "marin-worker.service", procs="")
    monkeypatch.setattr(cgroup_module, "CGROUP_ROOT", root)
    return slice_level, unit


def _child_groups(path: Path) -> list[str]:
    return [entry.name for entry in path.iterdir() if entry.is_dir()]


def test_container_scope_delegates_from_a_leaf_inside_itself(pod_tree):
    """The worker sits directly in the container scope, so discovery moves it into a child leaf.

    The delegation parent stays the scope: sandbox groups become siblings of
    the worker leaf, and the pod slice above the container is never touched.
    """
    scope, slice_level = pod_tree
    assert delegated_cgroup_parent() == scope
    assert (scope / "worker" / "cgroup.procs").read_text().split() == ["1234", "5678"]
    assert not list(scope.glob("shellbox-*"))
    assert _child_groups(slice_level) == ["cri-containerd.scope"]


def test_a_moved_worker_reuses_its_leaf_without_nesting(pod_tree, monkeypatch):
    """Discovery from inside the worker leaf delegates from the scope, not from a nested leaf."""
    scope, _ = pod_tree
    leaf = _make_cgroup(scope, "worker", procs="1234\n")
    (scope / "cgroup.procs").write_text("9999\n")
    monkeypatch.setattr(cgroup_module, "_current_cgroup", lambda: leaf)
    assert delegated_cgroup_parent() == scope
    assert (leaf / "cgroup.procs").read_text().split() == ["1234"]
    assert not (leaf / "worker").exists()


def test_an_empty_enclosing_unit_reuses_the_worker_subgroup(worker_tree, monkeypatch):
    """The systemd DelegateSubgroup layout already provides the leaf subgroup."""
    _slice_level, unit = worker_tree
    delegate = _make_cgroup(unit, "shellbox-delegate", procs="1234\n")
    monkeypatch.setattr(cgroup_module, "_current_cgroup", lambda: delegate)
    assert delegated_cgroup_parent() == unit
    assert not (unit / "worker").exists()
    assert (delegate / "cgroup.procs").read_text().split() == ["1234"]


def test_a_worker_directly_in_its_unit_moves_into_a_leaf_there(worker_tree, monkeypatch):
    """A plain delegated worker unit delegates from itself, not from the shared slice."""
    slice_level, unit = worker_tree
    (unit / "cgroup.procs").write_text("1234 5678\n")
    monkeypatch.setattr(cgroup_module, "_current_cgroup", lambda: unit)
    assert delegated_cgroup_parent() == unit
    assert (unit / "worker" / "cgroup.procs").read_text().split() == ["1234", "5678"]
    assert _child_groups(slice_level) == ["marin-worker.service"]
    assert not list(unit.glob("shellbox-*"))


def test_delegation_refuses_an_unwritable_hierarchy(worker_tree, monkeypatch):
    """Without a writable delegated unit there is no containment parent at all."""
    _slice_level, unit = worker_tree
    (unit / "cgroup.kill").chmod(0o444)
    monkeypatch.setattr(cgroup_module, "_current_cgroup", lambda: unit)
    assert delegated_cgroup_parent() is None
    assert not (unit / "worker").exists()


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
    """Discovery enforces limits and leaves no throwaway cgroups on a delegated worker."""
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
