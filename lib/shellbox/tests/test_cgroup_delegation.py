# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Delegation candidate and limit readback tests."""

from pathlib import Path

import pytest
from shellbox.backends.local import cgroup as cgroup_module
from shellbox.backends.local.cgroup import (
    CPU_PERIOD,
    _fallback_parents,
    cgroup_limits_applied,
    delegated_cgroup_parent,
)


def _make_cgroup(root: Path, relative: str, *, procs: str) -> Path:
    path = root / relative
    path.mkdir(parents=True, exist_ok=True)
    (path / "cgroup.procs").write_text(procs)
    (path / "cgroup.kill").write_text("")
    return path


@pytest.fixture
def pod_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A fake pod hierarchy: pod slice wrapping the worker's container scope."""
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "cgroup.controllers").write_text("cpuset cpu io memory pids\n")
    scope = _make_cgroup(
        root,
        "kubepods.slice/kubepods-burstable.slice/kubepods-burstable-podA.slice/cri-containerd.scope",
        procs="1234\n",
    )
    slice_level = _make_cgroup(root, "kubepods.slice/kubepods-burstable.slice/kubepods-burstable-podA.slice", procs="")
    monkeypatch.setattr(cgroup_module, "CGROUP_ROOT", root)
    monkeypatch.setattr(cgroup_module, "_current_cgroup", lambda: scope)
    return scope, slice_level


def test_pod_slice_is_the_fallback_for_a_container_scope(pod_tree):
    scope, slice_level = pod_tree
    assert _fallback_parents(scope) == [slice_level]


def test_worker_directly_in_a_unit_has_no_fallback(pod_tree):
    scope, _ = pod_tree
    unit = _make_cgroup(scope.parent.parent, "marin-worker.service", procs="1234\n")
    (unit.parent / "cgroup.procs").write_text("")
    assert _fallback_parents(unit) == []


def test_delegate_subgroup_prefers_its_empty_unit(pod_tree):
    scope, _ = pod_tree
    unit = _make_cgroup(scope.parent.parent, "marin-worker.service", procs="")
    delegate = _make_cgroup(unit, "shellbox-delegate", procs="1234\n")
    assert _fallback_parents(delegate) == [unit]


def test_shared_host_slices_are_never_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A worker directly under a shared slice must not enable controllers there."""
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "cgroup.controllers").write_text("cpuset cpu io memory pids\n")
    _make_cgroup(root, "system.slice", procs="")
    worker = _make_cgroup(root, "system.slice/marin-worker.service", procs="1234\n")
    monkeypatch.setattr(cgroup_module, "CGROUP_ROOT", root)
    assert _fallback_parents(worker) == []


def test_pod_slice_candidates_require_no_internal_processes(pod_tree):
    scope, slice_level = pod_tree
    (slice_level / "cgroup.procs").write_text("9999\n")
    assert _fallback_parents(scope) == []


def test_containment_survives_when_no_candidate_enforces_limits(pod_tree):
    scope, slice_level = pod_tree
    assert delegated_cgroup_parent() == slice_level
    for candidate in (scope, slice_level):
        assert not list(candidate.glob("shellbox-*"))


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
    """Discovery leaves no throwaway cgroups on a delegated worker."""
    parent = delegated_cgroup_parent()
    if parent is None:
        pytest.skip("Run inside a worker with a delegated cgroup v2 hierarchy")
    before = {path for path in parent.parent.rglob("shellbox-*")} | {path for path in parent.rglob("shellbox-*")}
    assert delegated_cgroup_parent() == parent
    after = {path for path in parent.parent.rglob("shellbox-*")} | {path for path in parent.rglob("shellbox-*")}
    assert before == after == set()
