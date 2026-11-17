# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Delegation discovery verifies limits instead of trusting writable subtree writes."""

from pathlib import Path

import pytest
from shellbox.backends.local import cgroup as cgroup_module
from shellbox.backends.local.cgroup import delegated_cgroup_parent
from shellbox.machine import UnsupportedMachineSpec


@pytest.fixture
def cgroup_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake cgroup v2 root with a pod slice, container scope and worker files."""
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "cgroup.controllers").write_text("cpuset cpu io memory pids\n")
    scope = root / "kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod.slice/cri-containerd.scope"
    scope.mkdir(parents=True)
    (scope / "cgroup.procs").write_text("1234\n")
    (scope / "cgroup.kill").write_text("")
    monkeypatch.setattr(cgroup_module, "CGROUP_ROOT", root)
    monkeypatch.setattr(cgroup_module, "_current_cgroup", lambda: scope)
    return root


def _patch_checks(monkeypatch: pytest.MonkeyPatch, writable: set[str], applying: set[str]) -> list[Path]:
    """check_cgroup_parent passes only for ``writable``; the limits probe only for ``applying``."""
    checked = []

    def fake_check(parent: Path) -> None:
        checked.append(parent)
        if parent not in writable:
            raise UnsupportedMachineSpec("Local process containment requires a delegated cgroup v2 directory")

    def fake_probe(parent: Path) -> bool:
        return parent in applying

    monkeypatch.setattr(cgroup_module, "check_cgroup_parent", fake_check)
    monkeypatch.setattr(cgroup_module, "probe_cgroup_limits", fake_probe)
    return checked


def test_scope_failure_falls_back_to_the_wrapping_pod_slice(cgroup_tree: Path, monkeypatch: pytest.MonkeyPatch):
    """Container scopes cannot enable domain controllers with the worker inside; the pod slice can."""
    scope = cgroup_module._current_cgroup()
    slice_level = scope.parent
    checked = _patch_checks(
        monkeypatch,
        writable={scope, slice_level},
        applying={slice_level},
    )
    assert delegated_cgroup_parent() == slice_level
    assert checked == [scope, slice_level]


def test_worker_scope_wins_when_it_can_enforce_limits(cgroup_tree: Path, monkeypatch: pytest.MonkeyPatch):
    """On a plain delegated worker nothing changes: the worker's own cgroup stays the parent."""
    scope = cgroup_module._current_cgroup()
    _patch_checks(monkeypatch, writable={scope, scope.parent}, applying={scope})
    assert delegated_cgroup_parent() == scope


def test_no_candidate_enforcing_limits_returns_none(cgroup_tree: Path, monkeypatch: pytest.MonkeyPatch):
    """A delegation that silently drops limits must never become the parent."""
    scope = cgroup_module._current_cgroup()
    _patch_checks(monkeypatch, writable={scope, scope.parent}, applying=set())
    assert delegated_cgroup_parent() is None


def test_delegate_subgroup_still_prefers_the_unit(cgroup_tree: Path, monkeypatch: pytest.MonkeyPatch):
    """Systemd DelegateSubgroup setups keep the existing preference for the unit parent."""
    scope = cgroup_module._current_cgroup()
    delegate = scope / "shellbox-delegate"
    delegate.mkdir()
    # the worker moved below the unit, so the unit itself has no direct processes
    (scope / "cgroup.procs").write_text("")
    (delegate / "cgroup.procs").write_text("1234\n")
    monkeypatch.setattr(cgroup_module, "_current_cgroup", lambda: delegate)
    checked = _patch_checks(
        monkeypatch,
        writable={delegate, scope, scope.parent},
        applying={scope, scope.parent},
    )
    assert delegated_cgroup_parent() == scope
    assert checked[0] == scope


def test_probe_rejecting_a_candidate_keeps_searching(cgroup_tree: Path, monkeypatch: pytest.MonkeyPatch):
    """check passes but the limits probe fails for the scope: the slice is still found."""
    scope = cgroup_module._current_cgroup()
    slice_level = scope.parent
    checked = _patch_checks(
        monkeypatch,
        writable={scope, slice_level},
        applying={slice_level},
    )
    # the scope passes check_cgroup_parent but drops the limits on readback
    assert delegated_cgroup_parent() == slice_level
    assert checked == [scope, slice_level]
