# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LOCAL resource-evidence verification for the capability survey."""

from pathlib import Path

from shellbox.machine import Backend
from shellbox.survey import (
    LocalResourceEvidence,
    cgroup_limits_applied,
    local_resource_evidence,
    production_checkers,
)


def _fake_cgroup(path: Path, *, quota: str, period: int, memory: int) -> Path:
    path.mkdir(parents=True)
    (path / "cpu.max").write_text(f"{quota} {period}\n")
    (path / "memory.max").write_text(f"{memory}\n")
    return path


def test_cgroup_limits_applied_reads_back_kernel_state(tmp_path: Path):
    applied = _fake_cgroup(tmp_path / "applied", quota="100000", period=100000, memory=1024 * 1024 * 1024)
    assert cgroup_limits_applied(applied, cpus=1, memory_mb=1024)

    unbounded = _fake_cgroup(tmp_path / "unbounded", quota="max", period=100000, memory="max")
    assert not cgroup_limits_applied(unbounded, cpus=1, memory_mb=1024)

    wrong_size = _fake_cgroup(tmp_path / "wrong", quota="200000", period=100000, memory=512 * 1024 * 1024)
    assert not cgroup_limits_applied(wrong_size, cpus=1, memory_mb=1024)

    missing = tmp_path / "missing"
    missing.mkdir()
    assert not cgroup_limits_applied(missing, cpus=1, memory_mb=1024)


def test_local_resource_evidence_without_machinery_stays_unverified():
    """Without a pool or delegated cgroup the evidence claims nothing."""
    evidence = local_resource_evidence(None)
    assert evidence == LocalResourceEvidence(storage_enforced=False, cgroup_verified=False, cgroup_parent=None)
    if evidence.cgroup_verified:
        # On a worker with delegation the probe ran; the parent must be recorded.
        assert evidence.cgroup_parent is not None


def test_production_checkers_imports_every_backend_module():
    """The rename guard: every checker's backend module must import by its label."""
    checkers = production_checkers()
    expected = {backend.value for backend in Backend}
    assert set(checkers) == expected
