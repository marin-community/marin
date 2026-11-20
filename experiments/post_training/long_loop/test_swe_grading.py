# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise branch bypass and candidate reconstruction without any accelerator."""

import subprocess
from pathlib import Path

import pytest

from experiments.post_training.long_loop.swe_grading import TaskPins, candidate_patch, restore_paths


def run_git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def pinned_repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    run_git(repo, "init", "-q")
    run_git(repo, "config", "user.name", "Fixture")
    run_git(repo, "config", "user.email", "fixture@example.invalid")
    (repo / "solution.py").write_text("VALUE = 1\n")
    (repo / "conftest.py").write_text("# trusted pytest config\n")
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_value.py").write_text("from solution import VALUE\ndef test_value(): assert VALUE == 1\n")
    run_git(repo, "add", ".")
    run_git(repo, "commit", "-qm", "reference")
    reference = run_git(repo, "rev-parse", "HEAD")
    (repo / "solution.py").write_text("VALUE = 0\n")
    run_git(repo, "commit", "-qam", "bug")
    bug = run_git(repo, "rev-parse", "HEAD")
    return repo, bug, reference


def test_default_branch_rejected_before_patch_extraction(pinned_repository):
    repo, bug, reference = pinned_repository
    run_git(repo, "checkout", "-q", reference)
    receipt, patch = candidate_patch(repo, bug, ["tests"], uid=None)
    assert receipt["reason"] == "wrong_ancestry"
    assert patch == b""


def test_candidate_patch_preserves_fix_but_not_pytest_hacks(pinned_repository, tmp_path):
    repo, bug, reference = pinned_repository
    (repo / "solution.py").write_text("VALUE = 1\n")
    (repo / "conftest.py").write_text("raise RuntimeError('agent forged pytest hook')\n")
    (repo / "tests/test_value.py").write_text("def test_value(): assert True\n")
    (repo / "helper.py").write_text("# new candidate source\n")
    run_git(repo, "add", "solution.py")
    run_git(repo, "commit", "-qm", "candidate fix")
    receipt, patch = candidate_patch(repo, bug, ["tests"], uid=None)
    assert receipt["accepted"]
    assert receipt["paths"] == ["helper.py", "solution.py"]
    clean = tmp_path / "clean"
    run_git(repo, "worktree", "add", "--detach", str(clean), bug)
    subprocess.run(["git", "apply", "--binary", "-"], cwd=clean, input=patch, check=True)
    pins = TaskPins("fixture", bug, reference, "bug", "", "", 30, 30)
    restore_paths(repo, clean, pins, ["tests"])
    assert (clean / "solution.py").read_text() == "VALUE = 1\n"
    assert (clean / "conftest.py").read_text() == "# trusted pytest config\n"
    assert "assert VALUE == 1" in (clean / "tests/test_value.py").read_text()
    assert (clean / "helper.py").exists()


def test_candidate_symlink_cannot_escape_reconstruction(pinned_repository):
    repo, bug, _ = pinned_repository
    (repo / "escape").symlink_to("/opt/long-loop-trusted/pins.json")
    receipt, patch = candidate_patch(repo, bug, [], uid=None)
    assert receipt["reason"] == "candidate_symlink"
    assert patch == b""
