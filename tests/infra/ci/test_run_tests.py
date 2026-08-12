# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the local affected-test runner."""

import subprocess
import sys
from pathlib import Path

from infra.ci.run_tests import (
    PackageSelection,
    PytestInvocation,
    local_invocation,
    pytest_command,
    pytest_invocations,
    worktree_diff,
)
from infra.ci.select_tests import MatrixLeg, SelectionResult


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def test_direct_script_entrypoint_resolves_workspace_imports() -> None:
    repo_root = Path(__file__).resolve().parents[3]

    subprocess.run(
        [sys.executable, "infra/ci/run_tests.py", "--base-ref", "HEAD", "--dry-run"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )


def test_worktree_diff_includes_branch_local_and_untracked_changes(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-b", "main")
    _git(tmp_path, "config", "user.name", "Test User")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "commit.gpgsign", "false")

    tracked = tmp_path / "tracked.py"
    staged = tmp_path / "staged.py"
    deleted = tmp_path / "deleted.py"
    tracked.write_text("BASE = 1\n")
    staged.write_text("BASE = 1\n")
    deleted.write_text("DELETE = 1\n")
    _git(tmp_path, "add", "tracked.py", "staged.py", "deleted.py")
    _git(tmp_path, "commit", "-m", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")

    tracked.write_text("BRANCH = 1\n")
    _git(tmp_path, "add", "tracked.py")
    _git(tmp_path, "commit", "-m", "branch change")
    staged.write_text("STAGED = 1\n")
    _git(tmp_path, "add", "staged.py")
    deleted.unlink()
    (tmp_path / "untracked.py").write_text("NEW = 1\n")

    diff = worktree_diff(base, tmp_path)

    assert diff.merge_base == base
    assert set(diff.changed_files) == {"tracked.py", "staged.py", "deleted.py", "untracked.py"}


def test_local_invocation_combines_ci_shards_in_one_workspace_process() -> None:
    selection = SelectionResult(
        reason="diff-driven",
        matrix=[
            MatrixLeg(
                label="levanter 1/2",
                package="marin-levanter",
                python="3.12",
                extras="",
                pytest_args="--durations=5 -n auto --dist=worksteal --tb=short",
                test_paths="lib/levanter/tests/test_a.py",
                setup="",
                timeout=15,
            ),
            MatrixLeg(
                label="levanter 2/2",
                package="marin-levanter",
                python="3.12",
                extras="",
                pytest_args="--durations=5 -n auto --dist=worksteal --tb=short",
                test_paths="lib/levanter/tests/test_b.py",
                setup="",
                timeout=15,
            ),
            MatrixLeg(
                label="marin",
                package="marin-core",
                python="3.12",
                extras="--extra cpu --extra dedup",
                pytest_args="--durations=5 -n auto --dist=worksteal --tb=short",
                test_paths="tests/test_training.py",
                setup="",
                timeout=15,
            ),
        ],
        suites=[],
        suite_test_paths={},
    )

    invocation = local_invocation(selection)

    assert invocation == PytestInvocation(
        python="3.12",
        extras=("cpu", "dedup"),
        pytest_args=("--durations=5", "-n", "auto", "--dist=worksteal", "--tb=short"),
        packages=(
            PackageSelection(
                label="levanter",
                test_paths=("lib/levanter/tests/test_a.py", "lib/levanter/tests/test_b.py"),
                source_build=False,
            ),
            PackageSelection(
                label="marin",
                test_paths=("tests/test_training.py",),
                source_build=False,
            ),
        ),
    )

    command = pytest_command(invocation, ("-x",))
    assert command[:2] == ("uv", "run")
    assert "--all-packages" in command
    assert "--no-default-groups" in command
    assert command.count("--extra") == 2
    marker_expression = command[command.index("-m") + 1]
    assert "not requires_cluster" in marker_expression
    assert "not torch" in marker_expression
    assert command[-4:] == (
        "lib/levanter/tests/test_a.py",
        "lib/levanter/tests/test_b.py",
        "tests/test_training.py",
        "-x",
    )


def test_haliax_runs_in_a_clean_jax_process_when_other_packages_are_selected() -> None:
    invocation = PytestInvocation(
        python="3.12",
        extras=("cpu",),
        pytest_args=("-n", "auto"),
        packages=(
            PackageSelection("haliax", ("lib/haliax/tests/test_axis.py",), False),
            PackageSelection("levanter", ("lib/levanter/tests/test_model.py",), False),
        ),
    )

    phases = pytest_invocations(invocation)

    assert [phase.test_paths for phase in phases] == [
        ("lib/haliax/tests/test_axis.py",),
        ("lib/levanter/tests/test_model.py",),
    ]
    assert "--no-sync" not in pytest_command(phases[0])
    assert "--no-sync" in pytest_command(phases[1], sync=False)
