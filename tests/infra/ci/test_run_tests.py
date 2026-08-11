# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the local affected-test runner."""

import subprocess
from pathlib import Path

from infra.ci.run_tests import PytestInvocation, local_invocations, pytest_command, worktree_diff
from infra.ci.select_tests import SelectionResult


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return result.stdout.strip()


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


def test_local_invocations_collapse_ci_shards_and_keep_package_configuration() -> None:
    selection = SelectionResult(
        reason="diff-driven",
        matrix=[
            {
                "label": "levanter 1/2",
                "package": "marin-levanter",
                "python": "3.12",
                "extras": "",
                "pytest_args": "--durations=5 -n auto --dist=worksteal --tb=short",
                "test_paths": "lib/levanter/tests/test_a.py",
                "setup": "",
                "timeout": 15,
            },
            {
                "label": "levanter 2/2",
                "package": "marin-levanter",
                "python": "3.12",
                "extras": "",
                "pytest_args": "--durations=5 -n auto --dist=worksteal --tb=short",
                "test_paths": "lib/levanter/tests/test_b.py",
                "setup": "",
                "timeout": 15,
            },
            {
                "label": "marin",
                "package": "marin-core",
                "python": "3.12",
                "extras": "--extra cpu --extra dedup",
                "pytest_args": "--durations=5 -n auto --dist=worksteal --tb=short",
                "test_paths": "tests/test_training.py",
                "setup": "",
                "timeout": 15,
            },
        ],
        suites=[],
    )

    invocations = local_invocations(selection)

    assert invocations == (
        PytestInvocation(
            label="levanter",
            package="marin-levanter",
            python="3.12",
            extras=(),
            pytest_args=("--durations=5", "-n", "auto", "--dist=worksteal", "--tb=short"),
            test_paths=("lib/levanter/tests/test_a.py", "lib/levanter/tests/test_b.py"),
            source_build=False,
        ),
        PytestInvocation(
            label="marin",
            package="marin-core",
            python="3.12",
            extras=("--extra", "cpu", "--extra", "dedup"),
            pytest_args=("--durations=5", "-n", "auto", "--dist=worksteal", "--tb=short"),
            test_paths=("tests/test_training.py",),
            source_build=False,
        ),
    )


def test_pytest_command_matches_unified_unit_defaults() -> None:
    invocation = PytestInvocation(
        label="marin",
        package="marin-core",
        python="3.12",
        extras=("--extra", "cpu"),
        pytest_args=("--durations=5", "-n", "auto", "--dist=worksteal", "--tb=short"),
        test_paths=("tests/test_training.py",),
        source_build=False,
    )

    assert pytest_command(invocation, ("-x",)) == (
        "uv",
        "run",
        "--isolated",
        "--python",
        "3.12",
        "--package",
        "marin-core",
        "--extra",
        "cpu",
        "--group",
        "test",
        "pytest",
        "--durations=5",
        "-n",
        "auto",
        "--dist=worksteal",
        "--tb=short",
        "tests/test_training.py",
        "-x",
    )
