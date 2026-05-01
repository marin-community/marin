# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/workflows/changes.py

Uses real git repositories created in tmp_path for integration-style coverage.
CLI tests use Click's CliRunner where no real git is needed; subprocess invocations
are used only where end-to-end git execution matters.
"""

import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from click.testing import CliRunner

_CHANGES_PY = Path(__file__).parents[3] / "scripts" / "workflows" / "changes.py"
_spec = importlib.util.spec_from_file_location("changes", _CHANGES_PY)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
sys.modules["changes"] = _mod

changed_paths = _mod.changed_paths
match_groups = _mod.match_groups
PathDecision = _mod.PathDecision
cli = _mod.cli


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GitRepo:
    """A minimal git repository with two commits for diff testing."""

    path: Path
    base_sha: str
    head_sha: str


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """Run a git command in repo, raising on failure."""
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _sha(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


@pytest.fixture()
def git_repo(tmp_path: Path) -> GitRepo:
    """Create a minimal two-commit git repo for diff tests.

    Commit A (base): lib/marin/core.py, pyproject.toml
    Commit B (head): adds lib/fray/runner.py, modifies lib/marin/core.py,
                     adds lib/docs/index.md
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "commit.gpgsign", "false")

    # Commit A — base
    (repo / "lib" / "marin").mkdir(parents=True)
    (repo / "lib" / "marin" / "core.py").write_text("# core\n")
    (repo / "pyproject.toml").write_text("[project]\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base commit")
    base_sha = _sha(repo)

    # Commit B — head
    (repo / "lib" / "fray").mkdir(parents=True)
    (repo / "lib" / "fray" / "runner.py").write_text("# runner\n")
    (repo / "lib" / "marin" / "core.py").write_text("# core updated\n")
    (repo / "lib" / "docs").mkdir(parents=True)
    (repo / "lib" / "docs" / "index.md").write_text("# docs\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "head commit")
    head_sha = _sha(repo)

    return GitRepo(path=repo, base_sha=base_sha, head_sha=head_sha)


@pytest.fixture()
def git_repo_with_rename(tmp_path: Path) -> GitRepo:
    """Repo where head renames a file from old_name.py to new_name.py."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "commit.gpgsign", "false")

    (repo / "lib").mkdir()
    (repo / "lib" / "old_name.py").write_text("# old\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    base_sha = _sha(repo)

    _git(repo, "mv", "lib/old_name.py", "lib/new_name.py")
    _git(repo, "commit", "-m", "rename")
    head_sha = _sha(repo)

    return GitRepo(path=repo, base_sha=base_sha, head_sha=head_sha)


@pytest.fixture()
def git_repo_with_deletion(tmp_path: Path) -> GitRepo:
    """Repo where head only deletes a file (no other changes)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "commit.gpgsign", "false")

    (repo / "lib").mkdir()
    (repo / "lib" / "gone.py").write_text("# gone\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    base_sha = _sha(repo)

    _git(repo, "rm", "lib/gone.py")
    _git(repo, "commit", "-m", "delete")
    head_sha = _sha(repo)

    return GitRepo(path=repo, base_sha=base_sha, head_sha=head_sha)


# ---------------------------------------------------------------------------
# Unit tests: changed_paths
# ---------------------------------------------------------------------------


def test_changed_paths_returns_sorted_relative_paths(git_repo: GitRepo) -> None:
    paths = changed_paths(git_repo.base_sha, git_repo.head_sha, repo=git_repo.path)
    # Should include the three paths modified/added in head commit.
    assert "lib/fray/runner.py" in paths
    assert "lib/marin/core.py" in paths
    assert "lib/docs/index.md" in paths
    # Must be sorted.
    assert list(paths) == sorted(paths)


def test_changed_paths_excludes_unmodified_files(git_repo: GitRepo) -> None:
    # pyproject.toml was in the base commit but not touched in head.
    paths = changed_paths(git_repo.base_sha, git_repo.head_sha, repo=git_repo.path)
    assert "pyproject.toml" not in paths


# ---------------------------------------------------------------------------
# Unit tests: match_groups (pure, no git)
# ---------------------------------------------------------------------------


def test_match_groups_single_positive_glob() -> None:
    paths = ["lib/marin/core.py", "lib/fray/runner.py"]
    (decision,) = match_groups(paths, {"marin": ["lib/marin/**"]})
    assert decision.matched is True
    assert "lib/marin/core.py" in decision.paths
    assert "lib/fray/runner.py" not in decision.paths


def test_match_groups_double_star_recursion() -> None:
    paths = ["lib/marin/sub/deep/module.py", "lib/other/file.py"]
    (decision,) = match_groups(paths, {"deep": ["lib/marin/**"]})
    assert decision.matched is True
    assert "lib/marin/sub/deep/module.py" in decision.paths
    assert "lib/other/file.py" not in decision.paths


def test_match_groups_negation_excludes_subpath() -> None:
    paths = ["docs/guide.md", "docs/_build/output.html", "docs/api.md"]
    (decision,) = match_groups(paths, {"docs": ["docs/**", "!docs/_build/**"]})
    assert decision.matched is True
    assert "docs/guide.md" in decision.paths
    assert "docs/api.md" in decision.paths
    # Negated path must not appear.
    assert "docs/_build/output.html" not in decision.paths


def test_match_groups_negation_all_excluded_gives_no_match() -> None:
    paths = ["docs/_build/output.html"]
    (decision,) = match_groups(paths, {"docs": ["docs/**", "!docs/_build/**"]})
    assert decision.matched is False
    assert decision.paths == ()


def test_match_groups_multiple_groups_partial_match() -> None:
    paths = ["lib/marin/core.py", "pyproject.toml"]
    decisions = match_groups(paths, {"marin": ["lib/marin/**", "pyproject.toml"], "fray": ["lib/fray/**"]})
    by_name = {d.name: d for d in decisions}
    assert by_name["marin"].matched is True
    assert by_name["fray"].matched is False


def test_match_groups_no_paths_gives_no_match() -> None:
    (decision,) = match_groups([], {"marin": ["lib/marin/**"]})
    assert decision.matched is False
    assert decision.paths == ()


def test_match_groups_preserves_group_order() -> None:
    paths = ["lib/a/f.py", "lib/b/f.py"]
    groups = {"b": ["lib/b/**"], "a": ["lib/a/**"]}
    decisions = match_groups(paths, groups)
    assert decisions[0].name == "b"
    assert decisions[1].name == "a"


def test_match_groups_rename_matches_new_path() -> None:
    # Simulate that changed_paths already returned the new path name (rename semantics).
    paths = ["lib/new_name.py"]
    (decision,) = match_groups(paths, {"lib": ["lib/**"]})
    assert decision.matched is True
    assert "lib/new_name.py" in decision.paths


# ---------------------------------------------------------------------------
# Integration tests: changed_paths with real git — rename and delete
# ---------------------------------------------------------------------------


def test_changed_paths_rename_shows_new_path(git_repo_with_rename: GitRepo) -> None:
    paths = changed_paths(git_repo_with_rename.base_sha, git_repo_with_rename.head_sha, repo=git_repo_with_rename.path)
    assert "lib/new_name.py" in paths
    assert "lib/old_name.py" not in paths


def test_changed_paths_deleted_file_not_included(git_repo_with_deletion: GitRepo) -> None:
    paths = changed_paths(
        git_repo_with_deletion.base_sha, git_repo_with_deletion.head_sha, repo=git_repo_with_deletion.path
    )
    assert "lib/gone.py" not in paths
    assert paths == ()


# ---------------------------------------------------------------------------
# CLI tests via CliRunner
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_cli_always_match_returns_matched_true_empty_paths(runner: CliRunner) -> None:
    result = runner.invoke(
        cli,
        [
            "match",
            "--always-match",
            "--group",
            "marin=lib/marin/**,tests/**",
            "--group",
            "fray=lib/fray/**",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.splitlines()[0])
    assert data["reason"] == "manual-or-scheduled"
    by_name = {g["name"]: g for g in data["groups"]}
    assert by_name["marin"]["matched"] is True
    assert by_name["marin"]["paths"] == []
    assert by_name["fray"]["matched"] is True
    assert by_name["fray"]["paths"] == []


def test_cli_usage_error_always_match_with_base(runner: CliRunner) -> None:
    result = runner.invoke(
        cli,
        [
            "match",
            "--always-match",
            "--base",
            "abc123",
            "--head",
            "def456",
            "--group",
            "marin=lib/marin/**",
        ],
    )
    assert result.exit_code != 0


def test_cli_usage_error_missing_base_or_head(runner: CliRunner) -> None:
    result = runner.invoke(
        cli,
        [
            "match",
            "--base",
            "abc123",
            "--group",
            "marin=lib/marin/**",
        ],
    )
    assert result.exit_code != 0


def test_cli_github_output_writes_lowercase_bools(runner: CliRunner, tmp_path: Path, git_repo: GitRepo) -> None:
    output_file = tmp_path / "github_output.txt"
    env = {"GITHUB_OUTPUT": str(output_file)}
    result = runner.invoke(
        cli,
        [
            "match",
            "--base",
            git_repo.base_sha,
            "--head",
            git_repo.head_sha,
            "--repo",
            str(git_repo.path),
            "--group",
            "marin=lib/marin/**",
            "--group",
            "fray=lib/fray/**",
            "--github-output",
        ],
        env=env,
    )
    assert result.exit_code == 0, result.output
    contents = output_file.read_text()
    lines = {line.split("=")[0]: line.split("=")[1] for line in contents.splitlines() if "=" in line}
    assert lines["marin"] == "true"
    assert lines["fray"] == "true"


def test_cli_github_output_no_error_when_env_unset(runner: CliRunner) -> None:
    """--github-output is a no-op when GITHUB_OUTPUT is not set."""
    result = runner.invoke(
        cli,
        [
            "match",
            "--always-match",
            "--group",
            "marin=lib/marin/**",
            "--github-output",
        ],
        env={},  # GITHUB_OUTPUT not set
    )
    assert result.exit_code == 0, result.output


def test_cli_diff_mode_produces_correct_json(runner: CliRunner, git_repo: GitRepo) -> None:
    result = runner.invoke(
        cli,
        [
            "match",
            "--base",
            git_repo.base_sha,
            "--head",
            git_repo.head_sha,
            "--repo",
            str(git_repo.path),
            "--group",
            "marin=lib/marin/**",
            "--group",
            "fray=lib/fray/**",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.splitlines()[0])
    assert data["reason"] == "diff"
    by_name = {g["name"]: g for g in data["groups"]}
    assert by_name["marin"]["matched"] is True
    assert "lib/marin/core.py" in by_name["marin"]["paths"]
    assert by_name["fray"]["matched"] is True
    assert "lib/fray/runner.py" in by_name["fray"]["paths"]


def test_cli_deleted_file_does_not_match(runner: CliRunner, git_repo_with_deletion: GitRepo) -> None:
    result = runner.invoke(
        cli,
        [
            "match",
            "--base",
            git_repo_with_deletion.base_sha,
            "--head",
            git_repo_with_deletion.head_sha,
            "--repo",
            str(git_repo_with_deletion.path),
            "--group",
            "lib=lib/**",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.splitlines()[0])
    (group,) = data["groups"]
    assert group["matched"] is False
    assert group["paths"] == []
