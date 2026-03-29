# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for git operations using real git repos."""

import subprocess
from pathlib import Path


from marin.dispatch.git_ops import PushResult, append_logbook, cleanup_worktree, commit_and_push, setup_worktree


def _init_bare_repo(path: Path) -> Path:
    """Create a bare git repo to act as 'origin'."""
    bare = path / "origin.git"
    bare.mkdir()
    subprocess.run(["git", "init", "--bare", "--initial-branch=main", str(bare)], check=True, capture_output=True)
    return bare


def _init_working_repo(path: Path, bare_path: Path, branch: str = "main") -> Path:
    """Clone from bare repo and create initial commit."""
    repo = path / "repo"
    subprocess.run(["git", "clone", str(bare_path), str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), check=True, capture_output=True)
    # Ensure we're on main branch.
    subprocess.run(["git", "checkout", "-B", "main"], cwd=str(repo), check=True, capture_output=True)
    # Create initial commit on main.
    (repo / "README.md").write_text("init")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "push", "-u", "origin", "main"], cwd=str(repo), check=True, capture_output=True)
    # Create the target branch.
    if branch != "main":
        subprocess.run(["git", "checkout", "-b", branch], cwd=str(repo), check=True, capture_output=True)
        subprocess.run(["git", "push", "origin", branch], cwd=str(repo), check=True, capture_output=True)
        subprocess.run(["git", "checkout", "main"], cwd=str(repo), check=True, capture_output=True)
    return repo


def test_append_logbook_creates_file(tmp_path):
    worktree = tmp_path / "wt"
    worktree.mkdir()
    append_logbook(worktree, "logs/test.md", "## Entry 1\nSome content.")
    logbook = worktree / "logs" / "test.md"
    assert logbook.exists()
    assert "## Entry 1" in logbook.read_text()


def test_append_logbook_appends(tmp_path):
    worktree = tmp_path / "wt"
    worktree.mkdir()
    logbook_path = worktree / "log.md"
    logbook_path.write_text("# Log\n")
    append_logbook(worktree, "log.md", "Entry 1")
    append_logbook(worktree, "log.md", "Entry 2")
    text = logbook_path.read_text()
    assert "Entry 1" in text
    assert "Entry 2" in text


def test_commit_and_push(tmp_path):
    bare = _init_bare_repo(tmp_path)
    repo = _init_working_repo(tmp_path, bare, branch="research/test")
    wt = setup_worktree(repo, "research/test")
    try:
        append_logbook(wt, "log.md", "## Test entry")
        result = commit_and_push(wt, "log.md", "test: add entry", "research/test")
        assert result == PushResult.SUCCESS
    finally:
        cleanup_worktree(repo, wt)


def test_commit_and_push_nothing_to_commit(tmp_path):
    bare = _init_bare_repo(tmp_path)
    repo = _init_working_repo(tmp_path, bare, branch="research/test")
    wt = setup_worktree(repo, "research/test")
    try:
        # Don't append anything — should return True (nothing to commit).
        result = commit_and_push(wt, "log.md", "noop", "research/test")
        assert result == PushResult.SUCCESS
    finally:
        cleanup_worktree(repo, wt)


def test_commit_and_push_conflict_retry(tmp_path):
    """Simulate a conflict by pushing from a second clone first."""
    bare = _init_bare_repo(tmp_path)
    repo = _init_working_repo(tmp_path, bare, branch="research/conflict")

    wt1 = setup_worktree(repo, "research/conflict")

    # Create a second clone to push a competing commit.
    clone2 = tmp_path / "clone2"
    subprocess.run(["git", "clone", str(bare), str(clone2)], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(clone2), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(clone2), check=True, capture_output=True)
    subprocess.run(["git", "checkout", "research/conflict"], cwd=str(clone2), check=True, capture_output=True)

    try:
        # Write and commit from wt1 but don't push yet.
        append_logbook(wt1, "log.md", "Entry from wt1")
        subprocess.run(["git", "add", "log.md"], cwd=str(wt1), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "wt1 entry"], cwd=str(wt1), check=True, capture_output=True)

        # Push a competing commit from clone2.
        (clone2 / "other.txt").write_text("conflict creator")
        subprocess.run(["git", "add", "other.txt"], cwd=str(clone2), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "clone2 commit"], cwd=str(clone2), check=True, capture_output=True)
        subprocess.run(["git", "push", "origin", "research/conflict"], cwd=str(clone2), check=True, capture_output=True)

        # Now commit_and_push from wt1 — first push will fail (non-fast-forward),
        # merge will succeed (no real conflict), and retry push will work.
        # Since we already committed, git add + status will show nothing new,
        # but the existing unpushed commit still needs to be pushed.
        # We need to call the push-retry part. Let's add another entry so there's
        # something to commit.
        append_logbook(wt1, "log.md", "Entry 2 from wt1")
        result = commit_and_push(wt1, "log.md", "wt1 second entry", "research/conflict")
        assert result == PushResult.SUCCESS
    finally:
        cleanup_worktree(repo, wt1)
