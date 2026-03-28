# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Git operations for the dispatcher: worktrees, logbook append, commit+push with retry."""

import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _run_git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = ["git", *args]
    logger.debug("git %s (cwd=%s)", " ".join(args), cwd)
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd), check=check)


def setup_worktree(repo_root: Path, branch: str) -> Path:
    """Create a temporary git worktree checked out at the given branch.

    If the branch does not exist locally or on the remote, it is created from HEAD.
    Returns the path to the worktree directory.
    """
    worktree_dir = Path(tempfile.mkdtemp(prefix="dispatch-wt-"))

    # Try to fetch the remote branch (harmless if it doesn't exist).
    _run_git(["fetch", "origin", branch], cwd=repo_root, check=False)

    # Check if the branch exists locally or on the remote.
    local_exists = _run_git(["rev-parse", "--verify", branch], cwd=repo_root, check=False).returncode == 0
    remote_exists = _run_git(["rev-parse", "--verify", f"origin/{branch}"], cwd=repo_root, check=False).returncode == 0

    if not local_exists and remote_exists:
        _run_git(["branch", branch, f"origin/{branch}"], cwd=repo_root, check=False)
    elif not local_exists and not remote_exists:
        # Bootstrap: create the branch from HEAD so first tick doesn't fail.
        _run_git(["branch", branch], cwd=repo_root)
        logger.info("Created new branch %s from HEAD", branch)

    _run_git(["worktree", "add", str(worktree_dir), branch], cwd=repo_root)
    logger.info("Created worktree at %s for branch %s", worktree_dir, branch)
    return worktree_dir


def cleanup_worktree(repo_root: Path, worktree_path: Path) -> None:
    """Remove a temporary git worktree."""
    _run_git(["worktree", "remove", "--force", str(worktree_path)], cwd=repo_root, check=False)
    logger.info("Cleaned up worktree %s", worktree_path)


def append_logbook(worktree_path: Path, logbook_rel_path: str, entry: str) -> None:
    """Append an entry to the logbook file, creating it if it does not exist."""
    logbook = worktree_path / logbook_rel_path
    logbook.parent.mkdir(parents=True, exist_ok=True)
    with open(logbook, "a") as f:
        f.write("\n" + entry.rstrip() + "\n")
    logger.info("Appended logbook entry to %s", logbook_rel_path)


def commit_and_push(
    worktree_path: Path,
    logbook_rel_path: str,
    message: str,
    branch: str,
    max_retries: int = 3,
) -> bool:
    """Stage the logbook, commit, and push. Retries with merge on conflict.

    Returns True on success, False if all retries are exhausted.
    """
    _run_git(["add", logbook_rel_path], cwd=worktree_path, check=False)

    # Commit if there are staged changes.
    status = _run_git(["status", "--porcelain"], cwd=worktree_path)
    if status.stdout.strip():
        _run_git(["commit", "-m", message], cwd=worktree_path)

    # Check for unpushed commits (covers both new commit and prior stranded commits).
    ahead = _run_git(["rev-list", "--count", f"origin/{branch}..HEAD"], cwd=worktree_path, check=False)
    if ahead.returncode == 0 and ahead.stdout.strip() == "0":
        logger.info("Nothing to push")
        return True

    for attempt in range(1, max_retries + 1):
        push = _run_git(["push", "origin", branch], cwd=worktree_path, check=False)
        if push.returncode == 0:
            logger.info("Pushed to %s on attempt %d", branch, attempt)
            return True

        logger.warning("Push failed (attempt %d/%d): %s", attempt, max_retries, push.stderr.strip())
        if attempt < max_retries:
            merge = _run_git(["pull", "--no-rebase", "origin", branch], cwd=worktree_path, check=False)
            if merge.returncode != 0:
                logger.error("Merge failed: %s", merge.stderr.strip())
                _run_git(["merge", "--abort"], cwd=worktree_path, check=False)
                return False

    return False
