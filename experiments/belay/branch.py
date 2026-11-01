# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Git checks and the weekly branch: every Belay run executes a clean, pushed commit of this week's branch."""

import contextlib
import datetime
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

from experiments.belay.config import CONFIG


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def repo_root() -> Path:
    return Path(git(Path.cwd(), "rev-parse", "--show-toplevel"))


def current_branch(repo: Path) -> str:
    return git(repo, "rev-parse", "--abbrev-ref", "HEAD")


def head_contains(repo: Path, ref: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", ref, "HEAD"], cwd=repo).returncode == 0


def remote_branch_exists(repo: Path, branch: str) -> bool:
    return bool(git(repo, "ls-remote", "--heads", "origin", branch))


def dirty_files(repo: Path) -> list[str]:
    return git(repo, "status", "--porcelain", "--untracked-files=no").splitlines()


def require_week_branch(repo: Path, today: datetime.date) -> str:
    """This week's branch name, after checking ``repo`` is on it and it contains ``CONFIG.base_branch``."""
    branch = current_branch(repo)
    expected = CONFIG.week_branch(today)
    if branch != expected:
        raise SystemExit(
            f"refusing: {repo} is on {branch}, not this week's branch {expected}.\n"
            "next: uv run python -m experiments.belay weekly-rollover"
        )
    git(repo, "fetch", "--quiet", "origin", CONFIG.base_branch)
    if not head_contains(repo, f"origin/{CONFIG.base_branch}"):
        raise SystemExit(
            f"refusing: {branch} does not contain origin/{CONFIG.base_branch}.\n"
            "next: uv run python -m experiments.belay weekly-rollover"
        )
    return branch


def require_clean_pushed_head(repo: Path, branch: str) -> str:
    """The HEAD commit, after checking the tree matches it and it is the tip of ``origin/<branch>``."""
    dirty = dirty_files(repo)
    if dirty:
        raise SystemExit(f"refusing: tracked files differ from HEAD: {dirty}")
    head = git(repo, "rev-parse", "HEAD")
    git(repo, "fetch", "--quiet", "origin", branch)
    if git(repo, "rev-parse", f"origin/{branch}") != head:
        raise SystemExit(f"refusing: HEAD {head[:10]} is not pushed to origin/{branch}.\nnext: git push origin {branch}")
    return head


def run_git(repo: Path, args: list[str], *, dry_run: bool) -> None:
    print(f"{'would run' if dry_run else 'run'}: git {' '.join(args)}")
    if not dry_run:
        git(repo, *args)


def cut_fresh_week(repo: Path, target: str, commits: list[str], *, dry_run: bool) -> str | None:
    """Check out ``target`` new from ``origin/<base branch>`` (the last accepted recipe) and cherry-pick ``commits``
    onto it, in order. Returns the first commit that does not apply cleanly (the checkout is then left at the base,
    nothing pushed), else None. The caller gates and pushes the result."""
    if dirty_files(repo):
        raise SystemExit(f"refusing: {repo} has uncommitted changes to tracked files")
    git(repo, "fetch", "--quiet", "origin", CONFIG.base_branch)
    print(f"cutting {target} from origin/{CONFIG.base_branch}")
    run_git(repo, ["checkout", "--quiet", "-B", target, f"origin/{CONFIG.base_branch}"], dry_run=dry_run)
    for commit in commits:
        print(f"{'would carry' if dry_run else 'carrying'} {git(repo, 'log', '-1', '--format=%h %s', commit)}")
        if dry_run:
            continue
        if subprocess.run(["git", "cherry-pick", "-x", commit], cwd=repo, capture_output=True).returncode != 0:
            git(repo, "cherry-pick", "--abort")
            git(repo, "reset", "--quiet", "--hard", f"origin/{CONFIG.base_branch}")
            return commit
    return None


@contextlib.contextmanager
def temporary_worktree(repo: Path, ref: str) -> Iterator[Path]:
    """A detached worktree of ``ref``, removed afterwards; for reading or merging without touching ``repo``."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "worktree"
        git(repo, "worktree", "add", "--quiet", "--detach", str(path), ref)
        try:
            yield path
        finally:
            git(repo, "worktree", "remove", "--force", str(path))


def named_checkout(repo: Path, branch: str) -> Path:
    """A worktree of ``repo`` on ``branch`` (created next to ``repo`` on first use), for a Track whose code is not on
    the weekly branch."""
    path = repo.parent / f"{repo.name}__{branch}"
    if not path.exists():
        git(repo, "fetch", "--quiet", "origin", branch)
        git(repo, "worktree", "add", "--quiet", "-B", branch, str(path), f"origin/{branch}")
        subprocess.run(["uv", "sync", "--quiet"], cwd=path, check=True)  # its own environment: its library code
    if current_branch(path) != branch:
        raise SystemExit(f"refusing: {path} is on {current_branch(path)}, not {branch}")
    return path
