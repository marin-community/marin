# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Git checks and the weekly branch: every Belay run executes a clean, pushed commit of this week's branch."""

import contextlib
import dataclasses
import datetime
import json
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


@dataclasses.dataclass
class Rollover:
    """A weekly cut in progress: the commits to carry (with the idea each belongs to), how many are on the branch,
    and the ports so far (original commit -> its port onto the recipe). Saved in the checkout's git directory while a
    carried commit waits for a port."""

    target: str
    commits: list[str]
    owners: dict[str, str]
    carried: int = 0
    ports: dict[str, str] = dataclasses.field(default_factory=dict)


ROLLOVER_FILE = "belay-rollover.json"


def rollover_path(repo: Path) -> Path:
    return Path(git(repo, "rev-parse", "--absolute-git-dir")) / ROLLOVER_FILE


def save_rollover(repo: Path, rollover: Rollover) -> None:
    rollover_path(repo).write_text(json.dumps(dataclasses.asdict(rollover), indent=2))


def load_rollover(repo: Path) -> Rollover:
    path = rollover_path(repo)
    if not path.exists():
        raise SystemExit("refusing: no weekly rollover is waiting for a port here (run weekly-rollover first)")
    return Rollover(**json.loads(path.read_text()))


def cut_fresh_week(repo: Path, rollover: Rollover, *, dry_run: bool) -> str | None:
    """Check out ``rollover.target`` new from ``origin/<base branch>`` (the last accepted recipe) and carry its
    commits onto it (``carry``). Returns the first commit that does not apply cleanly, else None."""
    if dirty_files(repo):
        raise SystemExit(f"refusing: {repo} has uncommitted changes to tracked files")
    git(repo, "fetch", "--quiet", "origin", CONFIG.base_branch)
    print(f"cutting {rollover.target} from origin/{CONFIG.base_branch}")
    run_git(repo, ["checkout", "--quiet", "-B", rollover.target, f"origin/{CONFIG.base_branch}"], dry_run=dry_run)
    if dry_run:
        for commit in rollover.commits:
            print(f"would carry {git(repo, 'log', '-1', '--format=%h %s', commit)}")
        return None
    return carry(repo, rollover)


def carry(repo: Path, rollover: Rollover) -> str | None:
    """Cherry-pick the rollover's remaining commits in order. A commit that conflicts is left mid-cherry-pick, with
    the rollover saved, for a port (``finish_port``); returns it. Returns None once every commit is on the branch."""
    for commit in rollover.commits[rollover.carried :]:
        print(f"carrying {git(repo, 'log', '-1', '--format=%h %s', commit)}")
        if subprocess.run(["git", "cherry-pick", "-x", commit], cwd=repo, capture_output=True).returncode != 0:
            save_rollover(repo, rollover)
            return commit
        rollover.carried += 1
    rollover_path(repo).unlink(missing_ok=True)
    return None


def conflicted_paths(repo: Path) -> list[str]:
    return git(repo, "diff", "--name-only", "--diff-filter=U").splitlines()


# Git's conflict markers, built so this file does not itself look conflicted to the merge-conflict lint.
OURS_MARKER, SPLIT_MARKER, THEIRS_MARKER = "<" * 7 + " ", "=" * 7, ">" * 7 + " "


def has_conflict_marker(text: str) -> bool:
    return any(line.startswith((OURS_MARKER, THEIRS_MARKER)) or line == SPLIT_MARKER for line in text.splitlines())


def finish_port(repo: Path, rollover: Rollover) -> str:
    """Commit the port of the rollover's conflicting commit, after checking every conflict is resolved and staged
    with no marker left; records it in ``rollover.ports`` and returns the new commit."""
    if current_branch(repo) != rollover.target:
        raise SystemExit(f"refusing: {repo} is on {current_branch(repo)}, not {rollover.target}")
    unresolved = conflicted_paths(repo)
    if unresolved:
        raise SystemExit(f"refusing: still conflicted: {', '.join(unresolved)}.\nnext: resolve them, git add them")
    staged = git(repo, "diff", "--cached", "--name-only").splitlines()
    marked = [p for p in staged if (repo / p).exists() and has_conflict_marker((repo / p).read_text())]
    if marked:
        raise SystemExit(f"refusing: conflict markers remain in {', '.join(marked)}.\nnext: remove them, git add")
    original = rollover.commits[rollover.carried]
    git(repo, "-c", "core.editor=true", "cherry-pick", "--continue")
    ported = git(repo, "rev-parse", "HEAD")
    rollover.ports[original] = ported
    rollover.carried += 1
    return ported


def abort_rollover(repo: Path) -> Rollover:
    """Drop a rollover waiting for a port: abort the cherry-pick, put the checkout back at the recipe, forget it."""
    rollover = load_rollover(repo)
    subprocess.run(["git", "cherry-pick", "--abort"], cwd=repo, capture_output=True)
    git(repo, "reset", "--quiet", "--hard", f"origin/{CONFIG.base_branch}")
    rollover_path(repo).unlink()
    return rollover


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
