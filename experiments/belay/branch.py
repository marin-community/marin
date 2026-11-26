# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Git checks and the weekly branch: every Belay run executes a clean, pushed commit of this week's branch."""

import contextlib
import dataclasses
import datetime
import json
import re
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


def enter_week(repo: Path, week: str) -> bool:
    """Put ``repo`` on this week's branch once it has been cut: check it out from origin, and merge in any recipe
    commits it lacks (new Belay tooling, or a change a maintainer adopted), pushing that merge. False when the week
    has not been cut yet. Refuses with uncommitted changes, or (aborting the merge) when the recipe conflicts."""
    git(repo, "fetch", "--quiet", "origin")
    if not remote_branch_exists(repo, week):
        return False
    if current_branch(repo) != week:
        if dirty_files(repo):
            raise SystemExit(f"refusing: {repo} has uncommitted changes to tracked files")
        git(repo, "checkout", "--quiet", "-B", week, f"origin/{week}")
    else:
        git(repo, "merge", "--quiet", "--ff-only", f"origin/{week}")
    if head_contains(repo, f"origin/{CONFIG.base_branch}"):
        return True
    merged = subprocess.run(
        ["git", "merge", "--quiet", "--no-edit", f"origin/{CONFIG.base_branch}"], cwd=repo, capture_output=True
    )
    if merged.returncode != 0:
        git(repo, "merge", "--abort")
        raise SystemExit(
            f"refusing: {CONFIG.base_branch} does not merge cleanly into {week}; nothing pushed.\n"
            f'next: uv run python -m experiments.belay escalate --title "{week}: the recipe does not merge" '
            '--body "A maintainer merges it, then sessions resume."'
        )
    git(repo, "push", "--quiet", "origin", week)
    print(f"merged origin/{CONFIG.base_branch} into {week}")
    return True


def latest_week(repo: Path, before: str) -> str | None:
    """The newest weekly branch on origin older than the week ``before``: where sessions work while ``before`` waits
    for its cut."""
    heads = git(repo, "ls-remote", "--heads", "origin", f"{CONFIG.week_branch_prefix}*").splitlines()
    weeks = [line.split("refs/heads/", 1)[1] for line in heads]
    return max((w for w in weeks if w < before), default=None)


def active_week(repo: Path, today: datetime.date) -> str | None:
    """The branch runs launch from: this week's once it is cut, else last week's (a blocked cut never stops work)."""
    week = CONFIG.week_branch(today)
    return week if remote_branch_exists(repo, week) else latest_week(repo, week)


def require_week_branch(repo: Path, today: datetime.date) -> str:
    """The active week's branch name (``active_week``), after checking ``repo`` is on it and it contains
    ``CONFIG.base_branch``."""
    branch = current_branch(repo)
    expected = active_week(repo, today) or CONFIG.week_branch(today)
    if branch != expected:
        raise SystemExit(
            f"refusing: {repo} is on {branch}, not this week's branch {expected}.\n"
            "next: uv run python -m experiments.belay start --events <the same events file>"
        )
    git(repo, "fetch", "--quiet", "origin", CONFIG.base_branch)
    if not head_contains(repo, f"origin/{CONFIG.base_branch}"):
        raise SystemExit(
            f"refusing: {branch} does not contain origin/{CONFIG.base_branch}.\n"
            "next: uv run python -m experiments.belay start --events <the same events file>"
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
    dropped: list[str] = dataclasses.field(default_factory=list)


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


ROLLOVER_BLOCKED_FILE = "belay-rollover-blocked.json"
# How long a checkout skips a cut that just failed before trying again (a maintainer may have fixed it meanwhile).
ROLLOVER_RETRY = datetime.timedelta(hours=2)


def mark_rollover_blocked(repo: Path, target: str, why: str) -> None:
    """Remember in the checkout that ``target``'s cut failed, so ``start`` works this pass on last week's branch."""
    path = Path(git(repo, "rev-parse", "--absolute-git-dir")) / ROLLOVER_BLOCKED_FILE
    at = datetime.datetime.now(datetime.UTC).isoformat()
    path.write_text(json.dumps({"target": target, "why": why, "at": at}))


def rollover_blocked(repo: Path, target: str) -> str | None:
    """Why ``target``'s cut failed in this checkout within ``ROLLOVER_RETRY``, else None (time to try the cut)."""
    path = Path(git(repo, "rev-parse", "--absolute-git-dir")) / ROLLOVER_BLOCKED_FILE
    if not path.exists():
        return None
    mark = json.loads(path.read_text())
    if mark["target"] != target:
        return None
    if datetime.datetime.now(datetime.UTC) - datetime.datetime.fromisoformat(mark["at"]) > ROLLOVER_RETRY:
        return None
    return mark["why"]


def clear_rollover_blocked(repo: Path) -> None:
    (Path(git(repo, "rev-parse", "--absolute-git-dir")) / ROLLOVER_BLOCKED_FILE).unlink(missing_ok=True)


# A name a Python diff adds at any depth with def/class, or at module level with an assignment.
DEFINED = re.compile(r"^\+\s*(?:async\s+)?(?:def|class)\s+(\w+)|^\+([A-Za-z_]\w*)\s*(?::[^=\n]*)?=(?!=)", re.M)
IDENTIFIER = re.compile(r"[A-Za-z_]\w*")


def added_python(repo: Path, commit: str) -> str:
    """The lines ``commit`` adds to Python files outside tests, as a zero-context diff (a test's module-level
    names are its own, not code another idea can need)."""
    return git(repo, "show", "--format=", "--unified=0", commit, "--", "*.py", ":(exclude)tests/*")


def defined_names(diff: str) -> set[str]:
    return {a or b for a, b in DEFINED.findall(diff)}


def used_names(diff: str) -> set[str]:
    added = [line[1:] for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
    return set(IDENTIFIER.findall("\n".join(added))) - defined_names(diff)


def defined_on(repo: Path, ref: str, name: str) -> bool:
    # POSIX classes, since git grep -E has no \\s or \\b.
    pattern = (
        rf"^[[:space:]]*(async[[:space:]]+)?(def|class)[[:space:]]+{name}([^A-Za-z0-9_]|$)|^{name}[[:space:]]*(:[^=]*)?="
    )
    grep = subprocess.run(["git", "grep", "-q", "-E", pattern, ref, "--", "*.py"], cwd=repo, capture_output=True)
    return grep.returncode == 0


def with_dependencies(
    repo: Path, carried: dict[str, str], others: dict[str, str], base: str
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Add to the carried commits (commit -> idea) the code they need from ideas left behind: a commit of
    ``others`` that defines a name a carried commit uses and ``base`` lacks brings in all of its idea's commits,
    repeatedly. Returns the commits in commit-date order, every commit's idea, and each added idea -> the idea that
    needed it. The scan reads names only; the gates on the cut branch are the real check."""
    owners = dict(carried)
    needed_by: dict[str, str] = {}
    present = {c: o for c, o in others.items() if subprocess.run(["git", "cat-file", "-e", c], cwd=repo).returncode == 0}
    definitions = {c: defined_names(added_python(repo, c)) for c in present}
    on_base: dict[str, bool] = {}
    queue = list(carried)
    while queue:
        commit = queue.pop()
        for name in sorted(used_names(added_python(repo, commit))):
            for other, names in definitions.items():
                if other in owners or name not in names:
                    continue
                if name not in on_base:
                    on_base[name] = defined_on(repo, base, name)
                if on_base[name]:
                    continue
                idea = present[other]
                needed_by.setdefault(idea, owners[commit])
                for sibling in [c for c, o in present.items() if o == idea and c not in owners]:
                    owners[sibling] = idea
                    queue.append(sibling)
    # Keep each commit as the registry spells it, so a port is recorded against the same string.
    spelled = {git(repo, "rev-parse", c): c for c in owners}
    ordered = git(repo, "log", "--no-walk=sorted", "--format=%H", *spelled).splitlines()[::-1]
    return [spelled[c] for c in ordered], owners, needed_by


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
