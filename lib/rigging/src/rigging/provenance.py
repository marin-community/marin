# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Provenance of a built artifact: a content-addressed tree hash plus the human
and launch context (branch, base commit, builder, origin, time, argv) needed to read it.

The ``tree_hash`` is content-addressed and deterministic — the same working-tree
content always produces the same hash regardless of when or by whom it is built —
so it is the natural deduplication key for image tags and version comparisons. The
remaining fields render that hash in a form a human can recognize and record the
launch that produced an artifact.

Two constructors capture it. :meth:`Provenance.from_git` is strict — it raises if git
is unavailable — for an image build, where the content-addressed ``tree_hash`` must
exist. :meth:`Provenance.capture` is best-effort — every field degrades to empty
outside a checkout — for stamping an artifact built by an arbitrary launch.
"""

import dataclasses
import getpass
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Provenance:
    """Identity, human context, and launch context of a built artifact.

    Attributes:
        tree_hash: Short git tree hash of the working-tree content. The
            deduplication key — content-addressed and stable across builds.
        base_commit: Short hash of the HEAD commit the tree was built from.
        dirty: Whether the working tree had uncommitted (tracked) changes.
        branch: Branch name at build time, or ``None`` for a detached HEAD.
        built_by: OS user that ran the build, or ``None`` if unknown.
        git_remote: ``origin`` remote URL, or ``None`` if unknown.
        created_at: ISO-8601 time the launch was captured, or ``""`` if uncaptured.
        command_line: ``sys.argv`` of the launching process; empty if uncaptured.
    """

    tree_hash: str
    base_commit: str
    dirty: bool
    branch: str | None
    built_by: str | None
    git_remote: str | None = None
    created_at: str = ""
    command_line: tuple[str, ...] = ()

    @classmethod
    def from_git(cls, repo_dir: Path | None = None) -> "Provenance":
        """Capture build provenance from the git repository at ``repo_dir`` (or cwd).

        Strict: raises if git is unavailable, since the content-addressed
        ``tree_hash`` is the reason to build at all.
        """
        cwd = str(repo_dir) if repo_dir is not None else None
        # `git stash create` captures any dirty state as a transient commit whose
        # tree we deref below — dropping the per-call timestamp a commit hash
        # would carry, so tree_hash stays purely content-addressed. Empty (no
        # output) means a clean tree, so fall back to HEAD.
        stash = _git(["stash", "create"], cwd)
        ref = stash or "HEAD"
        return cls(
            tree_hash=_git(["rev-parse", "--short", f"{ref}^{{tree}}"], cwd),
            base_commit=_git(["rev-parse", "--short", "HEAD"], cwd),
            dirty=bool(stash),
            branch=_git(["symbolic-ref", "--short", "-q", "HEAD"], cwd, check=False) or None,
            built_by=_getuser(),
        )

    @classmethod
    def capture(cls, repo_dir: Path | None = None) -> "Provenance":
        """Best-effort provenance of the current launch: git context plus the origin
        remote, capture time, and argv.

        Tolerant: every git field degrades to empty/``None`` outside a checkout or when
        the ``git`` binary is absent, so it never raises. Use to stamp an artifact built
        by an arbitrary run.
        """
        cwd = str(repo_dir) if repo_dir is not None else None

        def g(*args: str) -> str:
            # A missing `git` binary (bundle image without git) degrades like being outside
            # a checkout: git fields empty, launch context still captured.
            try:
                return _git(list(args), cwd, check=False)
            except FileNotFoundError:
                return ""

        stash = g("stash", "create")
        ref = stash or "HEAD"
        return cls(
            tree_hash=g("rev-parse", "--short", f"{ref}^{{tree}}"),
            base_commit=g("rev-parse", "--short", "HEAD"),
            dirty=bool(stash),
            branch=g("symbolic-ref", "--short", "-q", "HEAD") or None,
            built_by=_getuser(),
            git_remote=g("remote", "get-url", "origin") or None,
            created_at=datetime.now().isoformat(),
            command_line=tuple(sys.argv),
        )

    @classmethod
    def from_json(cls, raw: str) -> "Provenance":
        d = json.loads(raw)
        return cls(
            tree_hash=d["tree_hash"],
            base_commit=d["base_commit"],
            dirty=bool(d["dirty"]),
            branch=d.get("branch"),
            built_by=d.get("built_by"),
            git_remote=d.get("git_remote"),
            created_at=d.get("created_at", ""),
            command_line=tuple(d.get("command_line", ())),
        )

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @property
    def dedup_key(self) -> str:
        """The value to key image tags and version comparisons on."""
        return self.tree_hash

    def __str__(self) -> str:
        suffix = ""
        if self.branch:
            suffix += f" ({self.branch})"
        if self.built_by:
            suffix += f" ({self.built_by})"
        if self.dirty:
            return f"{self.tree_hash} (off of {self.base_commit}){suffix}"
        return f"{self.base_commit}{suffix}"


def _git(args: list[str], cwd: str | None, check: bool = True) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd)
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _getuser() -> str | None:
    # getpass.getuser() raises when no username resolves from the environment or the
    # password database; provenance is best-effort, so treat that as unknown.
    try:
        return getpass.getuser()
    except (OSError, KeyError):
        return None


# A path segment is lowercase alphanumerics plus '_' and '-'; collapse anything else.
_USER_SEGMENT_RE = re.compile(r"[^a-z0-9_-]+")


def username_segment() -> str:
    """The current user as a path-safe segment, for per-user artifact namespacing.

    Resolves the same OS login that stamps provenance ``built_by`` and reduces it to a clean
    segment: an email-like login drops its domain but keeps the whole local name
    (``russell.power@host`` → ``russell-power``), and the result is lowercased with any
    remaining character collapsed to ``-``. The full local name is kept so distinct users stay
    distinct. Raises ``RuntimeError`` if no username resolves — per-user namespacing must never
    silently fall back to a shared bucket.
    """
    raw = _getuser()
    if not raw:
        raise RuntimeError("cannot resolve a username for per-user namespacing (getpass.getuser found none)")
    name = raw.strip()
    if "@" in name:
        name = name.split("@", 1)[0]
    segment = _USER_SEGMENT_RE.sub("-", name.lower()).strip("-")
    if not segment:
        raise RuntimeError(f"username {raw!r} did not sanitize to a usable path segment")
    return segment
