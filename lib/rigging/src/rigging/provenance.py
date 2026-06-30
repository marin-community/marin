# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build provenance: a content-addressed tree hash plus the human context
(branch, base commit, builder) needed to read it.

The ``tree_hash`` is content-addressed and deterministic — the same working-tree
content always produces the same hash regardless of when or by whom it is built —
so it is the natural deduplication key for image tags and version comparisons.
The remaining fields exist purely to render that hash in a form a human can
recognize at a glance.
"""

import dataclasses
import getpass
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Provenance:
    """Identity and human context of a built artifact.

    Attributes:
        tree_hash: Short git tree hash of the working-tree content. The
            deduplication key — content-addressed and stable across builds.
        base_commit: Short hash of the HEAD commit the tree was built from.
        dirty: Whether the working tree had uncommitted (tracked) changes.
        branch: Branch name at build time, or ``None`` for a detached HEAD.
        built_by: OS user that ran the build, or ``None`` if unknown.
    """

    tree_hash: str
    base_commit: str
    dirty: bool
    branch: str | None
    built_by: str | None

    @classmethod
    def from_git(cls, repo_dir: Path | None = None) -> "Provenance":
        """Capture provenance from the git repository at ``repo_dir`` (or cwd).

        ``git stash create`` captures any dirty state as a transient commit
        whose tree we deref — that drops the per-call timestamp a commit hash
        would carry, keeping ``tree_hash`` purely content-addressed.
        """
        cwd = str(repo_dir) if repo_dir is not None else None
        stash = _git(["stash", "create"], cwd)
        ref = stash or "HEAD"
        return cls(
            tree_hash=_git(["rev-parse", "--short", f"{ref}^{{tree}}"], cwd),
            base_commit=_git(["rev-parse", "--short", "HEAD"], cwd),
            dirty=bool(stash),
            branch=_git(["symbolic-ref", "--short", "-q", "HEAD"], cwd, check=False) or None,
            built_by=getpass.getuser(),
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
