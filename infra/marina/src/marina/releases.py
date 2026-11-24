# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source identity for independently released, checked-in applets."""

import subprocess
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from rigging.provenance import git_output

RELEASE_METADATA = "dist/marina-release.json"


DEFAULT_RELEASE_BRANCH = "main"


def _clean_checkout_message(branch: str) -> str:
    if branch == DEFAULT_RELEASE_BRANCH:
        return "Releases require a clean checkout by default; use --allow-dirty to override or --local to preview."
    return "Releases require a clean checkout; --allow-dirty does not cover the release-branch guard."


def require_release_checkout(repo_dir: Path, *, branch: str = DEFAULT_RELEASE_BRANCH) -> str:
    """Require a clean checkout at a commit the release may cut from; return its commit.

    A ``main`` release must sit exactly at origin's current main commit. A declared
    release branch — an integration branch a release pipeline cuts from — accepts
    any commit origin's branch already contains, its tip or an ancestor, because the
    branch moves ahead of the release cut while CI and receipts run.
    """
    if git_output(["status", "--porcelain", "--untracked-files=all"], repo_dir):
        raise RuntimeError(_clean_checkout_message(branch))
    commit = git_output(["rev-parse", "HEAD"], repo_dir)
    # Query the remote directly: tracking refs and rerun CI checkouts may be stale.
    tip = git_output(["ls-remote", "--exit-code", "origin", f"refs/heads/{branch}"], repo_dir, timeout=30).split()[0]
    if commit == tip:
        return commit
    if branch != DEFAULT_RELEASE_BRANCH:
        # Ancestry needs the tip object locally: fetch the branch ref explicitly so a
        # tag that shares the branch's name can never shadow the ancestry check.
        git_output(["fetch", "--quiet", "origin", f"refs/heads/{branch}"], repo_dir, timeout=120)
        landed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "FETCH_HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if landed.returncode == 0:
            return commit
        raise RuntimeError(
            f"The checkout at {commit[:12]} is not landed on origin/{branch} (tip {tip[:12]}); "
            "release from a commit the declared branch contains."
        )
    raise RuntimeError(
        "Releases require origin's current main commit by default; "
        "update the checkout or use --allow-dirty to override."
    )


@contextmanager
def clean_frontend_build(app_dir: Path) -> Iterator[None]:
    """Build with empty output and restore preexisting local assets on exit."""
    dist = app_dir / "dist"
    # A sibling backup can be renamed without copying large local build outputs.
    with tempfile.TemporaryDirectory(prefix=".marina-release-", dir=app_dir) as directory:
        backup = Path(directory) / "dist"
        if dist.exists() or dist.is_symlink():
            dist.rename(backup)
        try:
            yield
        finally:
            if dist.exists() or dist.is_symlink():
                dist.rename(Path(directory) / "built")
            if backup.exists() or backup.is_symlink():
                backup.rename(dist)


@dataclass(frozen=True)
class ReleaseSource:
    commit: str
    path: str
    tree: str
    inputs: dict[str, str] = field(default_factory=dict)
    branch: str | None = None
    """The declared release branch a published record was cut from; None on a
    default main release and on checkout descriptions, which declare no branch."""


class ReleaseState(StrEnum):
    MODIFIED = "modified"
    UNRECORDED = "unrecorded"
    CURRENT = "current"
    DIFFERENT = "different"


@dataclass(frozen=True)
class ReleaseStatus:
    applet_id: str
    version: int
    state: ReleaseState
    released: ReleaseSource | None
    checkout: ReleaseSource


def release_source(app_dir: Path, source_paths: tuple[str, ...]) -> ReleaseSource:
    """Read the committed source identity without changing Git refs or the index."""
    root = Path(git_output(["rev-parse", "--show-toplevel"], app_dir))
    path = app_dir.resolve().relative_to(root).as_posix()
    commit = git_output(["rev-parse", "HEAD"], app_dir)
    tree = git_output(["rev-parse", f"{commit}:{path}"], app_dir)
    inputs = {source: git_output(["rev-parse", f"{commit}:{source}"], app_dir) for source in source_paths}
    return ReleaseSource(commit=commit, path=path, tree=tree, inputs=inputs)


def require_forward_move(repo_dir: Path, *, branch: str, live_commit: str) -> None:
    """Refuse a release that moves backward along its declared branch.

    The landing check accepts any commit the branch contains, so a stale checkout
    could otherwise republish older source over newer. The live release's commit
    must be an ancestor of this checkout's; it is reachable after the branch fetch
    the landing check performs. A different or undeclared lineage skips the check.
    """
    git_output(["fetch", "--quiet", "origin", f"refs/heads/{branch}"], repo_dir, timeout=120)
    forward = subprocess.run(
        ["git", "merge-base", "--is-ancestor", live_commit, "HEAD"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if forward.returncode != 0:
        raise RuntimeError(
            f"The live release's commit {live_commit[:12]} is ahead of this checkout on {branch!r}; "
            "release from a newer commit instead of republishing older source."
        )


def source_status(
    app_dir: Path, applet_id: str, version: int, released: ReleaseSource | None, source_paths: tuple[str, ...]
) -> ReleaseStatus:
    """Compare a published source identity with this directory's committed source."""
    expected = release_source(app_dir, source_paths)
    root = Path(git_output(["rev-parse", "--show-toplevel"], app_dir))
    paths = [".", *(str(root / path) for path in expected.inputs)]
    if git_output(["status", "--porcelain", "--untracked-files=all", "--", *paths], app_dir):
        state = ReleaseState.MODIFIED
    elif released is None:
        state = ReleaseState.UNRECORDED
    elif released.tree == expected.tree and released.path == expected.path and released.inputs == expected.inputs:
        state = ReleaseState.CURRENT
    else:
        state = ReleaseState.DIFFERENT
    return ReleaseStatus(applet_id=applet_id, version=version, state=state, released=released, checkout=expected)


def require_tracked_package(app_dir: Path, paths: Iterable[str], *, build_command: str | None) -> None:
    """Reject ignored backend files and unbuilt static assets from a release."""
    tracked = set(git_output(["ls-files", "-z", "--", "."], app_dir).split("\0"))
    untracked = {
        path for path in paths if path not in tracked and (build_command is None or not path.startswith("dist/"))
    }
    if untracked:
        raise ValueError(f"Release package contains files absent from Git: {', '.join(sorted(untracked))}")
