# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source identity for independently released, checked-in applets."""

import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from rigging.provenance import git_output

RELEASE_METADATA = "dist/marina-release.json"


def require_current_main(repo_dir: Path) -> str:
    """Require a clean checkout at origin's current main and return its full commit."""
    if git_output(["status", "--porcelain", "--untracked-files=all"], repo_dir):
        raise RuntimeError(
            "Releases require a clean checkout by default; use --allow-dirty to override or --local to preview."
        )
    commit = git_output(["rev-parse", "HEAD"], repo_dir)
    # Query the remote directly: tracking refs and rerun CI checkouts may be stale.
    remote = git_output(["ls-remote", "--exit-code", "origin", "refs/heads/main"], repo_dir, timeout=30)
    if commit != remote.split()[0]:
        raise RuntimeError(
            "Releases require origin's current main commit by default; "
            "update the checkout or use --allow-dirty to override."
        )
    return commit


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
