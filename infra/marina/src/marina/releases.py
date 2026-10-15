# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source identity for independently released, checked-in applets."""

import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from uuid import UUID

from rigging.provenance import git_output

RELEASE_CONFIG = "release.toml"
RELEASE_METADATA = "dist/marina-release.json"


@dataclass(frozen=True)
class ReleaseSource:
    commit: str
    path: str
    tree: str


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


def release_applet_id(app_dir: Path) -> str:
    """Read the existing applet UUID bound to this source directory."""
    with (app_dir / RELEASE_CONFIG).open("rb") as source:
        config = tomllib.load(source)
    if set(config) != {"applet_id"}:
        raise ValueError(f"{RELEASE_CONFIG} must contain only applet_id")
    return str(UUID(config["applet_id"]))


def release_source(app_dir: Path) -> ReleaseSource:
    """Read the committed source identity without changing Git refs or the index."""
    root = Path(git_output(["rev-parse", "--show-toplevel"], app_dir))
    path = app_dir.resolve().relative_to(root).as_posix()
    commit = git_output(["rev-parse", "HEAD"], app_dir)
    tree = git_output(["rev-parse", f"{commit}:{path}"], app_dir)
    return ReleaseSource(commit=commit, path=path, tree=tree)


def source_status(app_dir: Path, applet_id: str, version: int, released: ReleaseSource | None) -> ReleaseStatus:
    """Compare a published source identity with this directory's committed source."""
    expected = release_source(app_dir)
    if git_output(["status", "--porcelain", "--untracked-files=all", "--", "."], app_dir):
        state = ReleaseState.MODIFIED
    elif released is None:
        state = ReleaseState.UNRECORDED
    elif released.tree == expected.tree and released.path == expected.path:
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
