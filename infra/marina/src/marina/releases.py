# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source identity for independently released, checked-in applets."""

import subprocess
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

RELEASE_CONFIG = "release.toml"
RELEASE_METADATA = "dist/marina-release.json"


@dataclass(frozen=True)
class ReleaseSource:
    commit: str
    path: str
    tree: str


def release_applet_id(app_dir: Path) -> str:
    """Read the existing applet UUID bound to this source directory."""
    with (app_dir / RELEASE_CONFIG).open("rb") as source:
        config = tomllib.load(source)
    if set(config) != {"applet_id"}:
        raise ValueError(f"{RELEASE_CONFIG} must contain only applet_id")
    return str(UUID(config["applet_id"]))


def git_output(app_dir: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=app_dir, text=True).strip()


def release_source(app_dir: Path) -> ReleaseSource:
    """Read the committed source identity without changing Git refs or the index."""
    root = Path(git_output(app_dir, "rev-parse", "--show-toplevel"))
    path = app_dir.resolve().relative_to(root).as_posix()
    commit = git_output(app_dir, "rev-parse", "HEAD")
    tree = git_output(app_dir, "rev-parse", f"{commit}:{path}")
    return ReleaseSource(commit=commit, path=path, tree=tree)


def require_tracked_package(app_dir: Path, paths: Iterable[str], *, build_command: str | None) -> None:
    """Reject ignored backend files and unbuilt static assets from a release."""
    tracked = set(git_output(app_dir, "ls-files", "-z", "--", ".").split("\0"))
    untracked = {
        path for path in paths if path not in tracked and (build_command is None or not path.startswith("dist/"))
    }
    if untracked:
        raise ValueError(f"Release package contains files absent from Git: {', '.join(sorted(untracked))}")
