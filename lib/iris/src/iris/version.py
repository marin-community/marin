# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve the marin-iris client revision date.

Used by `LaunchJob` to gate root submissions against a server-side floor.
Resolution order: build-time stamp (set in wheels by the wheel builder), then
`git log` on the iris source tree (for editable installs), then empty string.
The server treats empty as the feature's introduction date so old clients get
a grace window after rollout.
"""

import subprocess
from pathlib import Path

from iris._build_info import BUILD_DATE

_CACHED: str | None = None


def _git_log_date(iris_root: Path, *path_filter: str) -> str:
    try:
        # %cs is short committer date (YYYY-MM-DD).
        out = subprocess.check_output(
            ["git", "log", "-1", "--format=%cs", *path_filter],
            cwd=iris_root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        ).strip()
        return out
    except (subprocess.SubprocessError, OSError):
        return ""


def _git_iris_date() -> str:
    # Path-filtered first (precise iris date for full clones), unfiltered
    # fallback (date of HEAD) for shallow CI clones where the tip commit
    # may not have touched lib/iris and `git log -- path` returns empty.
    iris_root = Path(__file__).resolve().parents[2]
    return _git_log_date(iris_root, "--", iris_root.as_posix()) or _git_log_date(iris_root)


def client_revision_date() -> str:
    """Return ISO date (YYYY-MM-DD) of this client build, or "" if unknown."""
    global _CACHED
    if _CACHED is None:
        _CACHED = BUILD_DATE or _git_iris_date()
    return _CACHED


def _reset_cache_for_tests() -> None:
    global _CACHED
    _CACHED = None
