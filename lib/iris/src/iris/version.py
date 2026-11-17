# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve the revision date of this marin-iris build.

`LaunchJob` sends it as the client's date; the controller resolves it for itself
too, and gates root submissions on the gap between the two. Both sides must
measure the same thing — the date of the last commit touching the iris tree — or
the comparison is meaningless.

Resolution order: `IRIS_REVISION_DATE`, baked into container images at build time
by `iris cluster build` (an image carries no `.git`, so this is the only way a
controller can identify itself); then the `BUILD_DATE` stamped into wheels and
sdists by `hatch_build.py`; then `git log` on the iris source tree, for a
checkout. A build that resolves to the empty string cannot identify itself.
"""

import os
import subprocess
from pathlib import Path

try:
    # Generated into built artifacts by hatch_build.py, and absent from a
    # checkout — an editable install resolves its date from git instead.
    from iris._build_info import BUILD_DATE  # pyrefly: ignore[missing-import]
except ImportError:
    BUILD_DATE = ""

# Set on container images by `iris.cli.build`, which resolves the date on the
# operator's machine, where the repository is present.
REVISION_DATE_ENV = "IRIS_REVISION_DATE"

_CACHED: str | None = None


def iris_tree_date(iris_root: Path) -> str:
    """ISO date (YYYY-MM-DD) of the last commit touching `iris_root`, or "" if unavailable.

    Empty when the path has no commits, is outside a checkout, or `git` is absent —
    all of which are ordinary for an installed package rather than a checkout.
    """
    try:
        # %cs is the short committer date.
        out = subprocess.check_output(
            ["git", "log", "-1", "--format=%cs", "--", iris_root.as_posix()],
            cwd=iris_root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        ).strip()
        return out
    except (subprocess.SubprocessError, OSError):
        return ""


def client_revision_date() -> str:
    """Return ISO date (YYYY-MM-DD) of this marin-iris build, or "" if unknown."""
    global _CACHED
    if _CACHED is None:
        _CACHED = (
            os.environ.get(REVISION_DATE_ENV, "") or BUILD_DATE or iris_tree_date(Path(__file__).resolve().parents[2])
        )
    return _CACHED


def _reset_cache_for_tests() -> None:
    global _CACHED
    _CACHED = None
