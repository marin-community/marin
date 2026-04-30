# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decide whether a PR is in scope for the zephyr perf gate, and which gate to run.

In scope = at least one non-test file under ``lib/zephyr/src/zephyr/**``.
Datakit / dedup / normalize / tokenize live under ``lib/marin/...`` and are
explicitly out of scope; ``lib/fray/**`` is also out of scope (flagged in the
reason but not auto-gated).

Gate 2 (full nemotron) is selected when the diff touches any file in
``HEAVY_PATHS``; otherwise Gate 1 (fineweb smoke).

Output is JSON on stdout. Exit code is always 0 unless the PR cannot be read.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)

INCLUDE_GLOBS = (
    "lib/zephyr/src/zephyr/*.py",
    "lib/zephyr/src/zephyr/**/*.py",
    "lib/zephyr/pyproject.toml",
)

EXCLUDE_GLOBS = (
    "lib/zephyr/tests/**",
    "lib/zephyr/src/zephyr/_test_helpers.py",
    "**/*.md",
)

HEAVY_PATHS = frozenset(
    {
        "lib/zephyr/src/zephyr/shuffle.py",
        "lib/zephyr/src/zephyr/plan.py",
        "lib/zephyr/src/zephyr/execution.py",
        "lib/zephyr/src/zephyr/external_sort.py",
        "lib/zephyr/src/zephyr/spill.py",
    }
)

FRAY_GLOB = "lib/fray/**"


def _matches_any(path: str, globs: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(path, g) for g in globs)


def _changed_files_from_pr(pr: int) -> list[str]:
    raw = subprocess.check_output(
        ["gh", "pr", "diff", str(pr), "--name-only"],
        text=True,
    )
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _changed_files_from_diff(merge_base: str, head: str) -> list[str]:
    raw = subprocess.check_output(
        ["git", "diff", "--name-only", f"{merge_base}...{head}"],
        text=True,
    )
    return [line.strip() for line in raw.splitlines() if line.strip()]


def classify(changed: list[str]) -> dict[str, object]:
    """Pure, side-effect-free classification.

    Exposed as a function so callers (CI, the orchestrator, tests) can pass an
    explicit file list without shelling out to git/gh.
    """
    zephyr_internals = [p for p in changed if _matches_any(p, INCLUDE_GLOBS) and not _matches_any(p, EXCLUDE_GLOBS)]

    if not zephyr_internals:
        return {
            "in_scope": False,
            "gate": None,
            "reason": "no non-test files under lib/zephyr/src/zephyr/** in diff",
            "touched_zephyr_files": [],
            "touched_fray_files": [p for p in changed if fnmatch.fnmatch(p, FRAY_GLOB)],
        }

    heavy_hits = sorted(set(zephyr_internals) & HEAVY_PATHS)
    gate = "2" if heavy_hits else "1"
    reason = f"heavy path(s) touched: {heavy_hits}" if heavy_hits else "non-heavy zephyr internals only"
    return {
        "in_scope": True,
        "gate": gate,
        "reason": reason,
        "touched_zephyr_files": sorted(zephyr_internals),
        "touched_fray_files": sorted(p for p in changed if fnmatch.fnmatch(p, FRAY_GLOB)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pr", type=int, help="GitHub PR number; uses `gh pr diff`.")
    src.add_argument(
        "--diff-range",
        help="Local git range, e.g. `origin/main...HEAD`. Format: <base>...<head>.",
    )
    src.add_argument(
        "--files-from",
        help="Read newline-delimited file list from this path (testing/debug).",
    )
    args = parser.parse_args()

    if args.files_from:
        with open(args.files_from) as f:
            changed = [line.strip() for line in f if line.strip()]
    elif args.pr is not None:
        changed = _changed_files_from_pr(args.pr)
    else:
        base, _, head = args.diff_range.partition("...")
        if not base or not head:
            parser.error("--diff-range must be of the form <base>...<head>")
        changed = _changed_files_from_diff(base, head)

    print(json.dumps(classify(changed), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
