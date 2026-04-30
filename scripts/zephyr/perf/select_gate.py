# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decide whether a PR is in scope for the zephyr perf gate.

This script is intentionally **advisory** — it does the mechanical part
(in-scope filtering + flagging hot files) and leaves the gate-vs-gate
decision to the agent. A trivial whitespace fix in ``shuffle.py`` should not
trigger an overnight nemotron run, and a one-line tweak elsewhere can still
move memory or CPU; only the agent reading the actual diff can tell the
difference.

In scope = at least one non-test file under ``lib/zephyr/src/zephyr/**``.
Datakit / dedup / normalize / tokenize live under ``lib/marin/...`` and are
explicitly out of scope. ``lib/fray/**`` is also out of scope (flagged in the
output but not auto-gated).

``hot_files_touched`` lists in-scope files with higher prior likelihood of
shuffle / memory / CPU impact (scatter pipeline, planner, executor, sort,
spill). The agent should inspect these first when assessing the diff.

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

HOT_FILES = frozenset(
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
    """Pure, side-effect-free in-scope classification.

    The script does **not** decide the gate — the agent does that, after
    reading the diff and assessing impact along the dimensions in
    ``.agents/skills/zephyr-perf/SKILL.md``. Output here is structured input
    for that assessment: which files matter, which are likely-hot, and a hint
    when a non-zephyr backend (fray) is also touched.
    """
    zephyr_internals = sorted(
        p for p in changed if _matches_any(p, INCLUDE_GLOBS) and not _matches_any(p, EXCLUDE_GLOBS)
    )
    fray_files = sorted(p for p in changed if fnmatch.fnmatch(p, FRAY_GLOB))
    hot_files = sorted(set(zephyr_internals) & HOT_FILES)

    if not zephyr_internals:
        return {
            "in_scope": False,
            "reason": "no non-test files under lib/zephyr/src/zephyr/** in diff",
            "touched_zephyr_files": [],
            "touched_fray_files": fray_files,
            "hot_files_touched": [],
            "next_step": "skip — out of scope for zephyr perf gate.",
        }

    if hot_files:
        next_step = (
            "Read the diff for the hot files first, then the rest. " "Assess along the SKILL dimensions and pick a gate."
        )
    else:
        next_step = "Read the diff. Assess along the SKILL dimensions and pick a gate."

    return {
        "in_scope": True,
        "reason": "zephyr internals touched; agent must assess the diff to choose a gate",
        "touched_zephyr_files": zephyr_internals,
        "touched_fray_files": fray_files,
        "hot_files_touched": hot_files,
        "next_step": next_step,
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
