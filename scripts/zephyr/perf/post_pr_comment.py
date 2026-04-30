# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Upsert a single canonical zephyr-perf-gate comment on a PR.

Idempotent: looks for an existing comment whose body starts with the
``<!-- zephyr-perf-gate -->`` sentinel and edits it in place; otherwise creates
a new comment. Re-runs of the gate replace prior comments instead of stacking.

Body must already start with the sentinel — this script does not insert one,
so the comparison output stays the canonical source of the comment shape.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)

SENTINEL = "<!-- zephyr-perf-gate -->"


def _list_comments(repo: str, pr: int) -> list[dict[str, object]]:
    raw = subprocess.check_output(
        ["gh", "api", "--paginate", f"repos/{repo}/issues/{pr}/comments"],
        text=True,
    )
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, list) else []


def _find_existing(comments: list[dict[str, object]]) -> int | None:
    for c in comments:
        body = c.get("body") or ""
        if isinstance(body, str) and body.lstrip().startswith(SENTINEL):
            cid = c.get("id")
            if isinstance(cid, int):
                return cid
    return None


def upsert(repo: str, pr: int, body: str) -> dict[str, object]:
    if not body.lstrip().startswith(SENTINEL):
        raise ValueError(f"comment body must start with the sentinel `{SENTINEL}` " "(produced by compare_perf_runs.py)")

    comments = _list_comments(repo, pr)
    existing = _find_existing(comments)

    if existing is not None:
        subprocess.check_call(
            [
                "gh",
                "api",
                "--method",
                "PATCH",
                f"repos/{repo}/issues/comments/{existing}",
                "-f",
                f"body={body}",
            ]
        )
        return {"action": "updated", "comment_id": existing}

    out = subprocess.check_output(
        [
            "gh",
            "api",
            "--method",
            "POST",
            f"repos/{repo}/issues/{pr}/comments",
            "-f",
            f"body={body}",
        ],
        text=True,
    )
    parsed = json.loads(out)
    return {"action": "created", "comment_id": parsed.get("id")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="marin-community/marin")
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--body", required=True, help="Path to a markdown file (output of compare_perf_runs.py).")
    args = parser.parse_args()

    with open(args.body) as f:
        body = f.read()

    result = upsert(args.repo, args.pr, body)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
