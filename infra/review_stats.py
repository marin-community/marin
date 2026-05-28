# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Opportunistic W&B logging for review-automation tools.

Reads one JSON event from stdin and appends rows to two flat W&B Tables
attached to a single persistent run (`review-stats`) in the
`marin-review-stats` project — one global table per metric, filter by
the `ts` column in the W&B UI:

  - `invocations` — one row per `pre-commit.py --review` / `/code-review` /
    `/review-pr` invocation. `finding_count = 0` rows are kept; they are the
    "tool ran but said nothing" signal.
  - `findings` — one row per individual finding emitted by the agent.

Join key between the two: `invocation_id`. Both tables denormalize `tool`,
`pr_number`, and `marin_user` so single-table queries in the W&B UI work.

Append-to-artifact pattern: fetch the latest artifact for the run, append,
re-log. Concurrent writers race on the artifact and the loser's row is
dropped — acceptable at our scale; if we ever care, shard by year.

Designed to be invoked fire-and-forget as a detached subprocess so missing
deps, missing auth, or slow networks never block the dev. Disable with
`MARIN_REVIEW_STATS=0`.

Expected stdin payload:

    {
      "invocation_id": "<uuid4>",
      "ts":            "2026-05-28T14:02:11Z",
      "tool":          "pre-commit-review",
      "invocation":    { variant, trigger, agent_cli, git_branch, merge_base_sha,
                         head_sha, pr_number, marin_user, lint_catalog_sha,
                         diff_files, diff_added_lines, diff_removed_lines,
                         finding_count, elapsed, agent_exit_code, timed_out },
      "findings":      [[file, line, code, confidence, message], ...]
    }
"""

from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
import uuid

WANDB_PROJECT = "marin-review-stats"
WANDB_RUN_ID = "review-stats"
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
LINT_CATALOG = ROOT_DIR / "infra" / "lint.md"

INVOCATION_COLUMNS = [
    "ts",
    "invocation_id",
    "tool",
    "variant",
    "trigger",
    "agent_cli",
    "git_branch",
    "merge_base_sha",
    "head_sha",
    "pr_number",
    "marin_user",
    "lint_catalog_sha",
    "diff_files",
    "diff_added_lines",
    "diff_removed_lines",
    "finding_count",
    "elapsed",
    "agent_exit_code",
    "timed_out",
]

FINDING_COLUMNS = [
    "ts",
    "invocation_id",
    "tool",
    "pr_number",
    "git_branch",
    "head_sha",
    "marin_user",
    "file",
    "line",
    "code",
    "confidence",
    "message",
]


def _load_existing_rows(wandb, run_id: str, log_key: str) -> list[list]:
    """Best-effort fetch of prior rows for `log_key` on the run.

    Returns [] on the very first invocation or if the artifact is missing
    for any other reason — both are normal and not errors.
    """
    artifact_name = f"run-{run_id}-{log_key}:latest"
    try:
        art = wandb.use_artifact(artifact_name)
        table = art.get(log_key)
        return [list(row) for row in table.data]
    except Exception:
        return []


def _build_invocation_row(event: dict) -> list:
    invocation = event.get("invocation") or {}
    row = {
        "ts": event.get("ts"),
        "invocation_id": event.get("invocation_id"),
        "tool": event.get("tool"),
        **invocation,
    }
    return [row.get(col) for col in INVOCATION_COLUMNS]


def _build_finding_rows(event: dict) -> list[list]:
    invocation = event.get("invocation") or {}
    ts = event.get("ts")
    invocation_id = event.get("invocation_id")
    tool = event.get("tool")
    pr_number = invocation.get("pr_number")
    git_branch = invocation.get("git_branch")
    head_sha = invocation.get("head_sha")
    marin_user = invocation.get("marin_user")

    rows: list[list] = []
    for finding in event.get("findings") or []:
        # finding is [file, line, code, confidence, message]
        if len(finding) != 5:
            continue
        rows.append([ts, invocation_id, tool, pr_number, git_branch, head_sha, marin_user, *finding])
    return rows


def _git(args: list[str]) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT_DIR, capture_output=True, text=True, timeout=2)
        return r.stdout.strip() or None
    except Exception:
        return None


def _fill_defaults(event: dict) -> dict:
    """Populate environment-derived fields the caller didn't supply.

    Callers (pre-commit.py, /review-pr, etc.) only need to specify what's
    specific to their invocation; the helper fills in everything inferable
    from local git state. Existing values are never overwritten.
    """
    event.setdefault("invocation_id", str(uuid.uuid4()))
    event.setdefault("ts", dt.datetime.now(dt.timezone.utc).isoformat())
    inv = event.setdefault("invocation", {})
    inv.setdefault("git_branch", _git(["rev-parse", "--abbrev-ref", "HEAD"]))
    inv.setdefault("head_sha", _git(["rev-parse", "HEAD"]))
    inv.setdefault("marin_user", _git(["config", "user.email"]))
    if LINT_CATALOG.exists():
        inv.setdefault("lint_catalog_sha", _git(["hash-object", str(LINT_CATALOG)]))
    return event


def main() -> int:
    if os.environ.get("MARIN_REVIEW_STATS", "1") == "0":
        return 0
    try:
        event = json.load(sys.stdin)
    except Exception:
        return 0

    try:
        import wandb
    except ImportError:
        return 0

    event = _fill_defaults(event)

    try:
        run = wandb.init(
            project=WANDB_PROJECT,
            id=WANDB_RUN_ID,
            resume="allow",
            settings=wandb.Settings(silent=True, _disable_stats=True, _disable_meta=True),
        )
    except Exception:
        return 0

    try:
        invocation_rows = _load_existing_rows(wandb, WANDB_RUN_ID, "invocations")
        invocation_rows.append(_build_invocation_row(event))
        run.log({"invocations": wandb.Table(columns=INVOCATION_COLUMNS, data=invocation_rows)})

        new_findings = _build_finding_rows(event)
        if new_findings:
            finding_rows = _load_existing_rows(wandb, WANDB_RUN_ID, "findings")
            finding_rows.extend(new_findings)
            run.log({"findings": wandb.Table(columns=FINDING_COLUMNS, data=finding_rows)})
    except Exception:
        pass
    finally:
        try:
            run.finish(quiet=True)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
