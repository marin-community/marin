# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Opportunistic Finelog logging for code-health automation.

Reads one JSON event from stdin and appends it to the
``codehealth.autolint.invocations`` and ``codehealth.autolint.findings``
namespaces defined in ``review_tables``.

Designed to be invoked fire-and-forget as a detached subprocess so missing
auth or a slow network never blocks the dev. Every failure is reported on
stderr and exits non-zero; the caller decides whether anyone sees it
(``linter.py`` keeps it in the run's log directory). Disable with
``MARIN_REVIEW_STATS=0``.

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

import argparse
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
import uuid

from finelog.client import LogClient

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
from infra.codehealth.review_tables import (  # noqa: E402
    DEFAULT_DEPLOYMENT,
    FINDINGS_NAMESPACE,
    INVOCATIONS_NAMESPACE,
    Finding,
    Invocation,
    append_rows,
    open_tables_client,
    parse_utc,
)
from infra.lint.catalog import catalog_sha, load_catalog  # noqa: E402

LINT_DIR = ROOT_DIR / "infra" / "lint"

# Bounds the detached writer so a wedged connection does not leave it running.
FLUSH_TIMEOUT = 30.0


def _git(args: list[str]) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT_DIR, capture_output=True, text=True, timeout=2)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _lint_catalog_sha() -> str | None:
    if not (LINT_DIR / "catalog.yaml").is_file():
        return None
    return catalog_sha(load_catalog(LINT_DIR))


def _parse_ts(value: str | None) -> dt.datetime:
    """Parse an event timestamp, defaulting to now."""
    return parse_utc(value) if value else dt.datetime.now(dt.UTC)


def fill_defaults(event: dict) -> dict:
    """Populate environment-derived fields the caller didn't supply.

    Callers (``linter.py``, ``/review-pr``) send only what is specific to their
    invocation. Existing values are never overwritten.
    """
    event.setdefault("invocation_id", str(uuid.uuid4()))
    event.setdefault("ts", dt.datetime.now(dt.UTC).isoformat())
    inv = event.setdefault("invocation", {})
    inv.setdefault("git_branch", _git(["rev-parse", "--abbrev-ref", "HEAD"]))
    inv.setdefault("head_sha", _git(["rev-parse", "HEAD"]))
    inv.setdefault("marin_user", _git(["config", "user.email"]))
    if LINT_DIR.is_dir():
        inv.setdefault("lint_catalog_sha", _lint_catalog_sha())
    inv["finding_count"] = len(event.get("findings") or [])
    return event


def build_invocation(event: dict) -> Invocation:
    inv = event.get("invocation") or {}
    return Invocation(
        ts=_parse_ts(event.get("ts")),
        invocation_id=event["invocation_id"],
        tool=event.get("tool") or "",
        variant=inv.get("variant"),
        trigger=inv.get("trigger"),
        agent_cli=inv.get("agent_cli"),
        git_branch=inv.get("git_branch"),
        merge_base_sha=inv.get("merge_base_sha"),
        head_sha=inv.get("head_sha"),
        pr_number=inv.get("pr_number"),
        marin_user=inv.get("marin_user"),
        lint_catalog_sha=inv.get("lint_catalog_sha"),
        diff_files=inv.get("diff_files"),
        diff_added_lines=inv.get("diff_added_lines"),
        diff_removed_lines=inv.get("diff_removed_lines"),
        finding_count=inv.get("finding_count"),
        elapsed=inv.get("elapsed"),
        agent_exit_code=inv.get("agent_exit_code"),
        timed_out=inv.get("timed_out"),
    )


def build_findings(event: dict) -> list[Finding]:
    """One row per ``[file, line, code, confidence, message]`` entry."""
    inv = event.get("invocation") or {}
    ts = _parse_ts(event.get("ts"))
    rows: list[Finding] = []
    for finding in event.get("findings") or []:
        if len(finding) != 5:
            continue
        file, line, code, confidence, message = finding
        rows.append(
            Finding(
                ts=ts,
                invocation_id=event["invocation_id"],
                tool=event.get("tool") or "",
                pr_number=inv.get("pr_number"),
                git_branch=inv.get("git_branch"),
                head_sha=inv.get("head_sha"),
                marin_user=inv.get("marin_user"),
                file=file,
                line=line,
                code=code,
                confidence=confidence,
                message=message,
            )
        )
    return rows


def write_event(client: LogClient, event: dict) -> int:
    """Append one event's rows. Returns the number of rows written."""
    append_rows(client, INVOCATIONS_NAMESPACE, Invocation, [build_invocation(event)], flush_timeout=FLUSH_TIMEOUT)
    written = 1

    findings = build_findings(event)
    if findings:
        append_rows(client, FINDINGS_NAMESPACE, Finding, findings, flush_timeout=FLUSH_TIMEOUT)
        written += len(findings)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", default=DEFAULT_DEPLOYMENT, help="Finelog deployment to write to.")
    parser.add_argument("--finelog-url", default=None, help="Connect to this address instead of a deployment.")
    args = parser.parse_args(argv)

    if os.environ.get("MARIN_REVIEW_STATS", "1") == "0":
        return 0

    event = fill_defaults(json.load(sys.stdin))
    with open_tables_client(args.deployment, args.finelog_url) as client:
        written = write_event(client, event)
    print(f"recorded {written} row(s) for invocation {event['invocation_id']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
