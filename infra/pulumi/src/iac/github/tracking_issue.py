# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep a single sticky GitHub issue in sync with a recurring scan's findings.

One open issue per (label, marker) holds the current findings. A re-scan that turns up the same
drift matches the fingerprint the body embeds and touches nothing, so an unresolved issue is
never re-filed or re-notified; new or changed drift rewrites the body; a clean scan comments and
closes. The `gh` runner is injected so the sync logic is testable without hitting GitHub.
"""

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

GhRunner = Callable[[list[str]], str]

AGENT_GENERATED_LABEL = "agent-generated"


class SyncAction(StrEnum):
    CREATED = "created"
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    RESOLVED = "resolved"
    CLEAN = "clean"


@dataclass(frozen=True)
class TrackingIssue:
    number: int
    body: str


def _run_gh(args: list[str]) -> str:
    result = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
    return result.stdout


def find_tracking_issue(*, repo: str, label: str, marker: str, run: GhRunner = _run_gh) -> TrackingIssue | None:
    """The open issue carrying `marker`, or None. Filters on the marker as well as the label so a
    human-relabelled issue can't collide with the scan's own."""
    output = run(["issue", "list", "--repo", repo, "--label", label, "--state", "open", "--json", "number,body"])
    for issue in json.loads(output):
        body = issue.get("body") or ""
        if marker in body:
            return TrackingIssue(number=issue["number"], body=body)
    return None


def sync_tracking_issue(
    *,
    repo: str,
    label: str,
    marker: str,
    title: str,
    body: str | None,
    fingerprint: str,
    changed_comment: str,
    resolved_comment: str,
    run: GhRunner = _run_gh,
) -> SyncAction:
    """Reconcile the sticky issue with this scan. `body` is None for a clean scan; otherwise it
    must embed `fingerprint` so an unchanged re-scan is a no-op.

    Editing an issue body notifies no one, so a changed finding set also posts `changed_comment`
    — otherwise newly surfaced drift (a first-sighted service agent) would land silently."""
    existing = find_tracking_issue(repo=repo, label=label, marker=marker, run=run)

    if body is not None:
        if existing is None:
            run(
                [
                    "issue",
                    "create",
                    "--repo",
                    repo,
                    "--title",
                    title,
                    "--label",
                    label,
                    "--label",
                    AGENT_GENERATED_LABEL,
                    "--body",
                    body,
                ]
            )
            return SyncAction.CREATED
        if f"fingerprint:{fingerprint}" in existing.body:
            return SyncAction.UNCHANGED
        run(["issue", "edit", str(existing.number), "--repo", repo, "--body", body])
        run(["issue", "comment", str(existing.number), "--repo", repo, "--body", changed_comment])
        return SyncAction.UPDATED

    if existing is None:
        return SyncAction.CLEAN
    run(["issue", "comment", str(existing.number), "--repo", repo, "--body", resolved_comment])
    run(["issue", "close", str(existing.number), "--repo", repo])
    return SyncAction.RESOLVED
