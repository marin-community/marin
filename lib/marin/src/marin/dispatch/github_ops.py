# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GitHub issue operations for the dispatcher, via the `gh` CLI."""

import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

REPO = "marin-community/marin"


def _get_issue_author(issue: int) -> str | None:
    """Look up the GitHub username of the issue creator."""
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{REPO}/issues/{issue}", "--jq", ".user.login"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def post_issue_comment(issue: int, body: str) -> bool:
    """Post a comment to a GitHub issue. Returns True on success."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(body)
        tmpfile = f.name

    try:
        result = subprocess.run(
            ["gh", "issue", "comment", str(issue), "--body-file", tmpfile, "--repo", REPO],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error("Failed to post comment on issue #%d: %s", issue, result.stderr.strip())
            return False
        logger.info("Posted comment on issue #%d", issue)
        return True
    finally:
        Path(tmpfile).unlink(missing_ok=True)


def post_escalation(issue: int, collection_name: str, error: str, branch: str) -> bool:
    author = _get_issue_author(issue)
    tag = f" cc @{author}" if author else ""
    body = (
        f"\U0001f916 **Dispatch Escalation** \u2014 collection `{collection_name}`\n\n"
        f"Unrecoverable failure after repeated attempts:\n"
        f"```\n{error}\n```\n"
        f"Manual intervention required.{tag}\n\n"
        f"Branch: [`{branch}`](https://github.com/{REPO}/tree/{branch})"
    )
    return post_issue_comment(issue, body)


def post_progress_comment(
    issue: int,
    body: str,
    collection_name: str,
    branch: str,
    logbook: str,
) -> bool:
    """Post an agent's issue comment with branch/logbook context appended."""
    context = (
        f"\n\n---\n"
        f"\U0001f916 *Dispatch: collection `{collection_name}`* | "
        f"[branch](https://github.com/{REPO}/tree/{branch}) | "
        f"[logbook](https://github.com/{REPO}/blob/{branch}/{logbook})"
    )
    return post_issue_comment(issue, body + context)
