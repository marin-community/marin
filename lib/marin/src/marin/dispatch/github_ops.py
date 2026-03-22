# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GitHub issue operations for the dispatcher, via the `gh` CLI."""

import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def post_issue_comment(issue: int, body: str) -> bool:
    """Post a comment to a GitHub issue. Returns True on success."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(body)
        tmpfile = f.name

    try:
        result = subprocess.run(
            ["gh", "issue", "comment", str(issue), "--body-file", tmpfile],
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


def post_escalation(issue: int, collection_name: str, error: str) -> bool:
    body = (
        f"\U0001f916 **Dispatch Escalation** \u2014 collection `{collection_name}`\n\n"
        f"Unrecoverable failure after repeated attempts:\n"
        f"```\n{error}\n```\n"
        f"Manual intervention required."
    )
    return post_issue_comment(issue, body)
