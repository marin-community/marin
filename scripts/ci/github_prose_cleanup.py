#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate and publish agent-edited GitHub descriptions."""

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

CLEANUP_FOOTER_MARKER = "<!-- marin-prose-cleanup -->"
GITHUB_BODY_SIZE_LIMIT = 65_536
UPDATED_DESCRIPTION_FOOTER_RESERVE = 512

_ARCHIVE_COMMENT_PREFIX = "🤖 Archived the description before automated prose cleanup."
_ARCHIVE_MARKER_PREFIX = "marin-prose-cleanup-archive"
_CLEANUP_FOOTER_RE = re.compile(
    r"\n{2,}---\n" r"\[Original description\]\([^\n)]+\)\s+" + re.escape(CLEANUP_FOOTER_MARKER) + r"\s*$"
)


class GithubItemKind(StrEnum):
    """GitHub description types supported by the cleanup workflow."""

    ISSUE = "issue"
    PULL_REQUEST = "pull_request"


@dataclass(frozen=True)
class CleanupResult:
    """Cleanup and archive payload consumed by the GitHub Actions wrapper."""

    kind: GithubItemKind
    number: int
    changed: bool
    cleaned_body: str
    original_body_hash: str
    archive_comment_body: str = ""
    archive_marker: str = ""
    skip_reason: str | None = None


def _body_without_cleanup_footer(body: str) -> str:
    return _CLEANUP_FOOTER_RE.sub("", body).rstrip()


def _archive_fence(original_body: str) -> str:
    def longest_run(character: str) -> int:
        return max((len(run) for run in re.findall(re.escape(character) + r"+", original_body)), default=0)

    backtick_length = max(3, longest_run("`") + 1)
    tilde_length = max(3, longest_run("~") + 1)
    if backtick_length <= tilde_length:
        return "`" * backtick_length
    return "~" * tilde_length


def _archive_comment(original_body: str, archive_marker: str) -> str:
    fence = _archive_fence(original_body)
    return (
        f"{_ARCHIVE_COMMENT_PREFIX}\n"
        f"{archive_marker}\n\n"
        "<details><summary>Original description</summary>\n\n"
        f"{fence}markdown\n{original_body}\n{fence}\n\n"
        "</details>"
    )


def _event_item(event: dict[str, Any]) -> tuple[GithubItemKind, dict[str, Any]]:
    pull_request = event.get("pull_request")
    if isinstance(pull_request, dict):
        return GithubItemKind.PULL_REQUEST, pull_request
    issue = event.get("issue")
    if isinstance(issue, dict):
        return GithubItemKind.ISSUE, issue
    raise ValueError("Event does not contain an issue or pull request")


def cleanup_result_from_event(event: dict[str, Any], rewritten_body: str) -> CleanupResult:
    """Build an archive payload from an event and an agent-edited body."""
    kind, item = _event_item(event)
    number = int(item["number"])
    original_body_value = item.get("body")
    original_body = original_body_value if isinstance(original_body_value, str) else ""
    body_hash = hashlib.sha256(original_body.encode("utf-8")).hexdigest()
    candidate_body = _body_without_cleanup_footer(rewritten_body).strip()
    if original_body.strip() and not candidate_body:
        return CleanupResult(
            kind=kind,
            number=number,
            changed=False,
            cleaned_body=original_body,
            original_body_hash=body_hash,
            skip_reason="rewrite_empty",
        )

    original_without_footer = _body_without_cleanup_footer(original_body).strip()
    if candidate_body == original_without_footer:
        return CleanupResult(
            kind=kind,
            number=number,
            changed=False,
            cleaned_body=original_body,
            original_body_hash=body_hash,
        )

    archive_marker = f"<!-- {_ARCHIVE_MARKER_PREFIX}:{body_hash} -->"
    archive_comment_body = _archive_comment(original_body, archive_marker)
    if len(archive_comment_body.encode("utf-8")) > GITHUB_BODY_SIZE_LIMIT:
        return CleanupResult(
            kind=kind,
            number=number,
            changed=False,
            cleaned_body=original_body,
            original_body_hash=body_hash,
            skip_reason="archive_too_large",
        )
    if len(candidate_body.encode("utf-8")) > GITHUB_BODY_SIZE_LIMIT - UPDATED_DESCRIPTION_FOOTER_RESERVE:
        return CleanupResult(
            kind=kind,
            number=number,
            changed=False,
            cleaned_body=original_body,
            original_body_hash=body_hash,
            skip_reason="description_too_large",
        )

    return CleanupResult(
        kind=kind,
        number=number,
        changed=True,
        cleaned_body=candidate_body,
        original_body_hash=body_hash,
        archive_comment_body=archive_comment_body,
        archive_marker=archive_marker,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=Path, required=True, help="GitHub event JSON")
    parser.add_argument("--output", type=Path, required=True, help="Cleanup result JSON")
    parser.add_argument("--rewritten-body-env", required=True, help="Environment variable containing the agent edit")
    parser.add_argument("--github-output", type=Path, help="GitHub Actions output file")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    event = json.loads(args.event.read_text(encoding="utf-8"))
    result = cleanup_result_from_event(event, os.environ[args.rewritten_body_env])
    args.output.write_text(json.dumps(asdict(result), ensure_ascii=False), encoding="utf-8")
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as github_output:
            github_output.write(f"changed={str(result.changed).lower()}\n")
            github_output.write(f"result={args.output}\n")
    status = "cleanup required" if result.changed else result.skip_reason or "no cleanup needed"
    print(f"{result.kind} #{result.number}: {status}")


if __name__ == "__main__":
    main()
