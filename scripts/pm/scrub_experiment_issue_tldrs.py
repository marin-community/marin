#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["PyGithub>=2.3.0"]
# ///
"""Select experiment issues for agent-authored TL;DR maintenance."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from github.Issue import Issue
    from github.Repository import Repository
else:
    Issue = Any
    Repository = Any

LOGGER = logging.getLogger(__name__)

EXPERIMENT_LABEL = "experiment"
TLDR_LABEL = "tldr"
SUMMARY_START = "<!-- experiment-tldr:start -->"
SUMMARY_END = "<!-- experiment-tldr:end -->"
MAX_COMMENT_COUNT = 15
BODY_BLOCK_RE = re.compile(
    rf"\n?{re.escape(SUMMARY_START)}.*?{re.escape(SUMMARY_END)}\n?",
    re.DOTALL,
)
DOC_ISSUE_MARKER_RE = re.compile(r"<!-- experiment-tldr:doc-issue=(\d+) -->")
URL_RE = re.compile(r"https?://[^\s>)\]]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="marin-community/marin", help="GitHub repo in owner/name form.")
    parser.add_argument("--issue-number", type=int, help="Only emit context for a single issue number.")
    parser.add_argument("--max-issues", type=int, default=10, help="Maximum issues to inspect in one run.")
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Refresh issues that already have a managed TL;DR block and the `tldr` label.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output for humans instead of compact JSON.",
    )
    return parser.parse_args()


def ensure_env(var_name: str) -> str:
    value = os.environ.get(var_name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {var_name}")
    return value


def dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def extract_urls(text: str) -> list[str]:
    return dedupe_preserving_order(URL_RE.findall(text or ""))


def issue_has_managed_block(body: str | None) -> bool:
    return bool(body and SUMMARY_START in body and SUMMARY_END in body)


def extract_existing_doc_issue_number(body: str | None) -> int | None:
    if not body:
        return None
    match = DOC_ISSUE_MARKER_RE.search(body)
    if match is None:
        return None
    return int(match.group(1))


def collect_issue_context(issue: Issue) -> tuple[str, list[str]]:
    comments = list(issue.get_comments())[:MAX_COMMENT_COUNT]
    comment_blocks = []
    candidate_links = extract_urls(issue.body or "")

    for comment in comments:
        candidate_links.extend(extract_urls(comment.body or ""))
        comment_author = json.dumps(comment.user.login)
        created_at = json.dumps(comment.created_at.isoformat())
        comment_blocks.append(
            "\n".join(
                [
                    f"<comment author={comment_author} created_at={created_at}>",
                    comment.body or "",
                    "</comment>",
                ]
            )
        )

    context = "\n".join(
        [
            f"<issue number={issue.number} state={json.dumps(issue.state)} url={json.dumps(issue.html_url)}>",
            f"<title>{issue.title}</title>",
            f"<labels>{','.join(label.name for label in issue.labels)}</labels>",
            "<body>",
            issue.body or "",
            "</body>",
            "<comments>",
            "\n".join(comment_blocks),
            "</comments>",
            "</issue>",
        ]
    )
    return context, dedupe_preserving_order(candidate_links)


def issue_needs_summary_refresh(issue: Issue, *, refresh_existing: bool) -> bool:
    has_managed_block = issue_has_managed_block(issue.body)
    has_tldr_label = any(label.name == TLDR_LABEL for label in issue.labels)
    if refresh_existing:
        return True
    if not has_managed_block:
        return True
    return not has_tldr_label


def issue_candidates(repo: Repository, args: argparse.Namespace) -> list[Issue]:
    if args.issue_number is not None:
        issue = repo.get_issue(args.issue_number)
        return [issue]

    candidates: list[Issue] = []
    for issue in repo.get_issues(
        state="all", labels=[repo.get_label(EXPERIMENT_LABEL)], sort="updated", direction="desc"
    ):
        if issue.pull_request is not None:
            continue
        if not issue_needs_summary_refresh(issue, refresh_existing=args.refresh_existing):
            continue
        candidates.append(issue)
        if len(candidates) >= args.max_issues:
            break
    return candidates


def serialize_issue(issue: Issue) -> dict[str, object]:
    issue_context, candidate_links = collect_issue_context(issue)
    labels = [label.name for label in issue.labels]
    return {
        "number": issue.number,
        "title": issue.title,
        "state": issue.state,
        "url": issue.html_url,
        "labels": labels,
        "has_managed_block": issue_has_managed_block(issue.body),
        "has_tldr_label": TLDR_LABEL in labels,
        "existing_doc_issue_number": extract_existing_doc_issue_number(issue.body),
        "candidate_links": candidate_links,
        "issue_context": issue_context,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    from github import Github

    github_token = ensure_env("GITHUB_TOKEN")
    github = Github(github_token)
    repo = github.get_repo(args.repo)
    candidates = issue_candidates(repo, args)
    LOGGER.info("Found %s issue(s) to inspect", len(candidates))

    payload = {
        "repo": args.repo,
        "candidate_count": len(candidates),
        "candidates": [serialize_issue(issue) for issue in candidates],
    }
    indent = 2 if args.pretty else None
    print(json.dumps(payload, indent=indent, sort_keys=bool(indent)))


if __name__ == "__main__":
    main()
