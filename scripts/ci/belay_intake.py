#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Transfer approved Marin Belay suggestions from the intake issue into the main Belay issue.

A comment on the intake issue is approved when an allowlisted user reacts to it with a thumbs-up made after the
comment's last edit, so editing an approved comment needs a fresh approval. The allowlist holds numeric GitHub user
ids, not logins: ids never change or get reused, while a login freed by a rename can be claimed by someone else.

An approved comment is copied (quoted, attributed, with @-mentions and marker syntax neutralized) into the main issue,
ending in a hidden marker, and the intake comment gets a rocket reaction. Only markers in comments posted by the
workflow's own bot count, so quoted text cannot mark other comments as transferred. A comment is copied once; a later
run adds a missing rocket if an earlier run posted the copy but stopped before reacting. The agent that runs the Belay
loop reads only the main issue, so only approved text reaches it.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

API = "https://api.github.com"
APPROVE_REACTION = "+1"
TRANSFERRED_REACTION = "rocket"
TRANSFER_BOT = "github-actions[bot]"
_MARKER_TEMPLATE = "<!-- belay-intake:{} -->"
_MARKER_RE = re.compile(re.escape(_MARKER_TEMPLATE).replace(r"\{\}", r"(\d+)") + r"\s*\Z")
# A zero-width space after "@" keeps copied text from pinging users or teams; one inside "<!--" keeps quoted text
# from forming an HTML comment (and so a marker).
_MENTION_RE = re.compile(r"@(?=[A-Za-z0-9-])")
_ZERO_WIDTH_SPACE = "​"


@dataclass(frozen=True)
class Reaction:
    user_id: int
    login: str
    content: str
    created_at: str


@dataclass(frozen=True)
class IntakeComment:
    comment_id: int
    author: str
    body: str
    url: str
    updated_at: str
    reactions: tuple[Reaction, ...]


def approver_of(comment: IntakeComment, approvers: frozenset[int]) -> str | None:
    """Login of the first allowlisted user who thumbs-upped the comment after its last edit, or None.

    GitHub timestamps are ISO 8601 UTC strings of fixed width, so string order is time order.
    """
    for reaction in comment.reactions:
        if (
            reaction.content == APPROVE_REACTION
            and reaction.user_id in approvers
            and reaction.created_at >= comment.updated_at
        ):
            return reaction.login
    return None


def pending_transfers(
    comments: list[IntakeComment], approvers: frozenset[int], transferred_ids: set[int]
) -> list[tuple[IntakeComment, str]]:
    """Approved comments whose copy is not yet in the main issue, in the given order, each with its approver."""
    pending = []
    for comment in comments:
        approver = approver_of(comment, approvers)
        if approver is not None and comment.comment_id not in transferred_ids:
            pending.append((comment, approver))
    return pending


def transferred_ids_in(target_comments: list[dict]) -> set[int]:
    """Intake comment ids already copied into the main issue, read only from the trailing marker of comments that
    the transfer bot posted."""
    ids = set()
    for comment in target_comments:
        if comment["user"]["login"] != TRANSFER_BOT:
            continue
        match = _MARKER_RE.search(comment["body"])
        if match:
            ids.add(int(match.group(1)))
    return ids


def transfer_body(comment: IntakeComment, approver: str) -> str:
    """The main-issue comment for an approved intake comment: attribution, the quoted text, and the marker."""
    text = _MENTION_RE.sub("@" + _ZERO_WIDTH_SPACE, comment.body).replace("<!--", "<" + _ZERO_WIDTH_SPACE + "!--")
    quoted = "\n".join(f"> {line}" if line else ">" for line in text.splitlines())
    return (
        f"📥 **Intake suggestion** from {comment.author}, approved by {approver} ([original]({comment.url})):\n\n"
        f"{quoted}\n\n{_MARKER_TEMPLATE.format(comment.comment_id)}"
    )


class GitHub:
    """The GitHub REST calls the transfer needs, for one repository."""

    def __init__(self, token: str, repo: str):
        self._token = token
        self._repo = repo

    def _request(self, method: str, url: str, payload: dict | None = None) -> tuple[object, dict]:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Authorization", f"Bearer {self._token}")
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("X-GitHub-Api-Version", "2022-11-28")
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
            return (json.loads(body) if body else None), dict(response.headers)

    def _paginate(self, path: str) -> list[dict]:
        items: list[dict] = []
        url: str | None = f"{API}/repos/{self._repo}{path}?per_page=100"
        while url:
            page, headers = self._request("GET", url)
            assert isinstance(page, list), page
            items += page
            url = _next_link(headers.get("Link", ""))
        return items

    def issue_comments(self, issue: int) -> list[dict]:
        return self._paginate(f"/issues/{issue}/comments")

    def comment_reactions(self, comment_id: int) -> list[dict]:
        return self._paginate(f"/issues/comments/{comment_id}/reactions")

    def post_comment(self, issue: int, body: str) -> None:
        self._request("POST", f"{API}/repos/{self._repo}/issues/{issue}/comments", {"body": body})

    def react(self, comment_id: int, content: str) -> None:
        self._request("POST", f"{API}/repos/{self._repo}/issues/comments/{comment_id}/reactions", {"content": content})


def _next_link(link_header: str) -> str | None:
    for part in link_header.split(","):
        url, _, rel = part.partition(";")
        if 'rel="next"' in rel:
            return url.strip().strip("<>")
    return None


def _to_intake(raw: dict, reactions: list[dict]) -> IntakeComment:
    return IntakeComment(
        comment_id=raw["id"],
        author=raw["user"]["login"],
        body=raw["body"],
        url=raw["html_url"],
        updated_at=raw["updated_at"],
        reactions=tuple(
            Reaction(user_id=r["user"]["id"], login=r["user"]["login"], content=r["content"], created_at=r["created_at"])
            for r in reactions
        ),
    )


def run(github: GitHub, *, intake_issue: int, target_issue: int, approvers: frozenset[int], dry_run: bool) -> int:
    """Copy every approved, not-yet-copied intake comment into the target issue and mark it with a rocket; also add
    a missing rocket to comments already copied. With ``dry_run`` only report. Returns the number of comments that
    were (or, with ``dry_run``, would be) copied."""
    transferred = transferred_ids_in(github.issue_comments(target_issue))
    candidates, unmarked = [], []
    for raw in github.issue_comments(intake_issue):
        summary = raw.get("reactions", {})
        if raw["id"] in transferred:
            if summary.get(TRANSFERRED_REACTION, 0) == 0:
                unmarked.append(raw["id"])
        elif summary.get(APPROVE_REACTION, 0) > 0:
            # Only comments with a thumbs-up and no copy yet need their full reaction list.
            candidates.append(_to_intake(raw, github.comment_reactions(raw["id"])))
    pending = pending_transfers(candidates, approvers, transferred)
    for comment, approver in pending:
        print(f"copy intake comment {comment.comment_id} by {comment.author} (approved by {approver})")
        if not dry_run:
            github.post_comment(target_issue, transfer_body(comment, approver))
            github.react(comment.comment_id, TRANSFERRED_REACTION)
    for comment_id in unmarked:
        print(f"add the missing rocket to already-copied intake comment {comment_id}")
        if not dry_run:
            github.react(comment_id, TRANSFERRED_REACTION)
    return len(pending)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake-issue", type=int, required=True)
    parser.add_argument("--target-issue", type=int, required=True)
    parser.add_argument(
        "--approver-ids", required=True, help="Comma-separated numeric GitHub user ids whose 👍 approves."
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    approvers = frozenset(int(a) for a in args.approver_ids.split(",") if a.strip())
    if not approvers:
        raise SystemExit("--approver-ids must name at least one user id")
    github = GitHub(os.environ["GITHUB_TOKEN"], os.environ["GITHUB_REPOSITORY"])
    try:
        count = run(
            github,
            intake_issue=args.intake_issue,
            target_issue=args.target_issue,
            approvers=approvers,
            dry_run=args.dry_run,
        )
    except urllib.error.HTTPError as e:
        raise SystemExit(f"GitHub API error {e.code}: {e.read().decode(errors='replace')}") from e
    print(f"{count} comment(s) {'would be ' if args.dry_run else ''}copied", file=sys.stderr)


if __name__ == "__main__":
    main()
