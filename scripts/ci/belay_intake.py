#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Transfer approved Marin Belay suggestions from the intake issue into the current weekly Belay issue.

A comment on the intake issue is approved when an allowlisted user reacts to it with a thumbs-up made after the
comment's last edit, so editing an approved comment needs a fresh approval. The allowlist holds numeric GitHub user
ids, not logins: ids never change or get reused, while a login freed by a rename can be claimed by someone else.

An approved comment first gets an experiment id in the Marin Belay registry (``experiments/belay/registry`` on the
registry branch): the next free ``X####``, with a ``status: approved`` entry built from the comment, recorded in the
registry's ``intake_ids.json`` so a retried run reuses it. It is then copied (quoted, attributed, headed with its id,
with @-mentions and marker syntax neutralized) into the week's Belay issue (the one open issue labeled
``belay-weekly``), the bot replies on the intake issue with the id, and the intake comment gets a rocket reaction.

The copy and the reply each end in a hidden marker naming the intake comment, and only markers in comments posted by
the workflow's own account count, so quoted text cannot mark other comments as transferred. A comment counts as
transferred once a reply or a copy carries its marker; the replies live on the intake issue, which never rolls over,
and copies from before the weekly issues are in ``LEGACY_TARGET_ISSUE``. A later run posts a missing reply or rocket
if an earlier run stopped part-way. The agent that runs the Belay loop reads only the Belay issues, so only approved
text reaches it.
"""

import argparse
import base64
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
# The account whose token the workflow uses; only its comments can carry transfer markers.
TRANSFER_BOT = "marin-belay-bot"
_MARKER_TEMPLATE = "<!-- belay-intake:{} -->"
_MARKER_RE = re.compile(re.escape(_MARKER_TEMPLATE).replace(r"\{\}", r"(\d+)") + r"\s*\Z")
_REPLY_MARKER_TEMPLATE = "<!-- belay-intake-reply:{} -->"
_REPLY_MARKER_RE = re.compile(re.escape(_REPLY_MARKER_TEMPLATE).replace(r"\{\}", r"(\d+)") + r"\s*\Z")
WEEKLY_LABEL = "belay-weekly"
# The single Belay issue used before the weekly issues; its transfer markers still count.
LEGACY_TARGET_ISSUE = 9451
# A zero-width space after "@" keeps copied text from pinging users or teams; one inside "<!--" keeps quoted text
# from forming an HTML comment (and so a marker).
_MENTION_RE = re.compile(r"@(?=[A-Za-z0-9-])")
_ZERO_WIDTH_SPACE = "​"

REGISTRY_BRANCH = "marin_belay_v1_registry"
REGISTRY_DIR = "experiments/belay/registry"
INTAKE_INDEX = f"{REGISTRY_DIR}/intake_ids.json"
# The registry's category list; an intake comment that names none of them is filed as UNSORTED.
CATEGORIES_FILE = f"{REGISTRY_DIR}/categories.json"
UNSORTED = "Unsorted"
_FIELD_RE = re.compile(r"^\s*\*\*(Title|Category|Idea):\*\*\s*(.+?)\s*$", re.MULTILINE)
_EXPERIMENT_FILE_RE = re.compile(r"^X(\d{4})-.*\.yaml$")


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


def transferred_ids_in(comments: list[dict], marker: re.Pattern = _MARKER_RE) -> set[int]:
    """Intake comment ids named by the trailing ``marker`` of comments that the transfer bot posted."""
    ids = set()
    for comment in comments:
        if comment["user"]["login"] != TRANSFER_BOT:
            continue
        match = marker.search(comment["body"])
        if match:
            ids.add(int(match.group(1)))
    return ids


def intake_fields(body: str) -> dict[str, str]:
    """The intake template's Title / Category / Idea fields present in ``body`` (see experiments/belay)."""
    return {name.lower(): value for name, value in _FIELD_RE.findall(body)}


def slugify(text: str, max_words: int = 6) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())[:max_words]
    return "-".join(words) or "intake"


def next_experiment_id(filenames: list[str]) -> str:
    """The id after the highest ``X####-*.yaml`` in ``filenames``."""
    numbers = [int(m.group(1)) for name in filenames if (m := _EXPERIMENT_FILE_RE.match(name))]
    return f"X{max(numbers, default=0) + 1:04d}"


def registry_entry(
    experiment_id: str, comment: IntakeComment, approver: str, categories: frozenset[str]
) -> tuple[str, str]:
    """``(filename, yaml)`` of the approved registry entry for an intake comment."""
    fields = intake_fields(comment.body)
    first_line = next((line.strip() for line in comment.body.splitlines() if line.strip()), "Intake suggestion")
    name = fields.get("title", first_line)[:120]
    category = fields.get("category", "")
    category = category if category in categories else UNSORTED
    summary = fields.get("idea", name)[:500]
    slug = slugify(name)
    # JSON strings are valid YAML scalars, so the entry needs no YAML library.
    q = json.dumps
    lines = [
        f"id: {experiment_id}",
        f"slug: {slug}",
        f"name: {q(name)}",
        f"category: {q(category)}",
        f"summary: {q(summary)}",
        "source: intake",
        "status: approved",
        "details:",
        *(f"- {q(line)}" for line in comment.body.splitlines() if line.strip()),
        "intake:",
        f"  url: {q(comment.url)}",
        f"  author: {q(comment.author)}",
        f"  approved_by: {q(approver)}",
    ]
    return f"{experiment_id}-{slug}.yaml", "\n".join(lines) + "\n"


def register(github: "GitHub", comment: IntakeComment, approver: str) -> str:
    """The experiment id of an approved intake comment, writing its registry entry the first time."""
    index_file = github.get_file(INTAKE_INDEX, REGISTRY_BRANCH)
    index = json.loads(index_file[0]) if index_file else {}
    key = str(comment.comment_id)
    if key in index:
        return index[key]
    experiment_id = next_experiment_id(github.list_dir(REGISTRY_DIR, REGISTRY_BRANCH))
    categories_file = github.get_file(CATEGORIES_FILE, REGISTRY_BRANCH)
    if categories_file is None:
        raise RuntimeError(f"{CATEGORIES_FILE} is missing on {REGISTRY_BRANCH}")
    categories = frozenset(json.loads(categories_file[0]))
    filename, entry = registry_entry(experiment_id, comment, approver, categories)
    github.put_file(
        f"{REGISTRY_DIR}/{filename}", REGISTRY_BRANCH, entry, f"Belay registry: {experiment_id} from intake {key}"
    )
    index[key] = experiment_id
    github.put_file(
        INTAKE_INDEX,
        REGISTRY_BRANCH,
        json.dumps(index, indent=1, sort_keys=True) + "\n",
        f"Belay registry: intake {key} -> {experiment_id}",
        sha=index_file[1] if index_file else None,
    )
    return experiment_id


def reply_body(comment_id: int, experiment_id: str, target_issue: int, repo: str) -> str:
    """The bot's reply on the intake issue once a suggestion is registered and copied."""
    registry = f"https://github.com/{repo}/tree/{REGISTRY_BRANCH}/{REGISTRY_DIR}"
    return (
        f"🤖 Registered as **{experiment_id}** in the [experiment registry]({registry}) and copied to "
        f"#{target_issue}.\n\n{_REPLY_MARKER_TEMPLATE.format(comment_id)}"
    )


def transfer_body(comment: IntakeComment, approver: str, experiment_id: str) -> str:
    """The weekly-issue copy of an approved intake comment: id, attribution, the quoted text, and the marker."""
    text = _MENTION_RE.sub("@" + _ZERO_WIDTH_SPACE, comment.body).replace("<!--", "<" + _ZERO_WIDTH_SPACE + "!--")
    quoted = "\n".join(f"> {line}" if line else ">" for line in text.splitlines())
    return (
        f"📥 **Intake suggestion {experiment_id}** from {comment.author}, approved by {approver} "
        f"([original]({comment.url})):\n\n"
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

    @property
    def repo(self) -> str:
        return self._repo

    def issue_comments(self, issue: int) -> list[dict]:
        return self._paginate(f"/issues/{issue}/comments")

    def open_issues_labeled(self, label: str) -> list[int]:
        """Numbers of the open issues (not pull requests) carrying ``label``."""
        items = self._paginate(f"/issues?labels={label}&state=open")
        return [item["number"] for item in items if "pull_request" not in item]

    def issue(self, number: int) -> dict:
        data, _ = self._request("GET", f"{API}/repos/{self._repo}/issues/{number}")
        assert isinstance(data, dict), data
        return data

    def create_issue(self, title: str, body: str, labels: list[str]) -> int:
        data, _ = self._request(
            "POST", f"{API}/repos/{self._repo}/issues", {"title": title, "body": body, "labels": labels}
        )
        assert isinstance(data, dict), data
        return data["number"]

    def lock(self, number: int) -> None:
        self._request("PUT", f"{API}/repos/{self._repo}/issues/{number}/lock", {})

    def close(self, number: int) -> None:
        self._request(
            "PATCH", f"{API}/repos/{self._repo}/issues/{number}", {"state": "closed", "state_reason": "completed"}
        )

    def comment_reactions(self, comment_id: int) -> list[dict]:
        return self._paginate(f"/issues/comments/{comment_id}/reactions")

    def post_comment(self, issue: int, body: str) -> None:
        self._request("POST", f"{API}/repos/{self._repo}/issues/{issue}/comments", {"body": body})

    def react(self, comment_id: int, content: str) -> None:
        self._request("POST", f"{API}/repos/{self._repo}/issues/comments/{comment_id}/reactions", {"content": content})

    def get_file(self, path: str, ref: str) -> tuple[str, str] | None:
        """``(text, blob sha)`` of ``path`` at branch ``ref``, or None if it does not exist."""
        try:
            data, _ = self._request("GET", f"{API}/repos/{self._repo}/contents/{path}?ref={ref}")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise
        assert isinstance(data, dict), data
        return base64.b64decode(data["content"]).decode(), data["sha"]

    def list_dir(self, path: str, ref: str) -> list[str]:
        data, _ = self._request("GET", f"{API}/repos/{self._repo}/contents/{path}?ref={ref}")
        assert isinstance(data, list), data
        return [item["name"] for item in data]

    def put_file(self, path: str, ref: str, text: str, message: str, sha: str | None = None) -> None:
        payload = {"message": message, "content": base64.b64encode(text.encode()).decode(), "branch": ref}
        if sha is not None:
            payload["sha"] = sha
        self._request("PUT", f"{API}/repos/{self._repo}/contents/{path}", payload)


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


def current_weekly_issue(github: GitHub) -> int:
    """The one open issue labeled ``WEEKLY_LABEL``."""
    issues = github.open_issues_labeled(WEEKLY_LABEL)
    if len(issues) != 1:
        raise RuntimeError(f"expected exactly one open {WEEKLY_LABEL} issue, found {sorted(issues)}")
    return issues[0]


def _transferred(github: GitHub, intake_raw: list[dict], target_issue: int) -> tuple[set[int], set[int]]:
    """Intake comment ids with a reply on the intake issue, and ids with a copy in this week's or the legacy issue."""
    replied = transferred_ids_in(intake_raw, _REPLY_MARKER_RE)
    copied = transferred_ids_in(github.issue_comments(target_issue))
    if target_issue != LEGACY_TARGET_ISSUE:
        copied |= transferred_ids_in(github.issue_comments(LEGACY_TARGET_ISSUE))
    return replied, copied


def _post_missing_replies(github: GitHub, ids: set[int], intake_issue: int, target_issue: int, dry_run: bool) -> None:
    """Reply for copies whose reply is missing (a run stopped between the two). Copies from before the registry have
    no id and get none."""
    index_file = github.get_file(INTAKE_INDEX, REGISTRY_BRANCH) if ids else None
    index = json.loads(index_file[0]) if index_file else {}
    for comment_id in sorted(i for i in ids if str(i) in index):
        print(f"post the missing reply for intake comment {comment_id}")
        if not dry_run:
            github.post_comment(intake_issue, reply_body(comment_id, index[str(comment_id)], target_issue, github.repo))


def run(github: GitHub, *, intake_issue: int, approvers: frozenset[int], dry_run: bool) -> int:
    """Register, copy and reply to every approved, not-yet-transferred intake comment, and mark it with a rocket;
    also post a missing reply or rocket for comments already copied. With ``dry_run`` only report. Returns the
    number of comments that were (or, with ``dry_run``, would be) copied."""
    target_issue = current_weekly_issue(github)
    intake_raw = github.issue_comments(intake_issue)
    replied, copied = _transferred(github, intake_raw, target_issue)
    transferred = replied | copied
    candidates, unmarked = [], []
    for raw in (r for r in intake_raw if r["user"]["login"] != TRANSFER_BOT):
        summary = raw.get("reactions", {})
        if raw["id"] in transferred:
            if summary.get(TRANSFERRED_REACTION, 0) == 0:
                unmarked.append(raw["id"])
        elif summary.get(APPROVE_REACTION, 0) > 0:
            # Only comments with a thumbs-up and no copy yet need their full reaction list.
            candidates.append(_to_intake(raw, github.comment_reactions(raw["id"])))
    pending = pending_transfers(candidates, approvers, transferred)
    for comment, approver in pending:
        print(
            f"copy intake comment {comment.comment_id} by {comment.author} (approved by {approver}) to #{target_issue}"
        )
        if not dry_run:
            experiment_id = register(github, comment, approver)
            github.post_comment(target_issue, transfer_body(comment, approver, experiment_id))
            github.post_comment(intake_issue, reply_body(comment.comment_id, experiment_id, target_issue, github.repo))
            github.react(comment.comment_id, TRANSFERRED_REACTION)
    _post_missing_replies(github, copied - replied, intake_issue, target_issue, dry_run)
    for comment_id in unmarked:
        print(f"add the missing rocket to already-copied intake comment {comment_id}")
        if not dry_run:
            github.react(comment_id, TRANSFERRED_REACTION)
    return len(pending)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake-issue", type=int, required=True)
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
            approvers=approvers,
            dry_run=args.dry_run,
        )
    except urllib.error.HTTPError as e:
        raise SystemExit(f"GitHub API error {e.code}: {e.read().decode(errors='replace')}") from e
    print(f"{count} comment(s) {'would be ' if args.dry_run else ''}copied", file=sys.stderr)


if __name__ == "__main__":
    main()
