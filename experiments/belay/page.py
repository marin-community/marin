# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Client for the Belay page's API: read its state, update an idea, post to the activity feed."""

import datetime
import json
import os
import re
import subprocess
import urllib.request
from dataclasses import asdict, dataclass, fields

from experiments.belay.config import CONFIG

# The service token is a credential, so it comes from the environment (the dispatcher sets it for each session).
TOKEN_ENV = "BELAY_SERVICE_TOKEN"
TOKEN_HEADER = "X-Belay-Token"
BOT_MARK = "🤖"
TERMINAL_RUN_STATES = ("succeeded", "failed", "killed", "worker_failed", "unschedulable")


@dataclass(frozen=True)
class Idea:
    id: int
    origin: str
    author: str
    title: str | None
    body: str
    url: str | None
    state: str
    registry_id: str | None
    decided_by: str | None
    decided_at: str | None
    note: str | None

    @property
    def name(self) -> str:
        return (self.title or self.body.strip().splitlines()[0])[:90]


@dataclass(frozen=True)
class PageRun:
    name: str
    job_id: str
    registry_id: str | None
    run: str | None
    state: str
    step: int | None

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_RUN_STATES


@dataclass(frozen=True)
class PageState:
    ideas: list[Idea]
    runs: list[PageRun]
    github_posting: bool

    def idea(self, idea_id: int) -> Idea:
        for idea in self.ideas:
            if idea.id == idea_id:
                return idea
        raise SystemExit(f"no idea #{idea_id} on the Belay page")

    def idea_for(self, registry_id: str) -> Idea | None:
        """The idea behind ``registry_id``; None for an experiment a maintainer launched without one."""
        ideas = [i for i in self.ideas if i.registry_id == registry_id]
        if len(ideas) > 1:
            raise SystemExit(f"{len(ideas)} ideas on the Belay page have registry id {registry_id}")
        return ideas[0] if ideas else None

    def runs_of(self, registry_id: str) -> list[PageRun]:
        return [r for r in self.runs if r.registry_id == registry_id]


def call(path: str, body: dict | None = None, *, token: str | None = None) -> dict:
    """GET (no body) or POST JSON to the Belay API; ``token`` is required for its ``agent/`` routes."""
    data = None if body is None else json.dumps(body).encode()
    headers = {"content-type": "application/json"}
    if token is not None:
        headers[TOKEN_HEADER] = token
    request = urllib.request.Request(f"{CONFIG.page_url}api/{path}", data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def service_token() -> str:
    token = os.environ.get(TOKEN_ENV)
    if not token:
        raise SystemExit(f"{TOKEN_ENV} is not set; the dispatcher passes the Belay service token to each session")
    return token


def _fields(cls: type, row: dict) -> dict:
    return {f.name: row[f.name] for f in fields(cls)}


def page_state() -> PageState:
    state = call("agent/state", token=service_token())
    return PageState(
        ideas=[Idea(**_fields(Idea, i)) for i in state["ideas"]],
        runs=[PageRun(**_fields(PageRun, r)) for r in state["runs"]],
        github_posting=state["settings"]["github_posting"],
    )


def report_idea(idea_id: int, report: dict, *, dry_run: bool) -> None:
    """Update an idea on the page with any of ``state``, ``registry_id``, ``registry_url``, ``plan``, ``result``,
    ``note``."""
    if dry_run:
        print(f"would report idea #{idea_id}: {json.dumps(report)}")
        return
    call(f"agent/ideas/{idea_id}/report", report, token=service_token())
    print(f"reported idea #{idea_id}: state {report.get('state', 'unchanged')}")


@dataclass(frozen=True)
class Post:
    kind: str
    """``plan``, ``result`` or ``escalation``."""
    title: str
    body: str
    registry_id: str | None = None
    idea_id: int | None = None


def post_activity(post: Post, *, github_posting: bool, dry_run: bool) -> None:
    """Post to the page's activity feed, and to the batch issue when an approver has turned GitHub posting on."""
    if dry_run:
        target = f"page and #{CONFIG.batch_issue}" if github_posting else "page only (GitHub posting is off)"
        print(f"would post {post.kind} to the {target}:\n--- {post.title}\n{post.body}\n---")
        return
    github_url = None
    if github_posting:
        comment = f"{BOT_MARK} **{post.title}**\n\n{post.body}"
        github_url = subprocess.run(
            ["gh", "issue", "comment", str(CONFIG.batch_issue), "--repo", CONFIG.repository, "--body", comment],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    call("agent/activity", {**asdict(post), "github_url": github_url}, token=service_token())
    print(f"posted {post.kind}: {post.title}")


HTML_COMMENT = re.compile(r"<!--.*?(-->|$)", re.DOTALL)
DETAILS = re.compile(r"<details\b[^>]*>(.*?)</details>", re.DOTALL | re.IGNORECASE)
# Zero-width characters and bidirectional controls, which render as nothing but can hide or reorder text.
INVISIBLE = re.compile("[\u200b-\u200f\u2028-\u202e\u2060-\u2069\ufeff]")


def visible_text(body: str) -> tuple[str, list[str]]:
    """``body`` as a reviewer sees it rendered, and what was removed or marked: HTML comments and invisible
    characters are dropped; ``<details>`` bodies are kept between markers."""
    notes = []
    if HTML_COMMENT.search(body):
        notes.append("removed HTML comments")
        body = HTML_COMMENT.sub("", body)
    if INVISIBLE.search(body):
        notes.append("removed zero-width or bidi control characters")
        body = INVISIBLE.sub("", body)
    if DETAILS.search(body):
        notes.append("marked collapsed <details> text")
        body = DETAILS.sub(lambda m: f"[collapsed on GitHub:]{m.group(1)}[end collapsed]", body)
    return body, notes


def require_unedited(updated_at: datetime.datetime, approved_at: datetime.datetime) -> None:
    """Refuse an intake comment edited after it was approved: the approval covered the earlier text."""
    if updated_at > approved_at:
        raise SystemExit(
            f"refusing: the intake comment was edited after approval ({updated_at:%Y-%m-%d %H:%M} > "
            f"{approved_at:%Y-%m-%d %H:%M} UTC); it needs re-approval."
        )


def comment_times(url: str) -> tuple[datetime.datetime, datetime.datetime | None]:
    """A GitHub issue comment's last edit and the time the intake bot marked it transferred (its 🚀), if any."""
    comment_id = url.rsplit("issuecomment-", 1)[1]
    api = f"repos/{CONFIG.repository}/issues/comments/{comment_id}"
    comment = json.loads(subprocess.run(["gh", "api", api], check=True, capture_output=True, text=True).stdout)
    reactions = json.loads(
        subprocess.run(["gh", "api", f"{api}/reactions"], check=True, capture_output=True, text=True).stdout
    )
    rockets = sorted(r["created_at"] for r in reactions if r["content"] == "rocket")
    parse = datetime.datetime.fromisoformat
    return parse(comment["updated_at"]), parse(rockets[0]) if rockets else None
