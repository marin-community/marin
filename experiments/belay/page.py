# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Client for the Belay page's API: read its state, update an idea, write an idea's agent thread, and the one
reply on the idea's #9514 comment."""

import datetime
import json
import os
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, replace

from experiments.belay.config import CONFIG

# The service token is a credential, so it comes from the environment (the dispatcher sets it for each session).
TOKEN_ENV = "BELAY_SERVICE_TOKEN"
TOKEN_HEADER = "X-Belay-Token"
SESSION_ENV = "BELAY_SESSION_ID"
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
    node_grant: int | None
    """A maintainer's node grant: this idea's launches may hold up to this many nodes, past the global cap."""
    hidden_removed: list[str] | None
    """What the app removed from the comment before storing the approved text (hidden HTML, invisible characters)."""
    reply_url: str | None
    """The session's one reply on the idea's #9514 comment."""

    @property
    def name(self) -> str:
        return (self.title or self.body.strip().splitlines()[0])[:90]


@dataclass(frozen=True)
class Expectation:
    name: str
    expected_at: str
    state: str | None

    @property
    def live(self) -> bool:
        return self.state is None or self.state not in TERMINAL_RUN_STATES


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
    node_cap: int
    """The global soft cap on nodes held by Belay runs, set by a maintainer on the admin page."""
    expectations: list[Expectation]
    """Every launched run's expected finish, and its state once the watcher has seen it."""
    sessions: list[dict]

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
        node_cap=state["settings"]["node_cap"],
        expectations=[Expectation(**_fields(Expectation, e)) for e in state["expectations"]],
        sessions=state["sessions"],
    )


def expect_finish(run_name: str, expected_at: datetime.datetime, *, dry_run: bool) -> None:
    """Tell the app when a launched run should finish."""
    if dry_run:
        return
    call("agent/runs/expected", {"name": run_name, "expected_at": expected_at.isoformat()}, token=service_token())


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
    session_id: str | None = None


def post_activity(post: Post, *, dry_run: bool) -> None:
    """Post to the page's activity feed (escalations and notes, which only the admin page shows). Posts carry the
    session that made them, so the admin page links each to its transcript."""
    post = replace(post, session_id=os.environ.get(SESSION_ENV))
    if dry_run:
        print(f"would post {post.kind} to the page:\n--- {post.title}\n{post.body}\n---")
        return
    call("agent/activity", asdict(post), token=service_token())
    print(f"posted {post.kind}: {post.title}")


def require_unedited(updated_at: datetime.datetime, approved_at: datetime.datetime) -> None:
    """Refuse an intake comment edited after it was approved: the approval covered the earlier text."""
    if updated_at > approved_at:
        raise SystemExit(
            f"refusing: the intake comment was edited after approval ({updated_at:%Y-%m-%d %H:%M} > "
            f"{approved_at:%Y-%m-%d %H:%M} UTC); it needs re-approval."
        )


def comment_updated_at(url: str) -> datetime.datetime:
    """When a #9514 comment was last edited (read with the caller's gh)."""
    comment_id = url.rsplit("issuecomment-", 1)[1]
    api = f"repos/{CONFIG.repository}/issues/comments/{comment_id}"
    comment = json.loads(subprocess.run(["gh", "api", api], check=True, capture_output=True, text=True).stdout)
    return datetime.datetime.fromisoformat(comment["updated_at"])


def gh_api(args: list[str]) -> str:
    return subprocess.run(["gh", "api", *args], check=True, capture_output=True, text=True).stdout


def thread_url(registry_id: str) -> str:
    """The idea's row on the Status page, which expands to its agent thread."""
    return f"{CONFIG.page_url}status#{registry_id}"


def read_thread(registry_id: str) -> str | None:
    """The idea's agent thread so far, or None before its first plan."""
    try:
        return call(f"agent/threads/{registry_id}", token=service_token())["body"]
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def publish_thread(registry_id: str, title: str, body: str, idea_id: int | None, kind: str, *, dry_run: bool) -> str:
    """Write the idea's agent thread on the page (created, or replaced in place) and return its Status link."""
    if dry_run:
        print(f"would write the {registry_id} thread ({kind}):\n{body}")
    else:
        call(
            f"agent/threads/{registry_id}",
            {"title": title, "body": body, "idea_id": idea_id, "kind": kind},
            token=service_token(),
        )
        print(f"wrote the {registry_id} thread: {title}")
    return thread_url(registry_id)


def reply_to_intake(
    idea: Idea, text: str, *, github_posting: bool, dry_run: bool, gh: Callable[[list[str]], str] = gh_api
) -> str | None:
    """Post the one reply on the idea's #9514 comment, or edit it once it exists; returns its URL. Only for intake
    ideas, and only when an approver has turned GitHub posting on."""
    if idea.url is None or "issuecomment-" not in idea.url:
        return None
    body = f"{BOT_MARK} {text}"
    if dry_run or not github_posting:
        print(f"would {'edit ' + idea.reply_url if idea.reply_url else 'post'} the #{CONFIG.intake_issue} reply: {body}")
        return idea.reply_url
    if idea.reply_url is None:
        api = f"repos/{CONFIG.repository}/issues/{CONFIG.intake_issue}/comments"
        url = json.loads(gh([api, "-X", "POST", "-f", f"body={body}"]))["html_url"]
        print(f"replied {url}")
        return url
    comment_id = idea.reply_url.rsplit("issuecomment-", 1)[1]
    gh([f"repos/{CONFIG.repository}/issues/comments/{comment_id}", "-X", "PATCH", "-f", f"body={body}"])
    print(f"edited {idea.reply_url}")
    return idea.reply_url
