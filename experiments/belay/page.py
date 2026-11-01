# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Client for the Belay page's API: read its state, update an idea, write an idea's agent thread, and the one
reply on the idea's #9514 comment.

On Marina (any page URL that is not localhost) every call carries an IAP ID token from ``marina.client``: the
person's cached ``iris login`` on a laptop, or the ambient service-account identity on Loom (``loom-vm``). Only a
local preview takes the service token, in ``X-Belay-Token``. A session's own writes also carry the session id and
secret its claim issued (``SessionCredentials``)."""

import datetime
import json
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from urllib.parse import urlparse

import requests
from marina.client import bearer_token

from experiments.belay.config import CONFIG

# The local preview's service token is a credential, so it comes from the environment (the local dispatcher sets
# it for each session). Marina callers never use it.
TOKEN_ENV = "BELAY_SERVICE_TOKEN"
TOKEN_HEADER = "X-Belay-Token"
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost"})
REQUEST_TIMEOUT = 60
# A session's credentials: from the environment (the local dispatcher claims for each session it starts), or from
# SESSION_FILE (a Loom session's own `claim` writes it; `finish` removes it).
SESSION_ENV = "BELAY_SESSION_ID"
SESSION_SECRET_ENV = "BELAY_SESSION_SECRET"
SESSION_HEADER = "X-Belay-Session"
SESSION_SECRET_HEADER = "X-Belay-Session-Secret"
SESSION_FILE = Path(tempfile.gettempdir()) / "belay-session.json"
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


class PageRequestError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"Belay API request failed ({status_code}): {detail}")
        self.status_code = status_code


@dataclass(frozen=True)
class SessionCredentials:
    """The session id and secret a claim issued; the app accepts a session's writes only with both."""

    id: str
    secret: str


def session_credentials() -> SessionCredentials | None:
    """This process's session: from the environment, else from SESSION_FILE; None outside a session."""
    if os.environ.get(SESSION_ENV) and os.environ.get(SESSION_SECRET_ENV):
        return SessionCredentials(os.environ[SESSION_ENV], os.environ[SESSION_SECRET_ENV])
    if SESSION_FILE.exists():
        return SessionCredentials(**json.loads(SESSION_FILE.read_text()))
    return None


def save_session(credentials: SessionCredentials) -> None:
    SESSION_FILE.touch(mode=0o600)
    SESSION_FILE.write_text(json.dumps(asdict(credentials)))


def is_local_page(page_url: str) -> bool:
    return urlparse(page_url).hostname in LOCAL_HOSTS


def call(
    path: str, body: dict | None = None, *, token: str | None = None, session: SessionCredentials | None = None
) -> dict:
    """GET (no body) or POST JSON to the Belay API. ``session`` (default: this process's session) goes with every
    call; ``token`` is the local preview's service token (default: from the environment, for ``agent/`` routes)."""
    session = session or session_credentials()
    headers = {"content-type": "application/json"}
    if session is not None:
        headers |= {SESSION_HEADER: session.id, SESSION_SECRET_HEADER: session.secret}
    url = f"{CONFIG.page_url}api/{path}"
    if is_local_page(CONFIG.page_url):
        token = token or (_local_service_token() if path.startswith("agent/") else None)
        return _local_call(url, body, headers | ({TOKEN_HEADER: token} if token else {}))
    if token is not None:
        raise ValueError("the service token is only for a local preview; Marina callers use their IAP identity")
    return _marina_call(url, body, headers | {"Authorization": f"Bearer {bearer_token()}"})


def _local_call(url: str, body: dict | None, headers: dict[str, str]) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise PageRequestError(error.code, error.read().decode()[:500]) from error


def _marina_call(url: str, body: dict | None, headers: dict[str, str]) -> dict:
    method = "GET" if body is None else "POST"
    response = requests.request(method, url, json=body, headers=headers, timeout=REQUEST_TIMEOUT)
    if response.status_code >= 400:
        raise PageRequestError(response.status_code, response.text[:500])
    return response.json()


def _local_service_token() -> str:
    token = os.environ.get(TOKEN_ENV)
    if not token:
        raise SystemExit(f"{TOKEN_ENV} is not set; the local dispatcher passes the service token to each session")
    return token


def _fields(cls: type, row: dict) -> dict:
    return {f.name: row[f.name] for f in fields(cls)}


def page_state() -> PageState:
    state = call("agent/state")
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
    call("agent/runs/expected", {"name": run_name, "expected_at": expected_at.isoformat()})


def report_idea(idea_id: int, report: dict, *, dry_run: bool) -> None:
    """Update an idea on the page with any of ``state``, ``registry_id``, ``registry_url``, ``plan``, ``result``,
    ``note``."""
    if dry_run:
        print(f"would report idea #{idea_id}: {json.dumps(report)}")
        return
    call(f"agent/ideas/{idea_id}/report", report)
    print(f"reported idea #{idea_id}: state {report.get('state', 'unchanged')}")


@dataclass(frozen=True)
class Post:
    kind: str
    """``plan``, ``result`` or ``escalation``."""
    title: str
    body: str
    registry_id: str | None = None
    idea_id: int | None = None


def post_activity(post: Post, *, dry_run: bool) -> None:
    """Post to the page's activity feed (escalations and notes, which only the admin page shows). The app records
    the posting session (from its credentials), so the admin page links each post to its transcript."""
    if dry_run:
        print(f"would post {post.kind} to the page:\n--- {post.title}\n{post.body}\n---")
        return
    call("agent/activity", asdict(post))
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
    """The idea's row on the Status page, which opens to its agent thread (a fragment route: Marina's canonical applet
    URL serves only the root document)."""
    return f"{CONFIG.page_url}#status/{registry_id}"


def read_thread(registry_id: str) -> str | None:
    """The idea's agent thread so far, or None before its first plan."""
    try:
        return call(f"agent/threads/{registry_id}")["body"]
    except PageRequestError as error:
        if error.status_code == 404:
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
