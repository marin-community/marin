# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin Belay: the dispatcher's event queue and the maintainers' admin page.

Maintainers approve intake suggestions here. Each approval, finished run or pending piece of work becomes an event;
the dispatcher claims every undispatched event for one agent session when dispatch is on and no session is active.
The session plans, submits runs and reports back through the agent endpoints, which feed the page's idea timelines
and activity feed. The watcher reports Belay's Iris runs, the intake issue's comments and storage use every tick.

The kernel serves three groups of routes under ``/belay/api``:

- ``/public/*``: read-only GETs with no caller check and no internal detail (no transcripts, paths, job ids, node or
  storage figures, people's Marina identities). The public page reads only these, so it can be served from Marina's
  GET/HEAD-only public applet host.
- ``/admin/*``: the maintainers' view and controls. The caller must be a signed-in Marina user listed in the
  ``approvers`` setting; anyone else gets 403.
- ``/agent/*``: the watcher's and sessions' event queue, session reports and syncs. The caller must be a Marina
  identity listed in ``session_identities`` (Loom's session service account) or an approver, or send the service
  token in the ``X-Belay-Token`` header, checked against ``service_token_sha256`` (unset: no token is accepted).
  A session's own writes (its reports, threads, activity, expected finishes, its finish and a drain claim) also need
  the session id and secret that its ``claim`` issued, in ``X-Belay-Session`` and ``X-Belay-Session-Secret``, while
  the session is running: another caller with the same identity can only claim, never write into that session.
"""

import datetime
import hashlib
import hmac
import json
import re
import secrets
from enum import StrEnum
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from marina.apps import RegisteredApi, Services, registered_api
from pydantic import BaseModel, ConfigDict, Field, field_validator
from rigging.server_auth import get_verified_identity
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

DEFAULT_SETTINGS: dict[str, Any] = {
    "dispatch_enabled": False,
    "github_posting": False,
    "node_cap": 8,
    "storage_limit_bytes": 1_000_000_000_000,
    # Who may use the admin view: Marina identities. Seeded from the intake bot's BELAY_APPROVER_IDS
    # (ops-belay-intake.yaml); an entry without an identity cannot sign in until a maintainer maps it.
    "approvers": [
        {"github": "ClassicLarry", "github_id": 42926649, "identity": "larry.dial@openathena.ai"},
        {"github": "dlwh", "github_id": 9633, "identity": None},
        {"github": "Helw150", "github_id": 9847335, "identity": None},
        {"github": "hammer", "github_id": 15233, "identity": None},
        {"github": "rjpower", "github_id": 607207, "identity": None},
        {"github": "ravwojdyla", "github_id": 1419010, "identity": None},
        {"github": "karansag", "github_id": 1545121, "identity": None},
        {"github": "yonromai", "github_id": 1596570, "identity": None},
        {"github": "atqamar", "github_id": 1707358, "identity": None},
        {"github": "muchanem", "github_id": 8000756, "identity": None},
    ],
    # Marina identities of the Belay sessions (Loom's VM service account): they may call the agent endpoints.
    "session_identities": ["loom-vm@hai-gcp-models.iam.gserviceaccount.com"],
    # When set, approvers must also hold Marina's admin role.
    "admin_role_required": False,
    # SHA-256 of an optional shared agent token; unset means no token is accepted (identities still are).
    "service_token_sha256": None,
}
# Settings that never leave the server, even to approvers.
SECRET_SETTINGS = frozenset({"service_token_sha256"})
ANONYMOUS = "anonymous"
TOKEN_HEADER = "X-Belay-Token"
SESSION_HEADER = "X-Belay-Session"
SESSION_SECRET_HEADER = "X-Belay-Session-Secret"
PUBLIC_EXCERPT = 600
PUBLIC_ACTIVITY = 30
TERMINAL_RUN_STATES = frozenset({"succeeded", "failed", "killed", "worker_failed", "unschedulable"})
# A session that has not reported back within this long is presumed lost, so the queue is not blocked forever.
# Loom ends a session after 4 hours; a draining session stops taking new work well before that.
SESSION_TIMEOUT = datetime.timedelta(hours=4, minutes=15)
# A run still going this long past its expected finish gets a run_overdue event, once.
OVERDUE_GRACE = datetime.timedelta(minutes=30)
INTAKE_URL_PREFIX = "https://github.com/marin-community/marin/issues/9514#issuecomment-"
# Re-offer approved work that no session has picked up after this long.
WORK_PENDING_INTERVAL = datetime.timedelta(hours=1)
TRANSCRIPT_LIMIT = 600_000
# A run first seen already finished gets a run_finished event only if it was submitted this recently.
FIRST_SEEN_WINDOW = datetime.timedelta(days=1)
LABEL_WIDTH = 90
RECENT_SESSIONS = 20
RECENT_ACTIVITY = 60

SCHEMA = (
    """CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY, value JSONB NOT NULL, updated_by TEXT, updated_at TIMESTAMPTZ NOT NULL DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS ideas (
        id BIGSERIAL PRIMARY KEY, origin TEXT NOT NULL, comment_id TEXT UNIQUE, author TEXT NOT NULL,
        body TEXT NOT NULL, url TEXT, posted_at TIMESTAMPTZ NOT NULL, state TEXT NOT NULL, title TEXT,
        registry_id TEXT, registry_url TEXT, decided_by TEXT, decided_at TIMESTAMPTZ, plan JSONB, result JSONB,
        note TEXT, node_grant INTEGER, hidden_removed JSONB, reply_url TEXT,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS runs (
        name TEXT PRIMARY KEY, job_id TEXT NOT NULL, registry_id TEXT, run TEXT, state TEXT NOT NULL,
        submitted_at TIMESTAMPTZ, finished_at TIMESTAMPTZ, nodes INTEGER NOT NULL DEFAULT 1, step INTEGER,
        total_steps INTEGER, loss DOUBLE PRECISION, wandb_url TEXT, updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        step_at TIMESTAMPTZ, steps_per_sec DOUBLE PRECISION)""",
    """CREATE TABLE IF NOT EXISTS events (
        id BIGSERIAL PRIMARY KEY, at TIMESTAMPTZ NOT NULL DEFAULT now(), kind TEXT NOT NULL, ref TEXT,
        summary TEXT NOT NULL, payload JSONB NOT NULL DEFAULT '{}', session_id TEXT)""",
    """CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY, started_at TIMESTAMPTZ NOT NULL DEFAULT now(), ended_at TIMESTAMPTZ, state TEXT NOT NULL,
        summary TEXT, transcript TEXT, turns INTEGER, input_tokens BIGINT, output_tokens BIGINT,
        cache_read_tokens BIGINT, secret_sha256 TEXT)""",
    # A sessions table created before the claim secret existed lacks this column.
    "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS secret_sha256 TEXT",
    """CREATE TABLE IF NOT EXISTS activity (
        id BIGSERIAL PRIMARY KEY, at TIMESTAMPTZ NOT NULL DEFAULT now(), kind TEXT NOT NULL, registry_id TEXT,
        idea_id BIGINT, session_id TEXT, title TEXT NOT NULL, body TEXT NOT NULL DEFAULT '', github_url TEXT)""",
    """CREATE TABLE IF NOT EXISTS threads (
        registry_id TEXT PRIMARY KEY, idea_id BIGINT, title TEXT NOT NULL, body TEXT NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS run_expectations (
        name TEXT PRIMARY KEY, expected_at TIMESTAMPTZ NOT NULL, overdue_noted BOOLEAN NOT NULL DEFAULT false)""",
    """CREATE TABLE IF NOT EXISTS registry (
        id INTEGER PRIMARY KEY, revision TEXT NOT NULL, built_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        data JSONB NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS watcher (
        id INTEGER PRIMARY KEY, ticked_at TIMESTAMPTZ, nodes_in_use INTEGER, storage_bytes BIGINT,
        storage_objects BIGINT, storage_at TIMESTAMPTZ, last_error TEXT)""",
)


RUNS_WITH_EXPECTATIONS = (
    "SELECT runs.*, e.expected_at FROM runs LEFT JOIN run_expectations e USING (name) "
    "ORDER BY submitted_at DESC NULLS LAST"
)
_TERMINAL_SQL = ", ".join(f"'{state}'" for state in sorted(TERMINAL_RUN_STATES))
UPSERT_RUN = f"""
INSERT INTO runs (name, job_id, registry_id, run, state, submitted_at, finished_at, nodes, step, total_steps, loss,
                  wandb_url, updated_at)
VALUES (:name, :job_id, :rid, :run, :state, :submitted_at, NULL, :nodes, :step, :total_steps, :loss, :wandb_url, now())
ON CONFLICT (name) DO UPDATE SET
    state = EXCLUDED.state,
    step = COALESCE(EXCLUDED.step, runs.step),
    total_steps = COALESCE(EXCLUDED.total_steps, runs.total_steps),
    loss = COALESCE(EXCLUDED.loss, runs.loss),
    wandb_url = COALESCE(EXCLUDED.wandb_url, runs.wandb_url),
    nodes = EXCLUDED.nodes,
    -- When a run is first seen already finished, its end time is unknown; otherwise it is when the watcher saw it end.
    finished_at = COALESCE(
        runs.finished_at,
        CASE WHEN CAST(:terminal AS BOOLEAN) AND runs.state NOT IN ({_TERMINAL_SQL}) THEN now() END
    ),
    updated_at = now()
"""


class IdeaState(StrEnum):
    PENDING = "pending"
    """An intake suggestion or maintainer proposal waiting for a maintainer."""
    DISMISSED = "dismissed"
    APPROVED = "approved"
    """Approved; no session has planned it yet."""
    PLANNED = "planned"
    """A session posted a plan and submitted (or is about to submit) runs."""
    RUNNING = "running"
    RECORDED = "recorded"
    """Results are in the registry."""
    DECLINED = "declined"
    """Closed without a run: a session found it outside the house rules, or a maintainer declined it."""


class EventKind(StrEnum):
    APPROVED = "approved"
    RUN_FINISHED = "run_finished"
    WORK_PENDING = "work_pending"
    RUN_OVERDUE = "run_overdue"


class SessionState(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    LOST = "lost"


class IntakeComment(BaseModel):
    comment_id: str
    author: str
    body: str
    url: str
    posted_at: datetime.datetime


class ThreadPost(BaseModel):
    """An idea's agent thread: its plan, each stage's table and its verdict, in one markdown post updated in place."""

    title: str
    body: str
    idea_id: int | None = None
    kind: str = "plan"
    """``plan`` or ``result``: the activity entry the update adds."""


class VisibleRequest(BaseModel):
    bodies: list[str]


class RunReport(BaseModel):
    name: str
    job_id: str
    state: str
    submitted_at: datetime.datetime | None = None
    nodes: int = 1
    step: int | None = None
    total_steps: int | None = None
    loss: float | None = None
    wandb_url: str | None = None


class RunsSync(BaseModel):
    runs: list[RunReport]


class Tick(BaseModel):
    nodes_in_use: int
    storage_bytes: int | None = None
    storage_objects: int | None = None
    error: str | None = None


class Proposal(BaseModel):
    title: str
    body: str


class Approver(BaseModel):
    github: str
    github_id: int | None = None
    identity: str | None = None
    """The Marina identity (IAP email) this approver signs in with."""


class SettingsUpdate(BaseModel):
    """The settings an approver may change from the admin view."""

    dispatch_enabled: bool | None = None
    github_posting: bool | None = None
    node_cap: int | None = Field(default=None, ge=1)
    """Soft cap on nodes held by Belay runs; launches read it from the app."""
    approvers: list[Approver] | None = None


class MaintainerNote(BaseModel):
    """A maintainer's note on a re-queue or a decline, and for a re-queue an optional node grant."""

    note: str = ""
    nodes: int | None = Field(default=None, ge=1)


class Approval(BaseModel):
    nodes: int | None = Field(default=None, ge=1)
    """A node grant: this idea's launches may hold up to this many nodes, past the global cap."""


class IntakeApproval(BaseModel):
    """A #9514 comment an approver approves from the admin page, as GitHub serves it."""

    comment_id: str
    url: str
    author: str
    body: str
    shown: str
    """The visible text the admin page showed the approver; the approval stores exactly this."""
    posted_at: datetime.datetime
    nodes: int | None = Field(default=None, ge=1)


class RunExpectation(BaseModel):
    name: str
    expected_at: datetime.datetime


class Claim(BaseModel):
    session_id: str
    kinds: list[EventKind] | None = None
    """Only these kinds of event (a draining session past its budget takes only run results); None takes all."""


class SessionFinish(BaseModel):
    state: SessionState
    """A terminal state: succeeded, failed or lost."""

    @field_validator("state")
    @classmethod
    def _terminal(cls, state: SessionState) -> SessionState:
        if state is SessionState.RUNNING:
            raise ValueError("a finish report needs a terminal state")
        return state

    summary: str
    transcript: str = ""
    input_tokens: int | None = None
    """Input tokens not read from the prompt cache (fresh input plus cache writes)."""
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    turns: int | None = None


class IdeaReport(BaseModel):
    """What a session may set on an idea; anything else (a node grant, say) is refused."""

    model_config = ConfigDict(extra="forbid")

    state: IdeaState | None = None
    title: str | None = None
    registry_id: str | None = None
    registry_url: str | None = None
    plan: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    note: str | None = None
    reply_url: str | None = None
    """The session's one reply on the idea's #9514 comment, which it edits when the verdict lands."""


class RegistryUpload(BaseModel):
    revision: str
    """The registry branch commit the data was built from."""
    data: dict[str, Any]


class ActivityPost(BaseModel):
    kind: str
    title: str
    body: str = ""
    registry_id: str | None = None
    idea_id: int | None = None
    session_id: str | None = None
    github_url: str | None = None


def migrate(connection: Connection) -> None:
    for statement in SCHEMA:
        connection.execute(text(statement))
    for key, value in DEFAULT_SETTINGS.items():
        connection.execute(
            text("INSERT INTO settings (key, value) VALUES (:k, CAST(:v AS JSONB)) ON CONFLICT (key) DO NOTHING"),
            {"k": key, "v": json.dumps(value)},
        )
    connection.execute(text("INSERT INTO watcher (id) VALUES (1) ON CONFLICT (id) DO NOTHING"))


def _signed_in_user() -> str | None:
    """The caller's Marina identity; None for an anonymous caller."""
    identity = get_verified_identity()
    if identity is not None and identity.user_id != ANONYMOUS:
        return identity.user_id
    return None


def _approver_identities(settings: dict[str, Any]) -> set[str]:
    return {a["identity"] for a in settings["approvers"] if a.get("identity")}


def agent_caller_ok(settings: dict[str, Any], user: str | None, token: str | None) -> bool:
    """Whether a caller may use the agent endpoints: a session identity, an approver, or the service token."""
    if _token_ok(settings, token):
        return True
    return user is not None and user in set(settings["session_identities"]) | _approver_identities(settings)


def _sha256(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def require_session(conn: Connection, session_id: str | None, secret: str | None) -> str:
    """The calling session's id, when it presents the secret its claim issued and is still running; 403 otherwise
    (a missing or wrong secret, or a session that finished, failed or was expired as lost)."""
    if not session_id or not secret:
        raise HTTPException(403, f"a session write needs {SESSION_HEADER} and {SESSION_SECRET_HEADER} from its claim")
    _expire_sessions(conn)
    rows = _rows(conn, "SELECT state, secret_sha256 FROM sessions WHERE id = :id", id=session_id)
    expected = rows[0]["secret_sha256"] if rows else None
    if not expected or not hmac.compare_digest(_sha256(secret), expected):
        raise HTTPException(403, f"wrong secret for session {session_id}")
    if rows[0]["state"] != SessionState.RUNNING:
        raise HTTPException(403, f"session {session_id} is {rows[0]['state']}")
    return session_id


def _token_ok(settings: dict[str, Any], token: str | None) -> bool:
    expected = settings["service_token_sha256"]
    if not expected or not token:
        return False
    return hmac.compare_digest(_sha256(token), expected)


def _label(idea: dict[str, Any]) -> str:
    """An idea's title, else the start of its text's first line."""
    return idea["title"] or idea["body"].strip().splitlines()[0][:LABEL_WIDTH]


def _session_active(conn: Connection) -> bool:
    return bool(_rows(conn, f"SELECT id FROM sessions WHERE state = '{SessionState.RUNNING}'"))


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _rows(conn: Connection, sql: str, **params: Any) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute(text(sql), params).mappings()]


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime.datetime) else value


def _plain(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: _iso(v) for k, v in row.items()} for row in rows]


def _settings(conn: Connection) -> dict[str, Any]:
    stored = {r["key"]: r["value"] for r in _rows(conn, "SELECT key, value FROM settings")}
    return {**DEFAULT_SETTINGS, **stored}


def _enqueue(conn: Connection, kind: EventKind, ref: str, summary: str, payload: dict[str, Any]) -> None:
    conn.execute(
        text("INSERT INTO events (kind, ref, summary, payload) VALUES (:kind, :ref, :summary, CAST(:payload AS JSONB))"),
        {"kind": kind.value, "ref": ref, "summary": summary, "payload": json.dumps(payload)},
    )


ESCALATED = "Escalated:"
# Approved ideas a session may take up: not one a session escalated, until a maintainer re-queues or declines it.
OFFERABLE_IDEAS = (
    f"SELECT id, title, body FROM ideas WHERE state = 'approved' AND (note IS NULL OR note NOT LIKE '{ESCALATED}%')"
)


def approver_login(settings: dict[str, Any], identity: str) -> str:
    """The GitHub login of the approver who signs in as ``identity`` (the name posts credit), else ``identity``."""
    for a in settings["approvers"]:
        if a.get("identity") == identity:
            return a["github"]
    return identity


def offer_pending_work(conn: Connection) -> int:
    """When nothing is running or queued, offer each approved idea no session has taken up (at most hourly);
    an idea a session escalated waits for a maintainer instead."""
    if _session_active(conn) or _rows(conn, "SELECT ref FROM events WHERE session_id IS NULL"):
        return 0
    offered = 0
    for idea in _rows(conn, OFFERABLE_IDEAS):
        last = _rows(
            conn,
            "SELECT max(at) AS at FROM events WHERE ref = :ref AND kind IN ('approved', 'work_pending')",
            ref=str(idea["id"]),
        )[0]["at"]
        if last is not None and _now() - last < WORK_PENDING_INTERVAL:
            continue
        offered += 1
        _enqueue(
            conn,
            EventKind.WORK_PENDING,
            str(idea["id"]),
            f"idea #{idea['id']} is approved but has no plan yet",
            {"idea_id": idea["id"], "title": idea["title"], "body": idea["body"]},
        )
    return offered


def approve_idea(conn: Connection, idea_id: int, caller: str, node_grant: int | None = None) -> bool:
    """Approve an idea and queue its one ``approved`` event; False unless it is pending.

    Approving an idea that already has its ``approved`` event queues nothing more, so it runs once. A pending
    idea's text freezes here: from now on sessions get the text as it was approved.
    """
    rows = _rows(conn, "SELECT * FROM ideas WHERE id = :id FOR UPDATE", id=idea_id)
    if not rows:
        raise HTTPException(404, f"no idea {idea_id}")
    idea = rows[0]
    if idea["state"] != IdeaState.PENDING:
        return False
    conn.execute(
        text(
            "UPDATE ideas SET state = 'approved', decided_by = :by, decided_at = now(), "
            "node_grant = COALESCE(CAST(:grant AS INTEGER), node_grant), updated_at = now() WHERE id = :id"
        ),
        {"by": caller, "grant": node_grant, "id": idea_id},
    )
    grant = f" with up to {node_grant} nodes" if node_grant else ""
    _activity(conn, ActivityPost(kind="approval", idea_id=idea_id, title=f"{caller} approved “{_label(idea)}”{grant}"))
    queued = _rows(conn, "SELECT id FROM events WHERE kind = 'approved' AND ref = :ref", ref=str(idea_id))
    if not queued:
        _enqueue(
            conn,
            EventKind.APPROVED,
            str(idea_id),
            f"idea #{idea_id} approved by {caller}",
            {
                "idea_id": idea_id,
                "origin": idea["origin"],
                "author": idea["author"],
                "title": idea["title"],
                "body": idea["body"],
                "url": idea["url"],
                "approved_by": caller,
            },
        )
    return True


ALREADY_RUN = "Already run as"
"""The note on an intake idea whose #9514 comment the registry already holds (it ran before the page tracked it)."""


def _comment_id(url: str) -> str:
    return url.rsplit("issuecomment-", 1)[-1]


def link_registered_intake(conn: Connection) -> int:
    """Mark each intake idea still waiting for a decision whose #9514 comment already has a registry entry as
    recorded under that entry's id, so it is never queued again; only a maintainer's re-queue runs it. Returns how
    many it marked."""
    rows = _rows(conn, "SELECT data FROM registry WHERE id = 1")
    if not rows:
        return 0
    registered = {
        _comment_id(i["intake"]["url"]): i["xid"] for i in rows[0]["data"]["ideas"] if (i.get("intake") or {}).get("url")
    }
    marked = 0
    for idea in _rows(conn, "SELECT id, url FROM ideas WHERE origin = 'intake' AND state = 'pending'"):
        xid = registered.get(_comment_id(idea["url"] or ""))
        if xid is None:
            continue
        conn.execute(
            text(
                "UPDATE ideas SET state = 'recorded', registry_id = :xid, note = :note, updated_at = now() "
                "WHERE id = :id"
            ),
            {"xid": xid, "note": f"{ALREADY_RUN} {xid}", "id": idea["id"]},
        )
        marked += 1
    return marked


# Hidden content a reviewer of the rendered comment would not see, removed before a text is stored.
HTML_COMMENT = re.compile(r"<!--.*?(-->|$)", re.DOTALL)
DETAILS = re.compile(r"<details\b[^>]*>(.*?)</details>", re.DOTALL | re.IGNORECASE)
# Zero-width characters and bidirectional controls, which render as nothing but can hide or reorder text.
INVISIBLE = re.compile("[\u200b-\u200f\u2028-\u202e\u2060-\u2069\ufeff]")


def visible_text(body: str) -> tuple[str, list[str]]:
    """``body`` as a reviewer sees it rendered, and what was removed or marked: HTML comments and invisible
    characters are dropped; ``<details>`` bodies are kept between markers. The approved text is this, so a session
    only ever reads what the approver saw."""
    notes = []
    if HTML_COMMENT.search(body):
        notes.append("hidden HTML comment removed")
        body = HTML_COMMENT.sub("", body)
    if INVISIBLE.search(body):
        notes.append("zero-width or bidi characters removed")
        body = INVISIBLE.sub("", body)
    if DETAILS.search(body):
        notes.append("collapsed <details> text marked")
        body = DETAILS.sub(lambda m: f"[collapsed on GitHub:]{m.group(1)}[end collapsed]", body)
    return body, notes


def record_intake_comment(conn: Connection, c: IntakeComment) -> None:
    """Add a #9514 comment as a pending intake idea with its visible text (a pending one takes the newer text; a
    decided one keeps the text it was decided on), then mark it recorded if the registry already holds it."""
    body, removed = visible_text(c.body)
    conn.execute(
        text(
            "INSERT INTO ideas (origin, comment_id, author, body, url, posted_at, state, hidden_removed) "
            "VALUES ('intake', :cid, :author, :body, :url, :posted_at, 'pending', CAST(:removed AS JSONB)) "
            "ON CONFLICT (comment_id) DO UPDATE SET body = EXCLUDED.body, hidden_removed = EXCLUDED.hidden_removed "
            "WHERE ideas.state = 'pending'"
        ),
        {
            "cid": c.comment_id,
            "author": c.author,
            "body": body,
            "url": c.url,
            "posted_at": c.posted_at,
            "removed": json.dumps(removed),
        },
    )
    link_registered_intake(conn)


# Ideas with no approval: their decided_at, if any, is not an approval time.
UNDECIDED_STATES = (IdeaState.PENDING, IdeaState.DISMISSED)
# Ideas a maintainer may decline: not yet planned or running, and not already closed.
DECLINABLE_STATES = (IdeaState.PENDING, IdeaState.APPROVED)


def decline_idea(conn: Connection, idea_id: int, login: str, note: str) -> dict[str, str]:
    """Close an idea without running it; refuses (writing nothing) an unknown idea or one already planned, running
    or closed."""
    rows = _rows(conn, "SELECT state FROM ideas WHERE id = :id FOR UPDATE", id=idea_id)
    if not rows:
        raise HTTPException(404, f"no idea {idea_id}")
    if rows[0]["state"] not in DECLINABLE_STATES:
        raise HTTPException(409, f"idea {idea_id} is {rows[0]['state']}; only a waiting idea can be declined")
    conn.execute(
        text("UPDATE ideas SET state = 'declined', note = :n, updated_at = now() WHERE id = :id"),
        {"n": f"Declined by {login}: {note or 'no reason given'}", "id": idea_id},
    )
    _activity(conn, ActivityPost(kind="approval", idea_id=idea_id, title=f"{login} declined idea #{idea_id}"))
    return {"state": IdeaState.DECLINED.value}


def finish_session(conn: Connection, session_id: str, report: SessionFinish) -> dict[str, str]:
    """Record a running session's report and, unless it succeeded, offer its events once more. The caller checked
    the session's secret (``require_session``), which also refuses a session no longer running."""
    rows = _rows(conn, "SELECT state FROM sessions WHERE id = :id FOR UPDATE", id=session_id)
    if not rows:
        raise HTTPException(404, f"no session {session_id}")
    conn.execute(
        text(
            "UPDATE sessions SET state = :state, ended_at = now(), summary = :summary, "
            "transcript = :transcript, turns = :turns, input_tokens = :input, output_tokens = :output, "
            "cache_read_tokens = :cache_read WHERE id = :id"
        ),
        {
            "state": report.state.value,
            "summary": report.summary,
            "transcript": report.transcript[-TRANSCRIPT_LIMIT:],
            "input": report.input_tokens,
            "output": report.output_tokens,
            "cache_read": report.cache_read_tokens,
            "turns": report.turns,
            "id": session_id,
        },
    )
    if report.state is not SessionState.SUCCEEDED:
        _requeue(conn, session_id)
    _activity(
        conn,
        ActivityPost(
            kind="session",
            session_id=session_id,
            title=f"Session {session_id} {report.state.value}",
            body=report.summary,
        ),
    )
    return {"state": report.state.value}


def claim_events(
    conn: Connection, session_id: str, kinds: list[EventKind] | None = None, secret: str | None = None
) -> dict[str, Any]:
    """Give every undispatched event (of ``kinds``, if given) to a new session, when dispatch is on and no session is
    active; a new session gets a ``session_secret`` that its later writes present. An existing session (a draining
    session, or a retried claim) must present that secret; it takes the newly queued events too, and gets back every
    event it holds."""
    conn.execute(text("SELECT pg_advisory_xact_lock(4917)"))
    _expire_sessions(conn)
    mine = _rows(conn, "SELECT state FROM sessions WHERE id = :id", id=session_id)
    if mine:
        require_session(conn, session_id, secret)
    else:
        if not _settings(conn)["dispatch_enabled"]:
            return {"launch": False, "reason": "dispatch is off"}
        if _session_active(conn):
            return {"launch": False, "reason": "a session is active"}
    wanted = [k.value for k in kinds] if kinds is not None else [k.value for k in EventKind]
    new = _rows(conn, "SELECT * FROM events WHERE session_id IS NULL AND kind = ANY(:kinds) ORDER BY id", kinds=wanted)
    issued = None
    if not mine:
        if not new:
            return {"launch": False, "reason": "no queued events"}
        issued = secrets.token_urlsafe(32)
        conn.execute(
            text("INSERT INTO sessions (id, state, secret_sha256) VALUES (:id, :state, :digest)"),
            {"id": session_id, "state": SessionState.RUNNING, "digest": _sha256(issued)},
        )
    if new:
        conn.execute(
            text("UPDATE events SET session_id = :s WHERE id = ANY(:ids)"),
            {"s": session_id, "ids": [e["id"] for e in new]},
        )
        summary = "; ".join(e["summary"] for e in new)
        verb = "Started session for" if not mine else "Session took"
        _activity(conn, ActivityPost(kind="dispatch", session_id=session_id, title=f"{verb}: {summary}"))
    events = _rows(conn, "SELECT * FROM events WHERE session_id = :s ORDER BY id", s=session_id)
    claimed = {"launch": True, "session_id": session_id, "events": _plain(events)}
    return {**claimed, "session_secret": issued} if issued else claimed


def expect_run(conn: Connection, expectation: RunExpectation) -> None:
    """Record when a launched run should finish (a session's wait and the overdue check read it)."""
    conn.execute(
        text(
            "INSERT INTO run_expectations (name, expected_at) VALUES (:n, :at) "
            "ON CONFLICT (name) DO UPDATE SET expected_at = EXCLUDED.expected_at, overdue_noted = false"
        ),
        {"n": expectation.name, "at": expectation.expected_at},
    )


def note_overdue_runs(conn: Connection) -> int:
    """Queue one run_overdue event for each run still going (or never seen) past its expected finish plus grace."""
    overdue = _rows(
        conn,
        "SELECT e.name, e.expected_at, r.state FROM run_expectations e LEFT JOIN runs r USING (name) "
        "WHERE NOT e.overdue_noted AND e.expected_at < :cutoff "
        f"AND (r.state IS NULL OR r.state NOT IN ({_TERMINAL_SQL}))",
        cutoff=_now() - OVERDUE_GRACE,
    )
    for run in overdue:
        experiment = run["name"].removeprefix("belay-").partition("-")[0]
        _enqueue(
            conn,
            EventKind.RUN_OVERDUE,
            run["name"],
            f"{run['name']} overdue (expected by {_iso(run['expected_at'])})",
            {
                "run": run["name"],
                "experiment": experiment,
                "state": run["state"],
                "expected_at": _iso(run["expected_at"]),
            },
        )
        conn.execute(text("UPDATE run_expectations SET overdue_noted = true WHERE name = :n"), {"n": run["name"]})
    return len(overdue)


def approve_intake_comment(conn: Connection, comment: IntakeApproval, login: str) -> dict[str, Any]:
    """Approve a #9514 comment straight from the admin page: record it as an intake idea and queue it, unless the
    registry already holds it. Its visible text, as stored here, is the only text a session reads."""
    if comment.url != f"{INTAKE_URL_PREFIX}{comment.comment_id}":
        raise HTTPException(422, f"{comment.url} is not #9514 comment {comment.comment_id}")
    if visible_text(comment.body)[0] != comment.shown:
        raise HTTPException(409, "the comment changed since the page showed it; Reload and review it again")
    record_intake_comment(
        conn,
        IntakeComment(
            comment_id=comment.comment_id,
            author=comment.author,
            body=comment.body,
            url=comment.url,
            posted_at=comment.posted_at,
        ),
    )
    idea = _rows(conn, "SELECT id, state, registry_id FROM ideas WHERE comment_id = :c", c=comment.comment_id)[0]
    if idea["state"] == IdeaState.RECORDED:
        raise HTTPException(409, f"already run as {idea['registry_id']}")
    if not approve_idea(conn, idea["id"], login, node_grant=comment.nodes):
        raise HTTPException(409, f"idea {idea['id']} is already {idea['state']}")
    return {"state": IdeaState.APPROVED.value, "idea_id": idea["id"]}


def requeue_idea(conn: Connection, idea_id: int, login: str, note: str, nodes: int | None = None) -> dict[str, str]:
    """Keep the maintainer's note (and any node grant) for the session and offer the idea to the next session. A
    retried re-queue whose event is still unclaimed queues nothing more."""
    rows = _rows(conn, "SELECT * FROM ideas WHERE id = :id FOR UPDATE", id=idea_id)
    if not rows or rows[0]["state"] not in (IdeaState.APPROVED, IdeaState.RECORDED):
        raise HTTPException(409, f"idea {idea_id} is neither approved nor recorded")
    conn.execute(
        text(
            "UPDATE ideas SET state = 'approved', note = :n, "
            "node_grant = COALESCE(CAST(:grant AS INTEGER), node_grant), updated_at = now() WHERE id = :id"
        ),
        {"n": f"Maintainer {login}: {note}" if note else None, "grant": nodes, "id": idea_id},
    )
    waiting = _rows(
        conn,
        "SELECT id FROM events WHERE kind = :kind AND ref = :ref AND session_id IS NULL",
        kind=EventKind.WORK_PENDING.value,
        ref=str(idea_id),
    )
    if waiting:
        return {"state": IdeaState.APPROVED.value}
    _activity(conn, ActivityPost(kind="approval", idea_id=idea_id, title=f"{login} re-queued “{_label(rows[0])}”"))
    _enqueue(
        conn,
        EventKind.WORK_PENDING,
        str(idea_id),
        f"idea #{idea_id} re-queued by {login}",
        {"idea_id": idea_id, "title": rows[0]["title"], "body": rows[0]["body"]},
    )
    return {"state": IdeaState.APPROVED.value}


def write_thread(conn: Connection, registry_id: str, post: ThreadPost) -> None:
    """Create or replace an idea's agent thread, and add an activity entry for the update."""
    conn.execute(
        text(
            "INSERT INTO threads (registry_id, idea_id, title, body) VALUES (:rid, :idea, :title, :body) "
            "ON CONFLICT (registry_id) DO UPDATE SET idea_id = COALESCE(EXCLUDED.idea_id, threads.idea_id), "
            "title = EXCLUDED.title, body = EXCLUDED.body, updated_at = now()"
        ),
        {"rid": registry_id, "idea": post.idea_id, "title": post.title, "body": post.body},
    )
    _activity(conn, ActivityPost(kind=post.kind, title=post.title, registry_id=registry_id, idea_id=post.idea_id))


def _activity(conn: Connection, post: ActivityPost) -> int:
    return conn.execute(
        text(
            "INSERT INTO activity (kind, registry_id, idea_id, session_id, title, body, github_url) "
            "VALUES (:kind, :registry_id, :idea_id, :session_id, :title, :body, :github_url) RETURNING id"
        ),
        post.model_dump(),
    ).scalar_one()


def _requeue(conn: Connection, session_id: str) -> None:
    """Offer a failed or lost session's events once more, so one crash does not drop an approval or a result."""
    for event in _rows(conn, "SELECT kind, ref, summary, payload FROM events WHERE session_id = :s", s=session_id):
        if event["payload"].get("retry_of"):
            continue
        payload = {**event["payload"], "retry_of": session_id}
        _enqueue(conn, EventKind(event["kind"]), event["ref"], f"{event['summary']} (retry)", payload)


def _expire_sessions(conn: Connection) -> None:
    lost = conn.execute(
        text(
            "UPDATE sessions SET state = 'lost', ended_at = now(), summary = 'No report within the session timeout.' "
            "WHERE state = 'running' AND started_at < :cutoff RETURNING id"
        ),
        {"cutoff": _now() - SESSION_TIMEOUT},
    ).scalars()
    for session_id in list(lost):
        _requeue(conn, session_id)


def _training(live: list[dict]) -> str:
    """``Training a, b · slowest 1,234/8,821`` for the live runs."""
    names = ", ".join(r["name"] for r in live)
    steps = [r for r in live if r["step"] and r["total_steps"]]
    if not steps:
        return f"Training {names}"
    r = min(steps, key=lambda r: r["step"] / r["total_steps"])
    return f"Training {names} · slowest {r['step']:,}/{r['total_steps']:,}"


def _next_step(
    settings: dict[str, Any], active: dict | None, queued: list[dict], live: list[dict], pending: int
) -> dict[str, str]:
    """The admin's one-line "what happens next", and its tone (``busy``, ``waiting``, ``blocked`` or ``idle``)."""
    if active is not None:
        return {"tone": "busy", "text": f"Session {active['id']} working: {active['events']}"}
    if queued and not settings["dispatch_enabled"]:
        return {"tone": "blocked", "text": f"Dispatch off · {len(queued)} queued"}
    if queued:
        return {"tone": "busy", "text": f"Starting a session: {queued[0]['summary']}"}
    if live:
        return {"tone": "waiting", "text": _training(live)}
    if pending:
        return {"tone": "idle", "text": f"{pending} to review"}
    return {"tone": "idle", "text": "Idle"}


# Weight of the newest step-rate observation in the running average behind ETAs.
RATE_WEIGHT = 0.5


def _update_rate(conn: Connection, run: "RunReport", previous: dict[str, Any] | None) -> None:
    """Track steps per second from successive watcher reports, for the status page's ETAs."""
    if run.step is None or (previous is not None and previous["step"] == run.step):
        return
    rate = previous["steps_per_sec"] if previous else None
    if previous and previous["step"] is not None and previous["step_at"] is not None and run.step > previous["step"]:
        elapsed = (_now() - previous["step_at"]).total_seconds()
        if elapsed > 0:
            observed = (run.step - previous["step"]) / elapsed
            rate = observed if rate is None else RATE_WEIGHT * observed + (1 - RATE_WEIGHT) * rate
    conn.execute(
        text("UPDATE runs SET step_at = now(), steps_per_sec = :rate WHERE name = :n"), {"rate": rate, "n": run.name}
    )


def _public_activity(ideas: list[dict], runs: list[dict], posts: list[dict]) -> list[dict[str, Any]]:
    """One line per public event, newest first: suggestions, approvals, plans, run starts and ends, results."""
    by_id = {i["id"]: i for i in ideas}
    by_registry = {i["registry_id"]: i for i in ideas if i["registry_id"]}
    items = []
    for i in ideas:
        if i["origin"] == "intake":
            items.append({"at": i["posted_at"], "kind": "intake", "text": f"New suggestion from {i['author']}"})
        title = _label(i)
        if i["decided_at"] and i["state"] not in UNDECIDED_STATES:
            items.append({"at": i["decided_at"], "kind": "approved", "text": f"Approved: {title}"})
        if i["state"] == IdeaState.DECLINED:
            items.append({"at": i["updated_at"], "kind": "declined", "text": f"Declined: {title}"})
    for r in runs:
        if r["submitted_at"]:
            items.append({"at": r["submitted_at"], "kind": "started", "text": f"Started {r['name']}"})
        if r["finished_at"]:
            items.append({"at": r["finished_at"], "kind": "finished", "text": f"{r['name']} {r['state']}"})
    for p in posts:
        idea = by_id.get(p["idea_id"]) or by_registry.get(p["registry_id"])
        label = idea["title"] if idea and idea["title"] else p["registry_id"] or ""
        if p["kind"] == "plan" and not (idea and idea["state"] == IdeaState.DECLINED):
            items.append(
                {"at": p["at"], "kind": "plan", "registry_id": p["registry_id"], "text": f"Plan posted: {label}"}
            )
        if p["kind"] == "result" and idea and idea["result"]:
            row = next((x for x in idea["result"].get("rows", []) if not x.get("control")), None)
            change = f" {row['delta']:+.4f}" if row and row.get("delta") is not None else ""
            items.append(
                {"at": p["at"], "kind": "result", "registry_id": p["registry_id"], "text": f"Result{change}: {label}"}
            )
    items.sort(key=lambda x: x["at"], reverse=True)
    return [{**x, "at": _iso(x["at"]), "registry_id": x.get("registry_id")} for x in items[:PUBLIC_ACTIVITY]]


def _pipeline(ideas: list[dict[str, Any]]) -> dict[str, int]:
    stages = (IdeaState.PENDING, IdeaState.APPROVED, IdeaState.PLANNED, IdeaState.RUNNING, IdeaState.RECORDED)
    return {stage.value: sum(1 for i in ideas if i["state"] == stage) for stage in stages}


def _public_next(active: bool, queued: int, live: list[dict], pending: int) -> dict[str, str]:
    """The public "what happens next": no session ids, people or controls."""
    if active:
        return {"tone": "busy", "text": "Agent working"}
    if queued:
        return {"tone": "busy", "text": "Queued for the agent"}
    if live:
        return {"tone": "waiting", "text": _training(live)}
    if pending:
        return {"tone": "idle", "text": f"{pending} awaiting review"}
    return {"tone": "idle", "text": "Idle"}


def _public_idea(idea: dict[str, Any]) -> dict[str, Any]:
    plan = idea["plan"] or {}
    return {
        "id": idea["id"],
        "title": _label(idea),
        "state": idea["state"],
        "origin": idea["origin"],
        # Intake authors posted publicly on GitHub; maintainers' Marina identities are not published.
        "author": idea["author"] if idea["origin"] == "intake" else None,
        "url": idea["url"],
        "body": idea["body"][:PUBLIC_EXCERPT],
        "posted_at": _iso(idea["posted_at"]),
        "approved_at": _iso(idea["decided_at"]) if idea["state"] not in UNDECIDED_STATES else None,
        "registry_id": idea["registry_id"],
        "registry_url": idea["registry_url"],
        "plan": (
            {k: plan.get(k) for k in ("summary", "protocol", "cost", "commit", "spec_url")}
            | {"runs": [{k: r.get(k) for k in ("name", "variant", "control")} for r in plan.get("runs", [])]}
            if idea["plan"]
            else None
        ),
        "result": idea["result"],
        "note": idea["note"] if idea["state"] == IdeaState.DECLINED else None,
        "updated_at": _iso(idea["updated_at"]),
    }


def _public_run(run: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "name",
        "registry_id",
        "state",
        "step",
        "total_steps",
        "steps_per_sec",
        "loss",
        "wandb_url",
        "submitted_at",
        "finished_at",
        "expected_at",
    )
    return {k: _iso(run[k]) for k in keys}


def create_api(services: Services) -> RegisteredApi:
    api = FastAPI()
    engine: Engine = services.engine()

    def approver() -> str:
        with engine.connect() as conn:
            settings = _settings(conn)
        user = _signed_in_user()
        if user is None:
            raise HTTPException(401, "Sign in to Marina to use the Belay admin view.")
        if user not in _approver_identities(settings):
            raise HTTPException(403, f"{user} is not a Belay approver. Ask a maintainer to add you in settings.")
        identity = get_verified_identity()
        if settings["admin_role_required"] and (identity is None or identity.role != "admin"):
            raise HTTPException(403, f"{user} needs Marina's admin role for the Belay admin view.")
        return user

    def service(x_belay_token: str | None = Header(default=None)) -> None:
        with engine.connect() as conn:
            settings = _settings(conn)
        identity = get_verified_identity()
        if not agent_caller_ok(settings, None if identity is None else identity.user_id, x_belay_token):
            raise HTTPException(
                401, f"Belay agent endpoints need a Belay session identity, an approver, or {TOKEN_HEADER}."
            )

    def session(
        x_belay_session: str | None = Header(default=None), x_belay_session_secret: str | None = Header(default=None)
    ) -> str:
        """The calling session (``require_session``), for a session's own writes."""
        with engine.begin() as conn:
            return require_session(conn, x_belay_session, x_belay_session_secret)

    public = APIRouter(prefix="/public")
    admin = APIRouter(prefix="/admin", dependencies=[Depends(approver)])
    agent = APIRouter(prefix="/agent", dependencies=[Depends(service)])

    @public.get("/state")
    def public_state() -> dict[str, Any]:
        """Read-only Belay status for anyone: pipeline, suggestions, idea timelines and runs. A session past its
        timeout is marked lost first, so a scheduled check of ``work`` never waits on a dead session."""
        with engine.begin() as conn:
            _expire_sessions(conn)
            note_overdue_runs(conn)
            ideas = _rows(conn, "SELECT * FROM ideas WHERE state <> 'dismissed' ORDER BY posted_at DESC")
            runs = _rows(conn, RUNS_WITH_EXPECTATIONS)
            threads = _rows(conn, "SELECT registry_id, title, body, updated_at FROM threads ORDER BY updated_at DESC")
            active = _session_active(conn)
            queued = len(_rows(conn, "SELECT id FROM events WHERE session_id IS NULL"))
            posts = _rows(
                conn,
                "SELECT at, kind, registry_id, idea_id FROM activity WHERE kind IN ('plan', 'result') ORDER BY at DESC "
                f"LIMIT {PUBLIC_ACTIVITY}",
            )
        live = [r for r in runs if r["state"] not in TERMINAL_RUN_STATES]
        pending = sum(1 for i in ideas if i["state"] == IdeaState.PENDING)
        return {
            "now": _now().isoformat(),
            "pipeline": _pipeline(ideas),
            "next": _public_next(active, queued, live, pending),
            # For a scheduled trigger's cheap check: start a session only when events wait and none is running.
            "work": {"queued_events": queued, "session_active": active},
            "ideas": [_public_idea(i) for i in ideas],
            "runs": [_public_run(r) for r in runs],
            # The agent's posts (plans, stage tables, verdicts): no transcripts or escalations, which stay admin-only.
            "threads": _plain(threads),
            "activity": _public_activity(ideas, runs, posts),
        }

    @public.get("/results")
    def public_results() -> dict[str, Any]:
        """The registry's results (``experiments.belay.registry build``), as the watcher last uploaded them."""
        with engine.connect() as conn:
            rows = _rows(conn, "SELECT revision, built_at, data FROM registry WHERE id = 1")
        if not rows:
            raise HTTPException(404, "the watcher has not uploaded the registry yet")
        return {**rows[0]["data"], "revision": rows[0]["revision"], "built_at": _iso(rows[0]["built_at"])}

    @admin.get("/me")
    def me(user: str = Depends(approver)) -> dict[str, str]:
        return {"user": user}

    @admin.get("/state")
    def state() -> dict[str, Any]:
        with engine.begin() as conn:
            _expire_sessions(conn)
            settings = _settings(conn)
            ideas = _rows(conn, "SELECT * FROM ideas ORDER BY posted_at DESC")
            runs = _rows(conn, RUNS_WITH_EXPECTATIONS)
            activity = _rows(conn, f"SELECT * FROM activity ORDER BY at DESC LIMIT {RECENT_ACTIVITY}")
            sessions = _rows(
                conn,
                "SELECT id, started_at, ended_at, state, summary, turns, input_tokens, output_tokens, "
                "cache_read_tokens, "
                f"length(transcript) AS transcript_chars FROM sessions ORDER BY started_at DESC LIMIT {RECENT_SESSIONS}",
            )
            events = _rows(conn, "SELECT id, at, kind, ref, summary, session_id FROM events ORDER BY id DESC LIMIT 200")
            watcher = _rows(conn, "SELECT * FROM watcher WHERE id = 1")[0]
        for s in sessions:
            s["events"] = [e for e in events if e["session_id"] == s["id"]]
        queued = [e for e in events if e["session_id"] is None]
        active = next((s for s in sessions if s["state"] == SessionState.RUNNING), None)
        if active is not None:
            active = {**active, "events": "; ".join(e["summary"] for e in active["events"]) or "its queued events"}
        live = [r for r in runs if r["state"] not in TERMINAL_RUN_STATES]
        pending = [i for i in ideas if i["state"] == IdeaState.PENDING]
        return {
            "now": _now().isoformat(),
            "settings": {k: v for k, v in settings.items() if k not in SECRET_SETTINGS},
            "pipeline": _pipeline(ideas),
            "next": _next_step(settings, active, queued, live, len(pending)),
            "ideas": _plain(ideas),
            "runs": _plain(runs),
            "activity": _plain(activity),
            "sessions": [{**_plain([s])[0], "events": _plain(s["events"])} for s in sessions],
            "queued": _plain(queued),
            "watcher": _plain([watcher])[0],
            "nodes_in_use": sum(r["nodes"] for r in live if r["state"] == "running"),
            "nodes_requested": sum(r["nodes"] for r in live),
        }

    @admin.post("/ideas")
    def propose(proposal: Proposal, caller: str = Depends(approver)) -> dict[str, int]:
        with engine.begin() as conn:
            idea_id = conn.execute(
                text(
                    "INSERT INTO ideas (origin, author, body, posted_at, state, title) "
                    "VALUES ('maintainer', :author, :body, now(), 'pending', :title) RETURNING id"
                ),
                {"author": caller, "body": proposal.body, "title": proposal.title},
            ).scalar_one()
            _activity(conn, ActivityPost(kind="proposal", idea_id=idea_id, title=f"{caller} proposed: {proposal.title}"))
        return {"id": idea_id}

    @admin.post("/intake/visible")
    def intake_visible(request: VisibleRequest) -> list[dict[str, Any]]:
        """What approving each #9514 comment would store: its visible text and what was removed."""
        return [{"text": t, "removed": r} for t, r in map(visible_text, request.bodies)]

    @admin.post("/intake/approve")
    def approve_intake(comment: IntakeApproval, caller: str = Depends(approver)) -> dict[str, Any]:
        """Approve a #9514 comment the admin page read live from GitHub."""
        with engine.begin() as conn:
            return approve_intake_comment(conn, comment, approver_login(_settings(conn), caller))

    @admin.post("/ideas/{idea_id}/approve")
    def approve(idea_id: int, approval: Approval | None = None, caller: str = Depends(approver)) -> dict[str, str]:
        grant = approval.nodes if approval is not None else None
        with engine.begin() as conn:
            if not approve_idea(conn, idea_id, approver_login(_settings(conn), caller), node_grant=grant):
                raise HTTPException(409, f"idea {idea_id} is not waiting for approval")
        return {"state": IdeaState.APPROVED.value}

    @admin.post("/ideas/{idea_id}/dismiss")
    def dismiss(idea_id: int, caller: str = Depends(approver)) -> dict[str, str]:
        with engine.begin() as conn:
            idea = _rows(conn, "SELECT * FROM ideas WHERE id = :id FOR UPDATE", id=idea_id)
            if not idea:
                raise HTTPException(404, f"no idea {idea_id}")
            if idea[0]["state"] != IdeaState.PENDING:
                raise HTTPException(409, f"idea {idea_id} is already {idea[0]['state']}")
            conn.execute(
                text(
                    "UPDATE ideas SET state = 'dismissed', decided_by = :by, decided_at = now(), updated_at = now() "
                    "WHERE id = :id"
                ),
                {"by": caller, "id": idea_id},
            )
            _activity(
                conn, ActivityPost(kind="approval", idea_id=idea_id, title=f"{caller} dismissed “{_label(idea[0])}”")
            )
        return {"state": IdeaState.DISMISSED.value}

    @admin.post("/ideas/{idea_id}/requeue")
    def requeue(idea_id: int, request: MaintainerNote, caller: str = Depends(approver)) -> dict[str, str]:
        """A maintainer's answer to an escalation, or a rerun of an idea that already has results."""
        with engine.begin() as conn:
            return requeue_idea(conn, idea_id, approver_login(_settings(conn), caller), request.note, request.nodes)

    @admin.post("/ideas/{idea_id}/decline")
    def decline(idea_id: int, request: MaintainerNote, caller: str = Depends(approver)) -> dict[str, str]:
        """A maintainer closes an idea without running it."""
        with engine.begin() as conn:
            return decline_idea(conn, idea_id, approver_login(_settings(conn), caller), request.note)

    @admin.post("/settings")
    def update_settings(update: SettingsUpdate, caller: str = Depends(approver)) -> dict[str, Any]:
        with engine.begin() as conn:
            before = _settings(conn)
            for key, value in update.model_dump(exclude_none=True).items():
                conn.execute(
                    text(
                        "INSERT INTO settings (key, value, updated_by, updated_at) "
                        "VALUES (:k, CAST(:v AS JSONB), :by, now()) "
                        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_by = :by, updated_at = now()"
                    ),
                    {"k": key, "v": json.dumps(value), "by": caller},
                )
                if key == "dispatch_enabled":
                    word = "on" if value else "off"
                    _activity(conn, ActivityPost(kind="control", title=f"{caller} turned dispatch {word}"))
                if key == "node_cap" and value != before["node_cap"]:
                    _activity(conn, ActivityPost(kind="control", title=f"{caller} set the node cap to {value}"))
            return {k: v for k, v in _settings(conn).items() if k not in SECRET_SETTINGS}

    @admin.get("/sessions/{session_id}/transcript", response_class=PlainTextResponse)
    def transcript(session_id: str) -> str:
        with engine.connect() as conn:
            rows = _rows(conn, "SELECT transcript FROM sessions WHERE id = :id", id=session_id)
        if not rows:
            raise HTTPException(404, f"no session {session_id}")
        return rows[0]["transcript"] or "(no transcript yet: the session is still running)"

    # Watcher and agent endpoints.

    @agent.post("/sync/runs")
    def sync_runs(sync: RunsSync) -> dict[str, int]:
        finished = 0
        with engine.begin() as conn:
            for r in sync.runs:
                previous = _rows(conn, "SELECT state, step, step_at, steps_per_sec FROM runs WHERE name = :n", n=r.name)
                experiment, _, run = r.name.removeprefix("belay-").partition("-")
                terminal = r.state in TERMINAL_RUN_STATES
                conn.execute(
                    text(UPSERT_RUN),
                    {**r.model_dump(), "rid": experiment, "run": run, "terminal": terminal},
                )
                _update_rate(conn, r, previous[0] if previous else None)
                was_live = bool(previous) and previous[0]["state"] not in TERMINAL_RUN_STATES
                # A run that ended before the watcher first saw it (unschedulable, an early crash) still needs its
                # session; older first-seen runs are history from before this app tracked them.
                recent = r.submitted_at is not None and _now() - r.submitted_at < FIRST_SEEN_WINDOW
                if terminal and (was_live or (not previous and recent)):
                    finished += 1
                    _enqueue(
                        conn,
                        EventKind.RUN_FINISHED,
                        r.name,
                        f"{r.name} {r.state}",
                        {"run": r.name, "job_id": r.job_id, "state": r.state, "experiment": experiment},
                    )
                if not terminal and experiment:
                    conn.execute(
                        text(
                            "UPDATE ideas SET state = 'running', updated_at = now() "
                            "WHERE registry_id = :rid AND state IN ('approved', 'planned')"
                        ),
                        {"rid": experiment},
                    )
        return {"finished": finished}

    @agent.post("/tick")
    def tick(update: Tick) -> dict[str, int]:
        """The watcher's heartbeat: node use, storage, and re-offering approved work no session has taken."""
        with engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE watcher SET ticked_at = now(), nodes_in_use = :nodes, last_error = :error, "
                    "storage_bytes = COALESCE(CAST(:bytes AS BIGINT), storage_bytes), storage_objects = "
                    "COALESCE(CAST(:objects AS BIGINT), storage_objects), storage_at = CASE WHEN CAST(:bytes AS BIGINT) "
                    "IS NULL THEN storage_at ELSE now() END "
                    "WHERE id = 1"
                ),
                {
                    "nodes": update.nodes_in_use,
                    "error": update.error,
                    "bytes": update.storage_bytes,
                    "objects": update.storage_objects,
                },
            )
            overdue = note_overdue_runs(conn)
            offered = offer_pending_work(conn)
        return {"work_pending": offered, "overdue": overdue}

    @agent.post("/dispatch/claim")
    def claim(request: Claim, x_belay_session_secret: str | None = Header(default=None)) -> dict[str, Any]:
        """``claim_events``: a new session's claim, or an existing session's (with its secret) drain claim."""
        with engine.begin() as conn:
            return claim_events(conn, request.session_id, request.kinds, x_belay_session_secret)

    @agent.post("/sessions/{session_id}/finish")
    def finish(session_id: str, report: SessionFinish, caller: str = Depends(session)) -> dict[str, str]:
        if caller != session_id:
            raise HTTPException(403, f"session {caller} cannot finish session {session_id}")
        with engine.begin() as conn:
            return finish_session(conn, session_id, report)

    @agent.get("/threads/{registry_id}")
    def read_thread(registry_id: str) -> dict[str, Any]:
        with engine.connect() as conn:
            rows = _rows(conn, "SELECT * FROM threads WHERE registry_id = :rid", rid=registry_id)
        if not rows:
            raise HTTPException(404, f"no thread for {registry_id}")
        return _plain(rows)[0]

    @agent.post("/threads/{registry_id}", dependencies=[Depends(session)])
    def post_thread(registry_id: str, post: ThreadPost) -> dict[str, str]:
        with engine.begin() as conn:
            write_thread(conn, registry_id, post)
        return {"registry_id": registry_id}

    @agent.post("/ideas/{idea_id}/report", dependencies=[Depends(session)])
    def report(idea_id: int, update: IdeaReport) -> dict[str, str]:
        fields = update.model_dump(exclude_none=True)
        if not fields:
            raise HTTPException(400, "nothing to update")
        assignments = ", ".join(
            f"{k} = CAST(:{k} AS JSONB)" if k in ("plan", "result") else f"{k} = :{k}" for k in fields
        )
        params = {k: json.dumps(v) if k in ("plan", "result") else v for k, v in fields.items()}
        with engine.begin() as conn:
            updated = conn.execute(
                text(f"UPDATE ideas SET {assignments}, updated_at = now() WHERE id = :id"), {**params, "id": idea_id}
            ).rowcount
        if not updated:
            raise HTTPException(404, f"no idea {idea_id}")
        return {"updated": ", ".join(fields)}

    @agent.post("/activity")
    def post_activity(post: ActivityPost, caller: str = Depends(session)) -> dict[str, int]:
        with engine.begin() as conn:
            return {"id": _activity(conn, post.model_copy(update={"session_id": caller}))}

    @agent.get("/registry")
    def registry_revision() -> dict[str, str | None]:
        with engine.connect() as conn:
            rows = _rows(conn, "SELECT revision FROM registry WHERE id = 1")
        return {"revision": rows[0]["revision"] if rows else None}

    @agent.post("/registry")
    def upload_registry(upload: RegistryUpload) -> dict[str, str]:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO registry (id, revision, built_at, data) VALUES (1, :rev, now(), CAST(:data AS JSONB)) "
                    "ON CONFLICT (id) DO UPDATE SET revision = EXCLUDED.revision, built_at = now(), data = EXCLUDED.data"
                ),
                {"rev": upload.revision, "data": json.dumps(upload.data)},
            )
            link_registered_intake(conn)
        return {"revision": upload.revision}

    @agent.get("/state")
    def agent_state() -> dict[str, Any]:
        """The full state, for sessions, plus every run's expected finish (including runs the watcher has not seen
        yet)."""
        with engine.connect() as conn:
            expectations = _rows(
                conn,
                "SELECT e.name, e.expected_at, r.state FROM run_expectations e LEFT JOIN runs r USING (name) "
                "ORDER BY name",
            )
        return {**state(), "expectations": _plain(expectations)}

    @agent.post("/runs/expected", dependencies=[Depends(session)])
    def runs_expected(expectation: RunExpectation) -> dict[str, str]:
        with engine.begin() as conn:
            expect_run(conn, expectation)
        return {"name": expectation.name}

    for router in (public, admin, agent):
        api.include_router(router)

    return registered_api(api)
