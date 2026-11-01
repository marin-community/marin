# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin Belay: the dispatcher's event queue and the maintainers' admin page.

Maintainers approve intake suggestions here. Each approval, finished run or pending piece of work becomes an event;
the dispatcher claims every undispatched event for one agent session when dispatch is on and no session is active.
The session plans, submits runs and reports back through the agent endpoints, which feed the page's idea timelines
and activity feed. The watcher reports Belay's Iris runs, the intake issue's comments and storage use every tick.

The kernel serves three groups of routes under ``/belay/api``:

- ``/public/*``: read-only GETs with no caller check and no internal detail (no transcripts, paths, job ids, node or
  storage figures, people's Marina identities). The public page and ``demo export-public`` read only these, so the
  same page can be served from Marina's GET/HEAD-only public applet host or from GitHub Pages.
- ``/admin/*``: the maintainers' view and controls. The caller must be a signed-in Marina user listed in the
  ``approvers`` setting; anyone else gets 403.
- ``/agent/*``: the watcher's and sessions' event queue, session reports and syncs. The caller must be a Marina
  identity listed in ``session_identities`` (Loom's session service account) or an approver; a local preview may
  instead send the service token in the ``X-Belay-Token`` header, checked against ``service_token_sha256``.
"""

import datetime
import hashlib
import hmac
import json
from enum import StrEnum
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from marina.apps import RegisteredApi, Services, registered_api
from pydantic import BaseModel
from rigging.server_auth import get_verified_identity
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

DEFAULT_SETTINGS: dict[str, Any] = {
    "dispatch_enabled": False,
    "github_posting": False,
    "node_cap": 8,
    "storage_limit_bytes": 1_000_000_000_000,
    "watch_interval": 120,
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
    # Local previews only: the identity an unauthenticated loopback caller acts as. Never set in production.
    "demo_approver": None,
    # SHA-256 of an optional shared agent token; unset means no token is accepted (identities still are).
    "service_token_sha256": None,
}
# Settings that never leave the server, even to approvers.
SECRET_SETTINGS = frozenset({"service_token_sha256"})
ANONYMOUS = "anonymous"
TOKEN_HEADER = "X-Belay-Token"
PUBLIC_EXCERPT = 600
PUBLIC_ACTIVITY = 30
TERMINAL_RUN_STATES = frozenset({"succeeded", "failed", "killed", "worker_failed", "unschedulable"})
# A session that has not reported back within this long is presumed lost, so the queue is not blocked forever.
SESSION_TIMEOUT = datetime.timedelta(hours=3)
# Re-offer approved work that no session has picked up after this long.
WORK_PENDING_INTERVAL = datetime.timedelta(hours=1)
TRANSCRIPT_LIMIT = 600_000
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
        note TEXT, updated_at TIMESTAMPTZ NOT NULL DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS runs (
        name TEXT PRIMARY KEY, job_id TEXT NOT NULL, registry_id TEXT, run TEXT, state TEXT NOT NULL,
        submitted_at TIMESTAMPTZ, finished_at TIMESTAMPTZ, nodes INTEGER NOT NULL DEFAULT 1, step INTEGER,
        total_steps INTEGER, loss DOUBLE PRECISION, wandb_url TEXT, updated_at TIMESTAMPTZ NOT NULL DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS events (
        id BIGSERIAL PRIMARY KEY, at TIMESTAMPTZ NOT NULL DEFAULT now(), kind TEXT NOT NULL, ref TEXT,
        summary TEXT NOT NULL, payload JSONB NOT NULL DEFAULT '{}', session_id TEXT)""",
    """CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY, started_at TIMESTAMPTZ NOT NULL DEFAULT now(), ended_at TIMESTAMPTZ, state TEXT NOT NULL,
        summary TEXT, transcript TEXT, cost_usd DOUBLE PRECISION, turns INTEGER)""",
    """CREATE TABLE IF NOT EXISTS activity (
        id BIGSERIAL PRIMARY KEY, at TIMESTAMPTZ NOT NULL DEFAULT now(), kind TEXT NOT NULL, registry_id TEXT,
        idea_id BIGINT, session_id TEXT, title TEXT NOT NULL, body TEXT NOT NULL DEFAULT '', github_url TEXT)""",
    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS step_at TIMESTAMPTZ",
    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS steps_per_sec DOUBLE PRECISION",
    """CREATE TABLE IF NOT EXISTS registry (
        id INTEGER PRIMARY KEY, revision TEXT NOT NULL, built_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        data JSONB NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS watcher (
        id INTEGER PRIMARY KEY, ticked_at TIMESTAMPTZ, nodes_in_use INTEGER, storage_bytes BIGINT,
        storage_objects BIGINT, storage_at TIMESTAMPTZ, last_error TEXT)""",
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
    UPSTREAM = "upstream"
    """Approved on GitHub before the cutover, so not queued; approving it on the page queues it."""
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
    approved_upstream: bool
    """The intake bot transferred it (🚀): it was approved on GitHub."""
    approved_at: datetime.datetime | None = None
    """When the bot transferred it, sent only for approvals after the watcher's cutover; such an approval queues it."""
    approved_by: str | None = None
    """The GitHub login whose 👍 approved it."""


class IntakeSync(BaseModel):
    comments: list[IntakeComment]


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
    node_cap: int | None = None
    approvers: list[Approver] | None = None


class Requeue(BaseModel):
    note: str = ""


class Claim(BaseModel):
    session_id: str


class SessionFinish(BaseModel):
    state: SessionState
    summary: str
    transcript: str = ""
    cost_usd: float | None = None
    turns: int | None = None


class IdeaReport(BaseModel):
    state: IdeaState | None = None
    title: str | None = None
    registry_id: str | None = None
    registry_url: str | None = None
    plan: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    note: str | None = None


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


def _signed_in_user(settings: dict[str, Any]) -> str | None:
    """The caller's Marina identity, or the configured demo approver for an unauthenticated local preview."""
    identity = get_verified_identity()
    if identity is not None and identity.user_id != ANONYMOUS:
        return identity.user_id
    return settings["demo_approver"]


def _approver_identities(settings: dict[str, Any]) -> set[str]:
    return {a["identity"] for a in settings["approvers"] if a.get("identity")}


def agent_caller_ok(settings: dict[str, Any], user: str | None, token: str | None) -> bool:
    """Whether a caller may use the agent endpoints: a session identity, an approver, or the service token."""
    if _token_ok(settings, token):
        return True
    return user is not None and user in set(settings["session_identities"]) | _approver_identities(settings)


def _token_ok(settings: dict[str, Any], token: str | None) -> bool:
    expected = settings["service_token_sha256"]
    if not expected or not token:
        return False
    return hmac.compare_digest(hashlib.sha256(token.encode()).hexdigest(), expected)


def _label(idea: dict[str, Any], width: int = LABEL_WIDTH) -> str:
    """An idea's title, else the start of its text's first line."""
    return idea["title"] or idea["body"].strip().splitlines()[0][:width]


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


GITHUB_APPROVER = "GitHub 👍 (intake bot)"
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
    if _rows(conn, "SELECT id FROM sessions WHERE state = 'running'") or _rows(
        conn, "SELECT ref FROM events WHERE session_id IS NULL"
    ):
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


def approve_idea(conn: Connection, idea_id: int, caller: str, at: datetime.datetime | None = None) -> bool:
    """Approve an idea and queue its one ``approved`` event; False when it is already approved or further along.

    Both approval paths (the admin page and a GitHub 👍 the watcher saw) end here, so an idea approved on both runs
    once. A pending idea's text freezes here: from now on sessions get the text as it was approved.
    """
    rows = _rows(conn, "SELECT * FROM ideas WHERE id = :id FOR UPDATE", id=idea_id)
    if not rows:
        raise HTTPException(404, f"no idea {idea_id}")
    idea = rows[0]
    if idea["state"] not in (IdeaState.PENDING, IdeaState.UPSTREAM):
        return False
    conn.execute(
        text(
            "UPDATE ideas SET state = 'approved', decided_by = :by, decided_at = COALESCE(CAST(:at AS TIMESTAMPTZ), "
            "now()), updated_at = now() WHERE id = :id"
        ),
        {"by": caller, "at": at, "id": idea_id},
    )
    label = _label(idea)
    _activity(conn, ActivityPost(kind="approval", idea_id=idea_id, title=f"{caller} approved “{label}”"))
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
    for idea in _rows(conn, "SELECT id, url FROM ideas WHERE origin = 'intake' AND state IN ('pending', 'upstream')"):
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


def sync_intake_comments(conn: Connection, comments: list[IntakeComment]) -> int:
    """Upsert #9514 comments as ideas, and approve (queue) those the watcher saw approved on GitHub after its
    cutover, except comments the registry already holds. Returns the number of new pending ideas."""
    added = 0
    for c in comments:
        state = IdeaState.UPSTREAM if c.approved_upstream else IdeaState.PENDING
        inserted = conn.execute(
            text(
                "INSERT INTO ideas (origin, comment_id, author, body, url, posted_at, state) "
                "VALUES ('intake', :cid, :author, :body, :url, :posted_at, :state) "
                "ON CONFLICT (comment_id) DO UPDATE SET body = CASE WHEN ideas.state = 'pending' "
                "THEN EXCLUDED.body ELSE ideas.body END RETURNING (xmax = 0)"
            ),
            {
                "cid": c.comment_id,
                "author": c.author,
                "body": c.body,
                "url": c.url,
                "posted_at": c.posted_at,
                "state": state.value,
            },
        ).scalar_one()
        if inserted and state is IdeaState.PENDING:
            added += 1
            _activity(conn, ActivityPost(kind="intake", title=f"New intake suggestion from {c.author}"))
    link_registered_intake(conn)
    for c in comments:
        if c.approved_at is not None:
            idea_id = _rows(conn, "SELECT id FROM ideas WHERE comment_id = :cid", cid=c.comment_id)[0]["id"]
            approve_idea(conn, idea_id, c.approved_by or GITHUB_APPROVER, c.approved_at)
    return added


# Ideas a maintainer may decline: not yet planned or running, and not already closed.
DECLINABLE_STATES = (IdeaState.PENDING, IdeaState.UPSTREAM, IdeaState.APPROVED)


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
    """Record a session's report and, unless it succeeded, offer its events once more. A session that is no longer
    running (finished, or expired as lost) is left as it is, so a retried report changes nothing."""
    rows = _rows(conn, "SELECT state FROM sessions WHERE id = :id FOR UPDATE", id=session_id)
    if not rows:
        raise HTTPException(404, f"no session {session_id}")
    if rows[0]["state"] != SessionState.RUNNING:
        return {"state": rows[0]["state"]}
    conn.execute(
        text(
            "UPDATE sessions SET state = :state, ended_at = now(), summary = :summary, "
            "transcript = :transcript, cost_usd = :cost, turns = :turns WHERE id = :id"
        ),
        {
            "state": report.state.value,
            "summary": report.summary,
            "transcript": report.transcript[-TRANSCRIPT_LIMIT:],
            "cost": report.cost_usd,
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
        if i["decided_at"] and i["state"] not in ("pending", "dismissed"):
            items.append({"at": i["decided_at"], "kind": "approved", "text": f"Approved: {title}"})
        if i["state"] == "declined":
            items.append({"at": i["updated_at"], "kind": "declined", "text": f"Declined: {title}"})
    for r in runs:
        if r["submitted_at"]:
            items.append({"at": r["submitted_at"], "kind": "started", "text": f"Started {r['name']}"})
        if r["finished_at"]:
            items.append({"at": r["finished_at"], "kind": "finished", "text": f"{r['name']} {r['state']}"})
    for p in posts:
        idea = by_id.get(p["idea_id"]) or by_registry.get(p["registry_id"])
        label = idea["title"] if idea and idea["title"] else p["registry_id"] or ""
        if p["kind"] == "plan" and not (idea and idea["state"] == "declined"):
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
        "approved_at": _iso(idea["decided_at"]) if idea["state"] not in ("pending", "dismissed") else None,
        "registry_id": idea["registry_id"],
        "registry_url": idea["registry_url"],
        "plan": (
            {k: plan.get(k) for k in ("summary", "protocol", "cost", "commit", "spec_url")}
            | {"runs": [{k: r.get(k) for k in ("name", "variant", "control")} for r in plan.get("runs", [])]}
            if idea["plan"]
            else None
        ),
        "result": idea["result"],
        "note": idea["note"] if idea["state"] == "declined" else None,
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
    )
    return {k: _iso(run[k]) for k in keys}


def create_api(services: Services) -> RegisteredApi:
    api = FastAPI()
    engine: Engine = services.engine()

    def approver() -> str:
        with engine.connect() as conn:
            settings = _settings(conn)
        user = _signed_in_user(settings)
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

    public = APIRouter(prefix="/public")
    admin = APIRouter(prefix="/admin", dependencies=[Depends(approver)])
    agent = APIRouter(prefix="/agent", dependencies=[Depends(service)])

    @public.get("/state")
    def public_state() -> dict[str, Any]:
        """Read-only Belay status for anyone: pipeline, suggestions, idea timelines and runs."""
        with engine.connect() as conn:
            ideas = _rows(
                conn, "SELECT * FROM ideas WHERE state NOT IN ('upstream', 'dismissed') ORDER BY posted_at DESC"
            )
            runs = _rows(conn, "SELECT * FROM runs ORDER BY submitted_at DESC NULLS LAST")
            active = bool(_rows(conn, "SELECT id FROM sessions WHERE state = 'running'"))
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
            runs = _rows(conn, "SELECT * FROM runs ORDER BY submitted_at DESC NULLS LAST")
            activity = _rows(conn, f"SELECT * FROM activity ORDER BY at DESC LIMIT {RECENT_ACTIVITY}")
            sessions = _rows(
                conn,
                "SELECT id, started_at, ended_at, state, summary, cost_usd, turns, "
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

    @admin.post("/ideas/{idea_id}/approve")
    def approve(idea_id: int, caller: str = Depends(approver)) -> dict[str, str]:
        with engine.begin() as conn:
            if not approve_idea(conn, idea_id, approver_login(_settings(conn), caller)):
                raise HTTPException(409, f"idea {idea_id} is already approved or further along")
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
    def requeue(idea_id: int, request: Requeue, caller: str = Depends(approver)) -> dict[str, str]:
        """A maintainer's answer to an escalation, or a rerun of an idea that already has results: keep the
        maintainer's note for the session and offer the idea to the next session."""
        with engine.begin() as conn:
            rows = _rows(conn, "SELECT * FROM ideas WHERE id = :id FOR UPDATE", id=idea_id)
            if not rows or rows[0]["state"] not in (IdeaState.APPROVED, IdeaState.RECORDED):
                raise HTTPException(409, f"idea {idea_id} is neither approved nor recorded")
            login = approver_login(_settings(conn), caller)
            note = f"Maintainer {login}: {request.note}" if request.note else None
            conn.execute(
                text("UPDATE ideas SET state = 'approved', note = :n, updated_at = now() WHERE id = :id"),
                {"n": note, "id": idea_id},
            )
            label = _label(rows[0])
            _activity(conn, ActivityPost(kind="approval", idea_id=idea_id, title=f"{login} re-queued “{label}”"))
            _enqueue(
                conn,
                EventKind.WORK_PENDING,
                str(idea_id),
                f"idea #{idea_id} re-queued by {login}",
                {"idea_id": idea_id, "title": rows[0]["title"], "body": rows[0]["body"]},
            )
        return {"state": IdeaState.APPROVED.value}

    @admin.post("/ideas/{idea_id}/decline")
    def decline(idea_id: int, request: Requeue, caller: str = Depends(approver)) -> dict[str, str]:
        """A maintainer closes an idea without running it."""
        with engine.begin() as conn:
            return decline_idea(conn, idea_id, approver_login(_settings(conn), caller), request.note)

    @admin.post("/settings")
    def update_settings(update: SettingsUpdate, caller: str = Depends(approver)) -> dict[str, Any]:
        with engine.begin() as conn:
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
            return {k: v for k, v in _settings(conn).items() if k not in SECRET_SETTINGS}

    @admin.get("/sessions/{session_id}/transcript", response_class=PlainTextResponse)
    def transcript(session_id: str) -> str:
        with engine.connect() as conn:
            rows = _rows(conn, "SELECT transcript FROM sessions WHERE id = :id", id=session_id)
        if not rows:
            raise HTTPException(404, f"no session {session_id}")
        return rows[0]["transcript"] or "(no transcript yet: the session is still running)"

    # Watcher and agent endpoints.

    @agent.post("/sync/intake")
    def sync_intake(sync: IntakeSync) -> dict[str, int]:
        with engine.begin() as conn:
            return {"new_pending": sync_intake_comments(conn, sync.comments)}

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
                if terminal and was_live:
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
            offered = offer_pending_work(conn)
        return {"work_pending": offered}

    @agent.post("/dispatch/claim")
    def claim(request: Claim) -> dict[str, Any]:
        """Give every undispatched event to a new session, when dispatch is on and no session is active."""
        with engine.begin() as conn:
            conn.execute(text("SELECT pg_advisory_xact_lock(4917)"))
            _expire_sessions(conn)
            if not _settings(conn)["dispatch_enabled"]:
                return {"launch": False, "reason": "dispatch is off"}
            if _rows(conn, "SELECT id FROM sessions WHERE state = 'running'"):
                return {"launch": False, "reason": "a session is active"}
            events = _rows(conn, "SELECT * FROM events WHERE session_id IS NULL ORDER BY id")
            if not events:
                return {"launch": False, "reason": "no queued events"}
            conn.execute(text("INSERT INTO sessions (id, state) VALUES (:id, 'running')"), {"id": request.session_id})
            conn.execute(
                text("UPDATE events SET session_id = :s WHERE id = ANY(:ids)"),
                {"s": request.session_id, "ids": [e["id"] for e in events]},
            )
            summary = "; ".join(e["summary"] for e in events)
            _activity(
                conn,
                ActivityPost(kind="dispatch", session_id=request.session_id, title=f"Started session for: {summary}"),
            )
        return {"launch": True, "session_id": request.session_id, "events": _plain(events)}

    @agent.post("/sessions/{session_id}/finish")
    def finish(session_id: str, report: SessionFinish) -> dict[str, str]:
        with engine.begin() as conn:
            return finish_session(conn, session_id, report)

    @agent.post("/ideas/{idea_id}/report")
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
    def post_activity(post: ActivityPost) -> dict[str, int]:
        with engine.begin() as conn:
            return {"id": _activity(conn, post)}

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
        """The full state, for sessions."""
        return state()

    for router in (public, admin, agent):
        api.include_router(router)

    return registered_api(api)
