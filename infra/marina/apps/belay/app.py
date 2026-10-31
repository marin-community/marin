# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin Belay: the dispatcher's event queue and the maintainers' admin page.

Maintainers approve intake suggestions here. Each approval, finished run or pending piece of work becomes an event;
the dispatcher claims every undispatched event for one agent session when dispatch is on and no session is active.
The session plans, submits runs and reports back through the agent endpoints, which feed the page's idea timelines
and activity feed. The watcher reports Belay's Iris runs, the intake issue's comments and storage use every tick.

The kernel serves these under ``/belay/api``. Browser endpoints: ``GET /state``, ``POST /ideas`` (propose),
``POST /ideas/{id}/approve|dismiss``, ``POST /settings``. Watcher and agent endpoints: ``POST /sync/intake``,
``POST /sync/runs``, ``POST /tick``, ``POST /dispatch/claim``, ``POST /sessions/{id}/finish``,
``POST /ideas/{id}/report``, ``POST /activity``. In production the agent endpoints admit only the watcher's and the
session's identities.
"""

import datetime
import json
from enum import StrEnum
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
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
}
TERMINAL_RUN_STATES = frozenset({"succeeded", "failed", "killed", "worker_failed", "unschedulable"})
# A session that has not reported back within this long is presumed lost, so the queue is not blocked forever.
SESSION_TIMEOUT = datetime.timedelta(hours=3)
# Re-offer approved work that no session has picked up after this long.
WORK_PENDING_INTERVAL = datetime.timedelta(hours=1)
TRANSCRIPT_LIMIT = 600_000
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
    """CREATE TABLE IF NOT EXISTS watcher (
        id INTEGER PRIMARY KEY, ticked_at TIMESTAMPTZ, nodes_in_use INTEGER, storage_bytes BIGINT,
        storage_objects BIGINT, storage_at TIMESTAMPTZ, last_error TEXT)""",
)


UPSERT_RUN = """
INSERT INTO runs (name, job_id, registry_id, run, state, submitted_at, finished_at, nodes, step, total_steps, loss,
                  wandb_url, updated_at)
VALUES (:name, :job_id, :rid, :run, :state, :submitted_at, CASE WHEN CAST(:terminal AS BOOLEAN) THEN now() END,
        :nodes, :step, :total_steps, :loss, :wandb_url, now())
ON CONFLICT (name) DO UPDATE SET
    state = EXCLUDED.state,
    step = COALESCE(EXCLUDED.step, runs.step),
    total_steps = COALESCE(EXCLUDED.total_steps, runs.total_steps),
    loss = COALESCE(EXCLUDED.loss, runs.loss),
    wandb_url = COALESCE(EXCLUDED.wandb_url, runs.wandb_url),
    nodes = EXCLUDED.nodes,
    finished_at = COALESCE(runs.finished_at, EXCLUDED.finished_at),
    updated_at = now()
"""


class IdeaState(StrEnum):
    PENDING = "pending"
    """An intake suggestion or maintainer proposal waiting for a maintainer."""
    UPSTREAM = "upstream"
    """Already approved on GitHub before this page existed; not shown."""
    DISMISSED = "dismissed"
    APPROVED = "approved"
    """Approved; no session has planned it yet."""
    PLANNED = "planned"
    """A session posted a plan and submitted (or is about to submit) runs."""
    RUNNING = "running"
    RECORDED = "recorded"
    """Results are in the registry."""
    DECLINED = "declined"
    """A session found it outside the house rules and explained why."""


class EventKind(StrEnum):
    APPROVED = "approved"
    RUN_FINISHED = "run_finished"
    WORK_PENDING = "work_pending"
    NOTE = "note"


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


class SettingsUpdate(BaseModel):
    dispatch_enabled: bool | None = None
    github_posting: bool | None = None
    node_cap: int | None = None


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


def _caller() -> str:
    identity = get_verified_identity()
    return identity.user_id if identity is not None else "local maintainer"


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


def _activity(conn: Connection, post: ActivityPost) -> int:
    return conn.execute(
        text(
            "INSERT INTO activity (kind, registry_id, idea_id, session_id, title, body, github_url) "
            "VALUES (:kind, :registry_id, :idea_id, :session_id, :title, :body, :github_url) RETURNING id"
        ),
        post.model_dump(),
    ).scalar_one()


def _expire_sessions(conn: Connection) -> None:
    conn.execute(
        text(
            "UPDATE sessions SET state = 'lost', ended_at = now(), summary = 'No report within the session timeout.' "
            "WHERE state = 'running' AND started_at < :cutoff"
        ),
        {"cutoff": _now() - SESSION_TIMEOUT},
    )


def _next_step(
    settings: dict[str, Any], active: dict | None, queued: list[dict], live: list[dict], pending: int
) -> dict[str, str]:
    """One plain sentence on what happens next, and its tone (``busy``, ``waiting``, ``blocked`` or ``idle``)."""
    if active is not None:
        return {"tone": "busy", "text": f"Session {active['id']} is working on {active['events']}. It exits on its own."}
    if queued and not settings["dispatch_enabled"]:
        return {
            "tone": "blocked",
            "text": f"Dispatch is off, so {len(queued)} queued event(s) wait. Turn dispatch on to start a session.",
        }
    if queued:
        return {"tone": "busy", "text": f"A session starts within a minute to handle: {queued[0]['summary']}."}
    if live:
        names = ", ".join(r["name"] for r in live)
        steps = [r for r in live if r["step"] and r["total_steps"]]
        progress = ""
        if steps:
            r = min(steps, key=lambda r: r["step"] / r["total_steps"])
            progress = f" (slowest at step {r['step']:,} of {r['total_steps']:,})"
        return {"tone": "waiting", "text": f"Waiting for {names} to finish{progress}; the next session records them."}
    if pending:
        return {"tone": "idle", "text": f"{pending} suggestion(s) wait for a maintainer. Approve one to start it."}
    return {"tone": "idle", "text": "Nothing is queued. Propose an idea or wait for a new intake suggestion."}


def create_api(services: Any) -> FastAPI:
    api = FastAPI()
    engine: Engine = services.engine()

    @api.get("/state")
    def state() -> dict[str, Any]:
        with engine.begin() as conn:
            _expire_sessions(conn)
            settings = _settings(conn)
            ideas = _rows(conn, "SELECT * FROM ideas WHERE state NOT IN ('upstream') ORDER BY posted_at DESC")
            runs = _rows(conn, "SELECT * FROM runs ORDER BY submitted_at DESC NULLS LAST")
            activity = _rows(conn, f"SELECT * FROM activity ORDER BY at DESC LIMIT {RECENT_ACTIVITY}")
            sessions = _rows(
                conn,
                "SELECT id, started_at, ended_at, state, summary, cost_usd, turns, "
                f"length(transcript) AS transcript_bytes FROM sessions ORDER BY started_at DESC LIMIT {RECENT_SESSIONS}",
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
        pipeline = {
            stage.value: sum(1 for i in ideas if i["state"] == stage)
            for stage in (
                IdeaState.PENDING,
                IdeaState.APPROVED,
                IdeaState.PLANNED,
                IdeaState.RUNNING,
                IdeaState.RECORDED,
            )
        }
        return {
            "now": _now().isoformat(),
            "settings": settings,
            "pipeline": pipeline,
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

    @api.post("/ideas")
    def propose(proposal: Proposal) -> dict[str, int]:
        caller = _caller()
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

    def _decide(idea_id: int, approve: bool) -> dict[str, str]:
        caller = _caller()
        with engine.begin() as conn:
            idea = _rows(conn, "SELECT * FROM ideas WHERE id = :id FOR UPDATE", id=idea_id)
            if not idea:
                raise HTTPException(404, f"no idea {idea_id}")
            idea = idea[0]
            if idea["state"] != IdeaState.PENDING:
                raise HTTPException(409, f"idea {idea_id} is already {idea['state']}")
            new_state = IdeaState.APPROVED if approve else IdeaState.DISMISSED
            conn.execute(
                text(
                    "UPDATE ideas SET state = :s, decided_by = :by, decided_at = now(), updated_at = now() "
                    "WHERE id = :id"
                ),
                {"s": new_state.value, "by": caller, "id": idea_id},
            )
            label = idea["title"] or idea["body"].strip().splitlines()[0][:90]
            verb = "approved" if approve else "dismissed"
            _activity(conn, ActivityPost(kind="approval", idea_id=idea_id, title=f"{caller} {verb} “{label}”"))
            if approve:
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
        return {"state": new_state.value}

    @api.post("/ideas/{idea_id}/approve")
    def approve(idea_id: int) -> dict[str, str]:
        return _decide(idea_id, approve=True)

    @api.post("/ideas/{idea_id}/dismiss")
    def dismiss(idea_id: int) -> dict[str, str]:
        return _decide(idea_id, approve=False)

    @api.post("/settings")
    def update_settings(update: SettingsUpdate) -> dict[str, Any]:
        caller = _caller()
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
            return _settings(conn)

    @api.get("/sessions/{session_id}/transcript", response_class=PlainTextResponse)
    def transcript(session_id: str) -> str:
        with engine.connect() as conn:
            rows = _rows(conn, "SELECT transcript FROM sessions WHERE id = :id", id=session_id)
        if not rows:
            raise HTTPException(404, f"no session {session_id}")
        return rows[0]["transcript"] or "(no transcript yet: the session is still running)"

    # Watcher and agent endpoints.

    @api.post("/sync/intake")
    def sync_intake(sync: IntakeSync) -> dict[str, int]:
        added = 0
        with engine.begin() as conn:
            for c in sync.comments:
                state = IdeaState.UPSTREAM if c.approved_upstream else IdeaState.PENDING
                inserted = conn.execute(
                    text(
                        "INSERT INTO ideas (origin, comment_id, author, body, url, posted_at, state) "
                        "VALUES ('intake', :cid, :author, :body, :url, :posted_at, :state) "
                        "ON CONFLICT (comment_id) DO UPDATE SET body = EXCLUDED.body RETURNING (xmax = 0)"
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
        return {"new_pending": added}

    @api.post("/sync/runs")
    def sync_runs(sync: RunsSync) -> dict[str, int]:
        finished = 0
        with engine.begin() as conn:
            for r in sync.runs:
                previous = _rows(conn, "SELECT state FROM runs WHERE name = :n", n=r.name)
                experiment, _, run = r.name.removeprefix("belay-").partition("-")
                terminal = r.state in TERMINAL_RUN_STATES
                conn.execute(
                    text(UPSERT_RUN),
                    {**r.model_dump(), "rid": experiment, "run": run, "terminal": terminal},
                )
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

    @api.post("/tick")
    def tick(update: Tick) -> dict[str, int]:
        """The watcher's heartbeat: node use, storage, and re-offering approved work no session has taken."""
        offered = 0
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
            active = _rows(conn, "SELECT id FROM sessions WHERE state = 'running'")
            queued = _rows(conn, "SELECT ref FROM events WHERE session_id IS NULL")
            if not active and not queued:
                for idea in _rows(conn, "SELECT id, title, body FROM ideas WHERE state = 'approved'"):
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
        return {"work_pending": offered}

    @api.post("/dispatch/claim")
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

    @api.post("/sessions/{session_id}/finish")
    def finish(session_id: str, report: SessionFinish) -> dict[str, str]:
        with engine.begin() as conn:
            updated = conn.execute(
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
            ).rowcount
            if not updated:
                raise HTTPException(404, f"no session {session_id}")
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

    @api.post("/ideas/{idea_id}/report")
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

    @api.post("/activity")
    def post_activity(post: ActivityPost) -> dict[str, int]:
        with engine.begin() as conn:
            return {"id": _activity(conn, post)}

    return api
