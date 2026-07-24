# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The echo HTTP API: search the corpus and read/append the shared work log.

A FastAPI service that encapsulates the `context-search` and `work-log` skills behind one
OpenAPI-documented interface (see `/docs`). See infra/echo/README.md for how it is wired.
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime

import schema
import sqlalchemy
from fastapi import FastAPI, Header, HTTPException, Query
from fastembed import TextEmbedding
from google.cloud.sql.connector import Connector
from pydantic import BaseModel, Field

EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # must match the corpus's embedding space
SOURCES = ("github", "discord")
KINDS = ("issue", "pr", "comment", "message")

state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    instance, database, user = os.environ["CLOUDSQL_CONNECTION"], os.environ["PGDATABASE"], os.environ["PGUSER"]
    with Connector() as connector:
        engine = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=lambda: connector.connect(instance, "pg8000", user=user, enable_iam_auth=True, db=database),
            pool_size=5,
            pool_pre_ping=True,
        )
        state["engine"] = engine
        state["model"] = TextEmbedding(EMBED_MODEL)
        try:
            yield
        finally:
            engine.dispose()


app = FastAPI(
    title="echo",
    description="Search Marin's GitHub+Discord corpus and read/append the shared agent work log.",
    lifespan=lifespan,
)


class Hit(BaseModel):
    id: int
    source: str
    kind: str
    date: datetime | None
    author: str | None
    title: str | None
    url: str
    snippet: str
    distance: float | None = Field(None, description="Cosine distance; lower is closer. Null for grep.")


class Chunk(Hit):
    text: str | None
    ref: str | None
    parent: str | None


class LogSummary(BaseModel):
    id: int
    at: datetime
    author: str
    project: str
    title: str


class LogEntry(LogSummary):
    body: str | None = None


class LogCreate(BaseModel):
    project: str = Field(description="Stable slug for the thread of work.")
    title: str = Field(description="One-line summary.")
    body: str | None = Field(None, description="Short markdown; link evidence inline.")


def engine() -> sqlalchemy.Engine:
    return state["engine"]  # type: ignore[return-value]


def snippet(row: sqlalchemy.Row) -> str:
    return " ".join((row.text or "").split())[:200]


def hit(row: sqlalchemy.Row) -> Hit:
    return Hit(snippet=snippet(row), **{c: getattr(row, c) for c in Hit.model_fields if c != "snippet"})


def escape_like(pattern: str) -> str:
    """Escape LIKE wildcards so `pattern` matches as an exact substring under ILIKE."""
    return pattern.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def iap_caller(header: str | None) -> str:
    """The authenticated identity from IAP's `X-Goog-Authenticated-User-Email` header.

    The header is `accounts.google.com:<email>`; behind IAP it is always present and
    cannot be spoofed by the client (IAP strips inbound copies).
    """
    return (header or "").split(":")[-1] or "unknown"


def filtered(query, source: str | None, kind: str | None, since: datetime | None):
    if source:
        query = query.where(schema.chunks.c.source == source)
    if kind:
        query = query.where(schema.chunks.c.kind == kind)
    if since:
        query = query.where(schema.chunks.c.date >= since)
    return query


@app.get("/healthz")
def healthz() -> dict[str, str]:
    with engine().connect() as conn:
        conn.execute(sqlalchemy.text("SELECT 1"))
    return {"status": "ok"}


@app.get("/search", response_model=list[Hit])
def search(
    q: str = Query(description="Natural-language query."),
    source: str | None = Query(None, enum=list(SOURCES)),
    kind: str | None = Query(None, enum=list(KINDS)),
    since: datetime | None = Query(None, description="ISO date lower bound on chunk date."),
    limit: int = Query(10, ge=1, le=100),
) -> list[Hit]:
    """Semantic search over the corpus, ranked by cosine distance (lower is closer)."""
    model: TextEmbedding = state["model"]  # type: ignore[assignment]
    vector = [float(v) for v in next(iter(model.embed([q])))]
    distance = schema.chunks.c.embedding.cosine_distance(vector)
    query = filtered(sqlalchemy.select(schema.chunks, distance.label("distance")), source, kind, since)
    query = query.order_by(distance).limit(limit)
    with engine().connect() as conn:
        # Without iterative scan a selective filter can empty the HNSW candidate set.
        conn.execute(sqlalchemy.text("SET hnsw.iterative_scan = relaxed_order"))
        return [hit(r) for r in conn.execute(query)]


@app.get("/grep", response_model=list[Hit])
def grep(
    pattern: str = Query(description="Exact substring (SQL wildcards are escaped)."),
    source: str | None = Query(None, enum=list(SOURCES)),
    kind: str | None = Query(None, enum=list(KINDS)),
    limit: int = Query(20, ge=1, le=100),
) -> list[Hit]:
    """Case-insensitive substring scan, newest first — for identifiers and exact strings."""
    query = filtered(sqlalchemy.select(schema.chunks, sqlalchemy.literal(None).label("distance")), source, kind, None)
    query = query.where(schema.chunks.c.text.ilike(f"%{escape_like(pattern)}%"))
    query = query.order_by(schema.chunks.c.date.desc()).limit(limit)
    with engine().connect() as conn:
        return [hit(r) for r in conn.execute(query)]


@app.get("/chunks/{chunk_id}", response_model=Chunk)
def chunk(chunk_id: int) -> Chunk:
    with engine().connect() as conn:
        row = conn.execute(sqlalchemy.select(schema.chunks).where(schema.chunks.c.id == chunk_id)).first()
    if row is None:
        raise HTTPException(404, f"no chunk {chunk_id}")
    fields = {c: getattr(row, c) for c in Chunk.model_fields if c not in ("distance", "snippet")}
    return Chunk(distance=None, snippet=snippet(row), **fields)


@app.get("/work_log", response_model=list[LogSummary])
def work_log(
    days: int = Query(7, ge=1, description="Look back this many days."),
    project: str | None = Query(None, description="Filter to one project slug."),
    limit: int = Query(30, ge=1, le=200),
) -> list[LogSummary]:
    """Recent work-log entries, newest first — summaries; use the id endpoint for bodies."""
    columns = [getattr(schema.work_log.c, c) for c in LogSummary.model_fields]
    # make_interval takes positional (years, months, weeks, days, ...).
    cutoff = sqlalchemy.func.now() - sqlalchemy.func.make_interval(0, 0, 0, days)
    query = sqlalchemy.select(*columns).where(schema.work_log.c.at > cutoff)
    if project:
        query = query.where(schema.work_log.c.project == project)
    query = query.order_by(schema.work_log.c.at.desc()).limit(limit)
    with engine().connect() as conn:
        return [LogSummary(**r._mapping) for r in conn.execute(query)]


@app.get("/work_log/{entry_id}", response_model=LogEntry)
def work_log_entry(entry_id: int) -> LogEntry:
    with engine().connect() as conn:
        row = conn.execute(sqlalchemy.select(schema.work_log).where(schema.work_log.c.id == entry_id)).first()
    if row is None:
        raise HTTPException(404, f"no work_log entry {entry_id}")
    return LogEntry(**{c: getattr(row, c) for c in LogEntry.model_fields})


@app.post("/work_log", response_model=LogEntry, status_code=201)
def add_work_log(entry: LogCreate, x_goog_authenticated_user_email: str | None = Header(None)) -> LogEntry:
    """Append one entry, attributed to the IAP-authenticated caller."""
    statement = (
        schema.work_log.insert()
        .values(
            author=iap_caller(x_goog_authenticated_user_email), project=entry.project, title=entry.title, body=entry.body
        )
        .returning(schema.work_log)
    )
    with engine().begin() as conn:
        row = conn.execute(statement).first()
    return LogEntry(**{c: getattr(row, c) for c in LogEntry.model_fields})
