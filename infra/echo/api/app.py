# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Search Echo activity and wiki notes, and read or append the shared work log.

See ``infra/echo/README.md`` for endpoints and access requirements.
"""

import os
import re
from collections.abc import Iterable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Any

import dashboard as echo_dashboard
import hybrid_search
import schema
import search_config
import sqlalchemy
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
from fastembed import TextEmbedding
from google.cloud.sql.connector import Connector
from pydantic import BaseModel, Field, field_validator

EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # must match the corpus's embedding space
SOURCES = ("github", "discord")
KINDS = ("issue", "pr", "comment", "message")
MAX_WIKI_TAG_LENGTH = 50
WIKI_TAG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    instance, database, user = os.environ["CLOUDSQL_CONNECTION"], os.environ["PGDATABASE"], os.environ["PGUSER"]
    with Connector() as connector:
        app.state.engine = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=lambda: connector.connect(instance, "pg8000", user=user, enable_iam_auth=True, db=database),
            pool_size=5,
            pool_pre_ping=True,
        )
        app.state.model = TextEmbedding(EMBED_MODEL)
        try:
            yield
        finally:
            app.state.engine.dispose()


app = FastAPI(
    title="echo",
    description="Search Marin activity and wiki notes, and read/append the shared agent work log.",
    lifespan=lifespan,
)


def get_engine(request: Request) -> sqlalchemy.Engine:
    return request.app.state.engine


def get_model(request: Request) -> TextEmbedding:
    return request.app.state.model


Engine = Annotated[sqlalchemy.Engine, Depends(get_engine)]
Model = Annotated[TextEmbedding, Depends(get_model)]

# Data endpoints live under /api so they don't collide with the dashboard SPA's client-side
# routes (e.g. /wiki/123), which fall through to index.html at the root.
api = APIRouter(prefix="/api")


class Hit(BaseModel):
    id: int
    source: str
    kind: str
    date: datetime | None
    author: str | None
    title: str | None
    url: str
    snippet: str
    score: float = Field(description="Hybrid reciprocal-rank score; higher is better.")
    distance: float | None = Field(None, description="Cosine distance; lower is closer.")
    lexical_score: float | None = Field(None, description="PostgreSQL full-text rank; higher is better.")


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


class WikiSummary(BaseModel):
    id: int
    created_at: datetime
    updated_at: datetime
    author: str
    title: str
    use_when: str = Field(description="One-sentence hint describing when an agent should load this entry.")
    tags: list[str]
    snippet: str
    reference_count: int
    score: float = 0.0
    distance: float | None = None
    lexical_score: float | None = None


class WikiEntry(WikiSummary):
    body: str


class WikiCreate(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    use_when: str = Field(
        min_length=1,
        max_length=300,
        description="One sentence describing when this entry is useful.",
    )
    tags: list[str] = Field(default_factory=list, max_length=schema.MAX_WIKI_TAGS)
    body: str = Field(min_length=1)

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, tags: list[str]) -> list[str]:
        return normalize_wiki_tags(tags)


def normalize_wiki_tags(tags: Iterable[str]) -> list[str]:
    """Normalize tag slugs and reject values that cannot be filtered reliably."""
    normalized = list(dict.fromkeys(tag.strip().lower() for tag in tags))
    for tag in normalized:
        if not tag or len(tag) > MAX_WIKI_TAG_LENGTH or WIKI_TAG_PATTERN.fullmatch(tag) is None:
            raise ValueError(
                f"invalid wiki tag {tag!r}; use lowercase kebab-case up to {MAX_WIKI_TAG_LENGTH} characters"
            )
    return normalized


def snippet(row: sqlalchemy.Row) -> str:
    return " ".join((row.text or "").split())[:200]


def hit(row: sqlalchemy.Row) -> Hit:
    return Hit(snippet=snippet(row), **{c: getattr(row, c) for c in Hit.model_fields if c != "snippet"})


def wiki_snippet(row: sqlalchemy.Row) -> str:
    return " ".join(row.body.split())[:240]


def wiki_summary(row: sqlalchemy.Row) -> WikiSummary:
    fields = {field: getattr(row, field) for field in WikiSummary.model_fields if field != "snippet"}
    return WikiSummary(snippet=wiki_snippet(row), **fields)


def wiki_entry(row: sqlalchemy.Row) -> WikiEntry:
    fields = {field: getattr(row, field) for field in WikiEntry.model_fields if field not in ("snippet", "body")}
    return WikiEntry(snippet=wiki_snippet(row), body=row.body, **fields)


def wiki_score_columns() -> tuple[sqlalchemy.ColumnElement[Any], ...]:
    return (
        sqlalchemy.literal(0.0).label("score"),
        sqlalchemy.literal(None).label("distance"),
        sqlalchemy.literal(None).label("lexical_score"),
    )


def vector(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


def query_embedding(model: TextEmbedding, query: str) -> list[float]:
    return vector(next(iter(model.query_embed([query]))))


def passage_embedding(model: TextEmbedding, title: str, use_when: str, tags: list[str], body: str) -> list[float]:
    tag_text = f"\n\nTags: {', '.join(tags)}" if tags else ""
    return vector(next(iter(model.passage_embed([f"{title}\n\nUse when: {use_when}{tag_text}\n\n{body}"]))))


def escape_like(pattern: str) -> str:
    """Escape LIKE wildcards so `pattern` matches as an exact substring under ILIKE."""
    return pattern.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def iap_caller(header: str | None) -> str:
    """The authenticated identity from IAP's `X-Goog-Authenticated-User-Email` header.

    IAP injects this for every authenticated request, browser or programmatic — a CLI or
    agent calling with an ADC-minted ID token gets its own identity here (a user's email,
    or a service account's). The value is `accounts.google.com:<email>`; IAP strips any
    client-supplied copy, so it cannot be spoofed. Falls back to `unknown` only if a
    request somehow reaches the app without IAP (misconfiguration).
    """
    return (header or "").split(":")[-1] or "unknown"


def chunk_filter_clauses(source: str | None, kind: str | None, since: datetime | None) -> list[str]:
    clauses = []
    if source:
        clauses.append("c.source = :source")
    if kind:
        clauses.append("c.kind = :kind")
    if since:
        clauses.append("c.date >= :since")
    return clauses


def wiki_filter_clauses(tags: list[str]) -> list[str]:
    return ["w.tags @> CAST(:tags AS text[])"] if tags else []


@app.get("/healthz")
def healthz(engine: Engine) -> dict[str, str]:
    with engine.connect() as conn:
        conn.execute(sqlalchemy.text("SELECT 1"))
    return {"status": "ok"}


@api.get("/search", response_model=list[Hit])
def search(
    engine: Engine,
    model: Model,
    q: str = Query(description="Natural-language query."),
    source: str | None = Query(None, enum=list(SOURCES)),
    kind: str | None = Query(None, enum=list(KINDS)),
    since: datetime | None = Query(None, description="ISO date lower bound on chunk date."),
    limit: int = Query(search_config.DEFAULT_SEARCH_LIMIT, ge=1, le=search_config.MAX_SEARCH_LIMIT),
) -> list[Hit]:
    """Hybrid full-text and semantic search over GitHub and Discord activity."""
    query = q.strip()
    if not query:
        raise HTTPException(422, "q must not be blank")
    params = {
        "q": query,
        "embedding": str(query_embedding(model, query)),
        "candidate_limit": hybrid_search.candidate_limit(limit),
        "limit": limit,
        "source": source,
        "kind": kind,
        "since": since,
    }
    statement = hybrid_search.chunk_search_statement(chunk_filter_clauses(source, kind, since))
    with engine.connect() as conn:
        conn.execute(hybrid_search.HNSW_ITERATIVE_SCAN)
        return [hit(row) for row in conn.execute(statement, params)]


@api.get("/grep", response_model=list[Hit])
def grep(
    engine: Engine,
    pattern: str = Query(description="Exact substring (SQL wildcards are escaped)."),
    source: str | None = Query(None, enum=list(SOURCES)),
    kind: str | None = Query(None, enum=list(KINDS)),
    limit: int = Query(20, ge=1, le=search_config.MAX_SEARCH_LIMIT),
) -> list[Hit]:
    """Case-insensitive substring scan, newest first — for identifiers and exact strings."""
    query = sqlalchemy.select(
        schema.chunks,
        sqlalchemy.literal(0.0).label("score"),
        sqlalchemy.literal(None).label("distance"),
        sqlalchemy.literal(None).label("lexical_score"),
    )
    if source:
        query = query.where(schema.chunks.c.source == source)
    if kind:
        query = query.where(schema.chunks.c.kind == kind)
    query = query.where(schema.chunks.c.text.ilike(f"%{escape_like(pattern)}%"))
    query = query.order_by(schema.chunks.c.date.desc()).limit(limit)
    with engine.connect() as conn:
        return [hit(r) for r in conn.execute(query)]


@api.get("/chunks/{chunk_id}", response_model=Chunk)
def chunk(chunk_id: int, engine: Engine) -> Chunk:
    with engine.connect() as conn:
        row = conn.execute(sqlalchemy.select(schema.chunks).where(schema.chunks.c.id == chunk_id)).first()
    if row is None:
        raise HTTPException(404, f"no chunk {chunk_id}")
    fields = {
        field: getattr(row, field)
        for field in Chunk.model_fields
        if field not in ("score", "distance", "lexical_score", "snippet")
    }
    return Chunk(score=0.0, distance=None, lexical_score=None, snippet=snippet(row), **fields)


@api.get("/work_log", response_model=list[LogSummary])
def work_log(
    engine: Engine,
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
    with engine.connect() as conn:
        return [
            LogSummary(**{field: getattr(row, field) for field in LogSummary.model_fields})
            for row in conn.execute(query)
        ]


@api.get("/work_log/{entry_id}", response_model=LogEntry)
def work_log_entry(entry_id: int, engine: Engine) -> LogEntry:
    with engine.connect() as conn:
        row = conn.execute(sqlalchemy.select(schema.work_log).where(schema.work_log.c.id == entry_id)).first()
    if row is None:
        raise HTTPException(404, f"no work_log entry {entry_id}")
    return LogEntry(**{c: getattr(row, c) for c in LogEntry.model_fields})


@api.post("/work_log", response_model=LogEntry, status_code=201)
def add_work_log(
    entry: LogCreate, engine: Engine, x_goog_authenticated_user_email: str | None = Header(None)
) -> LogEntry:
    """Append one entry, attributed to the IAP-authenticated caller."""
    statement = (
        schema.work_log.insert()
        .values(
            author=iap_caller(x_goog_authenticated_user_email), project=entry.project, title=entry.title, body=entry.body
        )
        .returning(schema.work_log)
    )
    with engine.begin() as conn:
        row = conn.execute(statement).first()
    return LogEntry(**{c: getattr(row, c) for c in LogEntry.model_fields})


@api.get("/wiki/search", response_model=list[WikiSummary])
def search_wiki(
    engine: Engine,
    model: Model,
    q: str = Query("", description="Query text. Blank returns recently updated notes."),
    tag: list[str] | None = Query(None, description="Require this tag; repeat to require several tags."),
    limit: int = Query(search_config.DEFAULT_SEARCH_LIMIT, ge=1, le=search_config.MAX_SEARCH_LIMIT),
) -> list[WikiSummary]:
    query = q.strip()
    try:
        tags = normalize_wiki_tags(tag or [])
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    with engine.connect() as conn:
        if not query:
            statement = sqlalchemy.select(
                schema.wiki_entries,
                *wiki_score_columns(),
            )
            if tags:
                statement = statement.where(schema.wiki_entries.c.tags.contains(tags))
            statement = statement.order_by(schema.wiki_entries.c.updated_at.desc()).limit(limit)
            return [wiki_summary(row) for row in conn.execute(statement)]
        conn.execute(hybrid_search.HNSW_ITERATIVE_SCAN)
        params = {
            "q": query,
            "embedding": str(query_embedding(model, query)),
            "candidate_limit": hybrid_search.candidate_limit(limit),
            "limit": limit,
            "tags": tags,
        }
        statement = hybrid_search.wiki_search_statement(wiki_filter_clauses(tags))
        return [wiki_summary(row) for row in conn.execute(statement, params)]


@api.get("/wiki/{entry_id}", response_model=WikiEntry)
def get_wiki_entry(entry_id: int, engine: Engine) -> WikiEntry:
    statement = sqlalchemy.select(
        schema.wiki_entries,
        *wiki_score_columns(),
    ).where(schema.wiki_entries.c.id == entry_id)
    with engine.connect() as conn:
        row = conn.execute(statement).first()
    if row is None:
        raise HTTPException(404, f"no wiki entry {entry_id}")
    return wiki_entry(row)


def wiki_write_values(entry: WikiCreate, model: TextEmbedding) -> dict[str, Any]:
    """Validated, re-embedded column values shared by wiki create and update."""
    title, use_when, body = entry.title.strip(), entry.use_when.strip(), entry.body.strip()
    if not title or not use_when or not body:
        raise HTTPException(422, "title, use_when, and body must not be blank")
    return {
        "title": title,
        "use_when": use_when,
        "tags": entry.tags,
        "body": body,
        "embedding": passage_embedding(model, title, use_when, entry.tags, body),
    }


@api.post("/wiki", response_model=WikiEntry, status_code=201)
def add_wiki_entry(
    entry: WikiCreate,
    engine: Engine,
    model: Model,
    x_goog_authenticated_user_email: str | None = Header(None),
) -> WikiEntry:
    statement = (
        schema.wiki_entries.insert()
        .values(author=iap_caller(x_goog_authenticated_user_email), **wiki_write_values(entry, model))
        .returning(schema.wiki_entries, *wiki_score_columns())
    )
    with engine.begin() as conn:
        row = conn.execute(statement).first()
    return wiki_entry(row)


@api.put("/wiki/{entry_id}", response_model=WikiEntry)
def update_wiki_entry(entry_id: int, entry: WikiCreate, engine: Engine, model: Model) -> WikiEntry:
    """Replace an entry's text and re-embed it. The original author and creation time stand."""
    statement = (
        schema.wiki_entries.update()
        .where(schema.wiki_entries.c.id == entry_id)
        .values(updated_at=sqlalchemy.func.now(), **wiki_write_values(entry, model))
        .returning(schema.wiki_entries, *wiki_score_columns())
    )
    with engine.begin() as conn:
        row = conn.execute(statement).first()
    if row is None:
        raise HTTPException(404, f"no wiki entry {entry_id}")
    return wiki_entry(row)


@api.post("/wiki/{entry_id}/references", response_model=WikiEntry)
def reference_wiki_entry(entry_id: int, engine: Engine) -> WikiEntry:
    statement = (
        schema.wiki_entries.update()
        .where(schema.wiki_entries.c.id == entry_id)
        .values(reference_count=schema.wiki_entries.c.reference_count + 1)
        .returning(
            schema.wiki_entries,
            *wiki_score_columns(),
        )
    )
    with engine.begin() as conn:
        row = conn.execute(statement).first()
    if row is None:
        raise HTTPException(404, f"no wiki entry {entry_id}")
    return wiki_entry(row)


app.include_router(api)
echo_dashboard.install_dashboard(app, echo_dashboard.dashboard_dist())
