# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Search Echo context, collect result feedback, and read or append the shared work log.

See ``infra/echo/README.md`` for endpoints and access requirements.
"""

import logging
import os
import re
import time
from collections.abc import Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, Protocol
from urllib.parse import quote

import dashboard as echo_dashboard
import hybrid_search
import repository_identity
import reranking
import schema
import search_config
import search_feedback
import search_history
import sqlalchemy
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from google.cloud.sql.connector import Connector
from pydantic import BaseModel, Field, field_validator

SOURCES = ("github", "discord")
KINDS = ("issue", "pr", "comment", "message")
MAX_WIKI_TAG_LENGTH = 50
WIKI_TAG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
FILE_BOILERPLATE_PREFIXES = (
    "# copyright",
    "// copyright",
    "/* copyright",
    "# spdx-license-identifier:",
    "// spdx-license-identifier:",
    "* spdx-license-identifier:",
)
QUERY_LINE_TERM_PATTERN = re.compile(r"[a-z0-9]+")
QUERY_LINE_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "do",
        "does",
        "for",
        "how",
        "i",
        "in",
        "is",
        "of",
        "or",
        "the",
        "to",
        "we",
        "what",
        "when",
        "where",
        "which",
    }
)
FILE_REFERENCE_LIMIT = 3
SEARCH_EXECUTION_HEADER = search_config.SEARCH_EXECUTION_HEADER
SEARCH_HISTORY_PAGE_LIMIT = 500
MILLISECONDS_PER_SECOND = 1_000
POSTGRES_UNDEFINED_COLUMN = "42703"
POSTGRES_UNDEFINED_TABLE = "42P01"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EchoConfig:
    public_url: str
    service_revision: str | None = None


DEFAULT_CONFIG = EchoConfig(
    public_url=search_config.PUBLIC_URL,
)


def environment_config() -> EchoConfig:
    return EchoConfig(
        public_url=os.environ.get("ECHO_PUBLIC_URL", DEFAULT_CONFIG.public_url).rstrip("/"),
        service_revision=os.environ.get("K_REVISION"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    instance, database, user = os.environ["CLOUDSQL_CONNECTION"], os.environ["PGDATABASE"], os.environ["PGUSER"]
    with Connector() as connector:
        app.state.config = environment_config()
        app.state.engine = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=lambda: connector.connect(instance, "pg8000", user=user, enable_iam_auth=True, db=database),
            pool_size=5,
            pool_pre_ping=True,
        )
        app.state.model = TextEmbedding(search_config.EMBED_MODEL, threads=search_config.INFERENCE_THREADS)
        app.state.reranker = reranking.text_cross_encoder()
        try:
            yield
        finally:
            app.state.engine.dispose()


app = FastAPI(
    title="echo",
    description="Search Marin context, record result feedback, and read or append the shared agent work log.",
    lifespan=lifespan,
)
app.state.config = DEFAULT_CONFIG


def get_engine(request: Request) -> sqlalchemy.Engine:
    return request.app.state.engine


def get_model(request: Request) -> TextEmbedding:
    return request.app.state.model


def get_reranker(request: Request) -> TextCrossEncoder:
    return request.app.state.reranker


def get_config(request: Request) -> EchoConfig:
    return request.app.state.config


Engine = Annotated[sqlalchemy.Engine, Depends(get_engine)]
Model = Annotated[TextEmbedding, Depends(get_model)]
Reranker = Annotated[TextCrossEncoder, Depends(get_reranker)]
Config = Annotated[EchoConfig, Depends(get_config)]

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


class SearchReference(BaseModel):
    line: int
    text: str
    url: str


class SearchResult(BaseModel):
    key: str | None = Field(default=None, description="Execution-specific key for grading this returned row.")
    id: str
    domain: search_config.SearchDomain
    title: str
    subtitle: str
    url: str
    snippet: str
    score: float = Field(description="Final reciprocal-rank fusion score; higher is better.")
    distance: float | None = Field(None, description="Cosine distance; lower is closer.")
    lexical_score: float | None = Field(None, description="PostgreSQL full-text rank; higher is better.")
    references: list[SearchReference] = Field(
        default_factory=list,
        description="Ranked line-level references within a repository file.",
    )
    rerank_score: float | None = Field(default=None, exclude=True)


class RepositoryFileDetail(BaseModel):
    id: str
    title: str
    subtitle: str
    url: str
    text: str


class SearchDomainOption(BaseModel):
    value: search_config.SearchDomain
    label: str


class SearchConfiguration(BaseModel):
    domains: list[SearchDomainOption]
    default_domains: list[search_config.SearchDomain]
    display_sha_characters: int


class SearchFeedbackGrade(BaseModel):
    key: str = Field(min_length=1, max_length=search_feedback.MAX_RESULT_KEY_CHARACTERS)
    grade: int = Field(ge=search_feedback.MIN_GRADE, le=search_feedback.MAX_GRADE)

    @field_validator("key")
    @classmethod
    def validate_key(cls, key: str) -> str:
        return search_feedback.checked_result_key(key)


class SearchFeedbackCreate(BaseModel):
    query: str = Field(min_length=1, max_length=search_feedback.MAX_QUERY_CHARACTERS)
    grades: list[SearchFeedbackGrade] = Field(default_factory=list, max_length=search_feedback.MAX_GRADES)
    note: str = Field(min_length=1, max_length=search_feedback.MAX_NOTE_CHARACTERS)
    execution_id: int | None = Field(None, gt=0)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, query: str) -> str:
        query = query.strip()
        if not query:
            raise ValueError("query must not be blank")
        return query

    @field_validator("grades")
    @classmethod
    def reject_duplicate_results(cls, grades: list[SearchFeedbackGrade]) -> list[SearchFeedbackGrade]:
        keys = [grade.key for grade in grades]
        if len(keys) != len(set(keys)):
            raise ValueError("each result may be graded once per feedback submission")
        return grades

    @field_validator("note")
    @classmethod
    def normalize_note(cls, note: str) -> str:
        note = note.strip()
        if not note:
            raise ValueError("note must not be blank")
        return note


class SearchFeedbackEntry(SearchFeedbackCreate):
    id: int
    created_at: datetime
    author: str


class SearchFeedbackResultGrade(BaseModel):
    key: str | None
    source_id: str
    grade: int
    title: str
    url: str


class SearchFeedbackListEntry(BaseModel):
    id: int
    created_at: datetime
    author: str
    query: str
    note: str
    execution_id: int | None
    grades: list[SearchFeedbackResultGrade]


class SearchExecutionResultEntry(BaseModel):
    id: int
    rank: int
    result_id: str
    domain: search_config.SearchDomain
    title: str | None
    url: str
    snippet: str
    score: float
    distance: float | None
    lexical_score: float | None
    rerank_score: float | None


class SearchExecutionEntry(BaseModel):
    id: int
    created_at: datetime
    author: str | None
    query: str
    normalized_query: str
    mode: search_history.SearchMode
    domains: list[str]
    filters: dict[str, Any]
    requested_limit: int
    returned_count: int
    duration_ms: float
    repository_commit: str | None
    service_revision: str | None
    results: list[SearchExecutionResultEntry]


class RepositoryIndexStatus(BaseModel):
    repository: str
    branch: str
    status: Literal["empty", "building", "ready"]
    commit_sha: str | None
    completed_files: int | None
    total_files: int | None
    started_at: datetime | None
    indexed_at: datetime | None


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


@dataclass(frozen=True)
class RepositoryIndexState:
    target: search_config.RepositoryTarget
    commit_sha: str
    completed_files: int | None
    total_files: int | None
    started_at: datetime | None
    indexed_at: datetime | None

    @property
    def building(self) -> bool:
        return self.started_at is not None


@dataclass(frozen=True)
class SearchCandidate:
    result: SearchResult
    text: str


@dataclass(frozen=True)
class SearchStageTimings:
    query_embedding_ms: float
    database_setup_ms: float
    wiki_retrieval_ms: float | None
    file_retrieval_ms: float | None
    activity_retrieval_ms: float | None
    rerank_ms: float


@dataclass(frozen=True)
class FederatedSearchRun:
    results: list[SearchResult]
    timings: SearchStageTimings


@dataclass(frozen=True)
class FeedbackResultMetadata:
    title: str
    url: str


@dataclass(frozen=True)
class StoredFeedbackResult:
    id: int
    execution_id: int
    source_id: str
    domain: str
    title: str | None
    url: str


@dataclass(frozen=True)
class StoredFeedbackGrade:
    result: StoredFeedbackResult
    grade: int


class RerankerModel(Protocol):
    def rerank(self, query: str, documents: Iterable[str], batch_size: int) -> Iterable[float]: ...


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
    return " ".join(row.body.split())[: search_config.FEDERATED_SUMMARY_CHARACTERS]


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


def hybrid_search_params(model: TextEmbedding, query: str, limit: int) -> dict[str, object]:
    weights = search_config.search_weights(query)
    return {
        "q": query,
        "embedding": str(query_embedding(model, search_config.expanded_query(query))),
        "candidate_limit": search_config.candidate_limit(limit),
        "limit": limit,
        "semantic_weight": weights.semantic,
        "lexical_weight": weights.lexical,
    }


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


def activity_domain_clause(domains: Iterable[str]) -> str:
    clauses = []
    selected = set(domains)
    if "discord" in selected:
        clauses.append("c.source = 'discord'")
    if "pr" in selected:
        clauses.append("(c.source = 'github' AND (c.kind = 'pr' OR (c.kind = 'comment' AND c.url LIKE '%/pull/%')))")
    if "issue" in selected:
        clauses.append(
            "(c.source = 'github' AND (c.kind = 'issue' OR (c.kind = 'comment' AND c.url LIKE '%/issues/%')))"
        )
    return f"({' OR '.join(clauses)})"


def activity_domain(source: str, kind: str, url: str) -> Literal["discord", "pr", "issue"]:
    if source == "discord":
        return "discord"
    if kind == "pr" or "/pull/" in url:
        return "pr"
    return "issue"


def activity_search_result(row: sqlalchemy.Row) -> SearchResult:
    domain = activity_domain(row.source, row.kind, row.url)
    title = row.title or snippet(row) or row.url
    details = [domain]
    if row.author:
        details.append(row.author)
    if row.date:
        details.append(row.date.isoformat())
    return SearchResult(
        id=f"{domain}:{row.id}",
        domain=domain,
        title=title,
        subtitle=" · ".join(details),
        url=row.url,
        snippet=snippet(row),
        score=row.score,
        distance=row.distance,
        lexical_score=row.lexical_score,
    )


def recorded_search_result(result: SearchResult) -> search_history.SearchResultRecord:
    return search_history.SearchResultRecord(
        result_id=result.id,
        domain=result.domain,
        title=result.title,
        url=result.url,
        snippet=result.snippet,
        score=result.score,
        distance=result.distance,
        lexical_score=result.lexical_score,
        rerank_score=result.rerank_score,
    )


def recorded_hit(result: Hit) -> search_history.SearchResultRecord:
    domain = activity_domain(result.source, result.kind, result.url)
    return search_history.SearchResultRecord(
        result_id=f"{domain}:{result.id}",
        domain=domain,
        title=result.title,
        url=result.url,
        snippet=result.snippet,
        score=result.score,
        distance=result.distance,
        lexical_score=result.lexical_score,
        rerank_score=None,
    )


def database_error_code(error: sqlalchemy.exc.DBAPIError) -> str | None:
    code = getattr(error.orig, "sqlstate", None) or getattr(error.orig, "pgcode", None)
    if isinstance(code, str):
        return code
    args = getattr(error.orig, "args", ())
    if isinstance(args, tuple) and args and isinstance(args[0], dict):
        value = args[0].get("C")
        return value if isinstance(value, str) else None
    return None


def record_live_search(
    engine: sqlalchemy.Engine, record: search_history.SearchExecutionRecord
) -> search_history.StoredSearchExecution | None:
    """Persist a search, returning ``None`` only while its additive schema is unavailable."""
    try:
        with engine.begin() as conn:
            return search_history.insert_execution(conn, record)
    except sqlalchemy.exc.DBAPIError as error:
        if database_error_code(error) not in {POSTGRES_UNDEFINED_COLUMN, POSTGRES_UNDEFINED_TABLE}:
            raise
        logger.warning("Search history schema is not available; apply pending Echo migrations", exc_info=True)
        return None


def attach_search_execution(
    response: Response,
    engine: sqlalchemy.Engine,
    record: search_history.SearchExecutionRecord,
) -> search_history.StoredSearchExecution | None:
    execution = record_live_search(engine, record)
    if execution is not None:
        response.headers[SEARCH_EXECUTION_HEADER] = str(execution.id)
    return execution


def wiki_search_result(row: sqlalchemy.Row, config: EchoConfig) -> SearchResult:
    return SearchResult(
        id=f"wiki:{row.id}",
        domain="wiki",
        title=row.title,
        subtitle=row.use_when,
        url=f"{config.public_url}/wiki/{row.id}",
        snippet=wiki_snippet(row),
        score=row.score,
        distance=row.distance,
        lexical_score=row.lexical_score,
    )


def repository_index_states(conn: sqlalchemy.Connection) -> dict[search_config.RepositoryTarget, RepositoryIndexState]:
    join_condition = sqlalchemy.and_(
        schema.repository_index_state.c.repository == schema.repository_index_builds.c.repository,
        schema.repository_index_state.c.branch == schema.repository_index_builds.c.branch,
    )
    repository = sqlalchemy.func.coalesce(
        schema.repository_index_builds.c.repository,
        schema.repository_index_state.c.repository,
    )
    branch = sqlalchemy.func.coalesce(
        schema.repository_index_builds.c.branch,
        schema.repository_index_state.c.branch,
    )
    rows = conn.execute(
        sqlalchemy.select(
            repository.label("repository"),
            branch.label("branch"),
            sqlalchemy.func.coalesce(
                schema.repository_index_builds.c.commit_sha,
                schema.repository_index_state.c.commit_sha,
            ).label("commit_sha"),
            schema.repository_index_builds.c.completed_files,
            schema.repository_index_builds.c.total_files,
            schema.repository_index_builds.c.started_at,
            schema.repository_index_state.c.indexed_at,
        ).select_from(schema.repository_index_state.outerjoin(schema.repository_index_builds, join_condition, full=True))
    )
    targets = {(target.repository, target.branch): target for target in search_config.REPOSITORY_TARGETS}
    states = {}
    for row in rows:
        target = targets.get((row.repository, row.branch))
        if target is not None:
            states[target] = RepositoryIndexState(
                target,
                row.commit_sha,
                row.completed_files,
                row.total_files,
                row.started_at,
                row.indexed_at,
            )
    return states


def query_line_terms(query: str) -> frozenset[str]:
    """Return content terms and light stems for ranking source lines."""
    terms = {term for term in QUERY_LINE_TERM_PATTERN.findall(query.casefold()) if term not in QUERY_LINE_STOP_WORDS}
    stems = set()
    for term in terms:
        if len(term) > 5 and term.endswith("ing"):
            stems.add(term[:-3])
        elif len(term) > 4 and term.endswith("ed"):
            stems.add(term[:-2])
        elif len(term) > 4 and term.endswith("es"):
            stems.add(term[:-2])
        elif len(term) > 3 and term.endswith("s"):
            stems.add(term[:-1])
    return frozenset(terms | stems)


def representative_file_lines(text: str, query: str, start_line: int) -> list[tuple[int, str]]:
    """Choose up to three non-boilerplate source lines that best explain a file match."""
    candidates = []
    lowered_query = query.casefold()
    terms = query_line_terms(query)
    lines = text.splitlines()
    for offset, line in enumerate(lines):
        stripped = " ".join(line.split())
        lowered_line = stripped.casefold()
        if not stripped or lowered_line.startswith(FILE_BOILERPLATE_PREFIXES):
            continue
        matched_terms = {term for term in terms if term in lowered_line}
        score = len(matched_terms) * 10 + sum(lowered_line.count(term) for term in matched_terms)
        if lowered_query and lowered_query in lowered_line:
            score += 100
        candidates.append((score, offset, stripped))
    if not candidates:
        return [(start_line, "")]
    if max(score for score, _, _ in candidates) == 0:
        selected = candidates[:FILE_REFERENCE_LIMIT]
    else:
        selected = sorted(candidates, key=lambda candidate: (-candidate[0], candidate[1]))[:FILE_REFERENCE_LIMIT]
    return [(start_line + offset, line) for _, offset, line in selected]


def repository_freshness(state: RepositoryIndexState) -> str:
    if state.building:
        return f"building {state.completed_files}/{state.total_files} since {state.started_at.isoformat()}"
    assert state.indexed_at is not None
    return f"indexed {state.indexed_at.isoformat()}"


def repository_blob_url(target: search_config.RepositoryTarget, commit_sha: str, path: str) -> str:
    return f"https://github.com/{target.repository}/blob/{commit_sha}/{quote(path, safe='/')}"


def default_feedback_result_metadata(result_id: str, config: EchoConfig) -> FeedbackResultMetadata:
    domain, _, value = result_id.partition(":")
    if domain == "wiki":
        return FeedbackResultMetadata(title=f"Wiki note #{value}", url=f"{config.public_url}/wiki/{value}")
    if domain == "file":
        reference = repository_identity.stored_repository_file_reference(result_id)
        return FeedbackResultMetadata(
            title=PurePosixPath(reference.path).name,
            url=repository_blob_url(reference.target, reference.target.branch, reference.path),
        )
    labels = {"discord": "Discord message", "pr": "Pull request result", "issue": "Issue result"}
    return FeedbackResultMetadata(
        title=f"{labels[domain]} #{value}",
        url=f"{config.public_url}/chunk/{value}",
    )


def stored_feedback_results(conn: sqlalchemy.Connection, search_result_ids: set[int]) -> dict[int, StoredFeedbackResult]:
    if not search_result_ids:
        return {}
    rows = conn.execute(
        sqlalchemy.select(
            schema.search_execution_results.c.id,
            schema.search_execution_results.c.execution_id,
            schema.search_execution_results.c.result_id,
            schema.search_execution_results.c.domain,
            schema.search_execution_results.c.title,
            schema.search_execution_results.c.url,
        ).where(schema.search_execution_results.c.id.in_(search_result_ids))
    )
    return {
        row.id: StoredFeedbackResult(
            id=row.id,
            execution_id=row.execution_id,
            source_id=row.result_id,
            domain=row.domain,
            title=row.title,
            url=row.url,
        )
        for row in rows
    }


def resolve_feedback_grades(
    conn: sqlalchemy.Connection,
    grades: list[SearchFeedbackGrade],
    execution_id: int | None,
) -> tuple[tuple[StoredFeedbackGrade, ...], int | None]:
    parsed = [(grade, *search_feedback.result_key_parts(grade.key)) for grade in grades]
    stored_results = stored_feedback_results(conn, {result_id for _, _, result_id in parsed})
    invalid_keys = [
        grade.key
        for grade, domain, result_id in parsed
        if result_id not in stored_results or stored_results[result_id].domain != domain
    ]
    if invalid_keys:
        raise HTTPException(422, "unknown search result keys: " + ", ".join(sorted(invalid_keys)))

    resolved = tuple(StoredFeedbackGrade(stored_results[result_id], grade.grade) for grade, _, result_id in parsed)
    graded_execution_ids = {grade.result.execution_id for grade in resolved}
    if len(graded_execution_ids) > 1:
        raise HTTPException(422, "graded search results must come from one execution")
    if not graded_execution_ids:
        return resolved, execution_id

    graded_execution_id = graded_execution_ids.pop()
    if execution_id is not None and execution_id != graded_execution_id:
        raise HTTPException(422, "graded search results were not returned by the linked search execution")
    return resolved, graded_execution_id


def validate_feedback_execution(
    conn: sqlalchemy.Connection,
    execution_id: int | None,
    author: str,
    query: str,
) -> None:
    if execution_id is None:
        return
    execution = conn.execute(
        sqlalchemy.select(schema.search_executions.c.author, schema.search_executions.c.query).where(
            schema.search_executions.c.id == execution_id
        )
    ).first()
    if execution is None:
        raise HTTPException(422, f"no search execution {execution_id}")
    if execution.author != author or execution.query != query:
        raise HTTPException(422, "execution_id must identify this caller's exact query")


def current_feedback_result_metadata(
    conn: sqlalchemy.Connection, result_ids: set[str], config: EchoConfig
) -> dict[str, FeedbackResultMetadata]:
    metadata = {result_id: default_feedback_result_metadata(result_id, config) for result_id in result_ids}
    wiki_ids = [
        int(value)
        for result_id in result_ids
        for domain, _, value in [result_id.partition(":")]
        if domain == "wiki" and int(value) <= search_feedback.MAX_NUMERIC_RESULT_ID
    ]
    if wiki_ids:
        for row in conn.execute(
            sqlalchemy.select(schema.wiki_entries.c.id, schema.wiki_entries.c.title).where(
                schema.wiki_entries.c.id.in_(wiki_ids)
            )
        ):
            result_id = f"wiki:{row.id}"
            metadata[result_id] = FeedbackResultMetadata(row.title, f"{config.public_url}/wiki/{row.id}")

    activity_result_ids = {
        int(value): result_id
        for result_id in result_ids
        for domain, _, value in [result_id.partition(":")]
        if domain in {"discord", "pr", "issue"} and int(value) <= search_feedback.MAX_NUMERIC_RESULT_ID
    }
    if activity_result_ids:
        for row in conn.execute(
            sqlalchemy.select(schema.chunks.c.id, schema.chunks.c.title, schema.chunks.c.url, schema.chunks.c.text)
            .where(schema.chunks.c.id.in_(activity_result_ids))
            .order_by(schema.chunks.c.id)
        ):
            result_id = activity_result_ids[row.id]
            fallback = metadata[result_id]
            title = row.title or " ".join((row.text or "").split())[: search_config.FEDERATED_SUMMARY_CHARACTERS]
            metadata[result_id] = FeedbackResultMetadata(title or fallback.title, row.url)
    return metadata


def repository_file_search_result(
    row: sqlalchemy.Row,
    state: RepositoryIndexState,
    query: str,
) -> SearchResult:
    reference = repository_identity.repository_file_reference(state.target, row.path)
    source_url = repository_blob_url(state.target, state.commit_sha, row.path)
    lines = representative_file_lines(row.text, query, row.start_line)
    references = [
        SearchReference(
            line=line,
            text=text[: search_config.FEDERATED_SUMMARY_CHARACTERS],
            url=f"{source_url}#L{line}",
        )
        for line, text in lines
    ]
    return SearchResult(
        id=reference.result_id,
        domain="file",
        title=row.title,
        subtitle=(
            f"{state.target.repository} · {row.path}:{references[0].line} · "
            f"{state.target.branch}@{state.commit_sha[: search_config.DISPLAY_SHA_CHARACTERS]} · "
            f"{repository_freshness(state)}"
        ),
        url=references[0].url,
        snippet=" · ".join(
            f"{state.target.repository}:{row.path}:{line_reference.line} {line_reference.text}"
            for line_reference in references
        ),
        score=row.score,
        distance=row.distance,
        lexical_score=row.lexical_score,
        references=references,
    )


def query_oriented_result(result: SearchResult, query: str) -> SearchResult:
    """Apply a small source-quality prior to prose queries over repository files."""
    if result.domain != "file" or search_config.is_identifier_query(query):
        return result
    try:
        path = repository_identity.parse_repository_file_id(result.id).path
    except ValueError:
        path = result.id.removeprefix("file:")
    filename = PurePosixPath(path).name
    score = result.score
    if path.lower().endswith(search_config.PROSE_FILE_SUFFIXES):
        score *= search_config.QUERY_PROSE_FILE_SCORE_MULTIPLIER
    if "tests" in PurePosixPath(path).parts or filename == "conftest.py" or filename.startswith("test_"):
        score *= search_config.QUERY_TEST_FILE_SCORE_MULTIPLIER
    return result.model_copy(update={"score": score})


def rerank_candidates(
    candidates: Iterable[SearchCandidate],
    query: str,
    reranker: RerankerModel,
    limit: int,
) -> list[SearchResult]:
    """Fuse the existing hybrid order with bounded cross-encoder ranks."""
    base = [SearchCandidate(query_oriented_result(candidate.result, query), candidate.text) for candidate in candidates]
    base.sort(key=lambda candidate: (-candidate.result.score, candidate.result.domain, candidate.result.id))
    selected = base[: search_config.RERANK_MAX_CANDIDATES]
    if not selected:
        return []
    scores = list(
        reranker.rerank(
            search_config.expanded_query(query),
            [candidate.text for candidate in selected],
            batch_size=search_config.RERANK_BATCH_SIZE,
        )
    )
    if len(scores) != len(selected):
        raise ValueError(f"reranker returned {len(scores)} scores for {len(selected)} candidates")
    model_order = sorted(range(len(selected)), key=lambda index: (-scores[index], index))
    model_ranks = {index: rank for rank, index in enumerate(model_order, start=1)}
    reranked = []
    for base_rank, candidate in enumerate(selected, start=1):
        score = search_config.RERANK_BASE_WEIGHT / (
            search_config.RRF_K + base_rank
        ) + search_config.RERANK_MODEL_WEIGHT / (search_config.RRF_K + model_ranks[base_rank - 1])
        reranked.append(
            candidate.result.model_copy(
                update={
                    "score": score,
                    "rerank_score": float(scores[base_rank - 1]),
                }
            )
        )
    reranked = [
        result
        for result in reranked
        if result.rerank_score is not None
        and result.rerank_score >= search_config.MIN_RERANK_SCORE_BY_DOMAIN[result.domain]
    ]
    reranked.sort(key=lambda result: (-result.score, result.domain, result.id))
    return reranked[:limit]


def wiki_candidates(
    conn: sqlalchemy.Connection,
    params: dict[str, object],
    config: EchoConfig,
) -> list[SearchCandidate]:
    return [
        SearchCandidate(
            wiki_search_result(row, config),
            f"{row.title}\n{row.use_when}\n\n{row.body}",
        )
        for row in conn.execute(hybrid_search.wiki_search_statement(), params)
    ]


def repository_file_candidates(
    conn: sqlalchemy.Connection,
    params: dict[str, object],
    retrieval_limit: int,
    query: str,
    targets: tuple[search_config.RepositoryTarget, ...],
) -> list[SearchCandidate]:
    states = repository_index_states(conn)
    if not states:
        return []
    candidates = []
    for target in targets:
        state = states.get(target)
        if state is None:
            continue
        # Rank inside each repository before merging. A large fork must not consume the
        # bounded HNSW and lexical pools before smaller repositories get a candidate.
        file_params = {
            **params,
            "candidate_limit": (
                search_config.candidate_limit(retrieval_limit) * search_config.FILE_CHUNK_CANDIDATE_MULTIPLIER
            ),
            "exact": escape_like(query),
            "substring": f"%{escape_like(query)}%",
            "repository_0": target.repository,
            "branch_0": target.branch,
        }
        for row in conn.execute(hybrid_search.repository_file_search_statement((target,)), file_params):
            result = repository_file_search_result(row, state, query)
            candidates.append(
                SearchCandidate(
                    result,
                    f"{result.snippet}\n\n{row.path}\n{row.title}\n\n{row.text}",
                )
            )
    return candidates


def activity_candidates(
    conn: sqlalchemy.Connection,
    params: dict[str, object],
    domains: list[search_config.SearchDomain],
) -> list[SearchCandidate]:
    statement = hybrid_search.chunk_search_statement([activity_domain_clause(domains)])
    return [
        SearchCandidate(
            activity_search_result(row),
            f"{row.title or ''}\n\n{row.text or ''}",
        )
        for row in conn.execute(statement, params)
    ]


def indexed_file_text(rows: Iterable[sqlalchemy.Row]) -> str:
    """Reconstruct indexed source while discarding repeated overlap lines."""
    lines: list[str] = []
    for row in rows:
        chunk_lines = row.text.split("\n")
        offset = row.start_line - 1
        if offset > len(lines):
            raise ValueError(f"repository file chunk starts at line {row.start_line} after line {len(lines)}")
        overlap = len(lines) - offset
        if overlap:
            assert lines[offset:] == chunk_lines[:overlap], "repository file chunks disagree in their overlap"
        lines.extend(chunk_lines[overlap:])
    return "\n".join(lines)


def wiki_filter_clauses(tags: list[str]) -> list[str]:
    return ["w.tags @> CAST(:tags AS text[])"] if tags else []


@app.get("/healthz")
def healthz(engine: Engine) -> dict[str, str]:
    with engine.connect() as conn:
        conn.execute(sqlalchemy.text("SELECT 1"))
    return {"status": "ok"}


@api.get("/search-configuration", response_model=SearchConfiguration)
def search_configuration() -> SearchConfiguration:
    return SearchConfiguration(
        domains=[
            SearchDomainOption(value=domain, label=search_config.SEARCH_DOMAIN_LABELS[domain])
            for domain in search_config.SEARCH_DOMAINS
        ],
        default_domains=list(search_config.DEFAULT_SEARCH_DOMAINS),
        display_sha_characters=search_config.DISPLAY_SHA_CHARACTERS,
    )


@api.get("/repository-index", response_model=list[RepositoryIndexStatus])
def repository_index_status(engine: Engine) -> list[RepositoryIndexStatus]:
    """Return freshness or build progress for every configured repository."""
    with engine.connect() as conn:
        states = repository_index_states(conn)
    statuses = []
    for target in search_config.REPOSITORY_TARGETS:
        state = states.get(target)
        statuses.append(
            RepositoryIndexStatus(
                repository=target.repository,
                branch=target.branch,
                status="empty" if state is None else "building" if state.building else "ready",
                commit_sha=state.commit_sha if state is not None else None,
                completed_files=state.completed_files if state is not None else None,
                total_files=state.total_files if state is not None else None,
                started_at=state.started_at if state is not None else None,
                indexed_at=state.indexed_at if state is not None else None,
            )
        )
    return statuses


@api.get("/search", response_model=list[Hit])
def search(
    engine: Engine,
    model: Model,
    config: Config,
    response: Response,
    q: str = Query(description="Natural-language query."),
    source: str | None = Query(None, enum=list(SOURCES)),
    kind: str | None = Query(None, enum=list(KINDS)),
    since: datetime | None = Query(None, description="ISO date lower bound on chunk date."),
    limit: int = Query(search_config.DEFAULT_SEARCH_LIMIT, ge=1, le=search_config.MAX_SEARCH_LIMIT),
    x_goog_authenticated_user_email: str | None = Header(None),
) -> list[Hit]:
    """Hybrid full-text and semantic search over GitHub and Discord activity."""
    query = q.strip()
    if not query:
        raise HTTPException(422, "q must not be blank")
    started_at = time.perf_counter()
    params = {
        **hybrid_search_params(model, query, limit),
        "source": source,
        "kind": kind,
        "since": since,
    }
    statement = hybrid_search.chunk_search_statement(chunk_filter_clauses(source, kind, since))
    with engine.connect() as conn:
        conn.execute(hybrid_search.HNSW_ITERATIVE_SCAN)
        results = [hit(row) for row in conn.execute(statement, params)]
    attach_search_execution(
        response,
        engine,
        search_history.SearchExecutionRecord(
            author=iap_caller(x_goog_authenticated_user_email),
            query=query,
            mode="activity",
            domains=(),
            filters={
                "source": source,
                "kind": kind,
                "since": since.isoformat() if since is not None else None,
            },
            requested_limit=limit,
            returned_count=len(results),
            duration_ms=(time.perf_counter() - started_at) * MILLISECONDS_PER_SECOND,
            service_revision=config.service_revision,
            results=tuple(recorded_hit(result) for result in results),
        ),
    )
    return results


def federated_search(
    engine: sqlalchemy.Engine,
    model: TextEmbedding,
    reranker: RerankerModel,
    config: EchoConfig,
    query: str,
    domains: list[search_config.SearchDomain],
    repository_targets: tuple[search_config.RepositoryTarget, ...],
    limit: int,
) -> FederatedSearchRun:
    """Search wiki, repository file, and activity domains and merge their hybrid ranks."""
    retrieval_limit = max(limit, search_config.RERANK_MIN_RESULTS_PER_DOMAIN)
    stage_started_at = time.perf_counter()
    params = hybrid_search_params(model, query, retrieval_limit)
    query_embedding_ms = (time.perf_counter() - stage_started_at) * MILLISECONDS_PER_SECOND
    candidates: list[SearchCandidate] = []
    wiki_retrieval_ms = None
    file_retrieval_ms = None
    activity_retrieval_ms = None
    stage_started_at = time.perf_counter()
    with engine.connect() as conn:
        conn.execute(hybrid_search.HNSW_ITERATIVE_SCAN)
        database_setup_ms = (time.perf_counter() - stage_started_at) * MILLISECONDS_PER_SECOND
        if "wiki" in domains:
            stage_started_at = time.perf_counter()
            candidates.extend(wiki_candidates(conn, params, config))
            wiki_retrieval_ms = (time.perf_counter() - stage_started_at) * MILLISECONDS_PER_SECOND
        if "file" in domains:
            stage_started_at = time.perf_counter()
            candidates.extend(repository_file_candidates(conn, params, retrieval_limit, query, repository_targets))
            file_retrieval_ms = (time.perf_counter() - stage_started_at) * MILLISECONDS_PER_SECOND
        activity_domains = [candidate for candidate in domains if candidate in ("discord", "pr", "issue")]
        if activity_domains:
            stage_started_at = time.perf_counter()
            candidates.extend(activity_candidates(conn, params, activity_domains))
            activity_retrieval_ms = (time.perf_counter() - stage_started_at) * MILLISECONDS_PER_SECOND
    stage_started_at = time.perf_counter()
    results = rerank_candidates(candidates, query, reranker, limit)
    rerank_ms = (time.perf_counter() - stage_started_at) * MILLISECONDS_PER_SECOND
    return FederatedSearchRun(
        results,
        SearchStageTimings(
            query_embedding_ms,
            database_setup_ms,
            wiki_retrieval_ms,
            file_retrieval_ms,
            activity_retrieval_ms,
            rerank_ms,
        ),
    )


def server_timing_header(timings: SearchStageTimings, history_ms: float, total_ms: float) -> str:
    metrics = [
        ("query_embedding", timings.query_embedding_ms),
        ("database_setup", timings.database_setup_ms),
        ("wiki_retrieval", timings.wiki_retrieval_ms),
        ("file_retrieval", timings.file_retrieval_ms),
        ("activity_retrieval", timings.activity_retrieval_ms),
        ("rerank", timings.rerank_ms),
        ("history", history_ms),
        ("total", total_ms),
    ]
    return ", ".join(f"{name};dur={duration:.1f}" for name, duration in metrics if duration is not None)


@api.get("/federated-search", response_model=list[SearchResult])
def federated_search_endpoint(
    engine: Engine,
    model: Model,
    reranker: Reranker,
    config: Config,
    response: Response,
    q: str = Query(description="Natural-language query."),
    domain: list[search_config.SearchDomain] | None = Query(
        None, description="Search this domain; repeat to select several."
    ),
    repository: str | None = Query(
        None,
        description="Repository-file scope: one configured owner/repository or all. Omission selects Marin.",
    ),
    limit: int = Query(search_config.DEFAULT_SEARCH_LIMIT, ge=1, le=search_config.MAX_SEARCH_LIMIT),
    x_goog_authenticated_user_email: str | None = Header(None),
) -> list[SearchResult]:
    query = q.strip()
    if not query:
        raise HTTPException(422, "q must not be blank")
    domains = list(dict.fromkeys(domain or search_config.DEFAULT_SEARCH_DOMAINS))
    repository_scope = search_config.LEGACY_REPOSITORY_TARGET.repository if repository is None else repository
    if repository_scope == search_config.ALL_REPOSITORIES:
        repository_targets = search_config.REPOSITORY_TARGETS
    else:
        try:
            repository_targets = (repository_identity.configured_repository_target(repository_scope),)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
    started_at = time.perf_counter()
    search_run = federated_search(engine, model, reranker, config, query, domains, repository_targets, limit)
    results = search_run.results
    search_duration_ms = (time.perf_counter() - started_at) * MILLISECONDS_PER_SECOND
    history_started_at = time.perf_counter()
    execution = attach_search_execution(
        response,
        engine,
        search_history.SearchExecutionRecord(
            author=iap_caller(x_goog_authenticated_user_email),
            query=query,
            mode="federated",
            domains=tuple(domains),
            filters={"repository": repository_scope} if "file" in domains else {},
            requested_limit=limit,
            returned_count=len(results),
            duration_ms=search_duration_ms,
            # A federated file result set may span several commits. Per-result pinned URLs
            # carry the exact provenance; the legacy scalar remains nullable.
            repository_commit=None,
            service_revision=config.service_revision,
            results=tuple(recorded_search_result(result) for result in results),
        ),
    )
    history_ms = (time.perf_counter() - history_started_at) * MILLISECONDS_PER_SECOND
    total_ms = (time.perf_counter() - started_at) * MILLISECONDS_PER_SECOND
    response.headers[search_config.SERVER_TIMING_HEADER] = server_timing_header(search_run.timings, history_ms, total_ms)
    logger.info("Federated search timing: %s", response.headers[search_config.SERVER_TIMING_HEADER])
    if execution is None:
        return results
    assert len(execution.search_result_ids) == len(results)
    return [
        result.model_copy(update={"key": f"{result.domain}:{result_id}"})
        for result, result_id in zip(results, execution.search_result_ids, strict=True)
    ]


@api.get("/grep", response_model=list[Hit])
def grep(
    engine: Engine,
    config: Config,
    response: Response,
    pattern: str = Query(description="Exact substring (SQL wildcards are escaped)."),
    source: str | None = Query(None, enum=list(SOURCES)),
    kind: str | None = Query(None, enum=list(KINDS)),
    limit: int = Query(20, ge=1, le=search_config.MAX_SEARCH_LIMIT),
    x_goog_authenticated_user_email: str | None = Header(None),
) -> list[Hit]:
    """Case-insensitive substring scan, newest first — for identifiers and exact strings."""
    query_text = pattern.strip()
    if not query_text:
        raise HTTPException(422, "pattern must not be blank")
    started_at = time.perf_counter()
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
    query = query.where(schema.chunks.c.text.ilike(f"%{escape_like(query_text)}%"))
    query = query.order_by(schema.chunks.c.date.desc()).limit(limit)
    with engine.connect() as conn:
        results = [hit(r) for r in conn.execute(query)]
    attach_search_execution(
        response,
        engine,
        search_history.SearchExecutionRecord(
            author=iap_caller(x_goog_authenticated_user_email),
            query=query_text,
            mode="grep",
            domains=(),
            filters={"source": source, "kind": kind},
            requested_limit=limit,
            returned_count=len(results),
            duration_ms=(time.perf_counter() - started_at) * MILLISECONDS_PER_SECOND,
            service_revision=config.service_revision,
            results=tuple(recorded_hit(result) for result in results),
        ),
    )
    return results


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


@api.get("/repository-files/{reference_value:path}", response_model=RepositoryFileDetail)
def repository_file(reference_value: str, engine: Engine) -> RepositoryFileDetail:
    """Return complete indexed text for one qualified repository-file identity."""
    try:
        reference = repository_identity.parse_repository_file_id(f"file:{reference_value}")
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    with engine.connect() as conn:
        state = repository_index_states(conn).get(reference.target)
        rows = conn.execute(
            sqlalchemy.select(schema.repository_file_chunks)
            .where(
                schema.repository_file_chunks.c.repository == reference.target.repository,
                schema.repository_file_chunks.c.branch == reference.target.branch,
                schema.repository_file_chunks.c.path == reference.path,
            )
            .order_by(schema.repository_file_chunks.c.chunk_index)
        ).all()
    if state is None or not rows:
        raise HTTPException(404, f"no indexed repository file {reference.result_id}")
    return RepositoryFileDetail(
        id=reference.result_id,
        title=rows[0].title,
        subtitle=(
            f"{reference.target.repository} · {reference.path} · {reference.target.branch}@"
            f"{state.commit_sha[: search_config.DISPLAY_SHA_CHARACTERS]} · {repository_freshness(state)}"
        ),
        url=repository_blob_url(reference.target, state.commit_sha, reference.path),
        text=indexed_file_text(rows),
    )


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


@api.get("/search-executions", response_model=list[SearchExecutionEntry])
def search_executions(
    engine: Engine,
    after_id: int = Query(0, ge=0),
    mode: search_history.SearchMode | None = None,
    limit: int = Query(SEARCH_HISTORY_PAGE_LIMIT, ge=1, le=SEARCH_HISTORY_PAGE_LIMIT),
) -> list[SearchExecutionEntry]:
    """Return durable search executions in stable ID order for evaluation exports."""
    statement = sqlalchemy.select(schema.search_executions).where(schema.search_executions.c.id > after_id)
    if mode is not None:
        statement = statement.where(schema.search_executions.c.mode == mode)
    statement = statement.order_by(schema.search_executions.c.id).limit(limit)
    with engine.connect() as conn:
        execution_rows = list(conn.execute(statement))
        execution_ids = [row.id for row in execution_rows]
        result_rows = (
            list(
                conn.execute(
                    sqlalchemy.select(schema.search_execution_results)
                    .where(schema.search_execution_results.c.execution_id.in_(execution_ids))
                    .order_by(schema.search_execution_results.c.execution_id, schema.search_execution_results.c.rank)
                )
            )
            if execution_ids
            else []
        )
    results_by_execution: dict[int, list[SearchExecutionResultEntry]] = {}
    for row in result_rows:
        results_by_execution.setdefault(row.execution_id, []).append(
            SearchExecutionResultEntry(
                **{field: getattr(row, field) for field in SearchExecutionResultEntry.model_fields}
            )
        )
    return [
        SearchExecutionEntry(
            **{field: getattr(row, field) for field in SearchExecutionEntry.model_fields if field != "results"},
            results=results_by_execution.get(row.id, []),
        )
        for row in execution_rows
    ]


@api.get("/feedback", response_model=list[SearchFeedbackListEntry])
def list_search_feedback(
    engine: Engine,
    config: Config,
    days: int = Query(30, ge=1, description="Look back this many days."),
    limit: int = Query(200, ge=1, le=500),
) -> list[SearchFeedbackListEntry]:
    """Return recent feedback with browseable metadata for each graded result."""
    cutoff = sqlalchemy.func.now() - sqlalchemy.func.make_interval(0, 0, 0, days)
    statement = (
        sqlalchemy.select(schema.search_feedback)
        .where(schema.search_feedback.c.created_at > cutoff)
        .order_by(schema.search_feedback.c.created_at.desc(), schema.search_feedback.c.id.desc())
        .limit(limit)
    )
    with engine.connect() as conn:
        feedback_rows = list(conn.execute(statement))
        if not feedback_rows:
            return []

        feedback_ids = [row.id for row in feedback_rows]
        grade_rows = list(
            conn.execute(
                sqlalchemy.select(schema.search_feedback_grades)
                .where(schema.search_feedback_grades.c.feedback_id.in_(feedback_ids))
                .order_by(
                    schema.search_feedback_grades.c.feedback_id,
                    schema.search_feedback_grades.c.grade.desc(),
                    schema.search_feedback_grades.c.result_id,
                )
            )
        )

        search_result_ids = {row.search_result_id for row in grade_rows if row.search_result_id is not None}
        stored_results = stored_feedback_results(conn, search_result_ids)
        missing_result_ids: set[str] = set()
        for grade in grade_rows:
            stored = stored_results.get(grade.search_result_id)
            if stored is None or not stored.title:
                missing_result_ids.add(grade.result_id)
        current_metadata = current_feedback_result_metadata(conn, missing_result_ids, config)

    grades_by_feedback: dict[int, list[SearchFeedbackResultGrade]] = {}
    for grade in grade_rows:
        stored = stored_results.get(grade.search_result_id)
        current = current_metadata.get(grade.result_id) or default_feedback_result_metadata(grade.result_id, config)
        metadata = FeedbackResultMetadata(stored.title or current.title, stored.url) if stored else current
        grades_by_feedback.setdefault(grade.feedback_id, []).append(
            SearchFeedbackResultGrade(
                key=f"{stored.domain}:{stored.id}" if stored else None,
                source_id=grade.result_id,
                grade=grade.grade,
                title=metadata.title,
                url=metadata.url,
            )
        )

    return [
        SearchFeedbackListEntry(
            **{field: getattr(row, field) for field in SearchFeedbackListEntry.model_fields if field != "grades"},
            grades=grades_by_feedback.get(row.id, []),
        )
        for row in feedback_rows
    ]


@api.post("/feedback", response_model=SearchFeedbackEntry, status_code=201)
def add_search_feedback(
    feedback: SearchFeedbackCreate,
    engine: Engine,
    x_goog_authenticated_user_email: str | None = Header(None),
) -> SearchFeedbackEntry:
    """Store one query's optional note and per-result relevance grades."""
    author = iap_caller(x_goog_authenticated_user_email)
    with engine.begin() as conn:
        resolved_grades, execution_id = resolve_feedback_grades(conn, feedback.grades, feedback.execution_id)
        validate_feedback_execution(conn, execution_id, author, feedback.query)

        statement = (
            schema.search_feedback.insert()
            .values(
                author=author,
                query=feedback.query,
                note=feedback.note,
                execution_id=execution_id,
            )
            .returning(schema.search_feedback)
        )
        row = conn.execute(statement).first()
        assert row is not None
        if resolved_grades:
            conn.execute(
                schema.search_feedback_grades.insert().values(
                    [
                        {
                            "feedback_id": row.id,
                            "result_id": grade.result.source_id,
                            "search_result_id": grade.result.id,
                            "grade": grade.grade,
                        }
                        for grade in resolved_grades
                    ]
                )
            )
    return SearchFeedbackEntry(
        id=row.id,
        created_at=row.created_at,
        author=row.author,
        query=feedback.query,
        grades=feedback.grades,
        note=feedback.note,
        execution_id=execution_id,
    )


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
        params = {**hybrid_search_params(model, query, limit), "tags": tags}
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
