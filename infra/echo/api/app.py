# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Search Echo activity and wiki notes, and read or append the shared work log.

See ``infra/echo/README.md`` for endpoints and access requirements.
"""

import os
import re
from collections.abc import Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, Protocol
from urllib.parse import quote

import dashboard as echo_dashboard
import hybrid_search
import schema
import search_config
import sqlalchemy
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
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


@dataclass(frozen=True)
class EchoConfig:
    public_url: str
    github_repository: str
    github_branch: str


DEFAULT_CONFIG = EchoConfig(
    public_url="https://echo.oa.dev",
    github_repository=search_config.INDEXED_REPOSITORY,
    github_branch=search_config.INDEXED_BRANCH,
)


def environment_config() -> EchoConfig:
    return EchoConfig(
        public_url=os.environ.get("ECHO_PUBLIC_URL", DEFAULT_CONFIG.public_url).rstrip("/"),
        github_repository=os.environ.get("GITHUB_REPOSITORY", DEFAULT_CONFIG.github_repository),
        github_branch=os.environ.get("GITHUB_BRANCH", DEFAULT_CONFIG.github_branch),
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
        app.state.model = TextEmbedding(search_config.EMBED_MODEL)
        app.state.reranker = TextCrossEncoder(search_config.RERANK_MODEL)
        try:
            yield
        finally:
            app.state.engine.dispose()


app = FastAPI(
    title="echo",
    description="Search Marin activity and wiki notes, and read/append the shared agent work log.",
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


class RerankerModel(Protocol):
    def rerank(self, query: str, documents: Iterable[str], batch_size: int = 64, **kwargs: Any) -> Iterable[float]:
        """Score documents for their relevance to one query."""


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
        "embedding": str(query_embedding(model, query)),
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


def activity_domain(row: sqlalchemy.Row) -> Literal["discord", "pr", "issue"]:
    if row.source == "discord":
        return "discord"
    if row.kind == "pr" or "/pull/" in row.url:
        return "pr"
    return "issue"


def activity_search_result(row: sqlalchemy.Row) -> SearchResult:
    domain = activity_domain(row)
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


def repository_index_state(
    conn: sqlalchemy.Connection,
    config: EchoConfig,
) -> RepositoryIndexState | None:
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
    row = conn.execute(
        sqlalchemy.select(
            sqlalchemy.func.coalesce(
                schema.repository_index_builds.c.commit_sha,
                schema.repository_index_state.c.commit_sha,
            ).label("commit_sha"),
            schema.repository_index_builds.c.completed_files,
            schema.repository_index_builds.c.total_files,
            schema.repository_index_builds.c.started_at,
            schema.repository_index_state.c.indexed_at,
        )
        .select_from(schema.repository_index_state.outerjoin(schema.repository_index_builds, join_condition, full=True))
        .where(repository == config.github_repository, branch == config.github_branch)
    ).first()
    if row is None:
        return None
    return RepositoryIndexState(
        row.commit_sha,
        row.completed_files,
        row.total_files,
        row.started_at,
        row.indexed_at,
    )


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


def repository_file_search_result(
    row: sqlalchemy.Row,
    state: RepositoryIndexState,
    query: str,
    config: EchoConfig,
) -> SearchResult:
    path = quote(row.path, safe="/")
    source_url = f"https://github.com/{config.github_repository}/blob/{state.commit_sha}/{path}"
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
        id=f"file:{row.path}",
        domain="file",
        title=row.title,
        subtitle=(
            f"{row.path}:{references[0].line} · "
            f"{config.github_branch}@{state.commit_sha[: search_config.DISPLAY_SHA_CHARACTERS]} · "
            f"{repository_freshness(state)}"
        ),
        url=references[0].url,
        snippet=" · ".join(f"{row.path}:{reference.line} {reference.text}" for reference in references),
        score=row.score,
        distance=row.distance,
        lexical_score=row.lexical_score,
        references=references,
    )


def query_oriented_result(result: SearchResult, query: str) -> SearchResult:
    """Apply a small source-quality prior to prose queries over repository files."""
    if result.domain != "file" or search_config.is_identifier_query(query):
        return result
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
    """Fuse the existing hybrid order with bounded full-text cross-encoder ranks."""
    base = [SearchCandidate(query_oriented_result(candidate.result, query), candidate.text) for candidate in candidates]
    base.sort(key=lambda candidate: (-candidate.result.score, candidate.result.domain, candidate.result.id))
    selected = base[: search_config.RERANK_MAX_CANDIDATES]
    if not selected:
        return []
    scores = list(reranker.rerank(query, [candidate.text for candidate in selected]))
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
        if result.rerank_score is not None and result.rerank_score >= search_config.MIN_RERANK_SCORE
    ]
    reranked.sort(key=lambda result: (-result.score, result.domain, result.id))
    return reranked[:limit]


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


@api.get("/repository-index", response_model=RepositoryIndexStatus)
def repository_index_status(engine: Engine, config: Config) -> RepositoryIndexStatus:
    """Return repository index freshness and current build progress."""
    with engine.connect() as conn:
        state = repository_index_state(conn, config)
    if state is None:
        return RepositoryIndexStatus(
            repository=config.github_repository,
            branch=config.github_branch,
            status="empty",
            commit_sha=None,
            completed_files=None,
            total_files=None,
            started_at=None,
            indexed_at=None,
        )
    return RepositoryIndexStatus(
        repository=config.github_repository,
        branch=config.github_branch,
        status="building" if state.building else "ready",
        commit_sha=state.commit_sha,
        completed_files=state.completed_files,
        total_files=state.total_files,
        started_at=state.started_at,
        indexed_at=state.indexed_at,
    )


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
        **hybrid_search_params(model, query, limit),
        "source": source,
        "kind": kind,
        "since": since,
    }
    statement = hybrid_search.chunk_search_statement(chunk_filter_clauses(source, kind, since))
    with engine.connect() as conn:
        conn.execute(hybrid_search.HNSW_ITERATIVE_SCAN)
        return [hit(row) for row in conn.execute(statement, params)]


@api.get("/federated-search", response_model=list[SearchResult])
def federated_search(
    engine: Engine,
    model: Model,
    reranker: Reranker,
    config: Config,
    q: str = Query(description="Natural-language query."),
    domain: list[search_config.SearchDomain] | None = Query(
        None, description="Search this domain; repeat to select several."
    ),
    limit: int = Query(search_config.DEFAULT_SEARCH_LIMIT, ge=1, le=search_config.MAX_SEARCH_LIMIT),
) -> list[SearchResult]:
    """Search wiki, repository file, and activity domains and merge their hybrid ranks."""
    query = q.strip()
    if not query:
        raise HTTPException(422, "q must not be blank")
    domains = list(dict.fromkeys(domain or search_config.DEFAULT_SEARCH_DOMAINS))
    retrieval_limit = max(limit, search_config.RERANK_MIN_RESULTS_PER_DOMAIN)
    params = hybrid_search_params(model, query, retrieval_limit)
    candidates: list[SearchCandidate] = []
    with engine.connect() as conn:
        conn.execute(hybrid_search.HNSW_ITERATIVE_SCAN)
        if "wiki" in domains:
            for row in conn.execute(hybrid_search.wiki_search_statement(), params):
                candidates.append(
                    SearchCandidate(
                        wiki_search_result(row, config),
                        f"{row.title}\n{row.use_when}\n\n{row.body}",
                    )
                )
        if "file" in domains:
            state = repository_index_state(conn, config)
            if state is not None:
                file_params = {
                    **params,
                    "candidate_limit": (
                        search_config.candidate_limit(retrieval_limit) * search_config.FILE_CHUNK_CANDIDATE_MULTIPLIER
                    ),
                    "repository": config.github_repository,
                    "branch": config.github_branch,
                    "exact": escape_like(query),
                    "substring": f"%{escape_like(query)}%",
                }
                for row in conn.execute(hybrid_search.repository_file_search_statement(), file_params):
                    candidates.append(
                        SearchCandidate(
                            repository_file_search_result(row, state, query, config),
                            f"{row.path}\n{row.title}\n\n{row.text}",
                        )
                    )
        activity_domains = [candidate for candidate in domains if candidate in ("discord", "pr", "issue")]
        if activity_domains:
            statement = hybrid_search.chunk_search_statement([activity_domain_clause(activity_domains)])
            for row in conn.execute(statement, params):
                candidates.append(
                    SearchCandidate(
                        activity_search_result(row),
                        f"{row.title or ''}\n\n{row.text or ''}",
                    )
                )
    return rerank_candidates(candidates, query, reranker, limit)


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


@api.get("/repository-files/{path:path}", response_model=RepositoryFileDetail)
def repository_file(path: str, engine: Engine, config: Config) -> RepositoryFileDetail:
    """Return the complete indexed text and pinned URL for one repository path."""
    with engine.connect() as conn:
        state = repository_index_state(conn, config)
        rows = conn.execute(
            sqlalchemy.select(schema.repository_file_chunks)
            .where(
                schema.repository_file_chunks.c.repository == config.github_repository,
                schema.repository_file_chunks.c.branch == config.github_branch,
                schema.repository_file_chunks.c.path == path,
            )
            .order_by(schema.repository_file_chunks.c.chunk_index)
        ).all()
    if state is None or not rows:
        raise HTTPException(404, f"no indexed repository file {path}")
    quoted_path = quote(path, safe="/")
    return RepositoryFileDetail(
        id=f"file:{path}",
        title=rows[0].title,
        subtitle=(
            f"{path} · {config.github_branch}@"
            f"{state.commit_sha[: search_config.DISPLAY_SHA_CHARACTERS]} · {repository_freshness(state)}"
        ),
        url=f"https://github.com/{config.github_repository}/blob/{state.commit_sha}/{quoted_path}",
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
