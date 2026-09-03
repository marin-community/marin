# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent PostgreSQL store for GitHub review and lint activity."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from itertools import batched

import sqlalchemy
from google.cloud.sql.connector import Connector
from pydantic import BaseModel, ConfigDict
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Connection, Engine

from .github_review_corpus import (
    GitHubUsage,
    PullRequestBundle,
    PullRequestFingerprint,
    PullRequestRecord,
    ReviewEventFingerprint,
    ReviewEventKey,
    ReviewEventRecord,
    review_event_fingerprint,
)

DEFAULT_CLOUDSQL_CONNECTION = "hai-gcp-models:us-central1:marin-metadata"
DEFAULT_DATABASE = "context"
DEFAULT_DATABASE_USER = "loom-vm@hai-gcp-models.iam"
DEFAULT_BACKFILL_DAYS = 30
MAX_SYNC_ATTEMPTS = 3
MAX_ACTIVITY_RESULTS = 500
STORED_ERROR_MAX_LENGTH = 4_000
DATABASE_WRITE_BATCH_SIZE = 100

metadata = MetaData()
json_type = JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")


class StoreModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SyncStatus(StrEnum):
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    ABANDONED = "abandoned"


class ProbeStatus(StrEnum):
    COMPLETE = "complete"
    FAILED = "failed"


sync_runs = Table(
    "codehealth_sync_runs",
    metadata,
    Column("sync_id", Text, primary_key=True),
    Column("repository", Text, nullable=False),
    Column("window_start", DateTime(timezone=True), nullable=False),
    Column("window_end", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True)),
    Column("status", Text, nullable=False),
    Column("attempt_count", Integer, nullable=False),
    Column("candidate_pull_requests", Integer),
    Column("reused_pull_requests", Integer),
    Column("github_usage", json_type),
    Column("finelog_watermark", json_type),
    Column("error", Text),
    Index("ix_codehealth_sync_runs_repository_started", "repository", "started_at"),
)

sync_pull_requests = Table(
    "codehealth_sync_pull_requests",
    metadata,
    Column("sync_id", Text, ForeignKey("codehealth_sync_runs.sync_id", ondelete="CASCADE"), primary_key=True),
    Column("pr_number", Integer, primary_key=True),
    Column("synced_at", DateTime(timezone=True), nullable=False),
)

pull_requests = Table(
    "codehealth_pull_requests",
    metadata,
    Column("repository", Text, primary_key=True),
    Column("pr_number", Integer, primary_key=True),
    Column("node_id", Text, nullable=False),
    Column("author", Text, nullable=False),
    Column("state", Text, nullable=False),
    Column("head_sha", Text, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("record_sha", Text, nullable=False),
    Column("record", json_type, nullable=False),
    Column("observed_at", DateTime(timezone=True), nullable=False),
    Index("ix_codehealth_pull_requests_updated", "updated_at"),
)

pull_request_versions = Table(
    "codehealth_pull_request_versions",
    metadata,
    Column("repository", Text, primary_key=True),
    Column("pr_number", Integer, primary_key=True),
    Column("record_sha", Text, primary_key=True),
    Column("record", json_type, nullable=False),
    Column("observed_at", DateTime(timezone=True), nullable=False),
)

review_events = Table(
    "codehealth_review_events",
    metadata,
    Column("event_id", Text, primary_key=True),
    Column("repository", Text, nullable=False),
    Column("pr_number", Integer, nullable=False),
    Column("kind", Text, nullable=False),
    Column("database_id", BigInteger, nullable=False),
    Column("author", Text, nullable=False),
    Column("is_human", Boolean, nullable=False),
    Column("activity_at", DateTime(timezone=True), nullable=False),
    Column("record_sha", Text, nullable=False),
    Column("record", json_type, nullable=False),
    Column("observed_at", DateTime(timezone=True), nullable=False),
    Index("ix_codehealth_review_events_pr_activity", "repository", "pr_number", "activity_at"),
    Index("ix_codehealth_review_events_author_activity", "author", "activity_at"),
)

review_event_versions = Table(
    "codehealth_review_event_versions",
    metadata,
    Column("event_id", Text, primary_key=True),
    Column("record_sha", Text, primary_key=True),
    Column("record", json_type, nullable=False),
    Column("observed_at", DateTime(timezone=True), nullable=False),
)

review_threads = Table(
    "codehealth_review_threads",
    metadata,
    Column("repository", Text, primary_key=True),
    Column("thread_id", Text, primary_key=True),
    Column("pr_number", Integer, nullable=False),
    Column("record_sha", Text, nullable=False),
    Column("record", json_type, nullable=False),
    Column("observed_at", DateTime(timezone=True), nullable=False),
)

changed_files = Table(
    "codehealth_changed_files",
    metadata,
    Column("repository", Text, primary_key=True),
    Column("pr_number", Integer, primary_key=True),
    Column("filename", Text, primary_key=True),
    Column("record", json_type, nullable=False),
    Column("observed_at", DateTime(timezone=True), nullable=False),
    Index("ix_codehealth_changed_files_filename", "filename"),
)

commits = Table(
    "codehealth_commits",
    metadata,
    Column("repository", Text, primary_key=True),
    Column("pr_number", Integer, primary_key=True),
    Column("sha", Text, primary_key=True),
    Column("position", Integer, nullable=False),
    Column("record", json_type, nullable=False),
    Column("observed_at", DateTime(timezone=True), nullable=False),
)

pull_request_diffs = Table(
    "codehealth_pull_request_diffs",
    metadata,
    Column("repository", Text, primary_key=True),
    Column("pr_number", Integer, primary_key=True),
    Column("head_sha", Text, primary_key=True),
    Column("diff", Text),
    Column("observed_at", DateTime(timezone=True), nullable=False),
)

source_contexts = Table(
    "codehealth_source_contexts",
    metadata,
    Column("event_id", Text, primary_key=True),
    Column("commit_sha", Text, primary_key=True),
    Column("path", Text, nullable=False),
    Column("anchor_line", Integer, nullable=False),
    Column("start_line", Integer, nullable=False),
    Column("end_line", Integer, nullable=False),
    Column("text", Text),
    Column("unavailable_reason", Text),
    Column("content_sha", Text, nullable=False),
    Column("fetched_at", DateTime(timezone=True), nullable=False),
)

lint_invocations = Table(
    "codehealth_lint_invocations",
    metadata,
    Column("invocation_id", Text, primary_key=True),
    Column("repository", Text, primary_key=True),
    Column("ts", DateTime(timezone=True), nullable=False),
    Column("pr_number", Integer),
    Column("head_sha", Text),
    Column("catalog_sha", Text),
    Column("successful", Boolean, nullable=False),
    Column("finding_count", Integer, nullable=False),
    Column("record", json_type, nullable=False),
    Index("ix_codehealth_lint_invocations_pr_ts", "repository", "pr_number", "ts"),
)

lint_findings = Table(
    "codehealth_lint_findings",
    metadata,
    Column("finding_id", Text, primary_key=True),
    Column("invocation_id", Text, nullable=False),
    Column("repository", Text, primary_key=True),
    Column("ts", DateTime(timezone=True), nullable=False),
    Column("pr_number", Integer),
    Column("code", Text, nullable=False),
    Column("record", json_type, nullable=False),
    ForeignKeyConstraint(
        ("repository", "invocation_id"),
        ("codehealth_lint_invocations.repository", "codehealth_lint_invocations.invocation_id"),
        ondelete="CASCADE",
    ),
    Index("ix_codehealth_lint_findings_pr_code", "repository", "pr_number", "code"),
)

lint_catalog_snapshots = Table(
    "codehealth_lint_catalog_snapshots",
    metadata,
    Column("catalog_sha", Text, primary_key=True),
    Column("observed_at", DateTime(timezone=True), nullable=False),
    Column("record", json_type, nullable=False),
)

rule_probes = Table(
    "codehealth_rule_probes",
    metadata,
    Column("probe_id", Text, primary_key=True),
    Column("idempotency_key", Text, nullable=False, unique=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("event_id", Text, nullable=False),
    Column("context_sha", Text, nullable=False),
    Column("rule_code", Text, nullable=False),
    Column("rule_sha", Text, nullable=False),
    Column("catalog_sha", Text, nullable=False),
    Column("model", Text, nullable=False),
    Column("effort", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("fired", Boolean),
    Column("confidence", Float),
    Column("finding", Text),
    Column("raw_output", Text),
    Column("error", Text),
    Column("elapsed", Float, nullable=False),
    Column("record", json_type, nullable=False),
)


class PullRequestActivity(StoreModel):
    repository: str
    number: int
    url: str
    title: str
    author: str
    state: str
    updated_at: str
    head_sha: str
    human_events: int
    lint_runs: int
    lint_findings: int
    rule_codes: tuple[str, ...]


class ReviewEventSummary(StoreModel):
    event_id: str
    kind: str
    database_id: int
    author: str
    is_human: bool
    is_agent_marked: bool
    body_preview: str
    activity_at: str
    source_url: str | None
    path: str | None
    line: int | None
    thread_is_resolved: bool | None
    lint_runs: int
    lint_findings: int


class TelemetryModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")


class LintInvocationRecord(TelemetryModel):
    invocation_id: str
    ts: dt.datetime
    pr_number: int | None = None
    head_sha: str | None = None
    lint_catalog_sha: str | None = None
    diff_added_lines: int | None = None
    diff_removed_lines: int | None = None


class LintFindingRecord(TelemetryModel):
    invocation_id: str
    ts: dt.datetime
    pr_number: int | None = None
    code: str | None = None


class LintActivity(StoreModel):
    invocations: tuple[LintInvocationRecord, ...]
    findings: tuple[LintFindingRecord, ...]


class SyncStatusSummary(StoreModel):
    sync_id: str
    repository: str
    window_start: str
    window_end: str
    status: SyncStatus
    attempt_count: int
    started_at: str
    completed_at: str | None
    candidate_pull_requests: int | None
    reused_pull_requests: int | None
    error: str | None


class StoredProbe(StoreModel):
    probe_id: str
    idempotency_key: str
    created_at: str
    event_id: str
    context_sha: str
    rule_code: str
    rule_sha: str
    catalog_sha: str
    model: str
    effort: str
    status: ProbeStatus
    fired: bool | None
    confidence: float | None
    finding: str | None
    raw_output: str | None
    elapsed: float
    error: str | None


class ReviewContext(StoreModel):
    event: ReviewEventRecord
    thread: tuple[ReviewEventRecord, ...]
    pull_request: PullRequestRecord
    diff: str | None
    source: str | None
    source_start_line: int | None
    source_end_line: int | None
    source_unavailable_reason: str | None
    lint_invocations: tuple[LintInvocationRecord, ...]
    lint_findings: tuple[LintFindingRecord, ...]
    context_sha: str


@dataclass(frozen=True)
class SyncRun:
    sync_id: str
    repository: str
    window_start: dt.datetime
    window_end: dt.datetime
    attempt_count: int


@dataclass(frozen=True)
class DatabaseConfig:
    instance: str
    database: str
    user: str


@dataclass(frozen=True)
class DatabaseResources:
    engine: Engine
    connector: Connector

    def close(self) -> None:
        self.engine.dispose()
        self.connector.close()


@dataclass(frozen=True)
class LintRecordRows:
    invocations: list[LintInvocationRecord]
    findings: list[LintFindingRecord]


def database_config_from_environment() -> DatabaseConfig:
    return DatabaseConfig(
        instance=os.environ.get("CLOUDSQL_CONNECTION", DEFAULT_CLOUDSQL_CONNECTION),
        database=os.environ.get("PGDATABASE", DEFAULT_DATABASE),
        user=os.environ.get("PGUSER", DEFAULT_DATABASE_USER),
    )


def create_database_resources(config: DatabaseConfig) -> DatabaseResources:
    """Connect to Marin's metadata database with the supplied IAM identity."""
    connector = Connector(refresh_strategy="lazy")
    engine = sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=lambda: connector.connect(
            config.instance,
            "pg8000",
            user=config.user,
            db=config.database,
            enable_iam_auth=True,
        ),
        pool_pre_ping=True,
    )
    return DatabaseResources(engine, connector)


@contextmanager
def database_engine(config: DatabaseConfig) -> Iterator[Engine]:
    """Own a database engine and connector for one CLI operation."""
    resources = create_database_resources(config)
    try:
        yield resources.engine
    finally:
        resources.close()


def create_schema(engine: Engine) -> None:
    """Create the codehealth tables in a test database."""
    metadata.create_all(engine)


def _dialect_insert(conn: Connection, table: Table):
    if conn.dialect.name == "postgresql":
        return pg_insert(table)
    if conn.dialect.name == "sqlite":
        return sqlite_insert(table)
    raise ValueError(f"unsupported review-store database dialect: {conn.dialect.name}")


def _upsert_many(
    conn: Connection,
    table: Table,
    values: Sequence[dict[str, object]],
    keys: Sequence[str],
) -> None:
    for value_batch in batched(values, DATABASE_WRITE_BATCH_SIZE):
        statement = _dialect_insert(conn, table).values(list(value_batch))
        updates = {column.name: statement.excluded[column.name] for column in table.columns if column.name not in keys}
        conn.execute(statement.on_conflict_do_update(index_elements=[table.c[key] for key in keys], set_=updates))


def _insert_ignore_many(
    conn: Connection,
    table: Table,
    values: Sequence[dict[str, object]],
    keys: Sequence[str],
) -> None:
    for value_batch in batched(values, DATABASE_WRITE_BATCH_SIZE):
        statement = _dialect_insert(conn, table).values(list(value_batch))
        conn.execute(statement.on_conflict_do_nothing(index_elements=[table.c[key] for key in keys]))


def _json_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dt.datetime):
        return utc_iso(value)
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return value


def _payload(record: BaseModel | dict[str, object]) -> dict[str, object]:
    value = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
    normalized = _json_value(value)
    assert isinstance(normalized, dict)
    return normalized


def record_sha(record: BaseModel | dict[str, object]) -> str:
    """Return the canonical SHA-256 identity for a persisted JSON record."""
    return hashlib.sha256(json.dumps(_payload(record), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stored_error_message(error: Exception) -> str:
    """Format one exception for a bounded persisted error field."""
    stderr = error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) and error.stderr else ""
    message = f"{type(error).__name__}: {error}"
    if stderr:
        message = f"{message}: {stderr}"
    return message[:STORED_ERROR_MAX_LENGTH]


def _utc_datetime(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.UTC)
    return value.astimezone(dt.UTC)


def _timestamp(value: str | None) -> dt.datetime:
    if value is None:
        return dt.datetime.min.replace(tzinfo=dt.UTC)
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(dt.UTC)


def _event_activity(event: ReviewEventRecord) -> dt.datetime:
    return max(_timestamp(event.created_at), _timestamp(event.updated_at), _timestamp(event.submitted_at))


def start_or_resume_sync(
    engine: Engine,
    repository: str,
    *,
    now: dt.datetime,
    days: int = DEFAULT_BACKFILL_DAYS,
) -> SyncRun:
    """Resume a bounded number of attempts, then abandon a poisoned window."""
    with engine.begin() as conn:
        row = (
            conn.execute(
                sqlalchemy.select(sync_runs)
                .where(
                    sync_runs.c.repository == repository,
                    sync_runs.c.status.in_((SyncStatus.RUNNING.value, SyncStatus.FAILED.value)),
                )
                .order_by(sync_runs.c.started_at.desc())
                .limit(1)
            )
            .mappings()
            .first()
        )
        if row is not None and int(row["attempt_count"]) < MAX_SYNC_ATTEMPTS:
            attempt_count = int(row["attempt_count"]) + 1
            conn.execute(
                sqlalchemy.update(sync_runs)
                .where(sync_runs.c.sync_id == row["sync_id"])
                .values(status=SyncStatus.RUNNING.value, attempt_count=attempt_count, error=None)
            )
            return SyncRun(str(row["sync_id"]), repository, row["window_start"], row["window_end"], attempt_count)
        if row is not None:
            conn.execute(
                sqlalchemy.update(sync_runs)
                .where(sync_runs.c.sync_id == row["sync_id"])
                .values(status=SyncStatus.ABANDONED.value)
            )
        end = now.astimezone(dt.UTC)
        start = end - dt.timedelta(days=days)
        sync_id = str(uuid.uuid4())
        conn.execute(
            sync_runs.insert().values(
                sync_id=sync_id,
                repository=repository,
                window_start=start,
                window_end=end,
                started_at=now,
                status=SyncStatus.RUNNING.value,
                attempt_count=1,
            )
        )
        return SyncRun(sync_id, repository, start, end, 1)


def latest_sync_status(engine: Engine, repository: str) -> SyncStatusSummary | None:
    """Return the latest fixed-window sync state for one repository."""
    with engine.begin() as conn:
        row = (
            conn.execute(
                sqlalchemy.select(sync_runs)
                .where(sync_runs.c.repository == repository)
                .order_by(sync_runs.c.started_at.desc())
                .limit(1)
            )
            .mappings()
            .first()
        )
    if row is None:
        return None
    return SyncStatusSummary(
        sync_id=str(row["sync_id"]),
        repository=str(row["repository"]),
        window_start=utc_iso(row["window_start"]),
        window_end=utc_iso(row["window_end"]),
        status=SyncStatus(str(row["status"])),
        attempt_count=int(row["attempt_count"]),
        started_at=utc_iso(row["started_at"]),
        completed_at=None if row["completed_at"] is None else utc_iso(row["completed_at"]),
        candidate_pull_requests=row["candidate_pull_requests"],
        reused_pull_requests=row["reused_pull_requests"],
        error=row["error"],
    )


def require_complete_sync(engine: Engine, repository: str) -> SyncStatusSummary:
    """Fail closed when the workbench does not represent a completed sync."""
    status = latest_sync_status(engine, repository)
    if status is None:
        raise RuntimeError(f"no review sync exists for {repository}")
    if status.status != SyncStatus.COMPLETE:
        raise RuntimeError(
            f"latest review sync for {repository} is {status.status.value} "
            f"(attempt {status.attempt_count}/{MAX_SYNC_ATTEMPTS}): {status.error or 'no error recorded'}"
        )
    return status


def completed_pull_request_numbers(engine: Engine, sync_id: str) -> set[int]:
    with engine.begin() as conn:
        return set(
            conn.execute(
                sqlalchemy.select(sync_pull_requests.c.pr_number).where(sync_pull_requests.c.sync_id == sync_id)
            ).scalars()
        )


def cached_pull_request_fingerprints(engine: Engine, repository: str) -> dict[int, PullRequestFingerprint]:
    """Return complete stored fingerprints suitable for avoiding unchanged hydration."""
    with engine.begin() as conn:
        pull_rows = conn.execute(
            sqlalchemy.select(pull_requests.c.pr_number, pull_requests.c.record).where(
                pull_requests.c.repository == repository
            )
        ).all()
        review_counts = dict(
            conn.execute(
                sqlalchemy.select(review_events.c.pr_number, sqlalchemy.func.count())
                .where(review_events.c.repository == repository, review_events.c.kind == "review")
                .group_by(review_events.c.pr_number)
            ).all()
        )
        thread_counts = dict(
            conn.execute(
                sqlalchemy.select(review_threads.c.pr_number, sqlalchemy.func.count())
                .where(review_threads.c.repository == repository)
                .group_by(review_threads.c.pr_number)
            ).all()
        )
    fingerprints: dict[int, PullRequestFingerprint] = {}
    for pr_number, record in pull_rows:
        fingerprints[int(pr_number)] = PullRequestFingerprint(
            updated_at=str(record["updated_at"]),
            head_sha=str(record["head_sha"]),
            base_sha=str(record["base_sha"]),
            changed_files=int(record["changed_files"]),
            commits=int(record["commits"]),
            reviews=int(review_counts.get(pr_number, 0)),
            review_threads=int(thread_counts.get(pr_number, 0)),
            issue_comments=int(record["issue_comments"]),
        )
    return fingerprints


def cached_review_event_fingerprints(engine: Engine, repository: str) -> dict[ReviewEventKey, ReviewEventFingerprint]:
    """Return current event identities used to reconcile GitHub activity seeds."""
    with engine.begin() as conn:
        rows = (
            conn.execute(sqlalchemy.select(review_events.c.record).where(review_events.c.repository == repository))
            .scalars()
            .all()
        )
    fingerprints: dict[ReviewEventKey, ReviewEventFingerprint] = {}
    for record in rows:
        event = ReviewEventRecord.model_validate(record)
        fingerprints[(event.pr_number, event.kind, event.database_id)] = review_event_fingerprint(
            event.body, event.updated_at
        )
    return fingerprints


def checkpoint_reused_pull_requests(
    engine: Engine,
    sync_id: str,
    pr_numbers: Sequence[int],
    *,
    observed_at: dt.datetime,
) -> None:
    """Record unchanged stored pull requests in the current sync."""
    rows = [{"sync_id": sync_id, "pr_number": pr_number, "synced_at": observed_at} for pr_number in pr_numbers]
    if not rows:
        return
    with engine.begin() as conn:
        _insert_ignore_many(
            conn,
            sync_pull_requests,
            rows,
            ("sync_id", "pr_number"),
        )


def store_bundles(
    engine: Engine,
    sync_id: str,
    bundles: Sequence[PullRequestBundle],
    *,
    observed_at: dt.datetime,
) -> None:
    """Commit one reconciled hydration batch and its resume checkpoints atomically."""
    if not bundles:
        return
    repository = bundles[0].pull_request.repository
    if any(bundle.pull_request.repository != repository for bundle in bundles):
        raise ValueError("one review-store batch cannot span repositories")
    pull_numbers = [bundle.pull_request.number for bundle in bundles]
    if len(pull_numbers) != len(set(pull_numbers)):
        raise ValueError("one review-store batch cannot contain a pull request twice")

    pull_version_rows: list[dict[str, object]] = []
    pull_rows: list[dict[str, object]] = []
    event_version_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    thread_rows: list[dict[str, object]] = []
    file_rows: list[dict[str, object]] = []
    commit_rows: list[dict[str, object]] = []
    diff_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    for bundle in bundles:
        pull = bundle.pull_request
        pull_payload = _payload(pull)
        pull_sha = record_sha(pull)
        pull_version_rows.append(
            {
                "repository": repository,
                "pr_number": pull.number,
                "record_sha": pull_sha,
                "record": pull_payload,
                "observed_at": observed_at,
            }
        )
        pull_rows.append(
            {
                "repository": repository,
                "pr_number": pull.number,
                "node_id": pull.node_id,
                "author": pull.author,
                "state": pull.state,
                "head_sha": pull.head_sha,
                "updated_at": _timestamp(pull.updated_at),
                "record_sha": pull_sha,
                "record": pull_payload,
                "observed_at": observed_at,
            }
        )
        for event in bundle.events:
            payload = _payload(event)
            event_sha = record_sha(event)
            event_version_rows.append(
                {"event_id": event.event_id, "record_sha": event_sha, "record": payload, "observed_at": observed_at}
            )
            event_rows.append(
                {
                    "event_id": event.event_id,
                    "repository": repository,
                    "pr_number": event.pr_number,
                    "kind": event.kind,
                    "database_id": event.database_id,
                    "author": event.author,
                    "is_human": event.is_human,
                    "activity_at": _event_activity(event),
                    "record_sha": event_sha,
                    "record": payload,
                    "observed_at": observed_at,
                }
            )
        thread_rows.extend(
            {
                "repository": repository,
                "thread_id": thread.thread_id,
                "pr_number": thread.pr_number,
                "record_sha": record_sha(thread),
                "record": _payload(thread),
                "observed_at": observed_at,
            }
            for thread in bundle.threads
        )
        file_rows.extend(
            {
                "repository": repository,
                "pr_number": file.pr_number,
                "filename": file.filename,
                "record": _payload(file),
                "observed_at": observed_at,
            }
            for file in bundle.files
        )
        commit_rows.extend(
            {
                "repository": repository,
                "pr_number": commit.pr_number,
                "sha": commit.sha,
                "position": position,
                "record": _payload(commit),
                "observed_at": observed_at,
            }
            for position, commit in enumerate(bundle.commits)
        )
        diff_rows.append(
            {
                "repository": repository,
                "pr_number": pull.number,
                "head_sha": pull.head_sha,
                "diff": bundle.diff,
                "observed_at": observed_at,
            }
        )
        checkpoint_rows.append({"sync_id": sync_id, "pr_number": pull.number, "synced_at": observed_at})

    with engine.begin() as conn:
        _insert_ignore_many(conn, pull_request_versions, pull_version_rows, ("repository", "pr_number", "record_sha"))
        _upsert_many(conn, pull_requests, pull_rows, ("repository", "pr_number"))
        for child_table in (review_events, review_threads, changed_files, commits):
            conn.execute(
                child_table.delete().where(
                    child_table.c.repository == repository,
                    child_table.c.pr_number.in_(pull_numbers),
                )
            )
        _insert_ignore_many(conn, review_event_versions, event_version_rows, ("event_id", "record_sha"))
        _upsert_many(conn, review_events, event_rows, ("event_id",))
        _upsert_many(conn, review_threads, thread_rows, ("repository", "thread_id"))
        _upsert_many(conn, changed_files, file_rows, ("repository", "pr_number", "filename"))
        _upsert_many(conn, commits, commit_rows, ("repository", "pr_number", "sha"))
        _upsert_many(conn, pull_request_diffs, diff_rows, ("repository", "pr_number", "head_sha"))
        _insert_ignore_many(conn, sync_pull_requests, checkpoint_rows, ("sync_id", "pr_number"))


def store_telemetry(
    engine: Engine,
    repository: str,
    invocations: Sequence[LintInvocationRecord],
    findings: Sequence[LintFindingRecord],
) -> None:
    """Mirror bounded Finelog activity for single-store exploration queries."""
    invocation_rows: list[dict[str, object]] = []
    for row in invocations:
        payload = _payload(row)
        invocation_rows.append(
            {
                "invocation_id": row.invocation_id,
                "repository": repository,
                "ts": _utc_datetime(row.ts),
                "pr_number": row.pr_number,
                "head_sha": row.head_sha,
                "catalog_sha": row.lint_catalog_sha,
                "successful": (
                    payload.get("agent_exit_code") is not None
                    and int(payload["agent_exit_code"]) == 0
                    and not bool(payload.get("timed_out"))
                ),
                "finding_count": int(payload.get("finding_count") or 0),
                "record": payload,
            }
        )
    finding_rows: list[dict[str, object]] = []
    for row in findings:
        payload = _payload(row)
        finding_rows.append(
            {
                "finding_id": record_sha(payload),
                "invocation_id": row.invocation_id,
                "repository": repository,
                "ts": _utc_datetime(row.ts),
                "pr_number": row.pr_number,
                "code": row.code or "",
                "record": payload,
            }
        )
    with engine.begin() as conn:
        _upsert_many(conn, lint_invocations, invocation_rows, ("repository", "invocation_id"))
        _upsert_many(conn, lint_findings, finding_rows, ("repository", "finding_id"))


def store_catalog_snapshot(
    engine: Engine,
    catalog_sha: str,
    record: dict[str, object],
    *,
    observed_at: dt.datetime,
) -> None:
    """Persist the catalog used by the checked-out weekly agent."""
    with engine.begin() as conn:
        _insert_ignore_many(
            conn,
            lint_catalog_snapshots,
            ({"catalog_sha": catalog_sha, "observed_at": observed_at, "record": _payload(record)},),
            ("catalog_sha",),
        )


def catalog_snapshot_shas(engine: Engine) -> set[str]:
    with engine.begin() as conn:
        return set(conn.execute(sqlalchemy.select(lint_catalog_snapshots.c.catalog_sha)).scalars())


def complete_sync(
    engine: Engine,
    sync_id: str,
    *,
    candidate_pull_requests: int,
    reused_pull_requests: int,
    github_usage: GitHubUsage,
    finelog_watermark: dict[str, object],
    completed_at: dt.datetime,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            sqlalchemy.update(sync_runs)
            .where(sync_runs.c.sync_id == sync_id)
            .values(
                status=SyncStatus.COMPLETE.value,
                completed_at=completed_at,
                candidate_pull_requests=candidate_pull_requests,
                reused_pull_requests=reused_pull_requests,
                github_usage=_payload(github_usage),
                finelog_watermark=finelog_watermark,
                error=None,
            )
        )


def fail_sync(engine: Engine, sync_id: str, error: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            sqlalchemy.update(sync_runs)
            .where(sync_runs.c.sync_id == sync_id)
            .values(status=SyncStatus.FAILED.value, error=error[:STORED_ERROR_MAX_LENGTH])
        )


def utc_iso(value: dt.datetime) -> str:
    return _utc_datetime(value).isoformat().replace("+00:00", "Z")


def _successful_lint_rows(
    conn: Connection,
    repository: str,
    *,
    pr_number: int | None = None,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
) -> LintRecordRows:
    statement = sqlalchemy.select(lint_invocations.c.record).where(
        lint_invocations.c.repository == repository,
        lint_invocations.c.successful.is_(True),
    )
    if pr_number is not None:
        statement = statement.where(lint_invocations.c.pr_number == pr_number)
    if start is not None:
        statement = statement.where(lint_invocations.c.ts >= start)
    if end is not None:
        statement = statement.where(lint_invocations.c.ts < end)
    invocation_rows = conn.execute(statement.order_by(lint_invocations.c.ts)).scalars().all()
    invocation_ids = [str(row["invocation_id"]) for row in invocation_rows]
    finding_rows = (
        conn.execute(
            sqlalchemy.select(lint_findings.c.record)
            .where(
                lint_findings.c.repository == repository,
                lint_findings.c.invocation_id.in_(invocation_ids) if invocation_ids else sqlalchemy.false(),
            )
            .order_by(lint_findings.c.ts)
        )
        .scalars()
        .all()
    )
    return LintRecordRows(
        [LintInvocationRecord.model_validate(row) for row in invocation_rows],
        [LintFindingRecord.model_validate(row) for row in finding_rows],
    )


def list_pull_request_activity(
    engine: Engine,
    *,
    start: dt.datetime,
    end: dt.datetime,
    repository: str,
    require_human: bool = False,
    require_lint: bool = False,
    limit: int = 100,
) -> tuple[PullRequestActivity, ...]:
    """List PRs with joined human-review and lint activity in a bounded window."""
    if not 1 <= limit <= MAX_ACTIVITY_RESULTS:
        raise ValueError(f"limit must be between 1 and {MAX_ACTIVITY_RESULTS}")
    require_complete_sync(engine, repository)
    with engine.begin() as conn:
        human_counts = (
            sqlalchemy.select(
                review_events.c.repository,
                review_events.c.pr_number,
                sqlalchemy.func.count().label("human_events"),
            )
            .where(
                review_events.c.is_human.is_(True),
                review_events.c.activity_at >= start,
                review_events.c.activity_at < end,
            )
            .group_by(review_events.c.repository, review_events.c.pr_number)
            .subquery()
        )
        invocation_counts = (
            sqlalchemy.select(
                lint_invocations.c.repository,
                lint_invocations.c.pr_number,
                sqlalchemy.func.count().label("lint_runs"),
            )
            .where(
                lint_invocations.c.repository == repository,
                lint_invocations.c.successful.is_(True),
                lint_invocations.c.pr_number.is_not(None),
                lint_invocations.c.ts >= start,
                lint_invocations.c.ts < end,
            )
            .group_by(lint_invocations.c.repository, lint_invocations.c.pr_number)
            .subquery()
        )
        human_count = sqlalchemy.func.coalesce(human_counts.c.human_events, 0)
        invocation_count = sqlalchemy.func.coalesce(invocation_counts.c.lint_runs, 0)
        statement = (
            sqlalchemy.select(
                pull_requests,
                human_count.label("human_events"),
                invocation_count.label("lint_runs"),
            )
            .outerjoin(
                human_counts,
                (human_counts.c.repository == pull_requests.c.repository)
                & (human_counts.c.pr_number == pull_requests.c.pr_number),
            )
            .outerjoin(
                invocation_counts,
                (invocation_counts.c.repository == pull_requests.c.repository)
                & (invocation_counts.c.pr_number == pull_requests.c.pr_number),
            )
            .where(
                pull_requests.c.repository == repository,
                pull_requests.c.updated_at >= start,
                pull_requests.c.updated_at < end,
            )
        )
        if require_human:
            statement = statement.where(human_count > 0)
        if require_lint:
            statement = statement.where(invocation_count > 0)
        rows = conn.execute(statement.order_by(pull_requests.c.updated_at.desc()).limit(limit)).mappings().all()
        pr_numbers = [int(row["pr_number"]) for row in rows]
        findings_by_pr: dict[int, list[str]] = {pr_number: [] for pr_number in pr_numbers}
        if pr_numbers:
            finding_rows = conn.execute(
                sqlalchemy.select(lint_invocations.c.pr_number, lint_findings.c.code)
                .select_from(lint_findings.join(lint_invocations))
                .where(
                    lint_invocations.c.pr_number.in_(pr_numbers),
                    lint_invocations.c.repository == repository,
                    lint_invocations.c.successful.is_(True),
                    lint_invocations.c.ts >= start,
                    lint_invocations.c.ts < end,
                )
            )
            for pr_number, code in finding_rows:
                findings_by_pr[int(pr_number)].append(str(code))
    return tuple(
        PullRequestActivity(
            repository=str(row["repository"]),
            number=int(row["pr_number"]),
            url=str(row["record"]["url"]),
            title=str(row["record"]["title"]),
            author=str(row["author"]),
            state=str(row["state"]),
            updated_at=utc_iso(row["updated_at"]),
            head_sha=str(row["head_sha"]),
            human_events=int(row["human_events"]),
            lint_runs=int(row["lint_runs"]),
            lint_findings=len(findings_by_pr[int(row["pr_number"])]),
            rule_codes=tuple(sorted(set(findings_by_pr[int(row["pr_number"])]))),
        )
        for row in rows
    )


def list_pr_review_events(engine: Engine, repository: str, pr_number: int) -> tuple[ReviewEventSummary, ...]:
    """List current review events for one PR with its lint totals."""
    require_complete_sync(engine, repository)
    with engine.begin() as conn:
        lint_rows = _successful_lint_rows(conn, repository, pr_number=pr_number)
        rows = (
            conn.execute(
                sqlalchemy.select(review_events)
                .where(review_events.c.repository == repository, review_events.c.pr_number == pr_number)
                .order_by(review_events.c.activity_at)
            )
            .mappings()
            .all()
        )
    return tuple(
        ReviewEventSummary(
            event_id=str(row["event_id"]),
            kind=str(row["kind"]),
            database_id=int(row["database_id"]),
            author=str(row["author"]),
            is_human=bool(row["is_human"]),
            is_agent_marked=bool(row["record"]["is_agent_marked"]),
            body_preview=str(row["record"]["body"])[:300],
            activity_at=utc_iso(row["activity_at"]),
            source_url=row["record"].get("source_url"),
            path=row["record"].get("path"),
            line=row["record"].get("line") or row["record"].get("original_line"),
            thread_is_resolved=row["record"].get("thread_is_resolved"),
            lint_runs=len(lint_rows.invocations),
            lint_findings=len(lint_rows.findings),
        )
        for row in rows
    )


def lint_activity(engine: Engine, *, repository: str, start: dt.datetime, end: dt.datetime) -> LintActivity:
    """Return successful invocation and finding rows for a bounded analysis window."""
    require_complete_sync(engine, repository)
    with engine.begin() as conn:
        rows = _successful_lint_rows(conn, repository, start=start, end=end)
    return LintActivity(
        invocations=tuple(LintInvocationRecord.model_validate(row) for row in rows.invocations),
        findings=tuple(LintFindingRecord.model_validate(row) for row in rows.findings),
    )


def review_context(engine: Engine, event_id: str) -> ReviewContext:
    """Return all stored evidence needed to judge one review event."""
    with engine.begin() as conn:
        event_row = (
            conn.execute(sqlalchemy.select(review_events).where(review_events.c.event_id == event_id)).mappings().one()
        )
        event = ReviewEventRecord.model_validate(event_row["record"])
    require_complete_sync(engine, event.repository)
    with engine.begin() as conn:
        pull_row = (
            conn.execute(
                sqlalchemy.select(pull_requests).where(
                    pull_requests.c.repository == event.repository, pull_requests.c.pr_number == event.pr_number
                )
            )
            .mappings()
            .one()
        )
        thread_rows = []
        if event.thread_id:
            thread_rows = (
                conn.execute(
                    sqlalchemy.select(review_events.c.record)
                    .where(
                        review_events.c.repository == event.repository,
                        review_events.c.pr_number == event.pr_number,
                    )
                    .order_by(review_events.c.activity_at)
                )
                .scalars()
                .all()
            )
            thread_rows = [row for row in thread_rows if row.get("thread_id") == event.thread_id]
        diff = conn.execute(
            sqlalchemy.select(pull_request_diffs.c.diff).where(
                pull_request_diffs.c.repository == event.repository,
                pull_request_diffs.c.pr_number == event.pr_number,
                pull_request_diffs.c.head_sha == pull_row["head_sha"],
            )
        ).scalar_one_or_none()
        source_row = (
            conn.execute(
                sqlalchemy.select(source_contexts)
                .where(source_contexts.c.event_id == event_id)
                .order_by(source_contexts.c.fetched_at.desc())
                .limit(1)
            )
            .mappings()
            .first()
        )
        lint_rows = _successful_lint_rows(conn, event.repository, pr_number=event.pr_number)
    identity = {
        "event": event.model_dump(mode="json"),
        "thread": thread_rows,
        "pull_request": pull_row["record"],
        "diff_sha": hashlib.sha256((diff or "").encode()).hexdigest(),
        "source_sha": None if source_row is None else source_row["content_sha"],
        "lint_invocations": lint_rows.invocations,
        "lint_findings": lint_rows.findings,
    }
    return ReviewContext(
        event=event,
        thread=tuple(ReviewEventRecord.model_validate(row) for row in thread_rows),
        pull_request=PullRequestRecord.model_validate(pull_row["record"]),
        diff=diff,
        source=None if source_row is None else source_row["text"],
        source_start_line=None if source_row is None else source_row["start_line"],
        source_end_line=None if source_row is None else source_row["end_line"],
        source_unavailable_reason=None if source_row is None else source_row["unavailable_reason"],
        lint_invocations=tuple(LintInvocationRecord.model_validate(row) for row in lint_rows.invocations),
        lint_findings=tuple(LintFindingRecord.model_validate(row) for row in lint_rows.findings),
        context_sha=record_sha(identity),
    )


def store_source_context(
    engine: Engine,
    *,
    event_id: str,
    commit_sha: str,
    path: str,
    anchor_line: int,
    start_line: int,
    end_line: int,
    text: str | None,
    unavailable_reason: str | None,
    fetched_at: dt.datetime,
) -> None:
    identity = text if text is not None else f"unavailable:{unavailable_reason or 'unknown'}"
    with engine.begin() as conn:
        _upsert_many(
            conn,
            source_contexts,
            (
                {
                    "event_id": event_id,
                    "commit_sha": commit_sha,
                    "path": path,
                    "anchor_line": anchor_line,
                    "start_line": start_line,
                    "end_line": end_line,
                    "text": text,
                    "unavailable_reason": unavailable_reason,
                    "content_sha": hashlib.sha256(identity.encode()).hexdigest(),
                    "fetched_at": fetched_at,
                },
            ),
            ("event_id", "commit_sha"),
        )


def stored_probe(engine: Engine, idempotency_key: str) -> StoredProbe | None:
    with engine.begin() as conn:
        row = conn.execute(
            sqlalchemy.select(rule_probes.c.record).where(rule_probes.c.idempotency_key == idempotency_key)
        ).scalar_one_or_none()
    return None if row is None else StoredProbe.model_validate(row)


def store_probe(engine: Engine, record: StoredProbe) -> None:
    payload = _payload(record)
    with engine.begin() as conn:
        _insert_ignore_many(
            conn,
            rule_probes,
            (
                {
                    "probe_id": record.probe_id,
                    "idempotency_key": record.idempotency_key,
                    "created_at": _timestamp(record.created_at),
                    "event_id": record.event_id,
                    "context_sha": record.context_sha,
                    "rule_code": record.rule_code,
                    "rule_sha": record.rule_sha,
                    "catalog_sha": record.catalog_sha,
                    "model": record.model,
                    "effort": record.effort,
                    "status": record.status.value,
                    "fired": record.fired,
                    "confidence": record.confidence,
                    "finding": record.finding,
                    "raw_output": record.raw_output,
                    "error": record.error,
                    "elapsed": record.elapsed,
                    "record": payload,
                },
            ),
            ("idempotency_key",),
        )
