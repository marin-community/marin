# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent PostgreSQL store for GitHub review and lint activity."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

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

from .github_review_corpus import PullRequestBundle, ReviewEventRecord

DEFAULT_CLOUDSQL_CONNECTION = "hai-gcp-models:us-central1:marin-metadata"
DEFAULT_DATABASE = "context"
DEFAULT_DATABASE_USER = "loom-vm@hai-gcp-models.iam"
DEFAULT_BACKFILL_DAYS = 30

metadata = MetaData()
json_type = JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")


class StoreModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SyncStatus(StrEnum):
    RUNNING = "running"
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
    Column("candidate_pull_requests", Integer),
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
    Column("ts", DateTime(timezone=True), nullable=False),
    Column("pr_number", Integer),
    Column("head_sha", Text),
    Column("catalog_sha", Text),
    Column("successful", Boolean, nullable=False),
    Column("finding_count", Integer, nullable=False),
    Column("record", json_type, nullable=False),
    Index("ix_codehealth_lint_invocations_pr_ts", "pr_number", "ts"),
)

lint_findings = Table(
    "codehealth_lint_findings",
    metadata,
    Column("finding_id", Text, primary_key=True),
    Column("invocation_id", Text, ForeignKey("codehealth_lint_invocations.invocation_id", ondelete="CASCADE")),
    Column("ts", DateTime(timezone=True), nullable=False),
    Column("pr_number", Integer),
    Column("code", Text, nullable=False),
    Column("record", json_type, nullable=False),
    Index("ix_codehealth_lint_findings_pr_code", "pr_number", "code"),
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
    Column("fired", Boolean, nullable=False),
    Column("confidence", Float),
    Column("finding", Text),
    Column("raw_output", Text, nullable=False),
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


class LintActivity(StoreModel):
    invocations: tuple[dict[str, object], ...]
    findings: tuple[dict[str, object], ...]


class ReviewContext(StoreModel):
    event: ReviewEventRecord
    thread: tuple[ReviewEventRecord, ...]
    pull_request: dict[str, object]
    diff: str | None
    source: str | None
    source_start_line: int | None
    source_end_line: int | None
    source_unavailable_reason: str | None
    lint_invocations: tuple[dict[str, object], ...]
    lint_findings: tuple[dict[str, object], ...]
    context_sha: str


@dataclass(frozen=True)
class SyncRun:
    sync_id: str
    repository: str
    window_start: dt.datetime
    window_end: dt.datetime


def create_engine_from_environment() -> tuple[Engine, Connector]:
    """Connect to Marin's existing metadata database with IAM authentication."""
    connector = Connector(refresh_strategy="lazy")
    instance = os.environ.get("CLOUDSQL_CONNECTION", DEFAULT_CLOUDSQL_CONNECTION)
    database = os.environ.get("PGDATABASE", DEFAULT_DATABASE)
    user = os.environ.get("PGUSER", DEFAULT_DATABASE_USER)
    engine = sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=lambda: connector.connect(instance, "pg8000", user=user, db=database, enable_iam_auth=True),
        pool_pre_ping=True,
    )
    return engine, connector


def create_schema(engine: Engine) -> None:
    """Create the codehealth tables in a test database."""
    metadata.create_all(engine)


def _dialect_insert(conn: Connection, table: Table):
    if conn.dialect.name == "postgresql":
        return pg_insert(table)
    if conn.dialect.name == "sqlite":
        return sqlite_insert(table)
    raise ValueError(f"unsupported review-store database dialect: {conn.dialect.name}")


def _upsert(conn: Connection, table: Table, values: dict[str, object], keys: Sequence[str]) -> None:
    statement = _dialect_insert(conn, table).values(values)
    updates = {column.name: statement.excluded[column.name] for column in table.columns if column.name not in keys}
    conn.execute(statement.on_conflict_do_update(index_elements=[table.c[key] for key in keys], set_=updates))


def _insert_ignore(conn: Connection, table: Table, values: dict[str, object], keys: Sequence[str]) -> None:
    statement = _dialect_insert(conn, table).values(values)
    conn.execute(statement.on_conflict_do_nothing(index_elements=[table.c[key] for key in keys]))


def _payload(record: BaseModel | dict[str, object]) -> dict[str, object]:
    return record.model_dump(mode="json") if isinstance(record, BaseModel) else record


def _record_sha(record: BaseModel | dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(_payload(record), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
    """Resume the newest incomplete window, or create a fixed new window."""
    with engine.begin() as conn:
        row = (
            conn.execute(
                sqlalchemy.select(sync_runs)
                .where(sync_runs.c.repository == repository, sync_runs.c.status != SyncStatus.COMPLETE.value)
                .order_by(sync_runs.c.started_at.desc())
                .limit(1)
            )
            .mappings()
            .first()
        )
        if row is not None:
            conn.execute(
                sqlalchemy.update(sync_runs)
                .where(sync_runs.c.sync_id == row["sync_id"])
                .values(status=SyncStatus.RUNNING.value, error=None)
            )
            return SyncRun(str(row["sync_id"]), repository, row["window_start"], row["window_end"])
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
            )
        )
        return SyncRun(sync_id, repository, start, end)


def completed_pull_requests(engine: Engine, sync_id: str) -> set[int]:
    with engine.begin() as conn:
        return set(
            conn.execute(
                sqlalchemy.select(sync_pull_requests.c.pr_number).where(sync_pull_requests.c.sync_id == sync_id)
            ).scalars()
        )


def store_bundle(engine: Engine, sync_id: str, bundle: PullRequestBundle, *, observed_at: dt.datetime) -> None:
    """Commit one reconciled PR bundle and its resume checkpoint atomically."""
    pull = bundle.pull_request
    repository = pull.repository
    pull_payload = _payload(pull)
    pull_sha = _record_sha(pull)
    with engine.begin() as conn:
        _insert_ignore(
            conn,
            pull_request_versions,
            {
                "repository": repository,
                "pr_number": pull.number,
                "record_sha": pull_sha,
                "record": pull_payload,
                "observed_at": observed_at,
            },
            ("repository", "pr_number", "record_sha"),
        )
        _upsert(
            conn,
            pull_requests,
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
            },
            ("repository", "pr_number"),
        )
        for child_table in (review_threads, changed_files, commits):
            conn.execute(
                child_table.delete().where(
                    child_table.c.repository == repository,
                    child_table.c.pr_number == pull.number,
                )
            )
        for event in bundle.events:
            payload = _payload(event)
            record_sha = _record_sha(event)
            _insert_ignore(
                conn,
                review_event_versions,
                {"event_id": event.event_id, "record_sha": record_sha, "record": payload, "observed_at": observed_at},
                ("event_id", "record_sha"),
            )
            _upsert(
                conn,
                review_events,
                {
                    "event_id": event.event_id,
                    "repository": repository,
                    "pr_number": event.pr_number,
                    "kind": event.kind,
                    "database_id": event.database_id,
                    "author": event.author,
                    "is_human": event.is_human,
                    "activity_at": _event_activity(event),
                    "record_sha": record_sha,
                    "record": payload,
                    "observed_at": observed_at,
                },
                ("event_id",),
            )
        for thread in bundle.threads:
            _upsert(
                conn,
                review_threads,
                {
                    "repository": repository,
                    "thread_id": thread.thread_id,
                    "pr_number": thread.pr_number,
                    "record_sha": _record_sha(thread),
                    "record": _payload(thread),
                    "observed_at": observed_at,
                },
                ("repository", "thread_id"),
            )
        for file in bundle.files:
            _upsert(
                conn,
                changed_files,
                {
                    "repository": repository,
                    "pr_number": file.pr_number,
                    "filename": file.filename,
                    "record": _payload(file),
                    "observed_at": observed_at,
                },
                ("repository", "pr_number", "filename"),
            )
        for position, commit in enumerate(bundle.commits):
            _upsert(
                conn,
                commits,
                {
                    "repository": repository,
                    "pr_number": commit.pr_number,
                    "sha": commit.sha,
                    "position": position,
                    "record": _payload(commit),
                    "observed_at": observed_at,
                },
                ("repository", "pr_number", "sha"),
            )
        _upsert(
            conn,
            pull_request_diffs,
            {
                "repository": repository,
                "pr_number": pull.number,
                "head_sha": pull.head_sha,
                "diff": bundle.diff,
                "observed_at": observed_at,
            },
            ("repository", "pr_number", "head_sha"),
        )
        _insert_ignore(
            conn,
            sync_pull_requests,
            {"sync_id": sync_id, "pr_number": pull.number, "synced_at": observed_at},
            ("sync_id", "pr_number"),
        )


def store_telemetry(
    engine: Engine,
    invocations: Sequence[dict[str, object]],
    findings: Sequence[dict[str, object]],
) -> None:
    """Mirror bounded Finelog activity for single-store exploration queries."""
    with engine.begin() as conn:
        for row in invocations:
            invocation_id = str(row["invocation_id"])
            _upsert(
                conn,
                lint_invocations,
                {
                    "invocation_id": invocation_id,
                    "ts": _timestamp(str(row["ts"])),
                    "pr_number": row.get("pr_number"),
                    "head_sha": row.get("head_sha"),
                    "catalog_sha": row.get("lint_catalog_sha"),
                    "successful": (
                        row.get("agent_exit_code") is not None
                        and int(row["agent_exit_code"]) == 0
                        and not bool(row.get("timed_out"))
                    ),
                    "finding_count": int(row.get("finding_count") or 0),
                    "record": row,
                },
                ("invocation_id",),
            )
        for row in findings:
            finding_id = _record_sha(row)
            _upsert(
                conn,
                lint_findings,
                {
                    "finding_id": finding_id,
                    "invocation_id": str(row["invocation_id"]),
                    "ts": _timestamp(str(row["ts"])),
                    "pr_number": row.get("pr_number"),
                    "code": str(row.get("code") or ""),
                    "record": row,
                },
                ("finding_id",),
            )


def complete_sync(
    engine: Engine,
    sync_id: str,
    *,
    candidate_pull_requests: int,
    github_usage: dict[str, object],
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
                github_usage=github_usage,
                finelog_watermark=finelog_watermark,
                error=None,
            )
        )


def fail_sync(engine: Engine, sync_id: str, error: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            sqlalchemy.update(sync_runs)
            .where(sync_runs.c.sync_id == sync_id)
            .values(status=SyncStatus.FAILED.value, error=error[:4000])
        )


def _iso(value: dt.datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.UTC)
    return value.astimezone(dt.UTC).isoformat().replace("+00:00", "Z")


def list_pull_request_activity(
    engine: Engine,
    *,
    start: dt.datetime,
    end: dt.datetime,
    repository: str | None = None,
    require_human: bool = False,
    require_lint: bool = False,
    limit: int = 100,
) -> tuple[PullRequestActivity, ...]:
    """List PRs with joined human-review and lint activity in a bounded window."""
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")
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
                lint_invocations.c.pr_number,
                sqlalchemy.func.count().label("lint_runs"),
            )
            .where(
                lint_invocations.c.pr_number.is_not(None),
                lint_invocations.c.ts >= start,
                lint_invocations.c.ts < end,
            )
            .group_by(lint_invocations.c.pr_number)
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
            .outerjoin(invocation_counts, invocation_counts.c.pr_number == pull_requests.c.pr_number)
            .where(pull_requests.c.updated_at >= start, pull_requests.c.updated_at < end)
        )
        if repository is not None:
            statement = statement.where(pull_requests.c.repository == repository)
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
            updated_at=_iso(row["updated_at"]),
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
    with engine.begin() as conn:
        invocations = (
            conn.execute(
                sqlalchemy.select(lint_invocations.c.invocation_id).where(lint_invocations.c.pr_number == pr_number)
            )
            .scalars()
            .all()
        )
        finding_count = conn.execute(
            sqlalchemy.select(sqlalchemy.func.count())
            .select_from(lint_findings)
            .where(lint_findings.c.invocation_id.in_(invocations) if invocations else sqlalchemy.false())
        ).scalar_one()
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
            activity_at=_iso(row["activity_at"]),
            source_url=row["record"].get("source_url"),
            path=row["record"].get("path"),
            line=row["record"].get("line") or row["record"].get("original_line"),
            thread_is_resolved=row["record"].get("thread_is_resolved"),
            lint_runs=len(invocations),
            lint_findings=int(finding_count),
        )
        for row in rows
    )


def lint_activity(engine: Engine, *, start: dt.datetime, end: dt.datetime) -> LintActivity:
    """Return successful invocation and finding rows for a bounded analysis window."""
    with engine.begin() as conn:
        invocation_rows = (
            conn.execute(
                sqlalchemy.select(lint_invocations.c.record)
                .where(
                    lint_invocations.c.successful.is_(True),
                    lint_invocations.c.ts >= start,
                    lint_invocations.c.ts < end,
                )
                .order_by(lint_invocations.c.ts)
            )
            .scalars()
            .all()
        )
        invocation_ids = [str(row["invocation_id"]) for row in invocation_rows]
        finding_rows = (
            conn.execute(
                sqlalchemy.select(lint_findings.c.record)
                .where(lint_findings.c.invocation_id.in_(invocation_ids) if invocation_ids else sqlalchemy.false())
                .order_by(lint_findings.c.ts)
            )
            .scalars()
            .all()
        )
    return LintActivity(invocations=tuple(invocation_rows), findings=tuple(finding_rows))


def review_context(engine: Engine, event_id: str) -> ReviewContext:
    """Return all stored evidence needed to judge one review event."""
    with engine.begin() as conn:
        event_row = (
            conn.execute(sqlalchemy.select(review_events).where(review_events.c.event_id == event_id)).mappings().one()
        )
        event = ReviewEventRecord.model_validate(event_row["record"])
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
        invocation_rows = (
            conn.execute(
                sqlalchemy.select(lint_invocations.c.record)
                .where(lint_invocations.c.pr_number == event.pr_number)
                .order_by(lint_invocations.c.ts)
            )
            .scalars()
            .all()
        )
        invocation_ids = [str(row["invocation_id"]) for row in invocation_rows]
        finding_rows = (
            conn.execute(
                sqlalchemy.select(lint_findings.c.record)
                .where(lint_findings.c.invocation_id.in_(invocation_ids) if invocation_ids else sqlalchemy.false())
                .order_by(lint_findings.c.ts)
            )
            .scalars()
            .all()
        )
    identity = {
        "event": event.model_dump(mode="json"),
        "thread": thread_rows,
        "pull_request": pull_row["record"],
        "diff_sha": hashlib.sha256((diff or "").encode()).hexdigest(),
        "source_sha": None if source_row is None else source_row["content_sha"],
        "lint_invocations": invocation_rows,
        "lint_findings": finding_rows,
    }
    return ReviewContext(
        event=event,
        thread=tuple(ReviewEventRecord.model_validate(row) for row in thread_rows),
        pull_request=pull_row["record"],
        diff=diff,
        source=None if source_row is None else source_row["text"],
        source_start_line=None if source_row is None else source_row["start_line"],
        source_end_line=None if source_row is None else source_row["end_line"],
        source_unavailable_reason=None if source_row is None else source_row["unavailable_reason"],
        lint_invocations=tuple(invocation_rows),
        lint_findings=tuple(finding_rows),
        context_sha=hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
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
        _upsert(
            conn,
            source_contexts,
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
            ("event_id", "commit_sha"),
        )


def stored_probe(engine: Engine, idempotency_key: str) -> dict[str, object] | None:
    with engine.begin() as conn:
        row = conn.execute(
            sqlalchemy.select(rule_probes.c.record).where(rule_probes.c.idempotency_key == idempotency_key)
        ).scalar_one_or_none()
    return row


def store_probe(engine: Engine, record: dict[str, object]) -> None:
    with engine.begin() as conn:
        _insert_ignore(
            conn,
            rule_probes,
            {
                "probe_id": record["probe_id"],
                "idempotency_key": record["idempotency_key"],
                "created_at": _timestamp(str(record["created_at"])),
                "event_id": record["event_id"],
                "context_sha": record["context_sha"],
                "rule_code": record["rule_code"],
                "rule_sha": record["rule_sha"],
                "catalog_sha": record["catalog_sha"],
                "model": record["model"],
                "effort": record["effort"],
                "fired": record["fired"],
                "confidence": record.get("confidence"),
                "finding": record.get("finding"),
                "raw_output": record["raw_output"],
                "elapsed": record["elapsed"],
                "record": record,
            },
            ("idempotency_key",),
        )
