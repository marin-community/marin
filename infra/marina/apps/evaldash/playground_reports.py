# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Playground reports: model-feedback reports saved by the marin-serve dashboard, with triage state.

The serve dashboard's Report issue button writes one immutable JSON object per report under
``{prefix}/{YYYY-MM-DD}/{id}.json`` (see ``marin.inference.model_feedback``). The scheduled ingest job
copies objects it has not seen into ``playground_reports``. Triage edits live in a separate table, so
reading a report again never overwrites them.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

import sqlalchemy
from pydantic import BaseModel, ConfigDict, ValidationError
from rigging.filesystem.factory import open_url, url_to_fs
from sqlalchemy import Boolean, CheckConstraint, Column, DateTime, ForeignKey, Index, Table, Text
from sqlalchemy.engine import Engine

from .results_db import _dialect_insert, json_type, metadata

logger = logging.getLogger(__name__)

DEFAULT_REPORT_PREFIXES = (
    "gs://marin-eval-metadata/model-feedback",
    "s3://marin-us-east-02a/marin/model-feedback",
)
MAX_REPORT_SUMMARIES = 1000


class ReportStatus(StrEnum):
    NEW = "new"
    TRIAGED = "triaged"
    FILED = "filed"
    WONT_FIX = "wont_fix"


reports = Table(
    "playground_reports",
    metadata,
    Column("report_id", Text, primary_key=True),
    Column("source_url", Text, nullable=False, unique=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("model", Text, nullable=False),
    Column("reporter", Text, nullable=False),
    Column("description", Text, nullable=False),
    Column("has_conversation", Boolean, nullable=False),
    Column("record", json_type, nullable=False),
    Index("playground_reports_created_at", "created_at"),
)

report_triage = Table(
    "playground_report_triage",
    metadata,
    Column(
        "report_id",
        Text,
        ForeignKey("playground_reports.report_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("status", Text, nullable=False),
    Column("notes", Text, nullable=False),
    Column("github_issue", Text),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("updated_by", Text),
    CheckConstraint(
        f"status IN ({', '.join(repr(status.value) for status in ReportStatus)})",
        name="playground_report_triage_status",
    ),
)


class FeedbackReport(BaseModel):
    """What the reporter entered, plus the conversation when they included it."""

    model_config = ConfigDict(extra="allow")
    description: str
    reporter: str = ""
    model: str
    conversation: dict[str, object] | None


class FeedbackRecord(BaseModel):
    """Version 1 of the object the serve dashboard writes for each report."""

    model_config = ConfigDict(extra="allow")
    version: Literal[1]
    id: str
    created_at: datetime
    serving: dict[str, object]
    report: FeedbackReport


class TriageUpdate(BaseModel):
    """A reviewer's edit to one report's triage state."""

    status: ReportStatus
    notes: str = ""
    github_issue: str | None = None


def list_report_paths(prefix: str) -> list[str]:
    """List ``{prefix}/{date}/*.json`` report objects; an absent prefix has none."""
    fs, root = url_to_fs(prefix)
    fs.invalidate_cache(root)
    try:
        days = [entry["name"] for entry in fs.ls(root, detail=True) if entry.get("type") == "directory"]
    except FileNotFoundError:
        return []
    paths: list[str] = []
    for day in sorted(days):
        fs.invalidate_cache(day)
        paths.extend(
            fs.unstrip_protocol(entry["name"])
            for entry in fs.ls(day, detail=True)
            if entry.get("type") == "file" and entry["name"].endswith(".json")
        )
    return paths


def report_row(url: str, record: FeedbackRecord) -> dict:
    """The ``playground_reports`` row for one parsed report object."""
    created_at = record.created_at if record.created_at.tzinfo else record.created_at.replace(tzinfo=UTC)
    return {
        "report_id": f"{created_at:%Y-%m-%d}-{record.id}",
        "source_url": url,
        "created_at": created_at,
        "model": record.report.model,
        "reporter": record.report.reporter.strip(),
        "description": record.report.description,
        "has_conversation": record.report.conversation is not None,
        "record": record.model_dump(mode="json"),
    }


def _read_report(url: str) -> FeedbackRecord | None:
    with open_url(url, "r") as handle:
        body = handle.read()
    try:
        return FeedbackRecord.model_validate_json(body)
    except ValidationError:
        # One malformed object must not block the others; it is retried, and logged, each pass.
        logger.warning("skipping unparseable playground report at %s", url, exc_info=True)
        return None


def ingest_reports(engine: Engine, prefixes: tuple[str, ...]) -> tuple[str, ...]:
    """Copy unseen report objects into the catalog; return the prefixes whose listing failed."""
    with engine.begin() as conn:
        known = set(conn.execute(sqlalchemy.select(reports.c.source_url)).scalars())
    failed: list[str] = []
    for prefix in prefixes:
        try:
            paths = list_report_paths(prefix)
        except Exception:
            logger.exception("playground reports: listing %s failed", prefix)
            failed.append(prefix)
            continue
        rows = []
        for url in paths:
            if url in known:
                continue
            record = _read_report(url)
            if record is not None:
                rows.append(report_row(url, record))
        if rows:
            with engine.begin() as conn:
                insert = _dialect_insert(conn, reports)
                conn.execute(insert.values(rows).on_conflict_do_nothing())
        logger.info("playground reports: %d objects, %d new under %s", len(paths), len(rows), prefix)
    return tuple(failed)


def _triage_fields(row: sqlalchemy.RowMapping) -> dict:
    return {
        "status": row["status"] or ReportStatus.NEW.value,
        "notes": row["notes"] or "",
        "github_issue": row["github_issue"],
        "triage_updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        "triage_updated_by": row["updated_by"],
    }


def _joined(*columns) -> sqlalchemy.Select:
    triage = (
        report_triage.c.status,
        report_triage.c.notes,
        report_triage.c.github_issue,
        report_triage.c.updated_at,
        report_triage.c.updated_by,
    )
    return sqlalchemy.select(*columns, *triage).select_from(
        reports.outerjoin(report_triage, reports.c.report_id == report_triage.c.report_id)
    )


def fetch_report_summaries(engine: Engine, limit: int = MAX_REPORT_SUMMARIES) -> list[dict]:
    """The newest reports without their conversations, each with its triage state."""
    stmt = _joined(
        reports.c.report_id,
        reports.c.created_at,
        reports.c.model,
        reports.c.reporter,
        reports.c.description,
        reports.c.has_conversation,
    ).order_by(reports.c.created_at.desc(), reports.c.report_id.desc())
    with engine.begin() as conn:
        rows = conn.execute(stmt.limit(limit)).mappings().all()
    return [
        {
            "report_id": row["report_id"],
            "created_at": _utc_iso(row["created_at"]),
            "model": row["model"],
            "reporter": row["reporter"],
            "description": row["description"],
            "has_conversation": row["has_conversation"],
            **_triage_fields(row),
        }
        for row in rows
    ]


def fetch_report(engine: Engine, report_id: str) -> dict | None:
    """One report's stored object, source, and triage state; None for an unknown ID."""
    stmt = _joined(reports.c.report_id, reports.c.source_url, reports.c.record).where(reports.c.report_id == report_id)
    with engine.begin() as conn:
        row = conn.execute(stmt).mappings().one_or_none()
    if row is None:
        return None
    return {"report_id": row["report_id"], "source_url": row["source_url"], **row["record"], **_triage_fields(row)}


def update_triage(engine: Engine, report_id: str, update: TriageUpdate, updated_by: str | None) -> dict | None:
    """Replace one report's triage state; None when the report is unknown."""
    values = {
        "report_id": report_id,
        "status": update.status.value,
        "notes": update.notes,
        "github_issue": (update.github_issue or "").strip() or None,
        "updated_at": datetime.now(UTC),
        "updated_by": updated_by,
    }
    with engine.begin() as conn:
        exists = conn.execute(sqlalchemy.select(reports.c.report_id).where(reports.c.report_id == report_id)).first()
        if exists is None:
            return None
        insert = _dialect_insert(conn, report_triage)
        changed = {name: insert.excluded[name] for name in values if name != "report_id"}
        conn.execute(
            insert.values(values).on_conflict_do_update(index_elements=[report_triage.c.report_id], set_=changed)
        )
    return fetch_report(engine, report_id)


def _utc_iso(value: datetime) -> str:
    return (value if value.tzinfo else value.replace(tzinfo=UTC)).astimezone(UTC).isoformat()
