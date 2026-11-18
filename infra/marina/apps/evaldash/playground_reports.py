# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Playground reports: model-feedback reports saved by the marin-serve dashboard.

The serve dashboard's Report issue button writes one immutable JSON object per report under
``{prefix}/{YYYY-MM-DD}/{id}.json`` (see ``marin.inference.model_feedback``). The scheduled ingest job
copies objects it has not seen into ``playground_reports``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Literal

import sqlalchemy
from pydantic import BaseModel, ConfigDict, ValidationError
from rigging.filesystem.factory import open_url, url_to_fs
from sqlalchemy import Boolean, Column, DateTime, Index, Table, Text
from sqlalchemy.engine import Engine

from .results_db import _dialect_insert, json_type, metadata

logger = logging.getLogger(__name__)

DEFAULT_REPORT_PREFIXES = (
    "gs://marin-eval-metadata/model-feedback",
    "s3://marin-us-east-02a/marin/model-feedback",
)
MAX_REPORT_SUMMARIES = 1000


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


def fetch_report_summaries(engine: Engine, limit: int = MAX_REPORT_SUMMARIES) -> list[dict]:
    """The newest reports without their conversations."""
    stmt = sqlalchemy.select(
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
        }
        for row in rows
    ]


def fetch_report(engine: Engine, report_id: str) -> dict | None:
    """One report's stored object and source; None for an unknown ID."""
    stmt = sqlalchemy.select(reports.c.report_id, reports.c.source_url, reports.c.record).where(
        reports.c.report_id == report_id
    )
    with engine.begin() as conn:
        row = conn.execute(stmt).mappings().one_or_none()
    if row is None:
        return None
    return {"report_id": row["report_id"], "source_url": row["source_url"], **row["record"]}


def _utc_iso(value: datetime) -> str:
    return (value if value.tzinfo else value.replace(tzinfo=UTC)).astimezone(UTC).isoformat()
