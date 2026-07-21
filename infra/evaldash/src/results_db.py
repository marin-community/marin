# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaldash's Postgres index of the canonical object-store eval records.

Each ``record.json`` (:mod:`marin.evaluation.records`) is upserted into a Cloud SQL Postgres instance
so the dashboard can filter and aggregate runs with SQL instead of scanning object storage. The record
is the source of truth; this table is a rebuildable index populated by evaldash's background
object-store ingestor, and the full record rides along in the ``record`` jsonb column so nothing is
lost.

This is an evaldash implementation detail. Evaluation launchers only write records to object storage;
the dashboard owns database configuration and periodically rebuilds this query index from those records.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime

import sqlalchemy
from marin.evaluation.records import EvalRunRecord
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Double,
    ForeignKey,
    MetaData,
    PrimaryKeyConstraint,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

DEFAULT_DB_INSTANCE = "hai-gcp-models:us-central1:marin-metadata"
DEFAULT_DB_NAME = "evals"
DEFAULT_DB_USER = "evals"
DEFAULT_DB_PASSWORD_SECRET = "cloudsql-evals-password"
GCP_PROJECT = "hai-gcp-models"

metadata = MetaData()

eval_runs = Table(
    "eval_runs",
    metadata,
    Column("run_id", Text, primary_key=True),
    Column("group_id", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("user_name", Text, nullable=False),
    Column("model_name", Text, nullable=False),
    Column("model_location", Text, nullable=False),
    Column("eval_name", Text, nullable=False),
    Column("mechanism", Text, nullable=False),
    Column("backend", Text, nullable=False),
    Column("platform", Text, nullable=False),
    Column("accelerator", Text, nullable=False),
    Column("region", Text),
    Column("status", Text, nullable=False),
    Column("results_path", Text),
    Column("git_sha", Text),
    Column("image_digest", Text),
    Column("error", Text),
    Column("record", JSONB, nullable=False),
)

eval_metrics = Table(
    "eval_metrics",
    metadata,
    Column("run_id", Text, ForeignKey("eval_runs.run_id", ondelete="CASCADE"), nullable=False),
    Column("task", Text, nullable=False),
    Column("metric", Text, nullable=False),
    Column("value", Double, nullable=False),
    PrimaryKeyConstraint("run_id", "task", "metric"),
)

# Per-model UI state, set from the dashboard and NEVER written by the record ingestor. ``eval_runs`` is
# a rebuildable index fully re-upserted every ingest pass, so mutable dashboard state (an archived
# model dropped from the headline matrix) lives in this side table keyed by model name instead.
model_state = Table(
    "model_state",
    metadata,
    Column("model_name", Text, primary_key=True),
    Column("archived", Boolean, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("updated_by", Text),
)


@dataclass(frozen=True)
class DbConfig:
    """Connection parameters for the eval-metadata Cloud SQL instance."""

    instance: str
    db: str
    user: str
    password: str


def connect_engine(instance: str, db: str, user: str, password: str) -> Engine:
    """Build a SQLAlchemy engine that dials the Cloud SQL instance through the Python connector.

    Uses the ``cloud-sql-python-connector`` + pg8000 creator pattern: every pooled connection is
    minted by the connector, which handles IAM-authenticated TLS to ``instance`` without a local proxy.
    """
    from google.cloud.sql.connector import Connector  # noqa: PLC0415  # lazy: keep import weight down

    connector = Connector()

    def getconn():
        return connector.connect(instance, "pg8000", user=user, password=password, db=db)

    return sqlalchemy.create_engine("postgresql+pg8000://", creator=getconn, pool_pre_ping=True)


def _secret_password(secret_id: str) -> str | None:
    """Fetch the latest version of ``secret_id`` from Secret Manager, or None if it cannot be read."""
    from google.cloud import secretmanager  # noqa: PLC0415  # lazy: keep import weight down

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{GCP_PROJECT}/secrets/{secret_id}/versions/latest"
    try:
        response = client.access_secret_version(name=name)
    except Exception:
        logger.warning(
            "could not read the eval-db password secret (EVAL_DB_PASSWORD_SECRET) from project %s",
            GCP_PROJECT,
            exc_info=True,
        )
        return None
    return response.payload.data.decode("utf-8")


def resolve_db_config() -> DbConfig | None:
    """Resolve the eval-DB connection from the environment, or None when no password is available.

    Reads ``EVAL_DB_INSTANCE``/``EVAL_DB_NAME``/``EVAL_DB_USER`` (with defaults). The password comes
    from ``EVAL_DB_PASSWORD`` if set, otherwise the latest version of the ``EVAL_DB_PASSWORD_SECRET``
    secret in Secret Manager. Returns None when neither yields a password; dashboard startup treats
    that as fatal.
    """
    instance = os.environ.get("EVAL_DB_INSTANCE", DEFAULT_DB_INSTANCE)
    db = os.environ.get("EVAL_DB_NAME", DEFAULT_DB_NAME)
    user = os.environ.get("EVAL_DB_USER", DEFAULT_DB_USER)
    password = os.environ.get("EVAL_DB_PASSWORD")
    if not password:
        secret_id = os.environ.get("EVAL_DB_PASSWORD_SECRET", DEFAULT_DB_PASSWORD_SECRET)
        password = _secret_password(secret_id)
    if not password:
        return None
    return DbConfig(instance=instance, db=db, user=user, password=password)


def ensure_schema(engine: Engine) -> None:
    """Create the ``eval_runs``, ``eval_metrics``, and ``model_state`` tables if they do not exist."""
    metadata.create_all(engine)


def fetch_archived_models(engine: Engine) -> set[str]:
    """The set of model names currently archived (hidden from the headline matrix by default)."""
    stmt = sqlalchemy.select(model_state.c.model_name).where(model_state.c.archived.is_(True))
    with engine.begin() as conn:
        return {row[0] for row in conn.execute(stmt).all()}


def set_model_archived(engine: Engine, model_name: str, archived: bool, updated_by: str | None) -> None:
    """Upsert one model's archive flag in the ``model_state`` side table."""
    values = {
        "model_name": model_name,
        "archived": archived,
        "updated_at": datetime.now(UTC),
        "updated_by": updated_by,
    }
    insert = pg_insert(model_state).values(**values)
    upsert = insert.on_conflict_do_update(
        index_elements=[model_state.c.model_name],
        set_={key: insert.excluded[key] for key in values if key != "model_name"},
    )
    with engine.begin() as conn:
        conn.execute(upsert)


def run_row(record: EvalRunRecord) -> dict:
    """Flatten a record to its ``eval_runs`` column values (with ``created_at`` as a datetime)."""
    return {
        "run_id": record.run_id,
        "group_id": record.group_id,
        "created_at": datetime.fromisoformat(record.created_at),
        "user_name": record.user,
        "model_name": record.model.name,
        "model_location": record.model.location,
        "eval_name": record.evaluation.name,
        "mechanism": record.evaluation.mechanism,
        "backend": record.model.backend,
        "platform": record.hardware.platform,
        "accelerator": record.hardware.accelerator,
        "region": record.hardware.region_or_cluster,
        "status": record.status.value,
        "results_path": record.results_path,
        "git_sha": record.provenance.git_sha,
        "image_digest": record.provenance.evalchemy_image,
        "error": record.error,
        "record": record.model_dump(mode="json", by_alias=True),
    }


def _metric_rows(record: EvalRunRecord) -> list[dict]:
    return [
        {"run_id": record.run_id, "task": task, "metric": metric, "value": float(value)}
        for task, metrics in record.metrics.items()
        for metric, value in metrics.items()
    ]


def upsert_record(engine: Engine, record: EvalRunRecord) -> None:
    """Upsert one run into ``eval_runs`` and replace its ``eval_metrics`` rows, in one transaction."""
    row = run_row(record)
    insert = pg_insert(eval_runs).values(**row)
    upsert = insert.on_conflict_do_update(
        index_elements=[eval_runs.c.run_id],
        set_={key: insert.excluded[key] for key in row if key != "run_id"},
    )
    metric_rows = _metric_rows(record)
    with engine.begin() as conn:
        conn.execute(upsert)
        conn.execute(eval_metrics.delete().where(eval_metrics.c.run_id == record.run_id))
        if metric_rows:
            conn.execute(eval_metrics.insert(), metric_rows)


def fetch_runs(
    engine: Engine,
    model: str | None = None,
    eval_name: str | None = None,
    user: str | None = None,
    status: str | None = None,
    group: str | None = None,
    limit: int = 200,
) -> list[dict]:
    """Return the most recent ``eval_runs`` rows matching the given filters, newest first."""
    stmt = sqlalchemy.select(eval_runs).order_by(eval_runs.c.created_at.desc()).limit(limit)
    for column, value in (
        (eval_runs.c.model_name, model),
        (eval_runs.c.eval_name, eval_name),
        (eval_runs.c.user_name, user),
        (eval_runs.c.status, status),
        (eval_runs.c.group_id, group),
    ):
        if value is not None:
            stmt = stmt.where(column == value)
    with engine.begin() as conn:
        rows = [dict(row) for row in conn.execute(stmt).mappings().all()]
    # timestamptz comes back as datetime; the callers (CLI table, dashboard JSON) want the
    # same ISO-8601 string the record carries.
    for row in rows:
        row["created_at"] = row["created_at"].isoformat()
    return rows
