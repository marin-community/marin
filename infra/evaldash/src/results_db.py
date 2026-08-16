# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PostgreSQL serving state and object-source inventory for EvalDash.

The dashboard boots entirely from ``eval_catalog_runs``. Object storage remains the durable recovery
source: the background reconciler records every discovered object in ``eval_record_sources`` and
materializes the highest-priority valid source for each run into the serving tables.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import sqlalchemy
from marin.evaluation.records import EvalRunRecord
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Double,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Connection, Engine

from .db_migrations import apply_migrations

logger = logging.getLogger(__name__)

DEFAULT_DB_INSTANCE = "hai-gcp-models:us-central1:marin-metadata"
DEFAULT_DB_NAME = "evals"
DEFAULT_DB_USER = "evals"
DEFAULT_DB_PASSWORD_SECRET = "cloudsql-evals-password"
GCP_PROJECT = "hai-gcp-models"

metadata = MetaData()
json_type = JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")

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
    Column("record", json_type, nullable=False),
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

catalog_runs = Table(
    "eval_catalog_runs",
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
    Column("record", json_type, nullable=False),
)

catalog_metrics = Table(
    "eval_catalog_metrics",
    metadata,
    Column("run_id", Text, ForeignKey("eval_catalog_runs.run_id", ondelete="CASCADE"), nullable=False),
    Column("task", Text, nullable=False),
    Column("metric", Text, nullable=False),
    Column("value", Double, nullable=False),
    PrimaryKeyConstraint("run_id", "task", "metric"),
)

model_state = Table(
    "model_state",
    metadata,
    Column("model_name", Text, primary_key=True),
    Column("archived", Boolean, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("updated_by", Text),
)

record_sources = Table(
    "eval_record_sources",
    metadata,
    Column("path", Text, primary_key=True),
    Column("prefix", Text, ForeignKey("eval_record_prefixes.prefix"), nullable=False),
    Column("run_id", Text),
    Column("object_version", Text),
    Column("record", json_type),
    Column("last_verified_at", DateTime(timezone=True), nullable=False),
    Column("next_verify_at", DateTime(timezone=True), nullable=False),
    Column("missing_since", DateTime(timezone=True)),
    Column("error", Text),
    CheckConstraint(
        "record IS NULL OR run_id IS NOT NULL",
        name="eval_record_sources_record_identity",
    ),
    Index("ix_eval_record_sources_prefix", "prefix"),
    Index("ix_eval_record_sources_run", "run_id"),
)

record_prefixes = Table(
    "eval_record_prefixes",
    metadata,
    Column("prefix", Text, primary_key=True),
    Column("priority", Integer, nullable=False),
    Column("active", Boolean, nullable=False),
    Column("last_probe_at", DateTime(timezone=True)),
    Column("last_success_at", DateTime(timezone=True)),
    Column("record_count", Integer),
    Column("error", Text),
)

catalog_state = Table(
    "eval_catalog_state",
    metadata,
    Column("singleton", Boolean, primary_key=True),
    Column("generation", BigInteger, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    CheckConstraint("singleton", name="eval_catalog_state_singleton"),
)


@dataclass(frozen=True)
class DbConfig:
    """Connection parameters for the eval-metadata Cloud SQL instance."""

    instance: str
    db: str
    user: str
    password: str


@dataclass(frozen=True)
class SourceState:
    """Persisted validation state for one record object."""

    path: str
    run_id: str | None
    object_version: str | None
    last_verified_at: datetime
    next_verify_at: datetime
    missing_since: datetime | None
    error: str | None


@dataclass(frozen=True)
class RecordObservation:
    """The result of checking one new or due record object."""

    path: str
    object_version: str | None
    verified_at: datetime
    next_verify_at: datetime
    changed: bool
    missing: bool = False
    run_id: str | None = None
    record: EvalRunRecord | None = None
    error: str | None = None


@dataclass(frozen=True)
class CatalogSnapshot:
    """One committed generation of canonical serving records."""

    records: list[EvalRunRecord]
    generation: int
    updated_at: datetime


@dataclass(frozen=True)
class _CatalogSource:
    run_id: str | None
    missing_since: datetime | None


def connect_engine(instance: str, db: str, user: str, password: str) -> Engine:
    """Build a SQLAlchemy engine that dials Cloud SQL through the Python connector."""
    from google.cloud.sql.connector import Connector  # noqa: PLC0415

    connector = Connector()

    def getconn():
        return connector.connect(instance, "pg8000", user=user, password=password, db=db)

    return sqlalchemy.create_engine("postgresql+pg8000://", creator=getconn, pool_pre_ping=True)


def _secret_password(secret_id: str) -> str | None:
    """Fetch the latest version of ``secret_id``, or None if it cannot be read."""
    from google.cloud import secretmanager  # noqa: PLC0415

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
    """Resolve the EvalDash database connection, or None when no password is available."""
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
    """Upgrade the database to the schema required by this EvalDash build."""
    apply_migrations(engine)
    inspector = sqlalchemy.inspect(engine)
    for table in metadata.sorted_tables:
        found_columns = {column["name"] for column in inspector.get_columns(table.name)}
        required_columns = {column.name for column in table.columns}
        if missing := required_columns - found_columns:
            raise RuntimeError(f"EvalDash database table {table.name!r} is missing columns: {sorted(missing)}")
        found_primary_key = set(inspector.get_pk_constraint(table.name)["constrained_columns"] or [])
        required_primary_key = {column.name for column in table.primary_key.columns}
        if found_primary_key != required_primary_key:
            raise RuntimeError(
                f"EvalDash database table {table.name!r} has primary key {sorted(found_primary_key)}, "
                f"expected {sorted(required_primary_key)}"
            )


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _lock_catalog(conn: Connection) -> None:
    stmt = sqlalchemy.select(catalog_state.c.singleton).where(catalog_state.c.singleton.is_(True))
    if conn.dialect.name == "postgresql":
        stmt = stmt.with_for_update()
    conn.execute(stmt).one()


def _dialect_insert(conn: Connection, table: Table):
    if conn.dialect.name == "postgresql":
        return pg_insert(table)
    if conn.dialect.name == "sqlite":
        return sqlite_insert(table)
    raise ValueError(f"EvalDash does not support database dialect {conn.dialect.name!r}")


def _upsert(conn: Connection, table: Table, rows: list[dict], key: str) -> None:
    if not rows:
        return
    insert = _dialect_insert(conn, table)
    update = {column.name: insert.excluded[column.name] for column in table.columns if column.name != key}
    conn.execute(insert.on_conflict_do_update(index_elements=[table.c[key]], set_=update), rows)


def fetch_archived_models(engine: Engine) -> set[str]:
    """The model names hidden from the headline matrix by default."""
    stmt = sqlalchemy.select(model_state.c.model_name).where(model_state.c.archived.is_(True))
    with engine.begin() as conn:
        return {row[0] for row in conn.execute(stmt).all()}


def set_model_archived(engine: Engine, model_name: str, archived: bool, updated_by: str | None) -> None:
    """Upsert one model's archive flag."""
    values = {
        "model_name": model_name,
        "archived": archived,
        "updated_at": datetime.now(UTC),
        "updated_by": updated_by,
    }
    with engine.begin() as conn:
        _upsert(conn, model_state, [values], "model_name")


def run_row(record: EvalRunRecord) -> dict:
    """Flatten a record to the legacy and catalog serving columns."""
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
        "image_digest": record.provenance.eval_runtime,
        "error": record.error,
        "record": record.model_dump(mode="json", by_alias=True),
    }


def _metric_rows(record: EvalRunRecord) -> list[dict]:
    return [
        {"run_id": record.run_id, "task": task, "metric": metric, "value": float(value)}
        for task, metrics in record.metrics.items()
        for metric, value in metrics.items()
    ]


def upsert_legacy_record(engine: Engine, record: EvalRunRecord) -> None:
    """Model an old revision's write to the pre-catalog tables during migration tests."""
    with engine.begin() as conn:
        _upsert(conn, eval_runs, [run_row(record)], "run_id")
        conn.execute(eval_metrics.delete().where(eval_metrics.c.run_id == record.run_id))
        if metric_rows := _metric_rows(record):
            conn.execute(eval_metrics.insert(), metric_rows)


def fetch_snapshot(engine: Engine) -> CatalogSnapshot:
    """Load one committed serving generation from PostgreSQL."""
    with engine.begin() as conn:
        state_stmt = sqlalchemy.select(catalog_state.c.generation, catalog_state.c.updated_at)
        if conn.dialect.name == "postgresql":
            state_stmt = state_stmt.with_for_update(read=True)
        generation, updated_at = conn.execute(state_stmt).one()
        rows = conn.execute(sqlalchemy.select(catalog_runs.c.record).order_by(catalog_runs.c.created_at)).scalars().all()
    return CatalogSnapshot(
        records=[EvalRunRecord.model_validate(row) for row in rows],
        generation=generation,
        updated_at=updated_at,
    )


def source_states(engine: Engine, prefix: str) -> dict[str, SourceState]:
    """Return the lightweight inventory needed to decide which objects need a HEAD request."""
    stmt = sqlalchemy.select(
        record_sources.c.path,
        record_sources.c.run_id,
        record_sources.c.object_version,
        record_sources.c.last_verified_at,
        record_sources.c.next_verify_at,
        record_sources.c.missing_since,
        record_sources.c.error,
    ).where(record_sources.c.prefix == prefix)
    with engine.begin() as conn:
        rows = conn.execute(stmt).mappings().all()
    return {
        row["path"]: SourceState(
            path=row["path"],
            run_id=row["run_id"],
            object_version=row["object_version"],
            last_verified_at=row["last_verified_at"],
            next_verify_at=row["next_verify_at"],
            missing_since=row["missing_since"],
            error=row["error"],
        )
        for row in rows
    }


def catalog_generation(engine: Engine) -> int:
    """Return the current serving commit token without loading record JSON."""
    with engine.begin() as conn:
        return conn.execute(sqlalchemy.select(catalog_state.c.generation)).scalar_one()


def prefix_statuses(engine: Engine) -> list[dict]:
    """Return persisted prefix health for status-page bootstrap."""
    with engine.begin() as conn:
        return [dict(row) for row in conn.execute(sqlalchemy.select(record_prefixes)).mappings().all()]


def _prefix_row(
    prefix: str,
    priority: int,
    probe_at: datetime | None,
    *,
    success_at: datetime | None,
    record_count: int | None,
    error: str | None,
) -> dict:
    return {
        "prefix": prefix,
        "priority": priority,
        "active": True,
        "last_probe_at": probe_at,
        "last_success_at": success_at,
        "record_count": record_count,
        "error": error,
    }


def configure_prefixes(engine: Engine, prefixes: tuple[str, ...]) -> None:
    """Make the configured prefix order authoritative while retaining inactive source rows."""
    if not prefixes:
        return
    configured = {prefix: priority for priority, prefix in enumerate(prefixes)}
    with engine.begin() as conn:
        _lock_catalog(conn)
        current = {row["prefix"]: dict(row) for row in conn.execute(sqlalchemy.select(record_prefixes)).mappings().all()}
        changed_prefixes = {
            prefix
            for prefix, row in current.items()
            if row["active"] != (prefix in configured)
            or (prefix in configured and row["priority"] != configured[prefix])
        }
        changed_prefixes.update(prefix for prefix in configured if prefix not in current)
        affected = set()
        if changed_prefixes:
            affected.update(
                conn.execute(
                    sqlalchemy.select(record_sources.c.run_id).where(
                        record_sources.c.prefix.in_(changed_prefixes), record_sources.c.run_id.is_not(None)
                    )
                ).scalars()
            )

        for prefix, priority in configured.items():
            previous = current.get(prefix)
            reactivated = previous is not None and not previous["active"]
            row = _prefix_row(
                prefix,
                priority,
                previous["last_probe_at"] if previous else None,
                success_at=None if reactivated else previous["last_success_at"] if previous else None,
                record_count=None if reactivated else previous["record_count"] if previous else None,
                error=None if reactivated else previous["error"] if previous else None,
            )
            _upsert(conn, record_prefixes, [row], "prefix")
        retired = set(current) - set(configured)
        if retired:
            conn.execute(record_prefixes.update().where(record_prefixes.c.prefix.in_(retired)).values(active=False))
        if _inventory_complete(conn):
            _materialize_runs(conn, {run_id for run_id in affected if run_id is not None})


def mark_prefix_failed(engine: Engine, prefix: str, probe_at: datetime, error: str) -> None:
    """Persist a failed listing without changing its last successful inventory."""
    with engine.begin() as conn:
        conn.execute(
            record_prefixes.update()
            .where(
                record_prefixes.c.prefix == prefix,
                sqlalchemy.or_(
                    record_prefixes.c.last_success_at.is_(None),
                    record_prefixes.c.last_success_at < probe_at,
                ),
                sqlalchemy.or_(
                    record_prefixes.c.last_probe_at.is_(None),
                    record_prefixes.c.last_probe_at < probe_at,
                ),
            )
            .values(last_probe_at=probe_at, error=error)
        )


def _inventory_complete(conn: Connection) -> bool:
    missing = conn.execute(
        sqlalchemy.select(sqlalchemy.func.count())
        .select_from(record_prefixes)
        .where(record_prefixes.c.active.is_(True), record_prefixes.c.last_success_at.is_(None))
    ).scalar_one()
    return missing == 0


def _catalog_run_ids(conn: Connection, run_ids: set[str]) -> set[str]:
    if not run_ids:
        return set()
    return set(
        conn.execute(sqlalchemy.select(catalog_runs.c.run_id).where(catalog_runs.c.run_id.in_(run_ids))).scalars()
    )


def _materialize_runs(conn: Connection, run_ids: set[str]) -> bool:
    if not run_ids:
        return False
    active_sources = record_sources.join(record_prefixes, record_sources.c.prefix == record_prefixes.c.prefix)
    stmt = (
        sqlalchemy.select(
            record_sources.c.run_id,
            record_sources.c.record,
            record_prefixes.c.priority,
            record_sources.c.path,
        )
        .select_from(active_sources)
        .where(record_sources.c.run_id.in_(run_ids))
        .where(record_prefixes.c.active.is_(True))
        .order_by(record_prefixes.c.priority, record_sources.c.path)
    )
    winners: dict[str, EvalRunRecord] = {}
    tracked: set[str] = set()
    for row in conn.execute(stmt).mappings():
        run_id = row["run_id"]
        tracked.add(run_id)
        if row["record"] is None:
            continue
        if run_id not in winners:
            winners[run_id] = EvalRunRecord.model_validate(row["record"])

    changed = set(winners) | (run_ids - tracked)
    if not changed:
        return False
    conn.execute(catalog_metrics.delete().where(catalog_metrics.c.run_id.in_(changed)))
    conn.execute(catalog_runs.delete().where(catalog_runs.c.run_id.in_(changed)))
    if winners:
        conn.execute(catalog_runs.insert(), [run_row(record) for record in winners.values()])
        metrics = [row for record in winners.values() for row in _metric_rows(record)]
        if metrics:
            conn.execute(catalog_metrics.insert(), metrics)
    conn.execute(
        catalog_state.update()
        .where(catalog_state.c.singleton.is_(True))
        .values(generation=catalog_state.c.generation + 1, updated_at=sqlalchemy.func.now())
    )
    return True


def reconcile_prefix(
    engine: Engine,
    prefix: str,
    paths: list[str],
    observations: list[RecordObservation],
    probe_at: datetime,
    confirm_missing_after: float,
) -> None:
    """Atomically apply one complete successful prefix listing to source and serving state."""
    missing_observations = {observation.path: observation for observation in observations if observation.missing}
    path_set = set(paths) - set(missing_observations)
    with engine.begin() as conn:
        _lock_catalog(conn)
        inventory_was_complete = _inventory_complete(conn)
        previous_success = conn.execute(
            sqlalchemy.select(record_prefixes.c.last_success_at).where(record_prefixes.c.prefix == prefix)
        ).scalar_one_or_none()
        if previous_success is not None and _utc(previous_success) >= probe_at:
            return
        existing = {
            row.path: _CatalogSource(
                run_id=row.run_id,
                missing_since=row.missing_since,
            )
            for row in conn.execute(
                sqlalchemy.select(
                    record_sources.c.path,
                    record_sources.c.run_id,
                    record_sources.c.missing_since,
                ).where(record_sources.c.prefix == prefix)
            )
        }
        absent = set(existing) - path_set
        deleted = {
            path
            for path in absent
            if existing[path].missing_since is not None
            and probe_at >= _utc(existing[path].missing_since) + timedelta(seconds=confirm_missing_after)
        }
        first_missing = absent - deleted
        affected = {run_id for path in deleted if (run_id := existing[path].run_id) is not None}
        if deleted:
            conn.execute(record_sources.delete().where(record_sources.c.path.in_(deleted)))
        newly_missing = {path for path in first_missing if existing[path].missing_since is None}
        if newly_missing:
            conn.execute(
                record_sources.update().where(record_sources.c.path.in_(newly_missing)).values(missing_since=probe_at)
            )
        for path, observation in missing_observations.items():
            if path not in existing or path in deleted:
                continue
            conn.execute(
                record_sources.update()
                .where(record_sources.c.path == path)
                .values(
                    last_verified_at=observation.verified_at,
                    next_verify_at=observation.next_verify_at,
                    error=observation.error,
                )
            )
        unchanged = [observation for observation in observations if not observation.changed and not observation.missing]
        if unchanged:
            stmt = (
                record_sources.update()
                .where(record_sources.c.path == sqlalchemy.bindparam("observed_path"))
                .values(
                    object_version=sqlalchemy.bindparam("observed_version"),
                    last_verified_at=sqlalchemy.bindparam("verified_at"),
                    next_verify_at=sqlalchemy.bindparam("next_verify_at"),
                    missing_since=None,
                    error=sqlalchemy.bindparam("observed_error"),
                )
            )
            conn.execute(
                stmt,
                [
                    {
                        "observed_path": observation.path,
                        "observed_version": observation.object_version,
                        "verified_at": observation.verified_at,
                        "next_verify_at": observation.next_verify_at,
                        "observed_error": observation.error,
                    }
                    for observation in unchanged
                ],
            )

        valid = [observation for observation in observations if observation.changed and observation.record is not None]
        for observation in valid:
            old = existing.get(observation.path)
            old_run_id = old.run_id if old is not None else None
            if old_run_id is not None:
                affected.add(old_run_id)
            affected.add(observation.record.run_id)
        _upsert(
            conn,
            record_sources,
            [
                {
                    "path": observation.path,
                    "prefix": prefix,
                    "run_id": observation.record.run_id,
                    "object_version": observation.object_version,
                    "record": observation.record.model_dump(mode="json", by_alias=True),
                    "last_verified_at": observation.verified_at,
                    "next_verify_at": observation.next_verify_at,
                    "missing_since": None,
                    "error": None,
                }
                for observation in valid
                if observation.record is not None
            ],
            "path",
        )

        invalid = [
            observation
            for observation in observations
            if observation.changed and observation.record is None and not observation.missing
        ]
        for observation in invalid:
            values = {
                "object_version": observation.object_version,
                "last_verified_at": observation.verified_at,
                "next_verify_at": observation.next_verify_at,
                "missing_since": None,
                "error": observation.error,
            }
            if observation.path in existing:
                conn.execute(record_sources.update().where(record_sources.c.path == observation.path).values(**values))
                continue
            conn.execute(
                record_sources.insert().values(
                    path=observation.path,
                    prefix=prefix,
                    run_id=observation.run_id,
                    record=None,
                    **values,
                )
            )

        conn.execute(
            record_prefixes.update()
            .where(record_prefixes.c.prefix == prefix)
            .values(last_probe_at=probe_at, last_success_at=probe_at, record_count=len(paths), error=None)
        )
        inventory_is_complete = _inventory_complete(conn)
        if inventory_is_complete and not inventory_was_complete:
            affected.update(
                conn.execute(
                    sqlalchemy.select(record_sources.c.run_id).where(record_sources.c.run_id.is_not(None))
                ).scalars()
            )
        elif not inventory_is_complete:
            affected -= _catalog_run_ids(conn, affected)
        _materialize_runs(conn, affected)


def prune_untracked_records(engine: Engine, prefixes: tuple[str, ...]) -> bool:
    """Delete pre-inventory serving rows after every configured prefix has listed successfully."""
    if not prefixes:
        return False
    with engine.begin() as conn:
        _lock_catalog(conn)
        succeeded = set(
            conn.execute(
                sqlalchemy.select(record_prefixes.c.prefix).where(
                    record_prefixes.c.prefix.in_(prefixes),
                    record_prefixes.c.active.is_(True),
                    record_prefixes.c.last_success_at.is_not(None),
                )
            ).scalars()
        )
        if succeeded != set(prefixes):
            return False
        active_sources = record_sources.join(record_prefixes, record_sources.c.prefix == record_prefixes.c.prefix)
        tracked = (
            sqlalchemy.select(record_sources.c.run_id)
            .select_from(active_sources)
            .where(record_sources.c.run_id.is_not(None), record_prefixes.c.active.is_(True))
        )
        stale = set(
            conn.execute(sqlalchemy.select(catalog_runs.c.run_id).where(catalog_runs.c.run_id.not_in(tracked))).scalars()
        )
        if not stale:
            return False
        conn.execute(catalog_metrics.delete().where(catalog_metrics.c.run_id.in_(stale)))
        conn.execute(catalog_runs.delete().where(catalog_runs.c.run_id.in_(stale)))
        conn.execute(
            catalog_state.update()
            .where(catalog_state.c.singleton.is_(True))
            .values(generation=catalog_state.c.generation + 1, updated_at=sqlalchemy.func.now())
        )
        return True
