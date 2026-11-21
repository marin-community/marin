# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist object discovery, validation, and precedence state."""

import sqlalchemy
from sqlalchemy.dialects.postgresql import JSONB


def upgrade(conn: sqlalchemy.Connection) -> None:
    metadata = sqlalchemy.MetaData()
    json_type = sqlalchemy.JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")
    legacy_runs = sqlalchemy.Table("eval_runs", metadata, autoload_with=conn)
    legacy_metrics = sqlalchemy.Table("eval_metrics", metadata, autoload_with=conn)
    catalog_runs = sqlalchemy.Table(
        "eval_catalog_runs",
        metadata,
        sqlalchemy.Column("run_id", sqlalchemy.Text, primary_key=True),
        sqlalchemy.Column("group_id", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("created_at", sqlalchemy.DateTime(timezone=True), nullable=False),
        sqlalchemy.Column("user_name", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("model_name", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("model_location", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("eval_name", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("mechanism", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("backend", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("platform", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("accelerator", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("region", sqlalchemy.Text),
        sqlalchemy.Column("status", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("results_path", sqlalchemy.Text),
        sqlalchemy.Column("git_sha", sqlalchemy.Text),
        sqlalchemy.Column("image_digest", sqlalchemy.Text),
        sqlalchemy.Column("error", sqlalchemy.Text),
        sqlalchemy.Column("record", json_type, nullable=False),
    )
    catalog_metrics = sqlalchemy.Table(
        "eval_catalog_metrics",
        metadata,
        sqlalchemy.Column(
            "run_id",
            sqlalchemy.Text,
            sqlalchemy.ForeignKey("eval_catalog_runs.run_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sqlalchemy.Column("task", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("metric", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("value", sqlalchemy.Double, nullable=False),
        sqlalchemy.PrimaryKeyConstraint("run_id", "task", "metric"),
    )
    sources = sqlalchemy.Table(
        "eval_record_sources",
        metadata,
        sqlalchemy.Column("path", sqlalchemy.Text, primary_key=True),
        sqlalchemy.Column(
            "prefix", sqlalchemy.Text, sqlalchemy.ForeignKey("eval_record_prefixes.prefix"), nullable=False
        ),
        sqlalchemy.Column("run_id", sqlalchemy.Text),
        sqlalchemy.Column("object_version", sqlalchemy.Text),
        sqlalchemy.Column("record", json_type),
        sqlalchemy.Column("last_verified_at", sqlalchemy.DateTime(timezone=True), nullable=False),
        sqlalchemy.Column("next_verify_at", sqlalchemy.DateTime(timezone=True), nullable=False),
        sqlalchemy.Column("missing_since", sqlalchemy.DateTime(timezone=True)),
        sqlalchemy.Column("error", sqlalchemy.Text),
        sqlalchemy.CheckConstraint(
            "record IS NULL OR run_id IS NOT NULL",
            name="eval_record_sources_record_identity",
        ),
    )
    sqlalchemy.Index("ix_eval_record_sources_prefix", sources.c.prefix)
    sqlalchemy.Index("ix_eval_record_sources_run", sources.c.run_id)
    prefixes = sqlalchemy.Table(
        "eval_record_prefixes",
        metadata,
        sqlalchemy.Column("prefix", sqlalchemy.Text, primary_key=True),
        sqlalchemy.Column("priority", sqlalchemy.Integer, nullable=False),
        sqlalchemy.Column("active", sqlalchemy.Boolean, nullable=False),
        sqlalchemy.Column("last_probe_at", sqlalchemy.DateTime(timezone=True)),
        sqlalchemy.Column("last_success_at", sqlalchemy.DateTime(timezone=True)),
        sqlalchemy.Column("record_count", sqlalchemy.Integer),
        sqlalchemy.Column("error", sqlalchemy.Text),
    )
    catalog = sqlalchemy.Table(
        "eval_catalog_state",
        metadata,
        sqlalchemy.Column("singleton", sqlalchemy.Boolean, primary_key=True),
        sqlalchemy.Column("generation", sqlalchemy.BigInteger, nullable=False),
        sqlalchemy.Column("updated_at", sqlalchemy.DateTime(timezone=True), nullable=False),
        sqlalchemy.CheckConstraint("singleton", name="eval_catalog_state_singleton"),
    )
    metadata.create_all(conn, tables=[catalog_runs, catalog_metrics, sources, prefixes, catalog])
    if conn.execute(sqlalchemy.select(sqlalchemy.func.count()).select_from(catalog_runs)).scalar_one() == 0:
        if run_rows := [dict(row) for row in conn.execute(sqlalchemy.select(legacy_runs)).mappings()]:
            conn.execute(catalog_runs.insert(), run_rows)
        if metric_rows := [dict(row) for row in conn.execute(sqlalchemy.select(legacy_metrics)).mappings()]:
            conn.execute(catalog_metrics.insert(), metric_rows)
    if conn.execute(sqlalchemy.select(catalog.c.singleton)).first() is None:
        conn.execute(catalog.insert().values(singleton=True, generation=0, updated_at=sqlalchemy.func.now()))
