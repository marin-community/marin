# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Create the original EvalDash serving tables.

The production database predates structured migrations. ``checkfirst=True`` adopts those
tables without changing their rows; a new database receives the same schema.
"""

import sqlalchemy
from sqlalchemy.dialects.postgresql import JSONB


def upgrade(conn: sqlalchemy.Connection) -> None:
    metadata = sqlalchemy.MetaData()
    json_type = sqlalchemy.JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")
    eval_runs = sqlalchemy.Table(
        "eval_runs",
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
    sqlalchemy.Table(
        "eval_metrics",
        metadata,
        sqlalchemy.Column(
            "run_id", sqlalchemy.Text, sqlalchemy.ForeignKey("eval_runs.run_id", ondelete="CASCADE"), nullable=False
        ),
        sqlalchemy.Column("task", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("metric", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("value", sqlalchemy.Double, nullable=False),
        sqlalchemy.PrimaryKeyConstraint("run_id", "task", "metric"),
    )
    sqlalchemy.Table(
        "model_state",
        metadata,
        sqlalchemy.Column("model_name", sqlalchemy.Text, primary_key=True),
        sqlalchemy.Column("archived", sqlalchemy.Boolean, nullable=False),
        sqlalchemy.Column("updated_at", sqlalchemy.DateTime(timezone=True), nullable=False),
        sqlalchemy.Column("updated_by", sqlalchemy.Text),
    )
    metadata.create_all(conn, tables=[eval_runs, metadata.tables["eval_metrics"], metadata.tables["model_state"]])
