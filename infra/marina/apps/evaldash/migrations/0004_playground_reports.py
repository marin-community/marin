# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Store marin-serve playground reports."""

import sqlalchemy
from sqlalchemy.dialects.postgresql import JSONB


def upgrade(conn: sqlalchemy.Connection) -> None:
    metadata = sqlalchemy.MetaData()
    json_type = sqlalchemy.JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")
    reports = sqlalchemy.Table(
        "playground_reports",
        metadata,
        sqlalchemy.Column("report_id", sqlalchemy.Text, primary_key=True),
        sqlalchemy.Column("source_url", sqlalchemy.Text, nullable=False, unique=True),
        sqlalchemy.Column("created_at", sqlalchemy.DateTime(timezone=True), nullable=False),
        sqlalchemy.Column("model", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("reporter", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("description", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("has_conversation", sqlalchemy.Boolean, nullable=False),
        sqlalchemy.Column("record", json_type, nullable=False),
        sqlalchemy.Index("playground_reports_created_at", "created_at"),
    )
    metadata.create_all(conn, tables=[reports])
