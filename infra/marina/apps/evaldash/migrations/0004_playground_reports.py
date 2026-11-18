# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Store marin-serve playground reports and their triage state."""

import sqlalchemy
from sqlalchemy.dialects.postgresql import JSONB

REPORT_STATUSES = ("new", "triaged", "filed", "wont_fix")


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
    triage = sqlalchemy.Table(
        "playground_report_triage",
        metadata,
        sqlalchemy.Column(
            "report_id",
            sqlalchemy.Text,
            sqlalchemy.ForeignKey("playground_reports.report_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sqlalchemy.Column("status", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("notes", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("github_issue", sqlalchemy.Text),
        sqlalchemy.Column("updated_at", sqlalchemy.DateTime(timezone=True), nullable=False),
        sqlalchemy.Column("updated_by", sqlalchemy.Text),
        sqlalchemy.CheckConstraint(
            f"status IN ({', '.join(repr(status) for status in REPORT_STATUSES)})",
            name="playground_report_triage_status",
        ),
    )
    metadata.create_all(conn, tables=[reports, triage])
