# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Store the released eval policy manifests and compliance files runs are admitted by."""

import sqlalchemy
from sqlalchemy.dialects.postgresql import JSONB


def upgrade(conn: sqlalchemy.Connection) -> None:
    metadata = sqlalchemy.MetaData()
    json_type = sqlalchemy.JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")
    admission = sqlalchemy.Table(
        "policy_admission",
        metadata,
        sqlalchemy.Column("singleton", sqlalchemy.Boolean, primary_key=True),
        sqlalchemy.Column("admission", json_type, nullable=False),
        sqlalchemy.Column("updated_at", sqlalchemy.DateTime(timezone=True), nullable=False),
        sqlalchemy.CheckConstraint("singleton", name="policy_admission_singleton"),
    )
    metadata.create_all(conn, tables=[admission])
