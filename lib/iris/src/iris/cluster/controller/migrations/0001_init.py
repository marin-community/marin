# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3

from iris.cluster.controller import schema
from sqlalchemy.dialects import sqlite as sqlite_dialect
from sqlalchemy.schema import CreateIndex, CreateTable


def migrate(conn: sqlite3.Connection) -> None:
    dialect = sqlite_dialect.dialect()
    for table in schema.metadata.sorted_tables:
        conn.execute(str(CreateTable(table, if_not_exists=True).compile(dialect=dialect)))
        for index in table.indexes:
            conn.execute(str(CreateIndex(index, if_not_exists=True).compile(dialect=dialect)))
