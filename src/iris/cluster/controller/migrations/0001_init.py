# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3

from iris.cluster.controller.schema import MAIN_TABLES, generate_full_ddl


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(generate_full_ddl(MAIN_TABLES))
