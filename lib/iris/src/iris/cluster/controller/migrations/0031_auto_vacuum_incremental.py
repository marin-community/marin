# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Switch the main DB to ``auto_vacuum=INCREMENTAL``.

SQLite only honors a change in auto_vacuum mode after a full VACUUM rewrites
the file. Once incremental mode is in effect, freed pages are tracked on the
freelist and reclaimed by ``PRAGMA incremental_vacuum`` -- which is cheap and
runs opportunistically before each TRUNCATE checkpoint.
"""

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    (current,) = conn.execute("PRAGMA main.auto_vacuum").fetchone()
    if current == 2:  # INCREMENTAL
        return
    # auto_vacuum cannot change inside a transaction, and the actual rewrite
    # only happens during VACUUM. VACUUM is also incompatible with an open
    # transaction.
    conn.commit()
    conn.execute("PRAGMA main.auto_vacuum = INCREMENTAL")
    conn.execute("VACUUM main")
