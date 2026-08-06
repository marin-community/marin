# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3

from grafana_role_migration import upgrade_sqlite


def test_upgrade_sqlite_updates_only_viewer_memberships(tmp_path):
    database_path = tmp_path / "grafana.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE org_user (user_id INTEGER, org_id INTEGER, role TEXT, updated DATETIME)")
        connection.executemany(
            "INSERT INTO org_user VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
            [
                (1, 1, "Viewer"),
                (2, 1, "Editor"),
                (3, 1, "Admin"),
                (4, 2, "Viewer"),
                (5, 1, "None"),
            ],
        )

    assert upgrade_sqlite(database_path) == 2

    with sqlite3.connect(database_path) as connection:
        memberships = connection.execute("SELECT user_id, org_id, role FROM org_user ORDER BY user_id").fetchall()
    assert memberships == [
        (1, 1, "Editor"),
        (2, 1, "Editor"),
        (3, 1, "Admin"),
        (4, 2, "Editor"),
        (5, 1, "None"),
    ]


def test_upgrade_sqlite_allows_fresh_database(tmp_path):
    assert upgrade_sqlite(tmp_path / "grafana.db") == 0
