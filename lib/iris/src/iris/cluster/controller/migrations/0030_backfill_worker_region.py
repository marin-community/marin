# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3

# Marin cluster zones (from examples/marin.yaml). Not for upstream — local hotfix
# to backfill region/zone/scale-group attributes on workers registered between
# #4681 and #4720, which stopped publishing these keys.
ZONES = (
    "us-central1-a",
    "us-central2-b",
    "us-east1-b",
    "us-east1-d",
    "us-east5-a",
    "us-east5-b",
    "us-west1-a",
    "us-west4-a",
    "europe-west4-a",
    "europe-west4-b",
)


def _zone_of(scale_group: str) -> str | None:
    for z in ZONES:
        if scale_group.endswith("-" + z):
            return z
    return None


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return column in {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def migrate(conn: sqlite3.Connection) -> None:
    # ``active`` was a workers column at the time this migration was authored.
    # It is dropped in 0042; on fresh DBs the column is absent at this point.
    active_predicate = "w.active=1 AND " if _has_column(conn, "workers", "active") else ""
    rows = conn.execute(
        "SELECT w.worker_id, w.scale_group FROM workers w "
        f"WHERE {active_predicate}w.scale_group != '' "
        "AND NOT EXISTS ("
        "  SELECT 1 FROM worker_attributes wa "
        "  WHERE wa.worker_id = w.worker_id AND wa.key = 'region'"
        ")"
    ).fetchall()
    for worker_id, sg in rows:
        zone = _zone_of(sg)
        if zone is None:
            continue
        region = zone.rsplit("-", 1)[0]
        for key, val in (("region", region), ("zone", zone), ("scale-group", sg)):
            conn.execute(
                "INSERT OR IGNORE INTO worker_attributes"
                "(worker_id, key, value_type, str_value, int_value, float_value)"
                " VALUES (?, ?, 'str', ?, NULL, NULL)",
                (worker_id, key, val),
            )
