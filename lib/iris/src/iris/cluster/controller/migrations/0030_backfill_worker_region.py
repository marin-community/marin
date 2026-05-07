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


def migrate(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT w.worker_id, w.scale_group FROM workers w "
        "WHERE w.active=1 AND w.scale_group != '' "
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
