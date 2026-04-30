# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic table-by-table JSON dump of the controller DB.

Produces a stable mapping ``{table_name: [row_dicts]}`` with rows sorted
by primary key. This is the canonical "did the DB end up in the same
state?" check used by the replay golden tests; SQL traces are reviewed
informationally.

Excludes SQLite internal tables and ``schema_migrations`` (which is
populated unconditionally on ``apply_migrations`` and adds noise without
testing any controller behavior).
"""

import base64
from typing import Any

from iris.cluster.controller.db import ControllerDB

EXCLUDED_TABLES: frozenset[str] = frozenset({"schema_migrations"})
"""Tables ignored by ``deterministic_dump`` — schema bookkeeping only."""


def _encode(value: Any) -> Any:
    """Render a SQLite cell value as a JSON-serializable scalar.

    ``bytes`` columns (notably ``job_workdir_files.data``) become
    base64 ASCII so the dump round-trips through ``json.dumps``.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(value)).decode("ascii")
    return value


def _list_user_tables(db: ControllerDB) -> list[str]:
    with db.read_snapshot() as snap:
        rows = snap.fetchall(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    return [str(row["name"]) for row in rows if str(row["name"]) not in EXCLUDED_TABLES]


def _primary_key_columns(db: ControllerDB, table: str) -> list[str]:
    """Return the table's primary key columns in declared order.

    Falls back to *all* columns (also in declared order) when the table
    has no PK so dumps remain stable regardless of insert order.
    """
    with db.read_snapshot() as snap:
        rows = snap.fetchall(f"PRAGMA table_info({table})")
    pk_cols = sorted(
        ((int(row["pk"]), str(row["name"]), int(row["cid"])) for row in rows if int(row["pk"]) > 0),
        key=lambda triple: triple[0],
    )
    if pk_cols:
        return [name for _, name, _ in pk_cols]
    # No PK declared — sort by every column for determinism.
    return [str(row["name"]) for row in sorted(rows, key=lambda r: int(r["cid"]))]


def deterministic_dump(db: ControllerDB) -> dict[str, list[dict[str, Any]]]:
    """Dump every user table as ``{table: [row_dicts]}``.

    Rows are returned as ordinary dicts in column-declaration order and
    sorted by primary key (or every column when no PK exists). Bytes
    columns are base64-encoded. Used as the canonical state-equivalence
    check for replay scenarios.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for table in _list_user_tables(db):
        pk = _primary_key_columns(db, table)
        order = ", ".join(pk)
        with db.read_snapshot() as snap:
            rows = snap.fetchall(f"SELECT * FROM {table} ORDER BY {order}")
        out[table] = [{key: _encode(row[key]) for key in row.keys()} for row in rows]
    return out
