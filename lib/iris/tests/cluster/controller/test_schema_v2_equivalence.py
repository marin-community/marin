# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structural equivalence between the migration-built DB and ``schema_v2`` metadata.

We do not require byte-exact DDL match — SA quotes differently and lays
column constraints in a different order. Instead we walk SQLite's PRAGMA
output (``table_info``, ``index_list``, ``index_xinfo``,
``foreign_key_list``) on both DBs and compare structural fingerprints.
"""

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.schema_v2 import auth_metadata, metadata

EXCLUDED_TABLES = {"schema_migrations"}

TYPE_AFFINITY: dict[str, str] = {
    "TEXT": "TEXT",
    "VARCHAR": "TEXT",
    "CHAR": "TEXT",
    "CLOB": "TEXT",
    "INTEGER": "INTEGER",
    "INT": "INTEGER",
    "BIGINT": "INTEGER",
    "TINYINT": "INTEGER",
    "BLOB": "BLOB",
    "REAL": "REAL",
    "FLOAT": "REAL",
    "DOUBLE": "REAL",
    "NUMERIC": "NUMERIC",
}


def _affinity(declared: str) -> str:
    """Map a SQLite declared type to its five storage affinities."""
    declared = declared.upper().strip()
    if declared in TYPE_AFFINITY:
        return TYPE_AFFINITY[declared]
    # SQLite affinity rules: substring match (simplified, sufficient here).
    if "INT" in declared:
        return "INTEGER"
    if any(token in declared for token in ("CHAR", "CLOB", "TEXT")):
        return "TEXT"
    if "BLOB" in declared or declared == "":
        return "BLOB"
    if any(token in declared for token in ("REAL", "FLOA", "DOUB")):
        return "REAL"
    return "NUMERIC"


def _normalize_default(value: Any) -> Any:
    """Normalize SQLite's stored default-value representation.

    SQLAlchemy renders ``server_default="0"`` as the SQL literal ``'0'``
    (quoted string), while hand-rolled DDL writes the bare token ``0``.
    SA renders ``server_default="''"`` as ``''''''`` — a quoted form of
    the two-character literal ``''``. Strip surrounding quotes repeatedly
    so both representations collapse to the same Python string.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    while len(value) >= 2 and value[0] == "'" and value[-1] == "'":
        value = value[1:-1].replace("''", "'")
    return value


def _table_fingerprint(conn: sqlite3.Connection, schema: str, table: str) -> dict[str, Any]:
    """Pull a structural fingerprint for a single table from a SQLite connection."""
    columns: list[dict[str, Any]] = []
    for _cid, name, decl_type, notnull, dflt, pk in conn.execute(f'PRAGMA "{schema}".table_info("{table}")'):
        columns.append(
            {
                "name": name,
                "affinity": _affinity(decl_type),
                "notnull": bool(notnull),
                "default": _normalize_default(dflt),
                "pk": int(pk),
            }
        )

    indexes: list[dict[str, Any]] = []
    for _seq, idx_name, unique, _origin, partial in conn.execute(f'PRAGMA "{schema}".index_list("{table}")'):
        # Skip auto-indexes created by sqlite for PRIMARY KEY / UNIQUE — they
        # are reflected by the column/UNIQUE-constraint comparison already.
        if idx_name.startswith("sqlite_autoindex_"):
            continue
        index_columns: list[tuple[str | None, int]] = []
        for _seqno, _cid, col_name, desc, _coll, key in conn.execute(f'PRAGMA "{schema}".index_xinfo("{idx_name}")'):
            if key:
                index_columns.append((col_name, int(desc)))
        sql_row = conn.execute(
            f"SELECT sql FROM \"{schema}\".sqlite_master WHERE type='index' AND name=?",
            (idx_name,),
        ).fetchone()
        where_clause = _extract_where(sql_row[0] if sql_row else None)
        indexes.append(
            {
                "name": idx_name,
                "unique": bool(unique),
                "partial": bool(partial),
                "columns": tuple(index_columns),
                "where": where_clause,
            }
        )
    indexes.sort(key=lambda i: i["name"])

    fks: list[dict[str, Any]] = []
    for _id, _seq, table_ref, from_col, to_col, on_update, on_delete, _match in conn.execute(
        f'PRAGMA "{schema}".foreign_key_list("{table}")'
    ):
        fks.append(
            {
                "from": from_col,
                "to_table": table_ref,
                "to_col": to_col,
                "on_update": on_update,
                "on_delete": on_delete,
            }
        )
    fks.sort(key=lambda f: (f["from"], f["to_table"], f["to_col"] or ""))

    return {"columns": columns, "indexes": indexes, "fks": fks}


def _extract_where(sql: str | None) -> str | None:
    """Pull the partial-index WHERE predicate from a CREATE INDEX statement."""
    if sql is None:
        return None
    upper = sql.upper()
    idx = upper.rfind(" WHERE ")
    if idx == -1:
        return None
    return _normalize_predicate(sql[idx + len(" WHERE ") :])


def _normalize_predicate(predicate: str) -> str:
    return " ".join(predicate.strip().split())


def _all_table_names(meta: sa.MetaData) -> list[str]:
    return sorted(t.name for t in meta.tables.values() if t.name not in EXCLUDED_TABLES)


@pytest.fixture(scope="module")
def canonical_conn(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[sqlite3.Connection, str, str]]:
    """Spin up a real ``ControllerDB`` and yield its underlying connection."""
    db_dir: Path = tmp_path_factory.mktemp("controller_canonical")
    db = ControllerDB(db_dir)
    yield db._conn, "main", "auth"
    db.close()


@pytest.fixture(scope="module")
def candidate_conn() -> Iterator[tuple[sqlite3.Connection, str, str]]:
    """Build an in-memory SQLite, ``create_all`` schema_v2 + auth metadata."""
    conn = sqlite3.connect(":memory:")
    main_engine = sa.create_engine(
        "sqlite://",
        creator=lambda: conn,
        poolclass=sa.pool.StaticPool,
    )
    metadata.create_all(main_engine)
    # Attach a second in-memory DB and create the auth tables there so both
    # main- and auth-schema lookups go through one connection.
    conn.execute("ATTACH DATABASE ':memory:' AS auth")
    auth_engine = sa.create_engine(
        "sqlite://",
        creator=lambda: conn,
        poolclass=sa.pool.StaticPool,
    )
    # SA's ``create_all`` defaults to the default schema. Override per-table
    # by emitting CreateTable / CreateIndex with the ``auth`` schema set.
    for tbl in auth_metadata.sorted_tables:
        tbl.schema = "auth"
    auth_metadata.create_all(auth_engine)
    yield conn, "main", "auth"
    conn.close()


MAIN_TABLES = _all_table_names(metadata)
AUTH_TABLES = _all_table_names(auth_metadata)


@pytest.mark.parametrize("table", MAIN_TABLES)
def test_main_table_equivalence(
    table: str,
    canonical_conn: tuple[sqlite3.Connection, str, str],
    candidate_conn: tuple[sqlite3.Connection, str, str],
) -> None:
    canon_conn, canon_schema, _ = canonical_conn
    cand_conn, cand_schema, _ = candidate_conn
    expected = _table_fingerprint(canon_conn, canon_schema, table)
    actual = _table_fingerprint(cand_conn, cand_schema, table)
    _assert_table_equivalent(table, expected, actual)


@pytest.mark.parametrize("table", AUTH_TABLES)
def test_auth_table_equivalence(
    table: str,
    canonical_conn: tuple[sqlite3.Connection, str, str],
    candidate_conn: tuple[sqlite3.Connection, str, str],
) -> None:
    canon_conn, _, canon_schema = canonical_conn
    cand_conn, _, cand_schema = candidate_conn
    expected = _table_fingerprint(canon_conn, canon_schema, table)
    actual = _table_fingerprint(cand_conn, cand_schema, table)
    _assert_table_equivalent(table, expected, actual)


def _assert_table_equivalent(table: str, expected: dict[str, Any], actual: dict[str, Any]) -> None:
    # Columns: ordered comparison so column position drift is caught.
    assert _column_signatures(expected["columns"]) == _column_signatures(actual["columns"]), f"column drift in {table!r}"

    # Indexes: keyed by name; compare full structure (columns, partial-where,
    # uniqueness).
    expected_idx = {i["name"]: i for i in expected["indexes"]}
    actual_idx = {i["name"]: i for i in actual["indexes"]}
    assert (
        expected_idx.keys() == actual_idx.keys()
    ), f"index set drift in {table!r}: expected={sorted(expected_idx)} actual={sorted(actual_idx)}"
    for name in expected_idx:
        e = expected_idx[name]
        a = actual_idx[name]
        assert e["unique"] == a["unique"], f"index {name!r}: unique mismatch"
        assert e["columns"] == a["columns"], f"index {name!r}: column list mismatch"
        assert e["where"] == a["where"], f"index {name!r}: partial WHERE mismatch"

    # Foreign keys: compared on (from, to_table, to_col, on_delete). SA does
    # not always emit ON UPDATE so we don't compare that column.
    def fk_sig(fk: dict[str, Any]) -> tuple[Any, ...]:
        return (fk["from"], fk["to_table"], fk["to_col"], fk["on_delete"] or "NO ACTION")

    expected_fks = sorted(map(fk_sig, expected["fks"]))
    actual_fks = sorted(map(fk_sig, actual["fks"]))
    assert expected_fks == actual_fks, f"FK drift in {table!r}"


def _column_signatures(columns: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    # SQLite reports PRIMARY KEY columns of non-WITHOUT-ROWID tables with
    # ``notnull = 0`` even though they are implicitly NOT NULL. SQLAlchemy's
    # ``create_all`` emits the explicit NOT NULL token, so the two raw
    # representations differ on that bit alone — collapse it by forcing
    # ``notnull`` to True whenever the column is part of the primary key.
    return [(c["name"], c["affinity"], c["notnull"] or c["pk"] > 0, c["default"], c["pk"]) for c in columns]
