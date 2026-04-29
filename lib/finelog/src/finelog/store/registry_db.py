# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sidecar DuckDB database persisting the registered-schema table.

The :class:`RegistryDB` persists each namespace's registered schema in a
DuckDB file at ``{data_dir}/_finelog_registry.duckdb``. Every
``RegisterTable`` mutates one row here. On finelog startup the registry is
read out and ``LogNamespace`` instances are rehydrated from the rows.

The DB is intentionally separate from the per-namespace Parquet directories:
schema metadata never lives in two places, so additive evolution is one
``UPDATE``, not a smear across every parquet footer in the namespace.

This module only knows how to ``CREATE``/``UPSERT``/``SELECT`` rows. The
register-time semantics (additive merge, conflict detection, ordering-key
validation) live in :mod:`finelog.store.schema` and the routing/locking
lives in :class:`finelog.store.duckdb_store.DuckDBLogStore`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import duckdb

from finelog.store.schema import Schema, schema_from_json, schema_to_json

logger = logging.getLogger(__name__)

REGISTRY_DB_FILENAME = "_finelog_registry.duckdb"


class RegistryDB:
    """Thin wrapper around the sidecar DuckDB database.

    Not concurrency-safe by itself — the caller (``DuckDBLogStore``) holds
    the registry's insertion mutex around any mutating call.
    """

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / REGISTRY_DB_FILENAME
        self._conn = duckdb.connect(str(self._path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS namespaces (
                namespace        TEXT PRIMARY KEY,
                schema_json      TEXT NOT NULL,
                key_column       TEXT NOT NULL,
                registered_at_ms BIGINT NOT NULL,
                last_modified_ms BIGINT NOT NULL
            )
            """
        )

    def close(self) -> None:
        self._conn.close()

    def get(self, namespace: str) -> Schema | None:
        row = self._conn.execute("SELECT schema_json FROM namespaces WHERE namespace = ?", [namespace]).fetchone()
        if row is None:
            return None
        return schema_from_json(row[0])

    def list_all(self) -> dict[str, Schema]:
        rows = self._conn.execute("SELECT namespace, schema_json FROM namespaces").fetchall()
        return {name: schema_from_json(payload) for name, payload in rows}

    def delete(self, namespace: str) -> None:
        """Remove the row for ``namespace`` if it exists. Idempotent."""
        self._conn.execute("DELETE FROM namespaces WHERE namespace = ?", [namespace])

    def upsert(self, namespace: str, schema: Schema, key_column: str) -> None:
        """Insert or evolve the row for ``namespace``.

        ``last_modified_ms`` is bumped on every call; ``registered_at_ms``
        is set on first insert and preserved on update.
        """
        now_ms = int(time.time() * 1000)
        existing = self._conn.execute(
            "SELECT registered_at_ms FROM namespaces WHERE namespace = ?", [namespace]
        ).fetchone()
        registered_at = existing[0] if existing is not None else now_ms
        self._conn.execute(
            """
            INSERT INTO namespaces (namespace, schema_json, key_column, registered_at_ms, last_modified_ms)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (namespace) DO UPDATE
              SET schema_json = excluded.schema_json,
                  key_column = excluded.key_column,
                  last_modified_ms = excluded.last_modified_ms
            """,
            [namespace, schema_to_json(schema), key_column, registered_at, now_ms],
        )
