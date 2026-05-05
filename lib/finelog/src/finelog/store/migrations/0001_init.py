# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Baseline registry schema.

Creates ``namespaces`` (one row per registered namespace, holding its
schema JSON) and ``segments`` (one row per locally-tracked parquet file,
maintained in lockstep with the in-memory ``_local_segments`` deque).

Both statements use ``IF NOT EXISTS``: existing deployments may already
have ``namespaces`` from a release that pre-dates the migrations runner,
or ``segments`` from an early build of the catalog branch. In either case
this migration is a no-op against the DDL but records v1 in
``schema_migrations`` so future versions have a defined starting point.
"""

from __future__ import annotations

import duckdb


def migrate(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS namespaces (
            namespace        TEXT PRIMARY KEY,
            schema_json      TEXT NOT NULL,
            registered_at_ms BIGINT NOT NULL,
            last_modified_ms BIGINT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS segments (
            namespace     TEXT   NOT NULL,
            path          TEXT   NOT NULL,
            state         TEXT   NOT NULL,
            min_seq       BIGINT NOT NULL,
            max_seq       BIGINT NOT NULL,
            row_count     BIGINT NOT NULL,
            byte_size     BIGINT NOT NULL,
            created_at_ms BIGINT NOT NULL,
            PRIMARY KEY (namespace, path)
        )
        """
    )
