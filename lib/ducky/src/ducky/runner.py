# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The DuckDB query runner: one connection, one query at a time.

The runner runs the user's SQL exactly once via ``COPY (<sql>) TO '<parquet>'`` and
then reads the written parquet back for the row count and a capped preview. Running
once keeps non-deterministic SQL (``random()``, ``now()``, unordered ``LIMIT``)
consistent between the spilled file and the preview.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re

import duckdb
from iris.env_resources import TaskResources

from ducky.config import DuckyConfig

logger = logging.getLogger(__name__)

# query_id is interpolated into the result path, so it must be a bare uuid4 hex.
_QUERY_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class DuckyError(Exception):
    """Base for ducky errors surfaced to the dashboard as a clean message."""


class QueryError(DuckyError):
    """DuckDB failed to plan or execute the SQL. Wraps the DuckDB message.

    Cross-region reads are not a distinct error in v1 — they surface here as an
    httpfs authentication failure, because the injected creds are scoped to
    same-region buckets.
    """


@dataclasses.dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    preview_rows: list[list]
    total_rows: int
    truncated: bool
    result_path: str


@dataclasses.dataclass(frozen=True)
class DuckDBResourceSettings:
    """DuckDB execution limits derived from the host envelope."""

    threads: int
    memory_limit_bytes: int


def duckdb_resource_settings(resources: TaskResources, memory_fraction: float) -> DuckDBResourceSettings:
    """Map a host resource snapshot to DuckDB ``threads`` / ``memory_limit``.

    ``threads`` is the host CPU count (at least 1). ``memory_limit_bytes`` is
    ``memory_fraction`` of host RAM, leaving headroom for Python/Arrow/OS; it is 0
    (meaning "leave DuckDB's default") when host memory is unknown.
    """
    threads = max(1, int(resources.cpu_cores))
    memory_limit_bytes = int(resources.memory_bytes * memory_fraction) if resources.memory_bytes > 0 else 0
    return DuckDBResourceSettings(threads=threads, memory_limit_bytes=memory_limit_bytes)


def _sql_literal(value: str) -> str:
    """Quote a string as a SQL literal, escaping embedded single quotes."""
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _coerce_cell(value: object) -> object:
    """Coerce a DuckDB/Arrow cell to a JSON-serializable scalar.

    Native scalars pass through; everything else (timestamp, decimal, interval,
    blob, list, struct) becomes its string form for the preview.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


class QueryRunner:
    """Owns one embedded DuckDB connection for the process lifetime.

    Not safe for concurrent use — ``run_query`` is serialized by the caller
    (single query at a time, per design).
    """

    def __init__(self, config: DuckyConfig, resources: TaskResources | None = None) -> None:
        self._config = config
        self._con = duckdb.connect()
        settings = duckdb_resource_settings(resources or TaskResources.from_environment(), config.memory_fraction)
        self._con.execute(f"SET threads = {settings.threads}")
        if settings.memory_limit_bytes > 0:
            self._con.execute(f"SET memory_limit = '{settings.memory_limit_bytes}B'")
        logger.info("DuckDB configured: threads=%d memory_limit_bytes=%d", settings.threads, settings.memory_limit_bytes)
        self._install_secrets()
        # A local scratch dir (smoke deploy / tests) needs the ducky/ subdir to exist;
        # object stores create the prefix implicitly.
        if "://" not in config.scratch_bucket:
            os.makedirs(f"{config.scratch_bucket.rstrip('/')}/ducky", exist_ok=True)

    def _install_secrets(self) -> None:
        """Load httpfs and create a DuckDB SECRET for each configured object-store backend."""
        cfg = self._config
        self._con.execute("INSTALL httpfs")
        self._con.execute("LOAD httpfs")
        if cfg.gcs_enabled:
            self._con.execute(
                f"CREATE OR REPLACE SECRET ducky_gcs "
                f"(TYPE GCS, KEY_ID {_sql_literal(cfg.gcs_hmac_key_id)}, SECRET {_sql_literal(cfg.gcs_hmac_secret)})"
            )
        if cfg.r2_enabled:
            self._con.execute(
                f"CREATE OR REPLACE SECRET ducky_r2 "
                f"(TYPE R2, ACCOUNT_ID {_sql_literal(cfg.r2_account_id)}, "
                f"KEY_ID {_sql_literal(cfg.r2_access_key)}, SECRET {_sql_literal(cfg.r2_secret_key)})"
            )
        if cfg.cw_enabled:
            self._con.execute(
                f"CREATE OR REPLACE SECRET ducky_cw "
                f"(TYPE S3, ENDPOINT {_sql_literal(cfg.cw_endpoint)}, URL_STYLE 'path', "
                f"KEY_ID {_sql_literal(cfg.cw_access_key)}, SECRET {_sql_literal(cfg.cw_secret_key)})"
            )

    def run_query(self, sql: str, query_id: str) -> QueryResult:
        """Run ``sql`` once, spill the full result to parquet, and return a capped preview.

        ``query_id`` must be a bare uuid4 hex (the server supplies it); anything else
        raises ``ValueError`` to prevent path injection. DuckDB failures — including
        cross-region/auth errors from httpfs — raise :class:`QueryError`.
        """
        if not _QUERY_ID_RE.match(query_id):
            raise ValueError(f"query_id must be a uuid4 hex, got {query_id!r}")

        result_path = f"{self._config.scratch_bucket.rstrip('/')}/ducky/{query_id}.parquet"
        path_literal = _sql_literal(result_path)

        # hive_partitioning=false: the scratch path embeds a `tmp/ttl=Nd/` segment, which
        # DuckDB would otherwise read back as a phantom `ttl` partition column.
        readback = f"read_parquet({path_literal}, hive_partitioning=false)"
        try:
            self._con.execute(f"COPY ({sql}) TO {path_literal} (FORMAT parquet)")
            count_row = self._con.execute(f"SELECT count(*) FROM {readback}").fetchone()
            assert count_row is not None  # count(*) always returns exactly one row
            total_rows = int(count_row[0])
            preview = self._con.execute(
                f"SELECT * FROM {readback} LIMIT {self._config.preview_row_cap}"
            ).fetch_arrow_table()
        except duckdb.Error as e:
            raise QueryError(str(e)) from e

        columns = list(preview.column_names)
        preview_rows = [[_coerce_cell(row[col]) for col in columns] for row in preview.to_pylist()]
        return QueryResult(
            columns=columns,
            preview_rows=preview_rows,
            total_rows=total_rows,
            truncated=total_rows > len(preview_rows),
            result_path=result_path,
        )

    def close(self) -> None:
        self._con.close()
