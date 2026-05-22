# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical local sample store. Append-only SQLite with WAL.

The SQLite file is the source of truth: every ProbeSample is committed here
before anything else. Finelog is best-effort secondary; if it's down we can
still reconstruct what happened from this file.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from probes.probe import ProbeSample

logger = logging.getLogger(__name__)


_ERROR_DETAIL_MAX_BYTES = 512

_SCHEMA = """
CREATE TABLE IF NOT EXISTS probe_samples (
    timestamp_us     INTEGER NOT NULL,
    probe_name       TEXT    NOT NULL,
    probe_kind       TEXT    NOT NULL,
    location         TEXT,
    outcome          TEXT    NOT NULL,
    latency_ms       INTEGER NOT NULL,
    error_class      TEXT,
    error_detail     TEXT,
    target_id        TEXT,
    extras_json      TEXT    NOT NULL DEFAULT '{}',
    daemon_instance  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS probe_samples_by_time
    ON probe_samples (timestamp_us);

CREATE INDEX IF NOT EXISTS probe_samples_by_probe_time
    ON probe_samples (probe_name, timestamp_us);
"""


class SqliteIntegrityError(RuntimeError):
    """Raised on startup if PRAGMA quick_check fails. The daemon exits 1."""


class SqliteSampleStore:
    """Append-only ProbeSample writer backed by SQLite.

    One row per commit (no batching) so that a crash mid-cycle loses at most
    the in-flight sample. WAL mode + synchronous=NORMAL is the standard
    durability/throughput compromise for this access pattern.
    """

    def __init__(self, path: Path):
        if not path.is_absolute():
            raise ValueError(f"sqlite_path must be absolute: {path}")
        if not path.parent.exists():
            raise ValueError(f"parent directory does not exist: {path.parent}")
        self._path = path
        self._conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._check_integrity()
        self._conn.executescript(_SCHEMA)

    def _check_integrity(self) -> None:
        row = self._conn.execute("PRAGMA quick_check").fetchone()
        if not row or row[0] != "ok":
            raise SqliteIntegrityError(f"PRAGMA quick_check failed for {self._path}: {row}")

    def write(self, sample: ProbeSample) -> None:
        detail = _truncate(sample.error_detail)
        self._conn.execute(
            """
            INSERT INTO probe_samples (
                timestamp_us, probe_name, probe_kind, location, outcome,
                latency_ms, error_class, error_detail, target_id,
                extras_json, daemon_instance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(sample.timestamp.timestamp() * 1_000_000),
                sample.probe_name,
                sample.probe_kind,
                sample.location,
                sample.outcome.value,
                sample.latency_ms,
                sample.error_class.value if sample.error_class is not None else None,
                detail,
                sample.target_id,
                sample.extras_json,
                sample.daemon_instance,
            ),
        )

    def count(self) -> int:
        """Test helper: total sample count."""
        return self._conn.execute("SELECT COUNT(*) FROM probe_samples").fetchone()[0]

    def disk_free_bytes(self) -> int:
        """Bytes free on the volume holding the database file. For heartbeat extras."""
        import shutil

        return shutil.disk_usage(self._path.parent).free

    def close(self) -> None:
        self._conn.close()


def _truncate(detail: str | None) -> str | None:
    if detail is None:
        return None
    encoded = detail.encode("utf-8", errors="replace")
    if len(encoded) <= _ERROR_DETAIL_MAX_BYTES:
        return detail
    return encoded[:_ERROR_DETAIL_MAX_BYTES].decode("utf-8", errors="replace")
