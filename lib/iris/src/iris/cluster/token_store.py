# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Named-cluster token store for Iris CLI.

Persists per-cluster credentials in ``~/.iris/tokens.sqlite`` so the CLI can
authenticate against multiple controllers without re-logging in each time.

SQLite gives atomic, concurrency-safe writes for free, so there is no
temp-file/rename dance and no corrupt-file tolerance to maintain: concurrent
CLI invocations and pytest-xdist workers serialize on the database lock.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

_DEFAULT_CLUSTER = "default"


@dataclass(frozen=True)
class ClusterCredential:
    url: str
    token: str
    # Long-lived OAuth refresh token for an IAP-fronted cluster, used to silently
    # re-mint the short-lived OIDC ID token. None for non-IAP clusters.
    iap_refresh_token: str | None = None


def cluster_name_from_url(url: str) -> str:
    """Derive a filesystem-safe cluster name from a URL.

    >>> cluster_name_from_url("http://localhost:10000")
    'localhost-10000'
    >>> cluster_name_from_url("http://controller.example.com:8080")
    'controller.example.com-8080'
    """
    parsed = urlparse(url)
    host = parsed.hostname or "unknown"
    if parsed.port:
        return f"{host}-{parsed.port}"
    return host


def store_token(
    cluster_name: str,
    url: str,
    token: str,
    *,
    iap_refresh_token: str | None = None,
    store_path: Path | None = None,
) -> None:
    """Upsert a cluster credential into the token store.

    A None ``iap_refresh_token`` preserves any existing one (so a later JWT-only
    refresh does not wipe the IAP refresh token), rather than clobbering it.
    """
    path = store_path or _default_store_path()
    with closing(_connect(path)) as conn:
        conn.execute(
            "INSERT INTO clusters (name, url, token, iap_refresh_token) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET url = excluded.url, token = excluded.token, "
            "iap_refresh_token = COALESCE(excluded.iap_refresh_token, clusters.iap_refresh_token)",
            (cluster_name, url, token, iap_refresh_token),
        )
        conn.commit()


def load_token(
    cluster_name: str,
    *,
    store_path: Path | None = None,
) -> ClusterCredential | None:
    """Load a specific cluster's credential, or None if not found."""
    path = store_path or _default_store_path()
    if not path.exists():
        return None
    with closing(_connect(path)) as conn:
        row = conn.execute(
            "SELECT url, token, iap_refresh_token FROM clusters WHERE name = ?", (cluster_name,)
        ).fetchone()
    return ClusterCredential(url=row[0], token=row[1], iap_refresh_token=row[2]) if row else None


def load_any_token(*, store_path: Path | None = None) -> ClusterCredential | None:
    """Return the first available credential, preferring the "default" cluster."""
    path = store_path or _default_store_path()
    if not path.exists():
        return None
    with closing(_connect(path)) as conn:
        row = conn.execute(
            "SELECT url, token, iap_refresh_token FROM clusters ORDER BY (name = ?) DESC, name LIMIT 1",
            (_DEFAULT_CLUSTER,),
        ).fetchone()
    return ClusterCredential(url=row[0], token=row[1], iap_refresh_token=row[2]) if row else None


def _default_store_path() -> Path:
    return Path.home() / ".iris" / "tokens.sqlite"


def _connect(store_path: Path) -> sqlite3.Connection:
    """Open the store, creating the file (mode 0600) and schema on first use."""
    store_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not store_path.exists()
    conn = sqlite3.connect(store_path)
    if is_new:
        os.chmod(store_path, 0o600)
    conn.execute("CREATE TABLE IF NOT EXISTS clusters (name TEXT PRIMARY KEY, url TEXT NOT NULL, token TEXT NOT NULL)")
    # Add the IAP refresh-token column to stores created before IAP support.
    # SQLite has no ADD COLUMN IF NOT EXISTS; the PRAGMA pre-check is a fast path,
    # and on the concurrent-writer race we re-read the schema (rather than match
    # on the error message, which is not a stable API) to confirm it now exists.
    if not _has_column(conn, "iap_refresh_token"):
        try:
            conn.execute("ALTER TABLE clusters ADD COLUMN iap_refresh_token TEXT")
        except sqlite3.OperationalError:
            if not _has_column(conn, "iap_refresh_token"):
                raise
    return conn


def _has_column(conn: sqlite3.Connection, column: str) -> bool:
    return any(row[1] == column for row in conn.execute("PRAGMA table_info(clusters)"))
