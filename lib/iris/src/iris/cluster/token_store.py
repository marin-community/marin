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
    store_path: Path | None = None,
) -> None:
    """Upsert a cluster credential into the token store."""
    path = store_path or _default_store_path()
    with closing(_connect(path)) as conn:
        conn.execute(
            "INSERT INTO clusters (name, url, token) VALUES (?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET url = excluded.url, token = excluded.token",
            (cluster_name, url, token),
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
        row = conn.execute("SELECT url, token FROM clusters WHERE name = ?", (cluster_name,)).fetchone()
    return ClusterCredential(url=row[0], token=row[1]) if row else None


def load_any_token(*, store_path: Path | None = None) -> ClusterCredential | None:
    """Return the first available credential, preferring the "default" cluster."""
    path = store_path or _default_store_path()
    if not path.exists():
        return None
    with closing(_connect(path)) as conn:
        row = conn.execute(
            "SELECT url, token FROM clusters ORDER BY (name = ?) DESC, name LIMIT 1",
            (_DEFAULT_CLUSTER,),
        ).fetchone()
    return ClusterCredential(url=row[0], token=row[1]) if row else None


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
    return conn
