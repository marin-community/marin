# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Named-cluster token store for Iris CLI.

Persists per-cluster credentials in ``~/.iris/tokens.json`` so the CLI can
authenticate against multiple controllers without re-logging in each time.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse


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
    data = _load_store(path)
    data["clusters"][cluster_name] = asdict(ClusterCredential(url=url, token=token))
    _save_store(path, data)


def load_token(
    cluster_name: str,
    *,
    store_path: Path | None = None,
) -> ClusterCredential | None:
    """Load a specific cluster's credential, or None if not found."""
    path = store_path or _default_store_path()
    data = _load_store(path)
    entry = data["clusters"].get(cluster_name)
    if entry is None:
        return None
    return ClusterCredential(url=entry["url"], token=entry["token"])


def load_any_token(*, store_path: Path | None = None) -> ClusterCredential | None:
    """Return the first available credential, preferring the "default" cluster."""
    path = store_path or _default_store_path()
    data = _load_store(path)
    clusters = data["clusters"]
    if not clusters:
        return None
    if "default" in clusters:
        entry = clusters["default"]
        return ClusterCredential(url=entry["url"], token=entry["token"])
    first = next(iter(clusters.values()))
    return ClusterCredential(url=first["url"], token=first["token"])


def _default_store_path() -> Path:
    return Path.home() / ".iris" / "tokens.json"


def _load_store(store_path: Path) -> dict:
    """Load the token store, migrating from the legacy single-token file if needed."""
    if not store_path.exists():
        _maybe_migrate_legacy(store_path)
    if store_path.exists():
        return json.loads(store_path.read_text())
    return {"clusters": {}}


def _save_store(store_path: Path, data: dict) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(data, indent=2) + "\n")
    os.chmod(store_path, 0o600)


def _maybe_migrate_legacy(store_path: Path) -> None:
    """Migrate ~/.iris/token (plain text) into the new JSON store under "default"."""
    legacy_path = store_path.parent / "token"
    if not legacy_path.exists():
        return
    token = legacy_path.read_text().strip()
    if not token:
        legacy_path.unlink()
        return
    data: dict = {"clusters": {"default": {"url": "", "token": token}}}
    _save_store(store_path, data)
    legacy_path.unlink()
