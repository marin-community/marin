# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bundle ID utilities and a single sqlite-backed BundleStore implementation."""

from __future__ import annotations

import hashlib
import io
import logging
import posixpath
import sqlite3
import threading
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen

logger = logging.getLogger(__name__)


def bundle_id_for_zip(blob: bytes) -> str:
    """Return canonical bundle id for zip bytes."""
    return hashlib.sha256(blob).hexdigest()


def normalize_workdir_relative_path(path: str) -> str:
    """Return a normalized relative path safe to write under a task workdir."""
    candidate = path.replace("\\", "/")
    if candidate.startswith("/"):
        raise ValueError(f"Invalid workdir file path (absolute paths are not allowed): {path}")
    normalized = posixpath.normpath(candidate)
    if normalized in {"", "."}:
        raise ValueError(f"Invalid workdir file path: {path}")
    if normalized.startswith("../") or normalized == "..":
        raise ValueError(f"Invalid workdir file path (path traversal): {path}")
    return normalized


class BundleStore:
    """Single sqlite-backed bundle store for controller and workers.

    Bundles are stored as zip bytes in sqlite by ``bundle_id``.
    Workers can lazily fetch missing bundles from the controller and cache
    them locally.
    """

    def __init__(
        self,
        db_path: Path,
        controller_address: str | None = None,
        max_total_bytes: int = 1_000_000_000,
        max_items: int = 1000,
    ):
        self._db_path = db_path
        self._controller_address = controller_address.rstrip("/") if controller_address else ""
        self._max_total_bytes = max_total_bytes
        self._max_items = max_items
        self._lock = threading.RLock()

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # We intentionally share one sqlite connection across threads; every
        # public DB operation must hold self._lock.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bundles (
                bundle_id TEXT PRIMARY KEY,
                zip_bytes BLOB NOT NULL,
                created_at_ms INTEGER NOT NULL,
                last_access_ms INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_bundles_last_access
            ON bundles(last_access_ms)
            """
        )
        self._conn.commit()

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    def write_zip(self, blob: bytes) -> str:
        """Write zip bytes if absent and return canonical bundle id."""
        bundle_id = bundle_id_for_zip(blob)
        now_ms = self._now_ms()
        with self._lock:
            row = self._conn.execute("SELECT 1 FROM bundles WHERE bundle_id = ?", (bundle_id,)).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO bundles(bundle_id, zip_bytes, created_at_ms, last_access_ms, size_bytes)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (bundle_id, blob, now_ms, now_ms, len(blob)),
                )
            else:
                self._conn.execute(
                    "UPDATE bundles SET last_access_ms = ? WHERE bundle_id = ?",
                    (now_ms, bundle_id),
                )
            self._conn.commit()
            self._evict_if_needed_locked()
        return bundle_id

    def get_zip(self, bundle_id: str) -> bytes:
        """Read zip bytes by bundle id.

        Note: this method mutates the DB (updates ``last_access_ms``), so it
        must hold ``self._lock`` like all other public DB methods.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT zip_bytes FROM bundles WHERE bundle_id = ?",
                (bundle_id,),
            ).fetchone()
            if row is None:
                raise FileNotFoundError(f"Bundle not found: {bundle_id}")
            self._conn.execute(
                "UPDATE bundles SET last_access_ms = ? WHERE bundle_id = ?",
                (self._now_ms(), bundle_id),
            )
            self._conn.commit()
            return bytes(row[0])

    def prefetch_bundle(self, bundle_id: str) -> None:
        """Ensure bundle is present locally, fetching from controller on miss."""
        try:
            self.get_zip(bundle_id)
            return
        except FileNotFoundError:
            logger.info("Bundle %s not found in local store, fetching from controller", bundle_id)

        if not self._controller_address:
            raise RuntimeError(f"Bundle {bundle_id} is not cached and controller address is not configured")

        url = f"{self._controller_address}/bundles/{bundle_id}.zip"
        for attempt in range(3):
            try:
                with urlopen(url, timeout=120) as resp:
                    blob = resp.read()
                break
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"Failed to fetch bundle {bundle_id}: {e}") from e
                time.sleep(0.25 * (2**attempt))

        actual = bundle_id_for_zip(blob)
        if actual != bundle_id:
            raise ValueError(f"Bundle hash mismatch while fetching {bundle_id}: got {actual}")
        self.write_zip(blob)

    def extract_bundle_to(self, bundle_id: str, dest: Path) -> None:
        """Extract a bundle zip into ``dest`` with zip-slip protection."""
        blob = self.get_zip(bundle_id)
        dest.mkdir(parents=True, exist_ok=True)
        base = dest.resolve()

        with zipfile.ZipFile(io.BytesIO(blob), "r") as zf:
            for member in zf.namelist():
                member_path = (dest / member).resolve()
                if not member_path.is_relative_to(base):
                    raise ValueError(f"Zip slip detected: {member} attempts to write outside extract path")
            zf.extractall(dest)

    def write_workdir_files(self, dest: Path, files: dict[str, bytes]) -> None:
        """Write workdir files under ``dest`` with path validation."""
        for name, data in files.items():
            normalized = normalize_workdir_relative_path(name)
            path = dest / normalized
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    def _evict_if_needed_locked(self) -> None:
        if self._max_items <= 0 and self._max_total_bytes <= 0:
            return
        row = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM bundles",
        ).fetchone()
        if row is None:
            return
        count = int(row[0])
        total = int(row[1])

        # Compute how many items to evict based on count and size limits.
        evict_for_items = max(0, count - self._max_items) if self._max_items > 0 else 0
        # For bytes, estimate conservatively: evict at least 1 item at a time
        # but start with the count-based minimum.
        n_to_evict = evict_for_items

        if self._max_total_bytes > 0 and total > self._max_total_bytes:
            # Need to evict by size too — fetch LRU candidates and accumulate
            # until we're under the byte limit.
            candidates = self._conn.execute(
                "SELECT bundle_id, size_bytes FROM bundles ORDER BY last_access_ms ASC",
            ).fetchall()
            freed = 0
            for i, (_bid, sz) in enumerate(candidates):
                if i >= n_to_evict and total - freed <= self._max_total_bytes:
                    break
                freed += int(sz)
                n_to_evict = i + 1

        if n_to_evict > 0:
            victims = self._conn.execute(
                "SELECT bundle_id FROM bundles ORDER BY last_access_ms ASC LIMIT ?",
                (n_to_evict,),
            ).fetchall()
            victim_ids = [str(v[0]) for v in victims]
            self._conn.executemany("DELETE FROM bundles WHERE bundle_id = ?", [(v,) for v in victim_ids])
            self._conn.commit()
