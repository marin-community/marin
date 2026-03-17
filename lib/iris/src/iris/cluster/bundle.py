# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bundle ID utilities and a flat-file BundleStore with in-memory LRU cache.

Bundles are stored as individual zip files on disk at ``{storage_dir}/{bundle_id}.zip``.
An in-memory index tracks known bundles and their sizes for LRU eviction. The index
is rebuilt by scanning the storage directory on startup, so no checkpoint/restore is
needed — the store survives restarts as long as the storage directory is durable.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import posixpath
import threading
import time
import zipfile
from collections import OrderedDict
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
    """Flat-file bundle store with in-memory LRU index.

    Bundles are persisted as ``{storage_dir}/{bundle_id}.zip``. On init the
    directory is scanned to rebuild the in-memory index, so the store survives
    process restarts without any explicit checkpoint/restore step.

    Workers can lazily fetch missing bundles from the controller and cache
    them locally.
    """

    def __init__(
        self,
        storage_dir: Path,
        controller_address: str | None = None,
        max_total_bytes: int = 1_000_000_000,
        max_items: int = 1000,
    ):
        self._storage_dir = storage_dir
        self._controller_address = controller_address.rstrip("/") if controller_address else ""
        self._max_total_bytes = max_total_bytes
        self._max_items = max_items
        self._lock = threading.RLock()

        self._storage_dir.mkdir(parents=True, exist_ok=True)

        # OrderedDict mapping bundle_id -> size_bytes, ordered by access time
        # (most recently accessed at end).
        self._index: OrderedDict[str, int] = OrderedDict()
        self._total_bytes = 0
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Scan storage directory and populate in-memory index, ordered by mtime."""
        entries: list[tuple[float, str, int]] = []
        for entry in self._storage_dir.iterdir():
            if entry.suffix == ".zip" and entry.is_file():
                bundle_id = entry.stem
                stat = entry.stat()
                entries.append((stat.st_mtime, bundle_id, stat.st_size))

        entries.sort(key=lambda e: e[0])
        self._index.clear()
        self._total_bytes = 0
        for _mtime, bundle_id, size in entries:
            self._index[bundle_id] = size
            self._total_bytes += size

    def _bundle_path(self, bundle_id: str) -> Path:
        return self._storage_dir / f"{bundle_id}.zip"

    def write_zip(self, blob: bytes) -> str:
        """Write zip bytes if absent and return canonical bundle id."""
        bundle_id = bundle_id_for_zip(blob)
        with self._lock:
            if bundle_id in self._index:
                self._index.move_to_end(bundle_id)
                return bundle_id

            path = self._bundle_path(bundle_id)
            path.write_bytes(blob)
            self._index[bundle_id] = len(blob)
            self._total_bytes += len(blob)
            self._evict_if_needed_locked()
        return bundle_id

    def get_zip(self, bundle_id: str) -> bytes:
        """Read zip bytes by bundle id."""
        with self._lock:
            if bundle_id not in self._index:
                raise FileNotFoundError(f"Bundle not found: {bundle_id}")
            self._index.move_to_end(bundle_id)

        path = self._bundle_path(bundle_id)
        return path.read_bytes()

    def _fetch_from_controller(self, bundle_id: str) -> None:
        """Fetch a bundle from the controller HTTP endpoint and store it locally.

        Retries up to 3 times with exponential backoff. Verifies the SHA-256
        hash of the downloaded zip matches ``bundle_id``.
        """
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
        """Extract a bundle zip into ``dest`` with zip-slip protection.

        If the bundle is not in the local cache and a controller address is
        configured, it is fetched on demand before extraction.
        """
        try:
            blob = self.get_zip(bundle_id)
        except FileNotFoundError:
            logger.info("Bundle %s not in local cache, fetching from controller", bundle_id)
            self._fetch_from_controller(bundle_id)
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

        count = len(self._index)
        evict_for_items = max(0, count - self._max_items) if self._max_items > 0 else 0
        n_to_evict = evict_for_items

        if self._max_total_bytes > 0 and self._total_bytes > self._max_total_bytes:
            # Walk LRU order (front = oldest) until under budget.
            freed = 0
            for i, (_bid, sz) in enumerate(self._index.items()):
                if i >= n_to_evict and self._total_bytes - freed <= self._max_total_bytes:
                    break
                freed += sz
                n_to_evict = i + 1

        if n_to_evict <= 0:
            return

        # Pop from front (LRU end) of the OrderedDict.
        for _ in range(n_to_evict):
            bundle_id, size = self._index.popitem(last=False)
            self._total_bytes -= size
            path = self._bundle_path(bundle_id)
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    def close(self) -> None:
        pass
