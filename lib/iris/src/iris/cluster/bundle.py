# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bundle ID utilities and a BundleStore with fsspec persistence and in-memory LRU cache.

Bundles are stored as ``{storage_dir}/{bundle_id}.zip`` using fsspec (supporting
local paths and GCS URIs). An in-memory LRU cache avoids repeated reads from
storage. The cache is populated lazily — no startup scan is performed.
"""

from __future__ import annotations

import hashlib
import io
import logging
import threading
import time
import zipfile
from collections import OrderedDict
from pathlib import Path
from urllib.request import urlopen

import fsspec.core

logger = logging.getLogger(__name__)


def bundle_id_for_zip(blob: bytes) -> str:
    """Return canonical content id (SHA-256 hex digest) for bytes."""
    return hashlib.sha256(blob).hexdigest()


class BundleStore:
    """Bundle store with fsspec persistence and in-memory LRU cache.

    Bundles are persisted as ``{storage_dir}/{bundle_id}.zip`` via fsspec, supporting
    both local paths and remote URIs (e.g. ``gs://bucket/path/bundles``).

    The in-memory cache is populated lazily: ``get_zip`` checks in-memory cache first,
    then falls back to fsspec storage, then (on workers) fetches from the controller.
    Eviction from the in-memory cache does NOT delete from fsspec storage.
    """

    def __init__(
        self,
        storage_dir: str,
        controller_address: str | None = None,
        max_cache_items: int = 1000,
        max_cache_bytes: int = 1_000_000_000,
    ):
        self._storage_dir = storage_dir
        self._controller_address = controller_address.rstrip("/") if controller_address else ""
        self._max_cache_items = max_cache_items
        self._max_cache_bytes = max_cache_bytes
        self._lock = threading.RLock()

        self._fs, self._fs_path = fsspec.core.url_to_fs(storage_dir)
        self._fs.mkdirs(self._fs_path, exist_ok=True)

        # OrderedDict mapping bundle_id -> blob bytes, ordered by access time
        # (most recently accessed at end).
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._cache_bytes = 0

    def _bundle_fs_path(self, bundle_id: str) -> str:
        return f"{self._fs_path}/{bundle_id}.zip"

    def _exists_in_storage(self, bundle_id: str) -> bool:
        return self._fs.exists(self._bundle_fs_path(bundle_id))

    def _read_from_storage(self, bundle_id: str) -> bytes:
        with self._fs.open(self._bundle_fs_path(bundle_id), "rb") as f:
            return f.read()

    def _write_to_storage(self, bundle_id: str, blob: bytes) -> None:
        with self._fs.open(self._bundle_fs_path(bundle_id), "wb") as f:
            f.write(blob)

    def _cache_put(self, bundle_id: str, blob: bytes) -> None:
        """Insert into in-memory cache and evict if needed. Caller must hold _lock."""
        if bundle_id in self._cache:
            self._cache.move_to_end(bundle_id)
            return
        self._cache[bundle_id] = blob
        self._cache_bytes += len(blob)
        self._evict_if_needed_locked()

    def write_zip(self, blob: bytes) -> str:
        """Write zip bytes if absent and return canonical bundle id."""
        bundle_id = bundle_id_for_zip(blob)
        with self._lock:
            if bundle_id in self._cache:
                self._cache.move_to_end(bundle_id)
                return bundle_id

            if not self._exists_in_storage(bundle_id):
                self._write_to_storage(bundle_id, blob)
            self._cache_put(bundle_id, blob)
        return bundle_id

    def get_zip(self, bundle_id: str) -> bytes:
        """Read zip bytes by bundle id.

        Checks in-memory cache, then fsspec storage. Raises FileNotFoundError
        if the bundle is not found in either location.
        """
        with self._lock:
            if bundle_id in self._cache:
                self._cache.move_to_end(bundle_id)
                return self._cache[bundle_id]

            if not self._exists_in_storage(bundle_id):
                raise FileNotFoundError(f"Bundle not found: {bundle_id}")

            blob = self._read_from_storage(bundle_id)
            self._cache_put(bundle_id, blob)
            return blob

    def _fetch_from_controller(self, content_id: str, url_path: str) -> None:
        """Fetch content from the controller HTTP endpoint and store it locally.

        Retries up to 3 times with exponential backoff. Verifies the SHA-256
        hash of the downloaded bytes matches ``content_id``.
        """
        if not self._controller_address:
            raise FileNotFoundError(f"Content {content_id} not found and no controller configured")

        url = f"{self._controller_address}/{url_path}"
        for attempt in range(3):
            try:
                with urlopen(url, timeout=120) as resp:
                    blob = resp.read()
                break
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"Failed to fetch {content_id}: {e}") from e
                time.sleep(0.25 * (2**attempt))

        actual = bundle_id_for_zip(blob)
        if actual != content_id:
            raise ValueError(f"Hash mismatch while fetching {content_id}: got {actual}")
        self.write_zip(blob)

    def get_or_fetch(self, content_id: str, url_path: str) -> bytes:
        """Get content by ID, fetching from controller if not in local cache/storage.

        Like ``get_zip`` but with automatic controller fallback. The ``url_path``
        is the HTTP path suffix used to fetch from the controller (e.g.
        ``bundles/{id}.zip`` or ``blobs/{id}``).
        """
        try:
            return self.get_zip(content_id)
        except FileNotFoundError:
            logger.info("Content %s not in local cache, fetching from controller", content_id)
            self._fetch_from_controller(content_id, url_path)
            return self.get_zip(content_id)

    def extract_bundle_to(self, bundle_id: str, dest: Path) -> None:
        """Extract a bundle zip into ``dest`` with zip-slip protection.

        If the bundle is not in the local cache/storage and a controller address is
        configured, it is fetched on demand before extraction.
        """
        blob = self.get_or_fetch(bundle_id, f"bundles/{bundle_id}.zip")

        dest.mkdir(parents=True, exist_ok=True)
        base = dest.resolve()

        with zipfile.ZipFile(io.BytesIO(blob), "r") as zf:
            for member in zf.namelist():
                member_path = (dest / member).resolve()
                if not member_path.is_relative_to(base):
                    raise ValueError(f"Zip slip detected: {member} attempts to write outside extract path")
            zf.extractall(dest)

    def _evict_if_needed_locked(self) -> None:
        """Evict oldest entries from in-memory cache. Caller must hold _lock."""
        if self._max_cache_items <= 0 and self._max_cache_bytes <= 0:
            return

        count = len(self._cache)
        evict_for_items = max(0, count - self._max_cache_items) if self._max_cache_items > 0 else 0
        n_to_evict = evict_for_items

        if self._max_cache_bytes > 0 and self._cache_bytes > self._max_cache_bytes:
            freed = 0
            for i, (_bid, blob) in enumerate(self._cache.items()):
                if i >= n_to_evict and self._cache_bytes - freed <= self._max_cache_bytes:
                    break
                freed += len(blob)
                n_to_evict = i + 1

        if n_to_evict <= 0:
            return

        for _ in range(n_to_evict):
            _bid, blob = self._cache.popitem(last=False)
            self._cache_bytes -= len(blob)

    def close(self) -> None:
        pass
