# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed BundleStore with fsspec persistence and in-memory LRU cache.

Two namespaces share the same store and cache:

* Bundles (zipped workdirs): ``{storage_dir}/{id}.zip``
* Blobs (large workdir files): ``{storage_dir}/blobs/{id}``

The in-memory cache is populated lazily and shared across both namespaces.
``get_zip``/``get_blob`` check cache → fsspec storage → controller HTTP
endpoint (when ``controller_address`` is set, i.e. on workers). Eviction
from the in-memory cache does not delete from fsspec storage.
"""

from __future__ import annotations

import hashlib
import io
import logging
import threading
import time
import zipfile
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from urllib.request import urlopen

import fsspec.core

logger = logging.getLogger(__name__)


def content_id(data: bytes) -> str:
    """Canonical content id (SHA-256 hex digest) for arbitrary bytes."""
    return hashlib.sha256(data).hexdigest()


class BundleStore:
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
        self._fs.mkdirs(f"{self._fs_path}/blobs", exist_ok=True)

        # bundle_id/blob_id -> bytes, ordered by access (MRU at end).
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._cache_bytes = 0

    def _bundle_fs_path(self, bundle_id: str) -> str:
        return f"{self._fs_path}/{bundle_id}.zip"

    def _blob_fs_path(self, blob_id: str) -> str:
        return f"{self._fs_path}/blobs/{blob_id}"

    def _cache_put(self, cid: str, data: bytes) -> None:
        """Insert into in-memory cache and evict if needed. Caller must hold _lock."""
        if cid in self._cache:
            self._cache.move_to_end(cid)
            return
        self._cache[cid] = data
        self._cache_bytes += len(data)
        self._evict_if_needed_locked()

    def _evict_if_needed_locked(self) -> None:
        while (self._max_cache_items > 0 and len(self._cache) > self._max_cache_items) or (
            self._max_cache_bytes > 0 and self._cache_bytes > self._max_cache_bytes
        ):
            _, evicted = self._cache.popitem(last=False)
            self._cache_bytes -= len(evicted)

    def _write_at(self, data: bytes, path: str) -> str:
        cid = content_id(data)
        with self._lock:
            if cid in self._cache:
                self._cache.move_to_end(cid)
                return cid
            if not self._fs.exists(path):
                with self._fs.open(path, "wb") as f:
                    f.write(data)
            self._cache_put(cid, data)
        return cid

    def _read_at(self, cid: str, path: str, url_path: str, writer: Callable[[bytes], str]) -> bytes:
        with self._lock:
            if cid in self._cache:
                self._cache.move_to_end(cid)
                return self._cache[cid]
            if self._fs.exists(path):
                with self._fs.open(path, "rb") as f:
                    data = f.read()
                self._cache_put(cid, data)
                return data
        # Outside lock: fetch from controller and route through the matching writer.
        self._fetch_from_controller(cid, url_path, writer)
        return self._read_at(cid, path, url_path, writer)

    def _fetch_from_controller(self, cid: str, url_path: str, writer: Callable[[bytes], str]) -> None:
        """Fetch content over HTTP, verify hash, and persist via ``writer``.

        Retries up to 3 times with exponential backoff. ``writer`` must be
        either ``write_zip`` or ``write_blob`` so the fetched bytes land in
        the correct namespace.
        """
        if not self._controller_address:
            raise FileNotFoundError(f"Content {cid} not found and no controller configured")

        url = f"{self._controller_address}/{url_path}"
        for attempt in range(3):
            try:
                with urlopen(url, timeout=120) as resp:
                    data = resp.read()
                break
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"Failed to fetch {cid}: {e}") from e
                time.sleep(0.25 * (2**attempt))

        actual = content_id(data)
        if actual != cid:
            raise ValueError(f"Hash mismatch while fetching {cid}: got {actual}")
        writer(data)

    def write_zip(self, blob: bytes) -> str:
        """Write zip bytes if absent and return canonical bundle id."""
        return self._write_at(blob, self._bundle_fs_path(content_id(blob)))

    def get_zip(self, bundle_id: str) -> bytes:
        """Read zip bytes by bundle id. Falls back to controller on workers."""
        return self._read_at(bundle_id, self._bundle_fs_path(bundle_id), f"bundles/{bundle_id}.zip", self.write_zip)

    def write_blob(self, data: bytes) -> str:
        """Write blob bytes if absent and return canonical blob id."""
        return self._write_at(data, self._blob_fs_path(content_id(data)))

    def get_blob(self, blob_id: str) -> bytes:
        """Read blob bytes by id. Falls back to controller on workers."""
        return self._read_at(blob_id, self._blob_fs_path(blob_id), f"blobs/{blob_id}", self.write_blob)

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

    def close(self) -> None:
        pass
