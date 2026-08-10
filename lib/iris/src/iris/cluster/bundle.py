# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed BundleStore with fsspec persistence and in-memory LRU cache.

A single content-addressed namespace serves both zipped workdirs and
externalized workdir files. Disk layout: ``{storage_dir}/{id}``. The
in-memory cache is populated lazily. ``get`` checks cache → fsspec
storage → controller HTTP endpoint (when ``controller_address`` is set,
i.e. on workers). Eviction from the in-memory cache does not delete from
fsspec storage.
"""

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

# Maximum size (bytes) of a submitted workspace bundle. Enforced by the client
# when creating the zip and re-checked by the controller when receiving it.
MAX_BUNDLE_SIZE_BYTES = 25 * 1024 * 1024


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

        # content id -> bytes, ordered by access (MRU at end).
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._cache_bytes = 0

    def _path_for(self, cid: str) -> str:
        return f"{self._fs_path}/{cid}"

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

    def write(self, data: bytes) -> str:
        """Write bytes if absent and return canonical content id."""
        cid = content_id(data)
        path = self._path_for(cid)
        with self._lock:
            # Always ensure disk has the bytes: a cache hit without disk write
            # would leave workers unable to fetch this id after a restart.
            if not self._fs.exists(path):
                with self._fs.open(path, "wb") as f:
                    f.write(data)
            self._cache_put(cid, data)
        return cid

    def get(self, cid: str) -> bytes:
        """Read bytes by content id. Falls back to controller on workers."""
        path = self._path_for(cid)
        with self._lock:
            if cid in self._cache:
                self._cache.move_to_end(cid)
                return self._cache[cid]
            if self._fs.exists(path):
                with self._fs.open(path, "rb") as f:
                    data = f.read()
                self._cache_put(cid, data)
                return data
        # Outside lock: fetch from controller and persist locally.
        data = self._fetch_from_controller(cid)
        self.write(data)
        return data

    def _fetch_from_controller(self, cid: str) -> bytes:
        """Fetch content over HTTP and verify hash. Retries up to 3 times."""
        if not self._controller_address:
            raise FileNotFoundError(f"Content {cid} not found and no controller configured")

        url = f"{self._controller_address}/blobs/{cid}"
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
        return data

    def extract_bundle_to(self, bundle_id: str, dest: Path) -> None:
        """Extract a bundle zip into ``dest`` with zip-slip protection."""
        blob = self.get(bundle_id)

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
