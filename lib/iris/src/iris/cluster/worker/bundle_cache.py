# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bundle cache for workspace bundles from GCS."""

import hashlib
import logging
import shutil
import threading
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Protocol

import fsspec

logger = logging.getLogger(__name__)


class BundleProvider(Protocol):
    """Protocol for bundle retrieval."""

    def get_bundle(self, gcs_path: str, expected_hash: str | None = None) -> Path: ...


class BundleCache:
    """Cache for workspace bundles downloaded from GCS.

    Assumes GCS paths are unique - uses path as cache key.
    Two-level caching: zip files + extracted directories.

    Supports both gs:// paths (requires GCS auth) and file:// paths
    for local testing.
    """

    def __init__(self, cache_dir: Path, max_bundles: int = 100):
        self._cache_dir = cache_dir
        self._bundles_dir = cache_dir / "bundles"
        self._extracts_dir = cache_dir / "extracts"
        self._max_bundles = max_bundles
        self._extract_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)

        self._bundles_dir.mkdir(parents=True, exist_ok=True)
        self._extracts_dir.mkdir(parents=True, exist_ok=True)

    def _path_to_key(self, gcs_path: str) -> str:
        return hashlib.sha256(gcs_path.encode()).hexdigest()[:16]

    def get_bundle(self, gcs_path: str, expected_hash: str | None = None) -> Path:
        key = self._path_to_key(gcs_path)
        extract_path = self._extracts_dir / key

        # All access is serialized under the per-key lock. The exists() check
        # must be inside the lock because _extract() creates the directory
        # before writing files — an unlocked check would race with an
        # in-progress extraction and return a partially-extracted bundle.
        with self._extract_locks[key]:
            if extract_path.exists():
                extract_path.touch()
                logger.debug("Bundle cache hit: %s", gcs_path)
                return extract_path

            # Download and extract
            zip_path = self._bundles_dir / f"{key}.zip"
            if not zip_path.exists():
                logger.debug("Downloading bundle: %s", gcs_path)
                self._download(gcs_path, zip_path)
            else:
                logger.debug("Bundle zip cached, extracting: %s", gcs_path)

            if expected_hash:
                actual_hash = self._compute_hash(zip_path)
                if actual_hash != expected_hash:
                    raise ValueError(f"Bundle hash mismatch: {actual_hash} != {expected_hash}")

            self._extract(zip_path, extract_path)
            self._evict_old_bundles()

            return extract_path

    def _download(self, gcs_path: str, local_path: Path) -> None:
        with fsspec.open(gcs_path, "rb") as src:
            with open(local_path, "wb") as dst:
                dst.write(src.read())

    def _extract(self, zip_path: Path, extract_path: Path) -> None:
        extract_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Validate all paths to prevent zip slip attacks
            for member in zf.namelist():
                member_path = (extract_path / member).resolve()
                if not member_path.is_relative_to(extract_path.resolve()):
                    raise ValueError(f"Zip slip detected: {member} attempts to write outside extract path")
            zf.extractall(extract_path)

    def _compute_hash(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    # Bundles accessed within this window are not eligible for eviction.
    # After get_bundle() returns a Path, callers do copytree() from it;
    # this grace period ensures that eviction doesn't race with that copy.
    _EVICTION_GRACE_SECONDS = 300

    def _evict_old_bundles(self) -> None:
        extracts = list(self._extracts_dir.iterdir())
        if len(extracts) <= self._max_bundles:
            return

        extracts.sort(key=lambda p: p.stat().st_mtime)
        for path in extracts[: len(extracts) - self._max_bundles]:
            # Sorted oldest-first — once we hit a recent bundle, the rest are newer.
            if time.time() - path.stat().st_mtime < self._EVICTION_GRACE_SECONDS:
                break

            key = path.name
            with self._extract_locks[key]:
                # Re-check under lock: another thread may have touched the bundle
                # between our mtime read and the lock acquisition.
                if not path.exists():
                    continue
                if time.time() - path.stat().st_mtime < self._EVICTION_GRACE_SECONDS:
                    continue

                if path.is_dir():
                    shutil.rmtree(path)
                zip_path = self._bundles_dir / f"{key}.zip"
                if zip_path.exists():
                    zip_path.unlink()
