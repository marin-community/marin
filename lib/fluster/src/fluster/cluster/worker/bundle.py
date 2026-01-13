# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bundle cache for workspace bundles from GCS."""

import hashlib
import threading
import zipfile
from collections import defaultdict
from pathlib import Path

import fsspec


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
        """Convert GCS path to cache key (hash)."""
        return hashlib.sha256(gcs_path.encode()).hexdigest()[:16]

    def get_bundle(self, gcs_path: str, expected_hash: str | None = None) -> Path:
        """Get bundle path, downloading if needed.

        Args:
            gcs_path: gs://bucket/path/bundle.zip or file:///local/path.zip
            expected_hash: Optional SHA256 hash for verification

        Returns:
            Path to extracted bundle directory
        """
        key = self._path_to_key(gcs_path)
        extract_path = self._extracts_dir / key

        if extract_path.exists():
            # Update access time for LRU
            extract_path.touch()
            return extract_path

        # Use a lock per bundle to prevent concurrent extractions to the same path
        with self._extract_locks[key]:
            # Double-check after acquiring lock - another task may have extracted it
            if extract_path.exists():
                extract_path.touch()
                return extract_path

            # Download and extract
            zip_path = self._bundles_dir / f"{key}.zip"
            if not zip_path.exists():
                self._download(gcs_path, zip_path)

            if expected_hash:
                actual_hash = self._compute_hash(zip_path)
                if actual_hash != expected_hash:
                    raise ValueError(f"Bundle hash mismatch: {actual_hash} != {expected_hash}")

            self._extract(zip_path, extract_path)
            self._evict_old_bundles()

            return extract_path

    def _download(self, gcs_path: str, local_path: Path) -> None:
        """Synchronous download implementation."""
        # fsspec handles gs://, file://, and other protocols
        with fsspec.open(gcs_path, "rb") as src:
            with open(local_path, "wb") as dst:
                dst.write(src.read())

    def _extract(self, zip_path: Path, extract_path: Path) -> None:
        """Synchronous extraction implementation with zip slip protection."""
        extract_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Validate all paths to prevent zip slip attacks
            for member in zf.namelist():
                member_path = (extract_path / member).resolve()
                if not member_path.is_relative_to(extract_path.resolve()):
                    raise ValueError(f"Zip slip detected: {member} attempts to write outside extract path")
            zf.extractall(extract_path)

    def _compute_hash(self, path: Path) -> str:
        """Synchronous hash computation implementation."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _evict_old_bundles(self) -> None:
        """LRU eviction when over max_bundles."""
        extracts = list(self._extracts_dir.iterdir())
        if len(extracts) <= self._max_bundles:
            return

        # Sort by mtime, remove oldest
        extracts.sort(key=lambda p: p.stat().st_mtime)
        for path in extracts[: len(extracts) - self._max_bundles]:
            if path.is_dir():
                import shutil

                shutil.rmtree(path)
            # Also remove corresponding zip
            zip_path = self._bundles_dir / f"{path.name}.zip"
            if zip_path.exists():
                zip_path.unlink()
