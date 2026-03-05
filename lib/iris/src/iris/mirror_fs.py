# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mirror filesystem: transparent cross-region file access for marin buckets.

``MirrorFileSystem`` is an fsspec filesystem registered under the ``mirror://``
protocol.  It provides transparent read access to files that may live in any
marin regional bucket, automatically copying them to the local zone on first
access.

Semantics:
  - **Reads**: If the file exists in the local zone bucket, serve it directly.
    Otherwise, scan all other ``marin-*`` data buckets.  When found, acquire a
    distributed lock on the destination path, copy the file to the local zone,
    and release the lock.
  - **Writes**: Go directly to the local zone bucket.
  - **Copy budget**: The filesystem tracks cumulative bytes copied across
    regions.  If the total exceeds ``MIRROR_COPY_LIMIT_BYTES`` (default 10 GB),
    ``MirrorCopyLimitExceeded`` is raised.

Usage::

    from iris.marin_fs import url_to_fs

    fs, path = url_to_fs("mirror://path/to/file")
    with fs.open(path) as f:
        data = f.read()
"""

import logging
import os
from typing import Any

import fsspec

from iris.distributed_lock import DistributedLock, default_holder_id
from iris.marin_fs import (
    REGION_TO_DATA_BUCKET,
    _REGION_TO_MARIN_BUCKET_OVERRIDES,
    marin_prefix,
    marin_region,
)

logger = logging.getLogger(__name__)

MIRROR_COPY_LIMIT_BYTES: int = 10 * 1024 * 1024 * 1024  # 10 GB


class MirrorCopyLimitExceeded(Exception):
    """Raised when cumulative cross-region copies exceed the budget."""


def _local_bucket() -> str | None:
    """Return the local marin data bucket name, or None if not on GCS."""
    prefix = marin_prefix()
    if not prefix.startswith("gs://"):
        return None
    # gs://marin-us-central2 → marin-us-central2
    return prefix.split("/")[2]


def _all_data_buckets() -> list[str]:
    """Return all known marin data bucket names."""
    return list(REGION_TO_DATA_BUCKET.values())


class MirrorFileSystem(fsspec.AbstractFileSystem):
    """Fsspec filesystem that mirrors files across marin regional buckets.

    Reads check the local bucket first, then scan other regions.  Files found
    in a remote region are copied to the local bucket under a distributed lock.
    Writes always target the local bucket.
    """

    protocol = "mirror"

    def __init__(
        self,
        *args: Any,
        copy_limit_bytes: int = MIRROR_COPY_LIMIT_BYTES,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._local_bucket = _local_bucket()
        self._all_buckets = _all_data_buckets()
        self._copy_limit_bytes = copy_limit_bytes
        self._bytes_copied: int = 0
        self._holder_id = default_holder_id()
        # Underlying GCS filesystem for actual I/O
        self._gcs: fsspec.AbstractFileSystem = fsspec.filesystem("gcs")

    # -- path helpers ---------------------------------------------------------

    def _local_gcs_path(self, path: str) -> str:
        """Convert a mirror path to a full GCS path in the local bucket."""
        return f"{self._local_bucket}/{path}"

    def _remote_gcs_path(self, bucket: str, path: str) -> str:
        """Convert a mirror path to a full GCS path in a given bucket."""
        return f"{bucket}/{path}"

    def _lock_path_for(self, path: str) -> str:
        """Return the lock file path for a mirror copy operation."""
        return f"gs://{self._local_bucket}/.mirror_locks/{path}.lock"

    # -- cross-region copy ----------------------------------------------------

    def _find_in_remote_buckets(self, path: str) -> str | None:
        """Scan non-local marin buckets for *path*. Returns bucket name or None."""
        for bucket in self._all_buckets:
            if bucket == self._local_bucket:
                continue
            remote_path = self._remote_gcs_path(bucket, path)
            try:
                if self._gcs.exists(remote_path):
                    return bucket
            except Exception:
                logger.debug("Error checking %s in bucket %s", path, bucket, exc_info=True)
        return None

    def _copy_to_local(self, source_bucket: str, path: str) -> None:
        """Copy a file from *source_bucket* to the local bucket under a lock."""
        local_path = self._local_gcs_path(path)
        remote_path = self._remote_gcs_path(source_bucket, path)

        # Check size and enforce copy budget
        try:
            size = self._gcs.size(remote_path)
        except Exception:
            size = 0

        if size and (self._bytes_copied + size) > self._copy_limit_bytes:
            raise MirrorCopyLimitExceeded(
                f"Copying {path} ({size / (1024**3):.2f} GB) would exceed the "
                f"{self._copy_limit_bytes / (1024**3):.0f} GB mirror copy limit "
                f"(already copied {self._bytes_copied / (1024**3):.2f} GB). "
                f"Consider running in the source region instead."
            )

        lock = DistributedLock(self._lock_path_for(path), self._holder_id)

        if not lock.try_acquire():
            # Another process is copying this file; wait for it.
            import time

            for _ in range(60):
                time.sleep(2)
                if self._gcs.exists(local_path):
                    return
                if not lock.has_active_holder():
                    break
            # Try once more after waiting
            if self._gcs.exists(local_path):
                return
            # Fall through and attempt copy ourselves
            if not lock.try_acquire():
                raise RuntimeError(f"Could not acquire mirror lock for {path} after waiting")

        try:
            # Double-check after acquiring lock (another holder may have finished)
            if self._gcs.exists(local_path):
                return

            logger.info("Mirror: copying gs://%s → gs://%s", remote_path, local_path)
            # Ensure parent directory exists
            parent = os.path.dirname(local_path)
            try:
                self._gcs.mkdirs(parent, exist_ok=True)
            except Exception:
                pass  # mkdirs on GCS is often a no-op

            self._gcs.copy(remote_path, local_path)
            self._bytes_copied += size or 0
        finally:
            lock.release()

    def _resolve_path(self, path: str) -> str:
        """Resolve a mirror path to a concrete GCS path, copying if needed.

        Returns the GCS path (without gs:// prefix) suitable for self._gcs.
        """
        local_path = self._local_gcs_path(path)
        if self._gcs.exists(local_path):
            return local_path

        source_bucket = self._find_in_remote_buckets(path)
        if source_bucket is None:
            raise FileNotFoundError(f"mirror://{path} not found in any marin bucket")

        self._copy_to_local(source_bucket, path)
        return local_path

    # -- fsspec interface: info/ls/exists -------------------------------------

    def _info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        path = self._strip_protocol(path)
        resolved = self._resolve_path(path)
        info = self._gcs.info(resolved, **kwargs)
        info["name"] = path
        return info

    def ls(self, path: str, detail: bool = True, **kwargs: Any) -> list[Any]:
        path = self._strip_protocol(path)
        local_path = self._local_gcs_path(path)
        try:
            results = self._gcs.ls(local_path, detail=detail, **kwargs)
        except FileNotFoundError:
            results = []

        bucket = self._local_bucket or ""
        prefix = bucket + "/"
        if detail:
            for entry in results:
                if entry["name"].startswith(prefix):
                    entry["name"] = entry["name"][len(prefix) :]
            return results
        else:
            return [r[len(prefix) :] if r.startswith(prefix) else r for r in results]

    def exists(self, path: str, **kwargs: Any) -> bool:
        path = self._strip_protocol(path)
        local_path = self._local_gcs_path(path)
        if self._gcs.exists(local_path, **kwargs):
            return True
        return self._find_in_remote_buckets(path) is not None

    # -- fsspec interface: read operations ------------------------------------

    def _open(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        path = self._strip_protocol(path)
        if "r" in mode:
            resolved = self._resolve_path(path)
            return self._gcs.open(resolved, mode, **kwargs)
        else:
            local_path = self._local_gcs_path(path)
            return self._gcs.open(local_path, mode, **kwargs)

    def cat_file(self, path: str, start: int | None = None, end: int | None = None, **kwargs: Any) -> bytes:
        path = self._strip_protocol(path)
        resolved = self._resolve_path(path)
        return self._gcs.cat_file(resolved, start=start, end=end, **kwargs)

    # -- fsspec interface: write operations ------------------------------------

    def _mkdir(self, path: str, create_parents: bool = True, **kwargs: Any) -> None:
        path = self._strip_protocol(path)
        local_path = self._local_gcs_path(path)
        self._gcs.mkdir(local_path, create_parents=create_parents, **kwargs)

    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        path = self._strip_protocol(path)
        local_path = self._local_gcs_path(path)
        self._gcs.makedirs(local_path, exist_ok=exist_ok)

    def put_file(self, lpath: str, rpath: str, **kwargs: Any) -> None:
        rpath = self._strip_protocol(rpath)
        local_path = self._local_gcs_path(rpath)
        self._gcs.put_file(lpath, local_path, **kwargs)

    def rm_file(self, path: str) -> None:
        path = self._strip_protocol(path)
        local_path = self._local_gcs_path(path)
        self._gcs.rm_file(local_path)

    def rm(self, path: str, recursive: bool = False, **kwargs: Any) -> None:
        path = self._strip_protocol(path)
        local_path = self._local_gcs_path(path)
        self._gcs.rm(local_path, recursive=recursive, **kwargs)

    def copy(self, path1: str, path2: str, **kwargs: Any) -> None:
        path1 = self._strip_protocol(path1)
        path2 = self._strip_protocol(path2)
        src = self._resolve_path(path1)
        dst = self._local_gcs_path(path2)
        self._gcs.copy(src, dst, **kwargs)

    # -- copy budget ----------------------------------------------------------

    @property
    def bytes_copied(self) -> int:
        """Total bytes copied across regions by this filesystem instance."""
        return self._bytes_copied


# Register the mirror:// protocol with fsspec.
fsspec.register_implementation("mirror", MirrorFileSystem)
