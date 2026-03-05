# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mirror filesystem: transparent cross-region file access for marin buckets.

``MirrorFileSystem`` is an fsspec filesystem registered under the ``mirror://``
protocol.  It provides transparent read access to files that may live in any
marin regional bucket, automatically copying them to the local zone on first
access.

Semantics:
  - **Reads**: If the file exists in the local prefix, serve it directly.
    Otherwise, scan all other ``marin-*`` data buckets.  When found, acquire a
    distributed lock on the destination path, copy the file to the local prefix,
    and release the lock.
  - **Writes**: Go directly to the local prefix.
  - **Copy budget**: The filesystem tracks cumulative bytes copied across
    regions.  If the total exceeds ``MIRROR_COPY_LIMIT_BYTES`` (default 10 GB),
    ``MirrorCopyLimitExceeded`` is raised.

The local prefix is determined by ``marin_prefix()`` — this works whether we're
on GCS (``gs://marin-us-east5``), or running locally (``/tmp/marin``).

Usage::

    from iris.marin_fs import url_to_fs

    fs, path = url_to_fs("mirror://path/to/file")
    with fs.open(path) as f:
        data = f.read()
"""

import logging
import time
from typing import Any

import fsspec

from iris.distributed_lock import DistributedLock, default_holder_id
from iris.marin_fs import (
    REGION_TO_DATA_BUCKET,
    marin_prefix,
)

logger = logging.getLogger(__name__)

MIRROR_COPY_LIMIT_BYTES: int = 10 * 1024 * 1024 * 1024  # 10 GB


class MirrorCopyLimitExceeded(Exception):
    """Raised when cumulative cross-region copies exceed the budget."""


def _all_data_bucket_prefixes() -> list[str]:
    """Return gs:// prefixes for all known marin data buckets."""
    return [f"gs://{bucket}" for bucket in REGION_TO_DATA_BUCKET.values()]


class MirrorFileSystem(fsspec.AbstractFileSystem):
    """Fsspec filesystem that mirrors files across marin regional buckets.

    Reads check the local prefix first, then scan other regions.  Files found
    in a remote region are copied to the local prefix under a distributed lock.
    Writes always target the local prefix.

    The local prefix comes from ``marin_prefix()`` — it may be a GCS bucket
    or a local directory like ``/tmp/marin``.
    """

    protocol = "mirror"

    def __init__(
        self,
        *args: Any,
        copy_limit_bytes: int = MIRROR_COPY_LIMIT_BYTES,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._local_prefix = marin_prefix()
        self._remote_prefixes = [p for p in _all_data_bucket_prefixes() if not self._local_prefix.startswith(p)]
        self._copy_limit_bytes = copy_limit_bytes
        self._bytes_copied: int = 0
        self._holder_id = default_holder_id()

    # -- underlying fs helpers ------------------------------------------------

    def _get_fs_and_path(self, url: str) -> tuple[Any, str]:
        """Return (fsspec_fs, path) for a full URL or local path."""
        return fsspec.core.url_to_fs(url)

    def _local_url(self, path: str) -> str:
        """Convert a mirror-relative path to a full URL under the local prefix."""
        return f"{self._local_prefix}/{path}"

    def _remote_url(self, prefix: str, path: str) -> str:
        """Convert a mirror-relative path to a full URL under a remote prefix."""
        return f"{prefix}/{path}"

    def _lock_path_for(self, path: str) -> str:
        """Return the lock file path for a mirror copy operation."""
        return f"{self._local_prefix}/.mirror_locks/{path}.lock"

    def _fs_exists(self, url: str) -> bool:
        """Check if a file exists at a full URL."""
        fs, fspath = self._get_fs_and_path(url)
        return fs.exists(fspath)

    def _fs_size(self, url: str) -> int | None:
        """Return file size at a full URL, or None if it cannot be determined."""
        fs, fspath = self._get_fs_and_path(url)
        size = fs.size(fspath)
        return size if size else None

    def _fs_copy(self, src_url: str, dst_url: str) -> None:
        """Copy a file from src_url to dst_url, potentially cross-filesystem."""
        src_fs, src_path = self._get_fs_and_path(src_url)
        dst_fs, dst_path = self._get_fs_and_path(dst_url)

        # Ensure parent directory exists
        parent = dst_path.rsplit("/", 1)[0] if "/" in dst_path else ""
        if parent:
            dst_fs.makedirs(parent, exist_ok=True)

        if type(src_fs) is type(dst_fs):
            src_fs.copy(src_path, dst_path)
        else:
            data = src_fs.cat_file(src_path)
            with dst_fs.open(dst_path, "wb") as f:
                f.write(data)

    # -- cross-region copy ----------------------------------------------------

    def _find_in_remote_prefixes(self, path: str) -> str | None:
        """Scan non-local marin prefixes for *path*. Returns the prefix or None."""
        for prefix in self._remote_prefixes:
            remote_url = self._remote_url(prefix, path)
            if self._fs_exists(remote_url):
                return prefix
        return None

    def _copy_to_local(self, source_prefix: str, path: str) -> None:
        """Copy a file from *source_prefix* to the local prefix under a lock."""
        local_url = self._local_url(path)
        remote_url = self._remote_url(source_prefix, path)

        size = self._fs_size(remote_url)
        if size is not None and (self._bytes_copied + size) > self._copy_limit_bytes:
            raise MirrorCopyLimitExceeded(
                f"Copying {path} ({size / (1024**3):.2f} GB) would exceed the "
                f"{self._copy_limit_bytes / (1024**3):.0f} GB mirror copy limit "
                f"(already copied {self._bytes_copied / (1024**3):.2f} GB). "
                f"Consider running in the source region instead."
            )

        lock = DistributedLock(self._lock_path_for(path), self._holder_id)

        if not lock.try_acquire():
            for _ in range(60):
                time.sleep(2)
                if self._fs_exists(local_url):
                    return
                if not lock.has_active_holder():
                    break
            if self._fs_exists(local_url):
                return
            if not lock.try_acquire():
                raise RuntimeError(f"Could not acquire mirror lock for {path} after waiting")

        try:
            # Double-check after acquiring lock
            if self._fs_exists(local_url):
                return

            logger.info("Mirror: copying %s → %s", remote_url, local_url)
            self._fs_copy(remote_url, local_url)
            if size is not None:
                self._bytes_copied += size
        finally:
            lock.release()

    def _resolve_path(self, path: str) -> str:
        """Resolve a mirror path to a concrete URL, copying if needed.

        Returns a full URL (e.g. ``gs://marin-us-east5/...`` or ``/tmp/marin/...``).
        """
        local_url = self._local_url(path)
        if self._fs_exists(local_url):
            return local_url

        source_prefix = self._find_in_remote_prefixes(path)
        if source_prefix is None:
            raise FileNotFoundError(f"mirror://{path} not found in any marin bucket")

        self._copy_to_local(source_prefix, path)
        return local_url

    # -- fsspec interface: info/ls/exists -------------------------------------

    def _info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        path = self._strip_protocol(path)
        resolved = self._resolve_path(path)
        fs, fspath = self._get_fs_and_path(resolved)
        info = fs.info(fspath, **kwargs)
        info["name"] = path
        return info

    def ls(self, path: str, detail: bool = True, **kwargs: Any) -> list[Any]:
        path = self._strip_protocol(path)
        local_url = self._local_url(path)
        fs, fspath = self._get_fs_and_path(local_url)
        try:
            results = fs.ls(fspath, detail=detail, **kwargs)
        except FileNotFoundError:
            results = []

        # Strip the local prefix from result paths so they look like mirror paths
        prefix = self._local_prefix.rstrip("/") + "/"
        # For GCS, fsspec strips the scheme, so the prefix in results won't have gs://
        stripped_prefix = prefix.replace("gs://", "").replace("file://", "")

        if detail:
            for entry in results:
                name = entry["name"]
                if name.startswith(stripped_prefix):
                    entry["name"] = name[len(stripped_prefix) :]
            return results
        else:
            return [r[len(stripped_prefix) :] if r.startswith(stripped_prefix) else r for r in results]

    def exists(self, path: str, **kwargs: Any) -> bool:
        path = self._strip_protocol(path)
        local_url = self._local_url(path)
        if self._fs_exists(local_url):
            return True
        return self._find_in_remote_prefixes(path) is not None

    # -- fsspec interface: read operations ------------------------------------

    def _open(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        path = self._strip_protocol(path)
        if "r" in mode:
            resolved = self._resolve_path(path)
            fs, fspath = self._get_fs_and_path(resolved)
            return fs.open(fspath, mode, **kwargs)
        else:
            local_url = self._local_url(path)
            fs, fspath = self._get_fs_and_path(local_url)
            parent = fspath.rsplit("/", 1)[0] if "/" in fspath else ""
            if parent:
                fs.makedirs(parent, exist_ok=True)
            return fs.open(fspath, mode, **kwargs)

    def cat_file(self, path: str, start: int | None = None, end: int | None = None, **kwargs: Any) -> bytes:
        path = self._strip_protocol(path)
        resolved = self._resolve_path(path)
        fs, fspath = self._get_fs_and_path(resolved)
        return fs.cat_file(fspath, start=start, end=end, **kwargs)

    # -- fsspec interface: write operations ------------------------------------

    def _mkdir(self, path: str, create_parents: bool = True, **kwargs: Any) -> None:
        path = self._strip_protocol(path)
        local_url = self._local_url(path)
        fs, fspath = self._get_fs_and_path(local_url)
        fs.mkdir(fspath, create_parents=create_parents, **kwargs)

    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        path = self._strip_protocol(path)
        local_url = self._local_url(path)
        fs, fspath = self._get_fs_and_path(local_url)
        fs.makedirs(fspath, exist_ok=exist_ok)

    def put_file(self, lpath: str, rpath: str, **kwargs: Any) -> None:
        rpath = self._strip_protocol(rpath)
        local_url = self._local_url(rpath)
        fs, fspath = self._get_fs_and_path(local_url)
        fs.put_file(lpath, fspath, **kwargs)

    def rm_file(self, path: str) -> None:
        path = self._strip_protocol(path)
        local_url = self._local_url(path)
        fs, fspath = self._get_fs_and_path(local_url)
        fs.rm_file(fspath)

    def rm(self, path: str, recursive: bool = False, **kwargs: Any) -> None:
        path = self._strip_protocol(path)
        local_url = self._local_url(path)
        fs, fspath = self._get_fs_and_path(local_url)
        fs.rm(fspath, recursive=recursive, **kwargs)

    def copy(self, path1: str, path2: str, **kwargs: Any) -> None:
        path1 = self._strip_protocol(path1)
        path2 = self._strip_protocol(path2)
        resolved_src = self._resolve_path(path1)
        local_dst = self._local_url(path2)
        self._fs_copy(resolved_src, local_dst)

    # -- copy budget ----------------------------------------------------------

    @property
    def bytes_copied(self) -> int:
        """Total bytes copied across regions by this filesystem instance."""
        return self._bytes_copied


# Register the mirror:// protocol with fsspec.
fsspec.register_implementation("mirror", MirrorFileSystem)
