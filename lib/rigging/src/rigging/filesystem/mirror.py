# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mirror filesystem: transparent cross-region file access.

Reads check the local prefix first, then scan all other ``marin-*`` GCS data
buckets and copy on first access under a distributed lock. Writes always target
the local prefix. Cross-region copies are charged against the shared
:class:`TransferBudget`.

This module and :mod:`~rigging.filesystem.distributed_lock` are the only ones in
the package that import ``botocore`` (for the S3 lock backend), so
:mod:`rigging.filesystem` registers the ``mirror://`` protocol lazily (by class
path) to keep it off a plain ``import rigging.filesystem``. fsspec imports this
module on demand the first time a ``mirror://`` filesystem is constructed.
"""

import logging
import time
from typing import Any, cast

import fsspec
from fsspec.callbacks import DEFAULT_CALLBACK, Callback

from rigging.filesystem.cluster_config import StoreType, data_config, marin_prefix
from rigging.filesystem.cross_region import TransferBudget, _global_transfer_budget, _mirror_budget_ctx
from rigging.filesystem.distributed_lock import create_lock, default_worker_id
from rigging.filesystem.storage_path import StoragePath, prefix_join

logger = logging.getLogger(__name__)


def _all_data_bucket_prefixes() -> list[str]:
    """Return gs:// prefixes for all of the active cluster's GCS data buckets."""
    return [f"gs://{spec.name}" for spec in data_config().region_buckets.values() if spec.store == StoreType.GCS]


def _mirror_remote_prefixes(local_prefix: str) -> list[str]:
    """Remote marin buckets to scan for mirror reads.

    The cross-region mirror only exists on GCS, and scanning GCS buckets
    requires GCP credentials.  Return an empty list unless the local prefix
    is itself a ``gs://`` URL — otherwise non-GCP runs (CoreWeave S3, local
    dev) would emit anonymous-caller 401s from gcsfs on every mirror read.
    """
    if not local_prefix.startswith("gs://"):
        return []
    return [p for p in _all_data_bucket_prefixes() if not local_prefix.startswith(p)]


class MirrorFileSystem(fsspec.AbstractFileSystem):
    """Fsspec filesystem that mirrors files across marin regional buckets.

    Reads check the local prefix first, then scan other regions.  Files found
    in a remote region are copied to the local prefix under a distributed lock.
    Writes always target the local prefix.

    Cross-region copies are charged against the shared ``TransferBudget``.
    """

    protocol = "mirror"

    def __init__(
        self,
        *args: Any,
        budget: TransferBudget | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._local_prefix = marin_prefix().rstrip("/")
        self._remote_prefixes = _mirror_remote_prefixes(self._local_prefix)
        self._budget = budget if budget is not None else _global_transfer_budget
        self._worker_id = default_worker_id()

    # -- budget resolution ----------------------------------------------------

    def _active_budget(self) -> TransferBudget:
        """Return the contextvar budget if set, otherwise the instance budget."""
        ctx_budget = _mirror_budget_ctx.get()
        if ctx_budget is not None:
            return ctx_budget
        return self._budget

    # -- underlying fs helpers ------------------------------------------------

    def _get_fs_and_path(self, url: str) -> tuple[Any, str]:
        """Return (fsspec_fs, path) for a full URL or local path."""
        return fsspec.core.url_to_fs(url)

    @property
    def _local_root(self) -> StoragePath:
        return StoragePath.parse(self._local_prefix)

    def _local_url(self, path: str) -> str:
        return str(self._local_root / path)

    def _remote_url(self, prefix: str, path: str) -> str:
        return prefix_join(prefix, path)

    def _lock_path_for(self, path: str) -> str:
        return str(self._local_root / ".mirror_locks" / f"{path}.lock")

    def _fs_exists(self, url: str) -> bool:
        fs, fspath = self._get_fs_and_path(url)
        return fs.exists(fspath)

    def _fs_size(self, url: str) -> int | None:
        fs, fspath = self._get_fs_and_path(url)
        return fs.size(fspath)

    def _fs_copy(self, src_url: str, dst_url: str) -> None:
        src_fs, src_path = self._get_fs_and_path(src_url)
        dst_fs, dst_path = self._get_fs_and_path(dst_url)

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
        for prefix in self._remote_prefixes:
            remote_url = self._remote_url(prefix, path)
            if self._fs_exists(remote_url):
                return prefix
        return None

    def _copy_to_local(self, source_prefix: str, path: str) -> None:
        local_url = self._local_url(path)
        remote_url = self._remote_url(source_prefix, path)

        lock = create_lock(self._lock_path_for(path), self._worker_id)

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
            if self._fs_exists(local_url):
                return

            size = self._fs_size(remote_url)
            if size is not None:
                self._active_budget().record(size, remote_url)

            logger.info("Mirror: copying %s → %s", remote_url, local_url)
            self._fs_copy(remote_url, local_url)
        finally:
            lock.release()

    def _resolve_path(self, path: str) -> str:
        """Resolve a mirror path to a concrete URL, copying if needed."""
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
        path = cast(str, self._strip_protocol(path))
        resolved = self._resolve_path(path)
        fs, fspath = self._get_fs_and_path(resolved)
        info = fs.info(fspath, **kwargs)
        info["name"] = path
        return info

    @staticmethod
    def _stripped_prefix(bucket_prefix: str) -> str:
        """Return the bucket prefix without scheme, with trailing slash."""
        return bucket_prefix.rstrip("/").replace("gs://", "").replace("file://", "") + "/"

    def ls(self, path: str, detail: bool = True, **kwargs: Any) -> list[Any]:
        path = cast(str, self._strip_protocol(path))
        # Union listings from local + all remote prefixes so that glob()
        # discovers files that only exist in other regions.  Local entries
        # take precedence when a relative path appears in multiple buckets.
        seen: dict[str, dict[str, Any]] = {}

        for prefix in [self._local_prefix, *self._remote_prefixes]:
            url = prefix_join(prefix, path)
            fs, fspath = self._get_fs_and_path(url)
            try:
                entries = fs.ls(fspath, detail=True, **kwargs)
            except FileNotFoundError:
                continue

            stripped = self._stripped_prefix(prefix)
            for entry in entries:
                rel_name = entry["name"]
                if rel_name.startswith(stripped):
                    rel_name = rel_name[len(stripped) :]
                if rel_name not in seen:
                    seen[rel_name] = {**entry, "name": rel_name}

        results = list(seen.values())
        if detail:
            return results
        return [e["name"] for e in results]

    def exists(self, path: str, **kwargs: Any) -> bool:
        path = cast(str, self._strip_protocol(path))
        local_url = self._local_url(path)
        if self._fs_exists(local_url):
            return True
        return self._find_in_remote_prefixes(path) is not None

    # -- fsspec interface: read operations ------------------------------------

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        autocommit: bool = True,
        cache_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        path = cast(str, self._strip_protocol(path))
        kwargs = {
            **kwargs,
            "block_size": block_size,
            "autocommit": autocommit,
            "cache_options": cache_options,
        }
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
        path = cast(str, self._strip_protocol(path))
        resolved = self._resolve_path(path)
        fs, fspath = self._get_fs_and_path(resolved)
        return fs.cat_file(fspath, start=start, end=end, **kwargs)

    # -- fsspec interface: write operations ------------------------------------

    def _mkdir(self, path: str, create_parents: bool = True, **kwargs: Any) -> None:
        path = cast(str, self._strip_protocol(path))
        local_url = self._local_url(path)
        fs, fspath = self._get_fs_and_path(local_url)
        fs.mkdir(fspath, create_parents=create_parents, **kwargs)

    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        path = cast(str, self._strip_protocol(path))
        local_url = self._local_url(path)
        fs, fspath = self._get_fs_and_path(local_url)
        fs.makedirs(fspath, exist_ok=exist_ok)

    def put_file(
        self,
        lpath: str,
        rpath: str,
        callback: Callback = DEFAULT_CALLBACK,
        mode: str = "overwrite",
        **kwargs: Any,
    ) -> None:
        rpath = cast(str, self._strip_protocol(rpath))
        local_url = self._local_url(rpath)
        fs, fspath = self._get_fs_and_path(local_url)
        fs.put_file(lpath, fspath, callback=callback, mode=mode, **kwargs)

    # fsspec's AbstractFileSystem.rm_file is typed as returning Never (its body
    # delegates to the unimplemented _rm), so a real None-returning override is
    # flagged. Parameters already match the base.
    # pyrefly: ignore[bad-override]
    def rm_file(self, path: str) -> None:
        path = cast(str, self._strip_protocol(path))
        local_url = self._local_url(path)
        fs, fspath = self._get_fs_and_path(local_url)
        fs.rm_file(fspath)

    def rm(self, path: str, recursive: bool = False, maxdepth: int | None = None, **kwargs: Any) -> None:
        path = cast(str, self._strip_protocol(path))
        local_url = self._local_url(path)
        fs, fspath = self._get_fs_and_path(local_url)
        fs.rm(fspath, recursive=recursive, maxdepth=maxdepth, **kwargs)

    def copy(
        self,
        path1: str,
        path2: str,
        recursive: bool = False,
        maxdepth: int | None = None,
        on_error: str | None = None,
        **kwargs: Any,
    ) -> None:
        # recursive/maxdepth/on_error are accepted for fsspec API compatibility;
        # the mirror only supports single-file copies via _fs_copy.
        path1 = cast(str, self._strip_protocol(path1))
        path2 = cast(str, self._strip_protocol(path2))
        resolved_src = self._resolve_path(path1)
        local_dst = self._local_url(path2)
        self._fs_copy(resolved_src, local_dst)

    @property
    def bytes_copied(self) -> int:
        """Total cross-region bytes transferred (shared budget)."""
        return self._budget.bytes_used


# Upgrade the package's lazy class-path registration to the concrete class now
# that this module is actually loaded.
fsspec.register_implementation("mirror", MirrorFileSystem, clobber=True)
