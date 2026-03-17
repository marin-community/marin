# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin filesystem helpers: prefix resolution, region-local temp storage,
and cross-region read guards.

Provides a unified API for resolving the marin storage prefix and building
GCS paths with lifecycle-managed TTL prefixes. The canonical temp-bucket
definitions live in ``infra/configure_temp_buckets.py``.

Resolution chain for the storage prefix:
  1. ``MARIN_PREFIX`` environment variable
  2. GCS instance metadata → ``gs://marin-{region}``
  3. ``/tmp/marin`` (local fallback)

Cross-region transfer budget:
  ``TransferBudget`` tracks cumulative cross-region GCS bytes across all
  filesystem instances in the process (default 10 GB).  Both
  ``CrossRegionGuardedFS`` (direct reads) and ``MirrorFileSystem`` (mirror
  copies) charge against the same global budget.  Prefer the guarded helpers
  (``url_to_fs``, ``open_url``, ``filesystem``) over the raw fsspec
  equivalents; they automatically wrap GCS filesystems in the guard.  Set
  the ``MARIN_I_WILL_PAY_FOR_ALL_FEES`` env var to override the guard.
"""

import dataclasses
import functools
import logging
import os
import pathlib
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import PurePath
from typing import Any

import fsspec

from iris.distributed_lock import create_lock, default_worker_id

logger = logging.getLogger(__name__)

_GCP_METADATA_ZONE_URL = "http://metadata.google.internal/computeMetadata/v1/instance/zone"

_DEFAULT_LOCAL_PREFIX = "/tmp/marin"

# Canonical mapping from GCP region to marin-tmp bucket name.
# Must stay in sync with infra/configure_temp_buckets.py BUCKETS dict.
REGION_TO_TMP_BUCKET: dict[str, str] = {
    "asia-northeast1": "marin-tmp-asia-northeast-1",
    "us-central1": "marin-tmp-us-central1",
    "us-central2": "marin-tmp-us-central2",
    "europe-west4": "marin-tmp-eu-west4",
    "eu-west4": "marin-tmp-eu-west4",
    "us-west4": "marin-tmp-us-west4",
    "us-east1": "marin-tmp-us-east1",
    "us-east5": "marin-tmp-us-east5",
}

# Special-case overrides for primary Marin buckets that do not follow the
# default `marin-{region}` naming convention.
_REGION_TO_MARIN_BUCKET_OVERRIDES: dict[str, str] = {
    "europe-west4": "marin-eu-west4",
}

# All known primary marin data buckets, keyed by region.
# Used by the mirror filesystem to scan for files across regions.
REGION_TO_DATA_BUCKET: dict[str, str] = {
    "us-central1": "marin-us-central1",
    "us-central2": "marin-us-central2",
    "us-east1": "marin-us-east1",
    "us-east5": "marin-us-east5",
    "us-west4": "marin-us-west4",
    "europe-west4": "marin-eu-west4",
    "asia-northeast1": "marin-asia-northeast1",
}

# Reverse lookup: bucket name → canonical GCP region.
# Derived from REGION_TO_DATA_BUCKET so that region_from_prefix can return
# canonical region names even when the bucket uses abbreviated naming
# (e.g. "marin-eu-west4" → "europe-west4" instead of "eu-west4").
_BUCKET_TO_REGION: dict[str, str] = {bucket: region for region, bucket in REGION_TO_DATA_BUCKET.items()}


# ---------------------------------------------------------------------------
# Low-level region helpers
# ---------------------------------------------------------------------------


def region_from_metadata() -> str | None:
    """Derive GCP region from the instance metadata server, or ``None``."""
    try:
        req = urllib.request.Request(
            _GCP_METADATA_ZONE_URL,
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            zone = resp.read().decode().strip().split("/")[-1]
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        return None
    if "-" not in zone:
        return None
    return zone.rsplit("-", 1)[0]


def region_from_prefix(prefix: str) -> str | None:
    """Extract the canonical GCP region from a ``gs://marin-{region}/…`` prefix.

    Uses ``_BUCKET_TO_REGION`` to normalize abbreviated bucket names
    (e.g. ``gs://marin-eu-west4`` → ``europe-west4``).
    """
    m = re.match(r"gs://([^/]+)", prefix)
    if not m:
        return None
    bucket = m.group(1)
    if bucket in _BUCKET_TO_REGION:
        return _BUCKET_TO_REGION[bucket]
    # Fall back to stripping the "marin-" prefix.
    if bucket.startswith("marin-"):
        return bucket[len("marin-") :]
    return None


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def marin_prefix() -> str:
    """Return the marin storage prefix. Never returns ``None``.

    Resolution order:
      1. ``MARIN_PREFIX`` environment variable
      2. GCS instance metadata → ``gs://marin-{region}``
      3. ``/tmp/marin``
    """
    prefix = os.environ.get("MARIN_PREFIX")
    if prefix:
        return prefix
    region = region_from_metadata()
    if region:
        bucket = _REGION_TO_MARIN_BUCKET_OVERRIDES.get(region, f"marin-{region}")
        return f"gs://{bucket}"
    return _DEFAULT_LOCAL_PREFIX


def marin_region() -> str | None:
    """Return the current GCP region, if detectable.

    Resolution order:
      1. GCS instance metadata server
      2. Infer from ``MARIN_PREFIX`` environment variable
    """
    return region_from_metadata() or region_from_prefix(os.environ.get("MARIN_PREFIX", ""))


def marin_temp_bucket(ttl_days: int, prefix: str = "") -> str:
    """Return a path on region-local temp storage. Never returns ``None``.

    For a GCS marin prefix with a known region, returns a path on the
    dedicated temp bucket::

        gs://marin-tmp-{region}/ttl={N}d/{prefix}

    Otherwise falls back to a flat path under the marin prefix::

        {marin_prefix}/tmp/{prefix}

    The temp buckets are provisioned by ``infra/configure_temp_buckets.py``
    with lifecycle rules that auto-delete objects under ``ttl=Nd/`` after
    *N* days.

    Args:
        ttl_days: Lifecycle TTL in days.  Should match one of the configured
            values (1-7, 14, 30) in ``infra/configure_temp_buckets.py``.
        prefix: Optional sub-path appended after the TTL directory.
    """
    mp = marin_prefix()

    if mp.startswith("gs://"):
        region = marin_region()
        if region:
            bucket = REGION_TO_TMP_BUCKET.get(region)
            if bucket:
                path = f"gs://{bucket}/ttl={ttl_days}d"
                if prefix:
                    path = f"{path}/{prefix.strip('/')}"
                return path

    if "://" not in mp:
        mp = f"file://{mp}"
    path = f"{mp}/tmp"
    if prefix:
        path = f"{path}/{prefix.strip('/')}"
    return path


# ---------------------------------------------------------------------------
# GCS utilities
# ---------------------------------------------------------------------------


def split_gcs_path(gs_uri: str) -> tuple[str, pathlib.Path]:
    """Split a GCS URI into ``(bucket, Path(path/to/resource))``.

    Returns ``(bucket, Path("."))`` when the URI has no object path component.
    """
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI `{gs_uri}`; expected URI of form `gs://BUCKET/path/to/resource`")

    parts = gs_uri[len("gs://") :].split("/", 1)
    if len(parts) == 1:
        return parts[0], pathlib.Path(".")
    return parts[0], pathlib.Path(parts[1])


def get_bucket_location(bucket_name_or_path: str) -> str:
    """Return the GCS bucket's location (lower-cased region string)."""
    from google.cloud import storage

    if bucket_name_or_path.startswith("gs://"):
        bucket_name = split_gcs_path(bucket_name_or_path)[0]
    else:
        bucket_name = bucket_name_or_path

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    return bucket.location.lower()


def check_path_in_region(key: str, path: str, region: str, local_ok: bool = False) -> None:
    """Validate that a GCS path's bucket is in the expected region.

    Raises ``ValueError`` if the path is local (and ``local_ok`` is False)
    or if the bucket's region doesn't match *region*.  Logs a warning
    (instead of raising) when the bucket's region can't be checked due
    to permission errors.
    """
    from google.api_core.exceptions import Forbidden as GcpForbiddenException

    if not path.startswith("gs://"):
        if local_ok:
            logger.warning(f"{key} is not a GCS path: {path}. This is fine if you're running locally.")
            return
        else:
            raise ValueError(f"{key} must be a GCS path, not {path}")
    try:
        bucket_region = get_bucket_location(path)
        if region.lower() != bucket_region.lower():
            raise ValueError(
                f"{key} is not in the same region ({bucket_region}) as the VM ({region}). "
                f"This can cause performance issues and billing surprises."
            )
    except GcpForbiddenException:
        logger.warning(f"Could not check region for {key}. Be sure it's in the same region as the VM.", exc_info=True)


def check_gcs_paths_same_region(
    obj: Any,
    *,
    local_ok: bool,
    region: str | None = None,
    skip_if_prefix_contains: Sequence[str] = ("train_urls", "validation_urls"),
    region_getter: Callable[[], str | None] | None = None,
    path_checker: Callable[[str, str, str, bool], None] | None = None,
) -> None:
    """Validate that ``gs://`` paths in ``obj`` live in the current VM region."""
    if region_getter is None:
        region_getter = marin_region
    if path_checker is None:
        path_checker = check_path_in_region

    if region is None:
        region = region_getter()
        if region is None:
            if local_ok:
                logger.warning("Could not determine the region of the VM. This is fine if you're running locally.")
                return
            raise ValueError("Could not determine the region of the VM. This is required for path checks.")

    _check_paths_recursively(
        obj,
        "",
        region=region,
        local_ok=local_ok,
        skip_if_prefix_contains=tuple(skip_if_prefix_contains),
        path_checker=path_checker,
    )


def _check_paths_recursively(
    obj: Any,
    path_prefix: str,
    *,
    region: str,
    local_ok: bool,
    skip_if_prefix_contains: tuple[str, ...],
    path_checker: Callable[[str, str, str, bool], None],
) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_prefix = f"{path_prefix}.{key}" if path_prefix else str(key)
            _check_paths_recursively(
                value,
                new_prefix,
                region=region,
                local_ok=local_ok,
                skip_if_prefix_contains=skip_if_prefix_contains,
                path_checker=path_checker,
            )
        return

    if isinstance(obj, list | tuple):
        for index, item in enumerate(obj):
            new_prefix = f"{path_prefix}[{index}]"
            _check_paths_recursively(
                item,
                new_prefix,
                region=region,
                local_ok=local_ok,
                skip_if_prefix_contains=skip_if_prefix_contains,
                path_checker=path_checker,
            )
        return

    if isinstance(obj, str | os.PathLike):
        path_str = _normalize_path_like(obj)
        if path_str.startswith("gs://"):
            if any(skip_token in path_prefix for skip_token in skip_if_prefix_contains):
                return
            path_checker(path_prefix, path_str, region, local_ok)
        return

    if dataclasses.is_dataclass(obj):
        for field in dataclasses.fields(obj):
            new_prefix = f"{path_prefix}.{field.name}" if path_prefix else field.name
            _check_paths_recursively(
                getattr(obj, field.name),
                new_prefix,
                region=region,
                local_ok=local_ok,
                skip_if_prefix_contains=skip_if_prefix_contains,
                path_checker=path_checker,
            )
        return

    if not isinstance(obj, str | int | float | bool | type(None)):
        logger.warning(f"Found unexpected type {type(obj)} at {path_prefix}. Skipping.")


def _normalize_path_like(path: str | os.PathLike) -> str:
    if isinstance(path, os.PathLike):
        path_str = os.fspath(path)
        if isinstance(path, PurePath):
            parts = path.parts
            if parts and parts[0] == "gs:" and not path_str.startswith("gs://"):
                remainder = "/".join(parts[1:])
                return f"gs://{remainder}" if remainder else "gs://"
        return path_str
    return path


# ---------------------------------------------------------------------------
# Cross-region read guard
# ---------------------------------------------------------------------------

MARIN_CROSS_REGION_OVERRIDE_ENV: str = "MARIN_I_WILL_PAY_FOR_ALL_FEES"
CROSS_REGION_TRANSFER_LIMIT_BYTES: int = 10 * 1024 * 1024 * 1024  # 10 GB

# GCS multi-region bucket locations are returned as "us", "eu", or "asia"
# rather than a specific region like "us-central1".  European regions use the
# prefix "europe-" (e.g. "europe-west4") so we map the multi-region label to
# the set of region prefixes it covers.
_MULTI_REGION_TO_PREFIXES: dict[str, tuple[str, ...]] = {
    "us": ("us-",),
    "eu": ("europe-", "eu-"),
    "asia": ("asia-",),
}


class TransferBudgetExceeded(Exception):
    """Raised when cumulative cross-region bytes exceed the budget."""

    def __init__(self, bytes_used: int, attempted: int, limit: int, path: str):
        self.bytes_used = bytes_used
        self.attempted = attempted
        self.limit = limit
        self.path = path
        super().__init__(
            f"Cross-region transfer budget exceeded: {path} "
            f"({attempted / (1024**2):.1f}MB) would bring total to "
            f"{(bytes_used + attempted) / (1024**3):.2f}GB, "
            f"exceeding the {limit / (1024**3):.0f}GB limit "
            f"(already transferred {bytes_used / (1024**3):.2f}GB). "
            f"Consider running in the source region instead."
        )


class TransferBudget:
    """Thread-safe cumulative byte budget for cross-region transfers.

    Shared by CrossRegionGuardedFS (direct reads) and MirrorFileSystem
    (mirror copies).  A single process-global instance tracks total
    cross-region bytes across all filesystem instances.
    """

    __slots__ = ("_bytes_used", "_limit_bytes", "_lock")

    def __init__(self, limit_bytes: int = CROSS_REGION_TRANSFER_LIMIT_BYTES):
        self._limit_bytes = limit_bytes
        self._bytes_used: int = 0
        self._lock = threading.Lock()

    @property
    def bytes_used(self) -> int:
        return self._bytes_used

    @property
    def limit_bytes(self) -> int:
        return self._limit_bytes

    def record(self, size: int, path: str) -> None:
        """Atomically record *size* bytes.  Raise if budget exceeded.

        Does NOT increment on failure — the transfer hasn't happened yet.
        """
        with self._lock:
            new_total = self._bytes_used + size
            if new_total > self._limit_bytes:
                raise TransferBudgetExceeded(self._bytes_used, size, self._limit_bytes, path)
            self._bytes_used = new_total

    def reset(self, limit_bytes: int | None = None) -> None:
        """Reset counter to zero.  For testing only."""
        with self._lock:
            self._bytes_used = 0
            if limit_bytes is not None:
                self._limit_bytes = limit_bytes


_global_transfer_budget = TransferBudget()


@functools.lru_cache(maxsize=1)
def _cached_marin_region() -> str | None:
    """Return the current VM region, cached for the process lifetime."""
    return marin_region()


@functools.lru_cache(maxsize=256)
def _cached_bucket_location(bucket_name: str) -> str | None:
    """Return the location of a GCS bucket, cached across calls."""
    try:
        return get_bucket_location(bucket_name)
    except Exception:
        logger.debug("Could not determine location for bucket %s", bucket_name, exc_info=True)
        return None


def _regions_match(vm_region: str, bucket_location: str) -> bool:
    """Return True if *vm_region* and *bucket_location* are the same region.

    Handles GCS multi-region buckets whose location is ``"us"``, ``"eu"``,
    or ``"asia"`` rather than a specific zone.
    """
    vm = vm_region.lower()
    bl = bucket_location.lower()
    if vm == bl:
        return True
    prefixes = _MULTI_REGION_TO_PREFIXES.get(bl)
    if prefixes is not None:
        return any(vm.startswith(p) for p in prefixes)
    return False


def _fs_is_gcs(fs: Any) -> bool:
    """Return True if *fs* is a GCS-backed fsspec filesystem."""
    proto = getattr(fs, "protocol", None)
    if isinstance(proto, tuple):
        return "gs" in proto or "gcs" in proto
    return proto in ("gs", "gcs")


def _is_gcs_url(url: str) -> bool:
    """Return True if *url* starts with a GCS scheme."""
    return url.startswith("gs://") or url.startswith("gcs://")


def _is_gcs_protocol(protocol: str) -> bool:
    """Return True if *protocol* names a GCS filesystem."""
    return protocol in ("gs", "gcs")


class CrossRegionGuardedFS:
    """Wrapper around a GCS fsspec filesystem that enforces a cross-region transfer budget.

    Intercepts read operations (``open``, ``cat``, ``cat_file``, ``get_file``,
    ``get``) and records each cross-region read against a shared
    ``TransferBudget``.  Raises ``TransferBudgetExceeded`` when the cumulative
    cross-region bytes exceed the budget.

    Only constructed for GCS filesystems — the entry points (``url_to_fs``,
    ``open_url``, ``filesystem``) decide whether to wrap.

    Args:
        fs: The GCS fsspec filesystem to wrap.
        cross_region_checker: Optional callback ``(bucket_name) -> bool``
            used **only** for testing.  When provided, bypasses the default
            region-comparison logic.
        budget: Transfer budget to charge reads against.  Defaults to the
            process-global singleton.
    """

    __slots__ = ("_budget", "_cross_region_checker", "_current_region", "_fs")

    def __init__(
        self,
        fs: Any,
        *,
        cross_region_checker: Callable[[str], bool] | None = None,
        budget: TransferBudget | None = None,
    ):
        self._fs = fs
        self._cross_region_checker = cross_region_checker
        self._current_region = None if cross_region_checker else _cached_marin_region()
        self._budget = budget if budget is not None else _global_transfer_budget

    # -- cross-region detection ----------------------------------------------

    def _is_cross_region(self, bucket_name: str) -> bool:
        if self._cross_region_checker is not None:
            return self._cross_region_checker(bucket_name)
        if self._current_region is None:
            return False
        bucket_location = _cached_bucket_location(bucket_name)
        if bucket_location is None:
            return False
        return not _regions_match(self._current_region, bucket_location)

    # -- read interception ---------------------------------------------------

    def open(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        if "r" in mode:
            self._guard_read(path)
        return self._fs.open(path, mode, **kwargs)

    def cat_file(self, path: str, start: int | None = None, end: int | None = None, **kwargs: Any) -> bytes:
        self._guard_read(path)
        return self._fs.cat_file(path, start=start, end=end, **kwargs)

    def cat(self, path: Any, recursive: bool = False, on_error: str = "raise", **kwargs: Any) -> Any:
        if isinstance(path, str):
            self._guard_read(path)
        elif isinstance(path, list):
            for p in path:
                self._guard_read(p)
        return self._fs.cat(path, recursive=recursive, on_error=on_error, **kwargs)

    def get_file(self, rpath: str, lpath: str, **kwargs: Any) -> None:
        self._guard_read(rpath)
        return self._fs.get_file(rpath, lpath, **kwargs)

    def get(self, rpath: Any, lpath: Any, recursive: bool = False, **kwargs: Any) -> None:
        """Guard each remote path before delegating the bulk download."""
        if isinstance(rpath, str):
            self._guard_read(rpath)
        elif isinstance(rpath, list):
            for p in rpath:
                self._guard_read(p)
        return self._fs.get(rpath, lpath, recursive=recursive, **kwargs)

    # -- guard logic ---------------------------------------------------------

    def _guard_read(self, path: str) -> None:
        if os.environ.get(MARIN_CROSS_REGION_OVERRIDE_ENV):
            return

        # fsspec strips the protocol, so paths look like "bucket/key".
        bucket = path.split("/")[0] if "/" in path else path
        if not self._is_cross_region(bucket):
            return

        try:
            size = self._fs.size(path)
        except Exception:
            logger.warning("Failed to stat %s for cross-region guard check", path, exc_info=True)
            return

        if size is not None:
            self._budget.record(size, f"gs://{path}")

    # -- transparent delegation ----------------------------------------------

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fs, name)


# ---------------------------------------------------------------------------
# Guarded fsspec entry points
#
# These are drop-in replacements for fsspec.core.url_to_fs, fsspec.open,
# and fsspec.filesystem that automatically wrap GCS filesystems in a
# CrossRegionGuardedFS.
# ---------------------------------------------------------------------------


def url_to_fs(url: str, **kwargs: Any) -> tuple[Any, str]:
    """Like ``fsspec.core.url_to_fs`` but wraps GCS filesystems in a cross-region guard.

    Returns ``(fs, path)``.  For non-GCS URLs the filesystem is returned
    unwrapped.  ``mirror://`` URLs are handled by :class:`MirrorFileSystem`.
    """
    fs, path = fsspec.core.url_to_fs(url, **kwargs)
    if _fs_is_gcs(fs):
        fs = CrossRegionGuardedFS(fs)
    return fs, path


def open_url(url: str, mode: str = "rb", **kwargs: Any) -> fsspec.core.OpenFile:
    """Like ``fsspec.open`` but checks the cross-region budget for GCS reads.

    For read modes on GCS URLs, eagerly stats the file and charges the
    transfer budget.  Then delegates to ``fsspec.open`` for the actual I/O.
    """
    if "r" in mode and _is_gcs_url(url):
        fs, path = fsspec.core.url_to_fs(url)
        guarded = CrossRegionGuardedFS(fs)
        guarded._guard_read(path)
    return fsspec.open(url, mode, **kwargs)


def filesystem(protocol: str, **kwargs: Any) -> Any:
    """Like ``fsspec.filesystem`` but wraps GCS filesystems in a cross-region guard."""
    fs = fsspec.filesystem(protocol, **kwargs)
    if _is_gcs_protocol(protocol):
        fs = CrossRegionGuardedFS(fs)
    return fs


# ---------------------------------------------------------------------------
# Mirror filesystem
#
# Transparent cross-region file access: reads check the local prefix first,
# then scan all other marin-* data buckets and copy on first access.
# ---------------------------------------------------------------------------


def _all_data_bucket_prefixes() -> list[str]:
    """Return gs:// prefixes for all known marin data buckets."""
    return [f"gs://{bucket}" for bucket in REGION_TO_DATA_BUCKET.values()]


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
        self._local_prefix = marin_prefix()
        self._remote_prefixes = [p for p in _all_data_bucket_prefixes() if not self._local_prefix.startswith(p)]
        self._budget = budget if budget is not None else _global_transfer_budget
        self._worker_id = default_worker_id()

    # -- underlying fs helpers ------------------------------------------------

    def _get_fs_and_path(self, url: str) -> tuple[Any, str]:
        """Return (fsspec_fs, path) for a full URL or local path."""
        return fsspec.core.url_to_fs(url)

    def _local_url(self, path: str) -> str:
        return f"{self._local_prefix}/{path}"

    def _remote_url(self, prefix: str, path: str) -> str:
        return f"{prefix}/{path}"

    def _lock_path_for(self, path: str) -> str:
        return f"{self._local_prefix}/.mirror_locks/{path}.lock"

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
                self._budget.record(size, remote_url)

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

    @property
    def bytes_copied(self) -> int:
        """Total cross-region bytes transferred (shared budget)."""
        return self._budget.bytes_used


# Register the mirror:// protocol with fsspec.
fsspec.register_implementation("mirror", MirrorFileSystem)
