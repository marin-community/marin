# Copyright 2025 The Marin Authors
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

Cross-region read guard:
  ``CrossRegionGuardedFS`` wraps an fsspec filesystem and blocks reads of
  large files from GCS buckets in a different region than the current VM.
  Prefer the guarded helpers (``url_to_fs``, ``open_url``, ``filesystem``)
  over the raw fsspec equivalents; they automatically wrap GCS filesystems
  in the guard.  Set the ``MARIN_I_WILL_PAY_FOR_ALL_FEES`` env var to a
  username to override the guard.
"""

import dataclasses
import functools
import logging
import os
import pathlib
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import PurePath
from typing import Any

import fsspec

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
    """Extract region from a ``gs://marin-{region}/…`` prefix string."""
    m = re.match(r"gs://marin-([^/]+)", prefix)
    return m.group(1) if m else None


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
        return f"gs://marin-{region}"
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

CROSS_REGION_READ_THRESHOLD_BYTES: int = 100 * 1024 * 1024  # 100 MB
MARIN_CROSS_REGION_OVERRIDE_ENV: str = "MARIN_I_WILL_PAY_FOR_ALL_FEES"

# GCS multi-region bucket locations are returned as "us", "eu", or "asia"
# rather than a specific region like "us-central1".  European regions use the
# prefix "europe-" (e.g. "europe-west4") so we map the multi-region label to
# the set of region prefixes it covers.
_MULTI_REGION_TO_PREFIXES: dict[str, tuple[str, ...]] = {
    "us": ("us-",),
    "eu": ("europe-", "eu-"),
    "asia": ("asia-",),
}


class CrossRegionReadError(Exception):
    """Raised when a cross-region GCS read exceeds the size threshold."""


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
    """Wrapper around an fsspec filesystem that blocks large cross-region GCS reads.

    Caches the VM region and GCS detection at construction time so that
    per-read overhead is minimal (no metadata-server round-trips).

    Intercepts read operations (``open``, ``cat``, ``cat_file``, ``get_file``,
    ``get``) and checks whether the target file lives in a GCS bucket in a
    different region than the current VM.  If the file exceeds
    *threshold_bytes*, raises ``CrossRegionReadError``.

    The guard is skipped when:
    * The ``MARIN_I_WILL_PAY_FOR_ALL_FEES`` env var is set.
    * The underlying filesystem is not GCS.
    * The bucket is in the same region (or region cannot be determined).
    * The file is smaller than the threshold.

    Args:
        fs: The fsspec filesystem to wrap.
        threshold_bytes: Maximum allowed file size for cross-region reads.
        cross_region_checker: Optional callback ``(bucket_name) -> bool``
            used **only** for testing.  When provided, bypasses the default
            region-comparison logic.
    """

    __slots__ = ("_cross_region_checker", "_current_region", "_fs", "_is_gcs", "_threshold_bytes")

    def __init__(
        self,
        fs: Any,
        *,
        threshold_bytes: int = CROSS_REGION_READ_THRESHOLD_BYTES,
        cross_region_checker: Callable[[str], bool] | None = None,
    ):
        self._fs = fs
        self._threshold_bytes = threshold_bytes
        self._is_gcs = _fs_is_gcs(fs)
        self._cross_region_checker = cross_region_checker
        # Cache the VM region once at construction so _guard_read never
        # hits the metadata server.
        self._current_region = None if cross_region_checker else _cached_marin_region()

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
        if not self._is_gcs:
            return

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

        if size is not None and size > self._threshold_bytes:
            msg = (
                f"Cross-region read blocked: gs://{path} is {size / (1024 * 1024):.1f}MB "
                f"(threshold: {self._threshold_bytes / (1024 * 1024):.0f}MB). "
                f"Set {MARIN_CROSS_REGION_OVERRIDE_ENV}=<your-username> to override."
            )
            logger.warning(msg)
            raise CrossRegionReadError(msg)

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
    unwrapped.
    """
    fs, path = fsspec.core.url_to_fs(url, **kwargs)
    if _fs_is_gcs(fs):
        fs = CrossRegionGuardedFS(fs)
    return fs, path


def open_url(url: str, mode: str = "rb", **kwargs: Any) -> fsspec.core.OpenFile:
    """Like ``fsspec.open`` but checks the cross-region guard for GCS reads.

    For read modes on GCS URLs, eagerly stats the file and raises
    ``CrossRegionReadError`` if it exceeds the size threshold in a
    cross-region bucket.  Then delegates to ``fsspec.open`` for the actual I/O.
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
