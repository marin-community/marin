# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cluster data config, storage-prefix/region resolution, region-local temp
storage, and GCS-location utilities.

A :class:`DataConfig` describes where a cluster's data lives: the
region-to-bucket mirror set, the URL scheme, and the temp TTL policy.
:func:`data_config` returns the active config — the one bound by
:func:`use_data_config`, else the cluster named by ``MARIN_CLUSTER`` (default
``marin``), loaded from ``config/<cluster>.yaml``. Every "where does data live"
answer flows through it: :func:`marin_prefix` is ``data_config().resolved_root()``,
and :func:`marin_temp_bucket` and the region helpers all read its fields.
Lifecycle rules on the ``marin-{region}`` buckets are managed by
``infra/configure_buckets.py``.
"""

import contextlib
import contextvars
import dataclasses
import functools
import logging
import os
import pathlib
import urllib.error
import urllib.request
from collections.abc import Callable, Generator, Mapping, Sequence
from enum import StrEnum
from pathlib import PurePath
from types import MappingProxyType
from typing import Any

import yaml
from google.api_core.exceptions import Forbidden as GcpForbiddenException
from google.cloud import storage

from rigging.config_discovery import list_cluster_configs, resolve_cluster_config
from rigging.filesystem.storage_path import StoragePath, split_gcs_path

logger = logging.getLogger(__name__)


def _bundled_cluster_config_dir() -> str | None:
    """Bundled cluster-config dir for an installed (wheel) rigging.

    Populated by the ``force-include`` in ``lib/rigging/pyproject.toml`` at
    ``rigging/clusters``. Returns ``None`` for a source/editable checkout, where
    the repo-root ``config/`` entry resolves against the marin workspace root.
    Mirrors ``iris.cli.connect._bundled_iris_config_dir``.

    This module lives at ``rigging/filesystem/cluster_config.py``, so the bundled
    ``rigging/clusters`` dir is two levels up from this file.
    """
    bundled = pathlib.Path(__file__).resolve().parent.parent / "clusters"
    return str(bundled) if bundled.is_dir() else None


# A per-user override dir for connecting to a live cluster (e.g. on a dev VM).
# Named so tests can filter it out of MARIN_CLUSTER_CONFIG_DIRS by identity
# rather than re-hardcoding the path, since it makes cluster-config resolution
# host-dependent: a document there may lack a `data:` block, silently swapping
# the loaded DataConfig out from under a test that expects the committed
# config/marin.yaml layout.
PER_USER_CLUSTER_CONFIG_DIR = "~/.config/marin/clusters"

# Cluster config search dirs, highest priority first: a per-user override, the
# repo-root ``config/`` directory (in-tree checkout), then the bundled copy for
# installed wheels. Relative paths resolve against the marin workspace root via
# :func:`rigging.config_discovery.resolve_cluster_config`.
MARIN_CLUSTER_CONFIG_DIRS: tuple[str, ...] = tuple(
    p
    for p in (
        PER_USER_CLUSTER_CONFIG_DIR,
        "config",
        _bundled_cluster_config_dir(),
    )
    if p is not None
)

_MARIN_PREFIX_ENV = "MARIN_PREFIX"
_MARIN_CLUSTER_ENV = "MARIN_CLUSTER"
_GCP_METADATA_ZONE_URL = "http://metadata.google.internal/computeMetadata/v1/instance/zone"
_DEFAULT_LOCAL_PREFIX = "/tmp/marin"


class StoreType(StrEnum):
    """Object-storage backend serving a bucket.

    Distinguishes the two S3-compatible backends — which share the ``s3://``
    scheme but differ in endpoint, addressing, and credentials — from GCS.
    """

    GCS = "gcs"
    R2 = "r2"
    COREWEAVE = "coreweave"


@dataclasses.dataclass(frozen=True)
class BucketSpec:
    """A single data bucket: its name and which backend serves it.

    Attributes:
        name: Bucket name without scheme, e.g. ``marin-us-east1`` or ``marin-na``.
        store: Backend serving the bucket.
        signing_region: S3 signing region for backends that require one
            (CoreWeave, e.g. ``US-EAST-02A``). Distinct from the *placement*
            region (the ``region_buckets`` key). ``None`` for GCS (not S3) and R2
            (which signs with ``"auto"``).
    """

    name: str
    store: StoreType
    signing_region: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    """Where a cluster's data lives — the single source for storage layout.

    Attributes:
        region_buckets: Region name -> :class:`BucketSpec` for the cross-region
            mirror set.
        scheme: URL scheme for the cluster's storage (e.g. ``"gs"`` or ``"s3"``).
        temp_path: Path segment for TTL-managed scratch data.
        ttl_days: Allowed TTL-day values for temp lifecycle rules.
        root: Explicit single-prefix root (e.g. ``"s3://marin-na/marin"``). Set
            only for clusters that do not use region-local bucket selection.
    """

    region_buckets: Mapping[str, BucketSpec]
    scheme: str = "gs"
    temp_path: str = "tmp"
    ttl_days: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 14, 30)
    root: str | None = None

    def resolved_root(self) -> str:
        """Resolve the storage root for this config. Never returns ``None``.

        Precedence: ``MARIN_PREFIX`` env > ``self.root`` > region-local bucket
        from ``region_buckets[<gcs metadata region>]`` > ``{scheme}://marin-{region}``
        for a detected-but-unmapped region > :data:`_DEFAULT_LOCAL_PREFIX`.

        The env/explicit value is canonicalized through :class:`StoragePath` (trailing
        ``/`` stripped, interior ``//`` collapsed) so downstream joins never double the
        separator.
        """
        env_prefix = os.environ.get(_MARIN_PREFIX_ENV)
        if env_prefix:
            return StoragePath.normalize(env_prefix)
        if self.root is not None:
            return StoragePath.normalize(self.root)
        region = region_from_metadata()
        if region is not None:
            spec = self.region_buckets.get(region)
            if spec is not None:
                return f"{self.scheme}://{spec.name}"
            return f"{self.scheme}://marin-{region}"
        return _DEFAULT_LOCAL_PREFIX


# The marin cluster's storage layout lives in ``config/marin.yaml`` (loaded as
# the default below). This in-code config is only a degraded fallback for when
# no config file is discoverable — e.g. an installed package running outside a
# marin checkout. Such contexts set ``MARIN_PREFIX`` (which wins in
# ``resolved_root``) or detect a region (constructing ``gs://marin-{region}``),
# so an empty ``region_buckets`` is sufficient.
_DEFAULT_CLUSTER = "marin"
_FALLBACK_DATA_CONFIG: DataConfig = DataConfig(region_buckets={})

_active_data_config: contextvars.ContextVar[DataConfig | None] = contextvars.ContextVar(
    "marin_data_config", default=None
)


def data_config() -> DataConfig:
    """Return the active :class:`DataConfig`.

    Resolution: the config bound by :func:`use_data_config` (a context-local
    override), else the cluster named by ``MARIN_CLUSTER`` (default ``marin``)
    loaded from its ``config/<cluster>.yaml``.
    """
    override = _active_data_config.get()
    if override is not None:
        return override
    return load_cluster_config()


@contextlib.contextmanager
def use_data_config(config: DataConfig) -> Generator[DataConfig, None, None]:
    """Bind *config* as the active :class:`DataConfig` for the duration of the block."""
    token = _active_data_config.set(config)
    try:
        yield config
    finally:
        _active_data_config.reset(token)


def load_cluster_config(cluster: str | None = None) -> DataConfig:
    """Load a cluster's :class:`DataConfig` from its ``config/<cluster>.yaml``.

    The cluster name is ``cluster`` arg > ``MARIN_CLUSTER`` env > ``marin``. A
    parsed ``data:`` block becomes the config; other keys (e.g. ``iris:``) are
    ignored. When the default ``marin`` config cannot be found (e.g. an installed
    package outside a checkout), returns :data:`_FALLBACK_DATA_CONFIG`; a missing
    *named* cluster raises ``FileNotFoundError``. Cached; call
    :func:`reset_data_config_cache` in tests after changing env or config files.
    """
    name = cluster or os.environ.get(_MARIN_CLUSTER_ENV) or _DEFAULT_CLUSTER
    return _load_cluster_config_cached(name)


@functools.cache
def _load_cluster_config_cached(cluster: str) -> DataConfig:
    try:
        config_path = resolve_cluster_config(cluster, MARIN_CLUSTER_CONFIG_DIRS)
    except FileNotFoundError:
        if cluster == _DEFAULT_CLUSTER:
            return _FALLBACK_DATA_CONFIG
        raise
    with config_path.open("rb") as f:
        document = yaml.safe_load(f) or {}
    data = document.get("data")
    if not data:
        return _FALLBACK_DATA_CONFIG
    return _parse_data_config(data)


def _parse_bucket_spec(value: object) -> BucketSpec:
    """Normalize one ``region_buckets`` YAML entry into a :class:`BucketSpec`.

    Each entry is an explicit mapping ``{bucket, store[, signing_region]}``. The
    ``store`` (``gcs``/``r2``/``coreweave``) is required because it cannot be
    inferred — R2 and CoreWeave share the ``s3`` scheme but need different
    endpoints and credentials. CoreWeave entries must carry a ``signing_region``.
    """
    if not isinstance(value, Mapping):
        raise ValueError(
            f"region_buckets entry must be a mapping {{bucket, store[, signing_region]}}, "
            f"got {type(value).__name__}: {value!r}"
        )
    if "bucket" not in value or "store" not in value:
        raise ValueError(f"region_buckets entry must set 'bucket' and 'store': {value!r}")
    store = StoreType(str(value["store"]))
    signing_region = str(value["signing_region"]) if value.get("signing_region") is not None else None
    if store is StoreType.COREWEAVE and signing_region is None:
        raise ValueError(f"CoreWeave bucket {value['bucket']!r} must specify a 'signing_region'.")
    return BucketSpec(name=str(value["bucket"]), store=store, signing_region=signing_region)


def _parse_data_config(data: Mapping[str, object]) -> DataConfig:
    """Build a :class:`DataConfig` from a parsed ``data:`` config block.

    Keys absent from the block fall back to the :class:`DataConfig` field
    defaults, so per-field defaults are defined once on the dataclass.
    """
    temp = data.get("temp") or {}
    raw_ttl = temp.get("ttl_days")
    root = data.get("root")
    scheme = str(data.get("scheme") or DataConfig.scheme)
    raw_buckets = data.get("region_buckets") or {}
    region_buckets = {region: _parse_bucket_spec(value) for region, value in raw_buckets.items()}
    return DataConfig(
        region_buckets=region_buckets,
        scheme=scheme,
        temp_path=str(temp.get("path") or DataConfig.temp_path),
        ttl_days=tuple(raw_ttl) if raw_ttl is not None else DataConfig.ttl_days,
        root=str(root) if root is not None else None,
    )


def reset_data_config_cache() -> None:
    """Clear the cluster-config and S3-bucket-registry caches. For tests."""
    _load_cluster_config_cached.cache_clear()
    s3_data_buckets.cache_clear()


# ---------------------------------------------------------------------------
# Region + prefix resolution
# ---------------------------------------------------------------------------


def region_from_metadata() -> str | None:
    """Derive the GCP region from the instance metadata server, or ``None``."""
    try:
        req = urllib.request.Request(_GCP_METADATA_ZONE_URL, headers={"Metadata-Flavor": "Google"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            zone = resp.read().decode().strip().split("/")[-1]
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        return None
    if "-" not in zone:
        return None
    return zone.rsplit("-", 1)[0]


def region_from_prefix(prefix: str) -> str | None:
    """Extract the canonical GCP region from a ``gs://marin-{region}/…`` prefix.

    Bucket names are normalized through the active config's ``region_buckets``
    (e.g. ``gs://marin-eu-west4`` -> ``europe-west4``); unknown ``marin-``
    buckets fall back to stripping the ``marin-`` prefix.
    """
    parsed = StoragePath.parse(prefix)
    if parsed.scheme != "gs" or not parsed.bucket:
        return None
    bucket = parsed.bucket
    for region, spec in data_config().region_buckets.items():
        if spec.name == bucket:
            return region
    if bucket.startswith("marin-"):
        return bucket[len("marin-") :]
    return None


def marin_region() -> str | None:
    """Return the current GCP region, from instance metadata or ``MARIN_PREFIX``."""
    return region_from_metadata() or region_from_prefix(os.environ.get(_MARIN_PREFIX_ENV, ""))


def marin_prefix() -> str:
    """Return the active cluster's storage prefix (``data_config().resolved_root()``)."""
    return data_config().resolved_root()


@functools.cache
def s3_data_buckets() -> Mapping[str, BucketSpec]:
    """R2/CoreWeave data buckets (name -> :class:`BucketSpec`) across all configs.

    These S3-compatible buckets carry ``tmp/ttl=Nd/`` lifecycle rules; used to
    route temp paths (:func:`marin_temp_bucket`) and to drive
    ``infra/configure_buckets.py``. The set is defined in ``config/*.yaml`` via
    each bucket's ``store`` type (``r2``/``coreweave``).

    Recognition must be cross-cluster — a launcher on a GCS cluster may target an
    R2/CoreWeave output prefix (see :func:`marin_temp_bucket`'s ``source_prefix``)
    — so this aggregates across all cluster configs rather than only the active
    one. Cached; :func:`reset_data_config_cache` clears it.
    """
    registry: dict[str, BucketSpec] = {}
    for cluster in list_cluster_configs(MARIN_CLUSTER_CONFIG_DIRS):
        for spec in load_cluster_config(cluster).region_buckets.values():
            if spec.store in (StoreType.R2, StoreType.COREWEAVE):
                registry.setdefault(spec.name, spec)
    return MappingProxyType(registry)


# ---------------------------------------------------------------------------
# Temp storage
# ---------------------------------------------------------------------------


def _s3_bucket_from_prefix(prefix: str | None) -> str | None:
    """Return the bucket from an ``s3://bucket/…`` prefix, or ``None``.

    Only recognizes buckets in :func:`s3_data_buckets` (the R2/CoreWeave buckets
    with lifecycle rules configured by ``infra/configure_buckets.py``), so unknown
    S3 buckets fall through to the flat non-TTL fallback instead of getting a
    ``tmp/ttl=Nd/`` path that would never be cleaned up.
    """
    if not prefix:
        return None
    parsed = StoragePath.parse(prefix)
    if parsed.scheme != "s3":
        return None
    return parsed.bucket if parsed.bucket in s3_data_buckets() else None


def _append_path_prefix(path: str, prefix: str) -> str:
    if prefix:
        return f"{path}/{prefix.strip('/')}"
    return path


def _resolve_ttl_days(ttl_days: int, allowed: tuple[int, ...]) -> int:
    """Map *ttl_days* to the smallest *allowed* value that is ``>= ttl_days``.

    Requests above the largest allowed value clamp to that maximum (with
    a warning) — temp data is by definition disposable, so capping the TTL
    is preferable to forcing the caller to handle an exception. Logs a
    warning whenever the requested value is rounded.
    """
    if ttl_days <= 0:
        raise ValueError(f"ttl_days={ttl_days} must be positive. Allowed values: {allowed}.")
    if ttl_days in allowed:
        return ttl_days
    for n in allowed:
        if n > ttl_days:
            logger.warning("ttl_days=%d not configured; rounding up to %d", ttl_days, n)
            return n
    capped = max(allowed)
    logger.warning("ttl_days=%d exceeds the configured maximum; clamping to %d", ttl_days, capped)
    return capped


def marin_temp_bucket(ttl_days: int, prefix: str = "", *, source_prefix: str | None = None) -> str:
    """Return a path on region-local temp storage. Never returns ``None``.

    For a GCS marin prefix with a known region, or an explicitly provided
    ``source_prefix`` with a known region, returns a path under the
    region-local marin bucket::

        gs://marin-{region}/tmp/ttl={N}d/{prefix}

    For a known S3-compatible prefix — an R2 or CoreWeave bucket in
    :func:`s3_data_buckets` — returns a path at the bucket root::

        s3://marin-na/tmp/ttl={N}d/{prefix}
        s3://marin-us-east-02a/tmp/ttl={N}d/{prefix}

    Otherwise falls back to a flat path under the marin prefix::

        {marin_prefix}/tmp/{prefix}

    Lifecycle rules on each ``marin-{region}`` GCS bucket and each R2/CoreWeave
    data bucket — managed by ``infra/configure_buckets.py`` — auto-delete objects
    under ``tmp/ttl=Nd/`` after *N* days.

    Args:
        ttl_days: Lifecycle TTL in days.  Values not in the active config's
            ``ttl_days`` are rounded up to the nearest configured value (with a
            warning); values above the maximum clamp to it.  Non-positive values
            raise :class:`ValueError`.
        prefix: Optional sub-path appended after the TTL directory.
        source_prefix: Optional path used to choose the temp bucket region.
            Useful when configuring a remote job from a launcher that may be in
            a different region than the job output path.
    """
    cfg = data_config()
    ttl_days = _resolve_ttl_days(ttl_days, cfg.ttl_days)

    mp = marin_prefix()

    # An explicit source_prefix fully determines the backend and region, taking
    # precedence over the ambient marin prefix and VM metadata so that an R2
    # source_prefix yields an R2 temp path even on a GCP launcher. Only when
    # source_prefix is absent do we derive the location from the marin prefix
    # (and VM metadata for the GCS region).
    if source_prefix is not None:
        region = region_from_prefix(source_prefix)
        s3_bucket = _s3_bucket_from_prefix(source_prefix)
    else:
        region = marin_region() if mp.startswith("gs://") else None
        s3_bucket = _s3_bucket_from_prefix(mp)

    if region:
        spec = cfg.region_buckets.get(region)
        if spec:
            path = f"gs://{spec.name}/{cfg.temp_path}/ttl={ttl_days}d"
            return _append_path_prefix(path, prefix)

    # R2 and CoreWeave temp lives at the bucket root so the `tmp/ttl=Nd/`
    # lifecycle prefix configured by infra/configure_buckets.py applies. The
    # bucket already pins the region (R2 is non-regional; CoreWeave encodes it in
    # the name, e.g. marin-us-east-02a), and the runtime marin prefix carries a
    # `marin/` data subdir (e.g. `s3://marin-na/marin`) that we deliberately strip.
    if s3_bucket:
        path = f"s3://{s3_bucket}/{cfg.temp_path}/ttl={ttl_days}d"
        return _append_path_prefix(path, prefix)

    if "://" not in mp:
        mp = f"file://{mp}"
    path = f"{mp}/{cfg.temp_path}"
    return _append_path_prefix(path, prefix)


# ---------------------------------------------------------------------------
# GCS utilities
# ---------------------------------------------------------------------------


def get_bucket_location(bucket_name_or_path: str) -> str:
    """Return the GCS bucket's location (lower-cased region string)."""
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

    for key, path in collect_gcs_paths(
        obj,
        path_prefix="",
        skip_if_prefix_contains=skip_if_prefix_contains,
    ):
        path_checker(key, path, region, local_ok)


def collect_gcs_paths(
    obj: Any,
    *,
    path_prefix: str = "",
    skip_if_prefix_contains: Sequence[str] = ("train_urls", "validation_urls"),
) -> list[tuple[str, str]]:
    """Collect ``(path_key, gs://...)`` entries found recursively in ``obj``."""
    paths: list[tuple[str, str]] = []
    _collect_gcs_paths_recursively(
        obj,
        path_prefix,
        skip_if_prefix_contains=tuple(skip_if_prefix_contains),
        out=paths,
    )
    return paths


def _collect_gcs_paths_recursively(
    obj: Any,
    path_prefix: str,
    *,
    skip_if_prefix_contains: tuple[str, ...],
    out: list[tuple[str, str]],
) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_prefix = f"{path_prefix}.{key}" if path_prefix else str(key)
            _collect_gcs_paths_recursively(
                value,
                new_prefix,
                skip_if_prefix_contains=skip_if_prefix_contains,
                out=out,
            )
        return

    if isinstance(obj, list | tuple | set):
        for index, item in enumerate(obj):
            new_prefix = f"{path_prefix}[{index}]"
            _collect_gcs_paths_recursively(
                item,
                new_prefix,
                skip_if_prefix_contains=skip_if_prefix_contains,
                out=out,
            )
        return

    if isinstance(obj, str | os.PathLike):
        path_str = _normalize_path_like(obj)
        if path_str.startswith("gs://"):
            if any(skip_token in path_prefix for skip_token in skip_if_prefix_contains):
                return
            out.append((path_prefix, path_str))
        return

    if dataclasses.is_dataclass(obj):
        for field in dataclasses.fields(obj):
            new_prefix = f"{path_prefix}.{field.name}" if path_prefix else field.name
            _collect_gcs_paths_recursively(
                getattr(obj, field.name),
                new_prefix,
                skip_if_prefix_contains=skip_if_prefix_contains,
                out=out,
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
