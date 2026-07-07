# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin filesystem: the cluster data config, storage-prefix/region resolution,
region-local temp storage, and cross-region read guards.

A :class:`DataConfig` describes where a cluster's data lives: the
region-to-bucket mirror set, the URL scheme, and the temp TTL policy.
:func:`data_config` returns the active config — the one
bound by :func:`use_data_config`, else the cluster named by ``MARIN_CLUSTER``
(default ``marin``), loaded from ``config/<cluster>.yaml``. Every "where does
data live" answer flows through it: :func:`marin_prefix` is
``data_config().resolved_root()``, and :func:`marin_temp_bucket`, the mirror
filesystem, and the region helpers all read its fields. Lifecycle rules on the
``marin-{region}`` buckets are managed by ``infra/configure_buckets.py``.

Prefix resolution chain (:meth:`DataConfig.resolved_root`):
  1. ``MARIN_PREFIX`` environment variable
  2. an explicit ``root`` (single-prefix clusters, e.g. R2)
  3. the region-local bucket ``region_buckets[<gcs metadata region>]``
  4. ``gs://marin-{region}`` for a detected-but-unmapped region
  5. ``/tmp/marin`` (local fallback)

Cross-region transfer budget:
  ``TransferBudget`` tracks cumulative cross-region GCS bytes across all
  filesystem instances in the process (default 10 GB).  Both
  ``CrossRegionGuardedFS`` (direct reads) and ``MirrorFileSystem`` (mirror
  copies) charge against the same global budget.  Prefer the guarded helpers
  (``url_to_fs``, ``open_url``, ``filesystem``) over the raw fsspec
  equivalents; they automatically wrap GCS filesystems in the guard.  Set
  the ``MARIN_I_WILL_PAY_FOR_ALL_FEES`` env var to override the guard.
"""

import contextlib
import contextvars
import dataclasses
import functools
import logging
import os
import pathlib
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Generator, Mapping, Sequence
from enum import StrEnum
from pathlib import PurePath
from types import MappingProxyType
from typing import Any, cast

import fsspec
import yaml
from fsspec.callbacks import DEFAULT_CALLBACK, Callback
from fsspec.implementations.local import LocalFileSystem
from google.api_core.exceptions import Forbidden as GcpForbiddenException
from google.cloud import storage

from rigging.config_discovery import list_cluster_configs, resolve_cluster_config
from rigging.distributed_lock import create_lock, default_worker_id
from rigging.timing import ExponentialBackoff, retry_with_backoff

logger = logging.getLogger(__name__)


def _bundled_cluster_config_dir() -> str | None:
    """Bundled cluster-config dir for an installed (wheel) rigging.

    Populated by the ``force-include`` in ``lib/rigging/pyproject.toml`` at
    ``rigging/clusters``. Returns ``None`` for a source/editable checkout, where
    the repo-root ``config/`` entry resolves against the marin workspace root.
    Mirrors ``iris.cli.connect._bundled_iris_config_dir``.
    """
    bundled = pathlib.Path(__file__).resolve().parent / "clusters"
    return str(bundled) if bundled.is_dir() else None


# Cluster config search dirs, highest priority first: a per-user override, the
# repo-root ``config/`` directory (in-tree checkout), then the bundled copy for
# installed wheels. Relative paths resolve against the marin workspace root via
# :func:`rigging.config_discovery.resolve_cluster_config`.
MARIN_CLUSTER_CONFIG_DIRS: tuple[str, ...] = tuple(
    p
    for p in (
        "~/.config/marin/clusters",
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


def _key_segments(key: str) -> tuple[str, ...]:
    return tuple(part for part in key.split("/") if part)


@dataclasses.dataclass(frozen=True)
class StoragePath:
    """A parsed storage location: URL scheme, authority, and key segments.

    Object-store keys are not normalized — a doubled ``/`` addresses a *different* key —
    so joins here are structural: a segment never contains a separator, which makes a
    doubled or trailing separator unrepresentable. ``parse`` -> ``str`` is
    therefore canonicalizing (interior ``//`` collapsed, trailing ``/`` stripped), not
    byte-preserving.

    Paths at rest (configs, artifact records, CLI args) stay ``str``: parse at a
    boundary, manipulate, and ``str()`` back out. See :func:`prefix_join` for the
    single-join convenience.
    """

    scheme: str | None
    """URL scheme (``gs``, ``s3``, ``mirror``), or ``None`` for a local path."""
    netloc: str
    """Bucket/authority; empty for an empty-authority scheme like ``mirror://``."""
    segments: tuple[str, ...]
    """Key segments; never empty strings."""
    rooted: bool = True
    """Whether a ``/`` precedes the key: a local path's absoluteness (``/tmp/x`` vs
    ``rel/x``), or the empty-authority join convention (``file:///x`` vs ``mirror://x``).
    Irrelevant when ``netloc`` is non-empty."""

    @staticmethod
    def parse(value: str) -> "StoragePath":
        if "://" in value:
            scheme, rest = value.split("://", 1)
            netloc, sep, key = rest.partition("/")
            # With an authority the key is always /-separated, so rooted is pinned True
            # to keep value equality and relative_to independent of a trailing slash.
            return StoragePath(
                scheme=scheme, netloc=netloc, segments=_key_segments(key), rooted=bool(netloc) or bool(sep)
            )
        return StoragePath(scheme=None, netloc="", segments=_key_segments(value), rooted=value.startswith("/"))

    @staticmethod
    def normalize(value: str) -> str:
        """``value`` in canonical single-separator form (``str(StoragePath.parse(value))``)."""
        return str(StoragePath.parse(value))

    def __truediv__(self, relative: str) -> "StoragePath":
        if "://" in relative or relative.startswith("/"):
            raise ValueError(f"cannot join non-relative path {relative!r} onto {self}")
        return dataclasses.replace(self, segments=self.segments + _key_segments(relative))

    def relative_to(self, base: "StoragePath") -> str:
        """The ``/``-joined segments of this path under ``base``.

        Structural containment — compares parsed segments, not string prefixes, so a
        doubled separator on either side cannot fork the answer.
        """
        same_root = (self.scheme, self.netloc, self.rooted) == (base.scheme, base.netloc, base.rooted)
        if not same_root or self.segments[: len(base.segments)] != base.segments:
            raise ValueError(f"{self} is not under {base}")
        return "/".join(self.segments[len(base.segments) :])

    @property
    def bucket(self) -> str:
        """The object-store bucket (the authority); empty for a local or empty-authority path."""
        return self.netloc

    @property
    def key(self) -> str:
        """The ``/``-joined key segments beneath the authority (no leading or trailing ``/``)."""
        return "/".join(self.segments)

    @property
    def name(self) -> str:
        """The last key segment (basename), or ``""`` at the authority/filesystem root."""
        return self.segments[-1] if self.segments else ""

    @property
    def parent(self) -> "StoragePath":
        """This path with its last key segment removed; unchanged once at the root."""
        if not self.segments:
            return self
        return dataclasses.replace(self, segments=self.segments[:-1])

    def __str__(self) -> str:
        key = "/".join(self.segments)
        if self.scheme is None:
            return f"/{key}" if self.rooted else key
        root = f"{self.scheme}://{self.netloc}"
        if not key:
            return root
        if self.netloc or self.rooted:
            return f"{root}/{key}"
        return f"{root}{key}"


def prefix_join(prefix: str, relative: str) -> str:
    """Join a relative path onto a storage prefix with exactly one ``/`` separator.

    Object-store keys are not normalized: a naive ``f"{prefix}/{relative}"`` join of a
    trailing-slash prefix produces a doubled separator — a *different* key — silently
    splitting writers from slash-collapsing readers. ``str``-in/``str``-out
    convenience over :class:`StoragePath` for a single join; parse once and use ``/``
    for repeated manipulation.
    """
    return str(StoragePath.parse(prefix) / relative)


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


# Finite botocore timeouts/retries for every S3/R2 filesystem we build.
# s3fs/aiobotocore default to *no* read or connect timeout, so a silently dead
# R2 connection wedges ``upload_part`` forever (#6487): the blocked socket never
# raises, the shard never completes, and the sequential stage barrier stalls the
# whole job. With finite timeouts the wedge becomes a retryable error that fails
# the shard, which the coordinator then re-queues.
_S3_CONNECT_TIMEOUT = 30
_S3_READ_TIMEOUT = 120
_S3_RETRY_MAX_ATTEMPTS = 5


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


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


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


def split_gcs_path(gs_uri: str) -> tuple[str, pathlib.Path]:
    """Split a GCS URI into ``(bucket, Path(path/to/resource))``.

    Returns ``(bucket, Path("."))`` when the URI has no object path component.
    """
    parsed = StoragePath.parse(gs_uri)
    if parsed.scheme != "gs":
        raise ValueError(f"Invalid GCS URI `{gs_uri}`; expected URI of form `gs://BUCKET/path/to/resource`")

    key = parsed.key
    return parsed.bucket, pathlib.Path(key) if key else pathlib.Path(".")


def rebase_file_path(
    base_in_path: str,
    file_path: str,
    base_out_path: str,
    new_extension: str | None = None,
    old_extension: str | None = None,
) -> str:
    """Rebase ``file_path`` from under ``base_in_path`` to under ``base_out_path``.

    The path below ``base_in_path`` is preserved beneath ``base_out_path``, optionally
    swapping the file extension. Containment and joins are structural (via
    :class:`StoragePath`), so a trailing or doubled separator on any argument cannot
    double the output separator. ``file_path`` must lie under ``base_in_path``;
    otherwise a ``ValueError`` is raised.

    Args:
        base_in_path: The base directory of the input file.
        file_path: The path of the input file, under ``base_in_path``.
        base_out_path: The base directory of the output file.
        new_extension: New file extension including the dot (e.g. ``".parquet"``).
        old_extension: When given with ``new_extension``, the suffix of ``file_path`` to
            replace; a ``ValueError`` is raised if ``file_path`` does not end with it.
            When omitted (but ``new_extension`` is set), everything after the last dot is
            replaced; with no dot, ``new_extension`` is appended.
    """
    rel_path = StoragePath.parse(file_path).relative_to(StoragePath.parse(base_in_path))

    if old_extension and not new_extension:
        raise ValueError("old_extension requires new_extension to be set")

    if new_extension:
        if old_extension:
            # endswith (not rfind) so a mismatch fails loudly instead of silently
            # truncating: rfind returns -1 and rel_path[:-1] would drop a character.
            if not rel_path.endswith(old_extension):
                raise ValueError(
                    f"Cannot rebase {file_path!r}: relative path {rel_path!r} does not end with "
                    f"old_extension={old_extension!r}"
                )
            rel_path = rel_path[: -len(old_extension)] + new_extension
        else:
            dot_idx = rel_path.rfind(".")
            rel_path = (rel_path[:dot_idx] if dot_idx != -1 else rel_path) + new_extension
    return prefix_join(base_out_path, rel_path)


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


# ---------------------------------------------------------------------------
# Cross-region read guard
# ---------------------------------------------------------------------------

MARIN_CROSS_REGION_OVERRIDE_ENV: str = "MARIN_I_WILL_PAY_FOR_ALL_FEES"
MARIN_MIRROR_BUDGET_ENV: str = "MARIN_MIRROR_BUDGET_GB"
_DEFAULT_TRANSFER_LIMIT_GB: int = 10


def _transfer_limit_bytes() -> int:
    raw = os.environ.get(MARIN_MIRROR_BUDGET_ENV, "")
    if raw:
        return int(float(raw) * 1024 * 1024 * 1024)
    return _DEFAULT_TRANSFER_LIMIT_GB * 1024 * 1024 * 1024


CROSS_REGION_TRANSFER_LIMIT_BYTES: int = _transfer_limit_bytes()

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
        # Pass the constructor arguments — not the rendered message — to
        # BaseException. The default exception reduce reconstructs via
        # ``TransferBudgetExceeded(*self.args)`` on unpickle, so ``args`` must
        # match this signature; storing the single message string instead made
        # the exception un-revivable (``TypeError: missing 3 required positional
        # arguments``) whenever it crossed a process boundary. The human-readable
        # message is rendered lazily by ``__str__``.
        super().__init__(bytes_used, attempted, limit, path)

    def __str__(self) -> str:
        return (
            f"Cross-region transfer budget exceeded: {self.path} "
            f"({self.attempted / (1024**2):.1f}MB) would bring total to "
            f"{(self.bytes_used + self.attempted) / (1024**3):.2f}GB, "
            f"exceeding the {self.limit / (1024**3):.0f}GB limit "
            f"(already transferred {self.bytes_used / (1024**3):.2f}GB). "
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

_mirror_budget_ctx: contextvars.ContextVar[TransferBudget | None] = contextvars.ContextVar(
    "_mirror_budget_ctx", default=None
)


def set_mirror_budget(budget_gb: float) -> contextvars.Token:
    """Set the MirrorFileSystem transfer budget for the current context.

    Returns a token that can be used to reset the budget.
    """
    budget = TransferBudget(limit_bytes=int(budget_gb * 1024 * 1024 * 1024))
    return _mirror_budget_ctx.set(budget)


def reset_mirror_budget(token: contextvars.Token) -> None:
    """Reset the MirrorFileSystem transfer budget to its previous value."""
    _mirror_budget_ctx.reset(token)


@contextlib.contextmanager
def mirror_budget(budget_gb: float) -> Generator[None, None, None]:
    """Context manager to scope a MirrorFileSystem transfer budget."""
    token = set_mirror_budget(budget_gb)
    try:
        yield
    finally:
        reset_mirror_budget(token)


@functools.lru_cache(maxsize=1)
def cached_marin_region() -> str | None:
    """Return the current VM region, cached for the process lifetime (the VM region is stable)."""
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


def _bucket_from_gcs_url(url: str) -> str | None:
    """Return the bucket name from a ``gs://``/``gcs://`` URL, or ``None``."""
    parsed = StoragePath.parse(url)
    if parsed.scheme in ("gs", "gcs"):
        return parsed.bucket
    return None


def _is_cross_region_url(url: str) -> bool:
    """Return True if *url* points to a GCS bucket in a different region than the VM."""
    if os.environ.get(MARIN_CROSS_REGION_OVERRIDE_ENV):
        return False
    bucket = _bucket_from_gcs_url(url)
    if bucket is None:
        return False
    vm_region = cached_marin_region()
    if vm_region is None:
        return False
    bucket_location = _cached_bucket_location(bucket)
    if bucket_location is None:
        return False
    return not _regions_match(vm_region, bucket_location)


def is_cross_region_url(url: str) -> bool:
    """Return True if reading *url* would cross regions and be charged to the budget.

    Cheap: only cached region lookups, no listing or stat.  Callers can use this
    to skip an expensive size computation when a read would not be charged
    anyway (local paths, same-region buckets, unknown VM region, override set).
    """
    return _is_cross_region_url(url)


def record_transfer(size: int, url: str, *, budget: TransferBudget | None = None) -> None:
    """Charge *size* bytes against the cross-region transfer budget.

    Always safe to call: no-op for non-GCS URLs, same-region buckets, when the
    VM region is unknown, or when the override env var is set.  Raises
    :class:`TransferBudgetExceeded` if the recorded transfer would push the
    cumulative total past the budget.

    Used by callers (e.g. tensorstore-based code) that bypass fsspec but still
    want to charge against the shared cross-region transfer budget.

    Args:
        size: Number of bytes to charge.
        url: GCS URL (``gs://bucket/key``) being read or written.  Used both
            to decide whether the transfer is cross-region and as the path
            string in any raised :class:`TransferBudgetExceeded`.
        budget: Budget to charge against.  Defaults to the process-global
            singleton shared with :class:`CrossRegionGuardedFS` and
            :class:`MirrorFileSystem`.
    """
    if size <= 0:
        return
    if not _is_cross_region_url(url):
        return
    (budget if budget is not None else _global_transfer_budget).record(size, url)


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
        self._current_region = None if cross_region_checker else cached_marin_region()
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


def _with_s3_timeout_defaults(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Inject finite botocore timeouts/retries into S3 filesystem kwargs.

    Caller-supplied ``config_kwargs`` values win; we only fill in keys the
    caller did not set. See :data:`_S3_READ_TIMEOUT` and #6487.

    We seed ``config_kwargs`` from the ``FSSPEC_S3`` config block first. fsspec
    builds the filesystem by shallow-merging ``{**conf, **kwargs}``, so a bare
    ``config_kwargs`` here would *replace* (not merge with) any ``config_kwargs``
    in ``FSSPEC_S3`` -- silently dropping settings like
    ``{"s3": {"addressing_style": "virtual"}}`` that S3-compatible endpoints
    (CoreWeave object storage) require, which then hangs/path-style-rejects.
    """
    conf_config_kwargs = (fsspec.config.conf.get("s3") or {}).get("config_kwargs") or {}
    config_kwargs = {**conf_config_kwargs, **dict(kwargs.get("config_kwargs") or {})}
    config_kwargs.setdefault("connect_timeout", _S3_CONNECT_TIMEOUT)
    config_kwargs.setdefault("read_timeout", _S3_READ_TIMEOUT)
    config_kwargs.setdefault("retries", {"max_attempts": _S3_RETRY_MAX_ATTEMPTS, "mode": "standard"})
    return {**kwargs, "config_kwargs": config_kwargs}


def url_to_fs(url: str, **kwargs: Any) -> tuple[Any, str]:
    """Like ``fsspec.core.url_to_fs`` but wraps GCS filesystems in a cross-region guard.

    Returns ``(fs, path)``.  For non-GCS URLs the filesystem is returned
    unwrapped.  ``mirror://`` URLs are handled by :class:`MirrorFileSystem`.
    S3/R2 URLs get finite timeouts injected (#6487).
    """
    if url.startswith("s3://"):
        kwargs = _with_s3_timeout_defaults(kwargs)
    fs, path = fsspec.core.url_to_fs(url, **kwargs)
    if _fs_is_gcs(fs):
        fs = CrossRegionGuardedFS(fs)
    return fs, path


def is_remote_path(path: str) -> bool:
    """True if ``path`` resolves to a remote filesystem (e.g. ``gs://``, ``s3://``) rather than the
    local disk. A bare path or ``file://`` URL is local; anything with a remote scheme is not."""
    fs, _ = url_to_fs(path)
    return not isinstance(fs, LocalFileSystem)


def open_url(url: str, mode: str = "rb", **kwargs: Any) -> fsspec.core.OpenFile:
    """Like ``fsspec.open`` but checks the cross-region budget for GCS reads.

    For read modes on GCS URLs, eagerly stats the file and charges the
    transfer budget.  Then delegates to ``fsspec.open`` for the actual I/O.
    """
    if "r" in mode and _is_gcs_url(url):
        fs, path = fsspec.core.url_to_fs(url)
        guarded = CrossRegionGuardedFS(fs)
        guarded._guard_read(path)
    if url.startswith("s3://"):
        kwargs = _with_s3_timeout_defaults(kwargs)
    return cast(fsspec.core.OpenFile, fsspec.open(url, mode, **kwargs))


def filesystem(protocol: str, **kwargs: Any) -> Any:
    """Like ``fsspec.filesystem`` but wraps GCS filesystems in a cross-region guard.

    S3/R2 filesystems get finite timeouts injected (#6487)."""
    if protocol in ("s3", "s3a"):
        kwargs = _with_s3_timeout_defaults(kwargs)
    fs = fsspec.filesystem(protocol, **kwargs)
    if _is_gcs_protocol(protocol):
        fs = CrossRegionGuardedFS(fs)
    return fs


# ---------------------------------------------------------------------------
# Atomic write-and-rename


def unique_temp_path(output_path: str) -> str:
    """Return a unique temporary path derived from ``output_path``.

    Appends ``.tmp.<uuid>`` to avoid collisions when multiple writers target the
    same output path (e.g. during network-partition induced worker races).
    """
    return f"{output_path}.tmp.{uuid.uuid4().hex}"


# AWS error codes that are safe to retry on a server-side multipart copy
# (``s3fs.S3FileSystem.mv``). ``InvalidPart`` is the R2-specific symptom:
# every ``UploadPartCopy`` returns 200 but ``CompleteMultipartUpload`` then
# claims one or more parts are missing.
_TRANSIENT_S3_ERROR_CODES = frozenset(
    {
        "InvalidPart",
        "InternalError",
        "ServiceUnavailable",
        "SlowDown",
        "RequestTimeout",
        "RequestTimeTooSkewed",
    }
)

# Fragments matched against ``str(exc)`` for the case where s3fs has already
# translated the underlying ``botocore.ClientError`` into an ``OSError`` and
# the structured error code is no longer reachable.
_TRANSIENT_S3_MESSAGE_FRAGMENTS = (
    "specified parts could not be found",
    "InternalError",
    "ServiceUnavailable",
    "SlowDown",
    "RequestTimeout",
)


def _is_transient_s3_error(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code")
        if code in _TRANSIENT_S3_ERROR_CODES:
            return True
    msg = str(exc)
    return any(frag in msg for frag in _TRANSIENT_S3_MESSAGE_FRAGMENTS)


def _mv_with_retry(fs: Any, src: str, dst: str) -> None:
    retry_with_backoff(
        lambda: fs.mv(src, dst, recursive=True),
        retryable=_is_transient_s3_error,
        max_attempts=4,
        backoff=ExponentialBackoff(initial=1.0, maximum=8.0, factor=2.0),
        operation=f"atomic_rename fs.mv {src} -> {dst}",
    )


@contextlib.contextmanager
def atomic_rename(output_path: str, fs: Any = None) -> Generator[str, None, None]:
    """Atomic write-and-rename via a sibling temp key.

    Yields ``<output_path>.tmp.<uuid>``; on clean exit, ``fs.mv`` renames the
    temp into the final path. On exception, the temp key is best-effort
    deleted and the original exception re-raised.

    Callers may pass a pre-constructed ``fs`` to reuse a configured
    filesystem (e.g. an ``S3FileSystem`` with ``fixed_upload_size=True``)
    instead of letting atomic_rename build a default one from ``output_path``.

    Example:
        with atomic_rename("output.jsonl.gz") as tmp_path:
            write_data(tmp_path)
        # File is now at output.jsonl.gz
    """
    temp_path = unique_temp_path(output_path)
    if fs is None:
        fs = url_to_fs(output_path)[0]
    try:
        yield temp_path
        _mv_with_retry(fs, temp_path, output_path)
    except Exception:
        # Best-effort cleanup: temp file may not exist (writer crashed before
        # creating it) so we tolerate any rm error and re-raise the original.
        with contextlib.suppress(Exception):
            fs.rm(temp_path)
        raise


# ---------------------------------------------------------------------------
# Mirror filesystem
#
# Transparent cross-region file access: reads check the local prefix first,
# then scan all other marin-* data buckets and copy on first access.
# ---------------------------------------------------------------------------


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


# Register the mirror:// protocol with fsspec.
fsspec.register_implementation("mirror", MirrorFileSystem)
