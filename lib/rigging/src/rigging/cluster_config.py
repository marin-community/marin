# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cluster storage configuration: per-cluster storage profiles and user-home paths.

A ``StorageProfile`` describes where a cluster's data lives: the region-to-bucket
mirror set, the URL scheme, the user-directory segment, and the temp TTL policy.
Profiles are loaded from ``data:`` blocks in cluster YAML files discovered under
:data:`MARIN_CLUSTER_CONFIG_DIRS`.

With no config present, :func:`load_cluster_config` returns
:data:`DEFAULT_STORAGE_PROFILE`, whose ``resolved_root()`` delegates to
:func:`rigging.filesystem.marin_prefix` so the default storage layout is
byte-identical to the historical behavior.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache

import yaml

from rigging.config_discovery import resolve_cluster_config
from rigging.filesystem import ALLOWED_TTL_DAYS, REGION_TO_DATA_BUCKET, marin_prefix, marin_region

# Cluster config search dirs, highest priority first: the repo-root ``config/``
# directory, then a per-user override location. Resolved against the marin
# workspace root by :func:`rigging.config_discovery.resolve_cluster_config`.
MARIN_CLUSTER_CONFIG_DIRS: tuple[str, ...] = ("config", "~/.config/marin/clusters")

_MARIN_PREFIX_ENV = "MARIN_PREFIX"
_MARIN_CLUSTER_ENV = "MARIN_CLUSTER"


@dataclass(frozen=True)
class StorageProfile:
    """Storage layout for a single cluster.

    Attributes:
        region_buckets: Region name -> bucket name for the cross-region mirror set.
        scheme: URL scheme for the cluster's storage (e.g. ``"gs"`` or ``"s3"``).
        user_segment: Path segment under which per-user directories live.
        temp_path: Path segment for TTL-managed scratch data.
        ttl_days: Allowed TTL-day values for temp lifecycle rules.
        root: Explicit single-prefix root (e.g. ``"s3://marin-na/marin"``). Set
            only for clusters that do not use region-local bucket selection.
    """

    region_buckets: Mapping[str, str]
    scheme: str = "gs"
    user_segment: str = "users"
    temp_path: str = "tmp"
    ttl_days: tuple[int, ...] = ALLOWED_TTL_DAYS
    root: str | None = None

    def resolved_root(self) -> str:
        """Resolve the storage root for this profile.

        Precedence: ``MARIN_PREFIX`` env > ``self.root`` > region-local bucket
        from ``region_buckets[marin_region()]`` > :func:`marin_prefix`. With
        ``root=None`` and no region match it returns :func:`marin_prefix`
        unchanged.
        """
        env_prefix = os.environ.get(_MARIN_PREFIX_ENV)
        if env_prefix:
            return env_prefix
        if self.root is not None:
            return self.root
        region = marin_region()
        if region is not None:
            bucket = self.region_buckets.get(region)
            if bucket is not None:
                return f"{self.scheme}://{bucket}"
        return marin_prefix()

    def user_home(self, user: str, *, base: str | None = None) -> str:
        """Return the per-user home directory under ``user_segment``."""
        return os.path.join(base or self.resolved_root(), self.user_segment, user)


DEFAULT_STORAGE_PROFILE: StorageProfile = StorageProfile(
    region_buckets=dict(REGION_TO_DATA_BUCKET),
    scheme="gs",
    user_segment="users",
    temp_path="tmp",
    ttl_days=ALLOWED_TTL_DAYS,
    root=None,
)


def _parse_storage_profile(data: Mapping[str, object]) -> StorageProfile:
    """Build a :class:`StorageProfile` from a parsed ``data:`` config block."""
    region_buckets = dict(data.get("region_buckets") or {})
    scheme = str(data.get("scheme", "gs"))
    user_segment = str(data.get("user_segment", "users"))
    root = data.get("root")

    temp = data.get("temp") or {}
    temp_path = str(temp.get("path", "tmp"))
    raw_ttl = temp.get("ttl_days")
    ttl_days = tuple(raw_ttl) if raw_ttl is not None else ALLOWED_TTL_DAYS

    return StorageProfile(
        region_buckets=region_buckets,
        scheme=scheme,
        user_segment=user_segment,
        temp_path=temp_path,
        ttl_days=ttl_days,
        root=str(root) if root is not None else None,
    )


@cache
def _load_cluster_config_cached(cluster: str | None) -> StorageProfile:
    if cluster is None:
        return DEFAULT_STORAGE_PROFILE

    config_path = resolve_cluster_config(cluster, MARIN_CLUSTER_CONFIG_DIRS)
    with config_path.open("rb") as f:
        document = yaml.safe_load(f) or {}

    data = document.get("data")
    if not data:
        return DEFAULT_STORAGE_PROFILE
    return _parse_storage_profile(data)


def load_cluster_config(cluster: str | None = None) -> StorageProfile:
    """Resolve a cluster name to its :class:`StorageProfile`.

    The cluster is resolved as ``cluster`` arg > ``MARIN_CLUSTER`` env > ``None``.
    When a ``config/<cluster>.yaml`` is discovered, its ``data:`` block is parsed
    into a profile; the ``iris:`` (and any other) keys are ignored. When no
    cluster is resolved, returns :data:`DEFAULT_STORAGE_PROFILE`.

    The result is cached; call :func:`reset_cluster_config_cache` in tests after
    changing the environment or config files.
    """
    resolved = cluster if cluster is not None else os.environ.get(_MARIN_CLUSTER_ENV) or None
    return _load_cluster_config_cached(resolved)


def reset_cluster_config_cache() -> None:
    """Clear the :func:`load_cluster_config` cache. For tests."""
    _load_cluster_config_cached.cache_clear()
