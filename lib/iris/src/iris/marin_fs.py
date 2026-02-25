# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin filesystem helpers: prefix resolution and region-local temp storage.

Provides a unified API for resolving the marin storage prefix and building
GCS paths with lifecycle-managed TTL prefixes. The canonical temp-bucket
definitions live in ``infra/configure_temp_buckets.py``.

Resolution chain for the storage prefix:
  1. ``MARIN_PREFIX`` environment variable
  2. GCS instance metadata → ``gs://marin-{region}``
  3. ``/tmp/marin`` (local fallback)
"""

import logging
import os
import pathlib
import re
import urllib.error
import urllib.request

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
