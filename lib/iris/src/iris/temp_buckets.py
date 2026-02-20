# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared temp-bucket utilities for region-local scratch storage.

The canonical bucket definitions live in ``infra/configure_temp_buckets.py``.
This module provides a region-to-bucket mapping and helpers for building
GCS paths with lifecycle-managed TTL prefixes.
"""

import logging
import os
import re
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_GCP_METADATA_ZONE_URL = "http://metadata.google.internal/computeMetadata/v1/instance/zone"

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


def region_from_marin_prefix() -> str | None:
    """Derive region from ``MARIN_PREFIX`` (e.g. ``gs://marin-us-central2/…``)."""
    marin_prefix = os.environ.get("MARIN_PREFIX", "")
    m = re.match(r"gs://marin-([^/]+)", marin_prefix)
    return m.group(1) if m else None


def get_temp_bucket_path(ttl_days: int, prefix: str = "") -> str | None:
    """Return a GCS path on the region-local temp bucket with the given TTL.

    Returns a path of the form ``gs://marin-tmp-{region}/ttl={ttl_days}d/{prefix}``,
    or ``None`` if the region cannot be determined or has no temp bucket configured.

    Region is resolved by first querying the GCP metadata server, then falling
    back to inferring the region from the ``MARIN_PREFIX`` environment variable.

    The temp buckets are provisioned by ``infra/configure_temp_buckets.py`` with
    lifecycle rules that auto-delete objects under ``ttl=Nd/`` after *N* days.

    Args:
        ttl_days: Lifecycle TTL in days.  Should match one of the configured
            values (1-7, 14, 30) in ``infra/configure_temp_buckets.py``.
        prefix: Optional sub-path appended after the TTL directory.
    """
    region = region_from_metadata() or region_from_marin_prefix()
    if region is None:
        logger.debug("Cannot determine region from metadata or MARIN_PREFIX; temp bucket unavailable.")
        return None

    bucket = REGION_TO_TMP_BUCKET.get(region)
    if bucket is None:
        logger.warning("No temp bucket configured for region %s", region)
        return None

    path = f"gs://{bucket}/ttl={ttl_days}d"
    if prefix:
        path = f"{path}/{prefix.strip('/')}"
    return path
