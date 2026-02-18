# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
gcs_utils.py

Helpful functions for manipulating Google Cloud Storage (path manipulation, parsing, etc.).
"""

import logging
import os
import re
from pathlib import Path

import requests
from google.cloud import storage

logger = logging.getLogger(__name__)

# Canonical mapping from GCP region to marin-tmp bucket name.
# Must stay in sync with infra/configure_temp_buckets.py BUCKETS dict.
# Iris (lib/iris/src/iris/cluster/worker/env_probe.py) and Zephyr
# (lib/zephyr/src/zephyr/execution.py) maintain their own copies because
# they cannot depend on the marin package.
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


def split_gcs_path(gs_uri: str) -> tuple[str, Path]:
    """Split a GCS URI of the form `gs://BUCKET/path/to/resource` into a tuple (BUCKET, Path(path/to/resource))."""
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI `{gs_uri}`; expected URI of form `gs://BUCKET/path/to/resource`")

    # Split on `/` --> return (BUCKET, Path(".")) if single element, otherwise (BUCKET, Path(path/to/resource))
    maybe_bucket_path = gs_uri[len("gs://") :].split("/", 1)
    if len(maybe_bucket_path) == 1:
        return maybe_bucket_path[0], Path(".")

    return maybe_bucket_path[0], Path(maybe_bucket_path[1])


def get_vm_region():
    """Get the current VM's region using the Google Cloud Metadata API."""
    metadata_url = "http://metadata.google.internal/computeMetadata/v1/instance/zone"
    headers = {"Metadata-Flavor": "Google"}
    try:
        response = requests.get(metadata_url, headers=headers)
        if response.status_code == 200:
            # The response contains the full zone (e.g., projects/<project-id>/zones/<zone>)
            zone = response.text.split("/")[-1]
            # Remove the last part to get the region (e.g., us-central1-a -> us-central1)
            region = "-".join(zone.split("-")[:-1])
            return region
        else:
            raise ValueError(f"Failed to get VM region: {response.text}")
    except requests.exceptions.ConnectionError as e:
        raise ValueError("Failed to connect to Google Cloud Metadata API") from e


def get_bucket_location(bucket_name_or_path):
    """Get the GCS bucket's location."""
    if bucket_name_or_path.startswith("gs://"):
        bucket_name = split_gcs_path(bucket_name_or_path)[0]
    else:
        bucket_name = bucket_name_or_path

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    # this returns upper case regions, which isn't consistent with the rest of the codebase
    return bucket.location.lower()


def _region_from_marin_prefix() -> str | None:
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
        ttl_days: Lifecycle TTL in days. Should match one of the configured
            values (1-7, 14, 30) in ``infra/configure_temp_buckets.py``.
        prefix: Optional sub-path appended after the TTL directory.
    """
    try:
        region = get_vm_region()
    except ValueError:
        region = _region_from_marin_prefix()
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
