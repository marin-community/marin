# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
gcs_utils.py

Helpful functions for manipulating Google Cloud Storage (path manipulation, parsing, etc.).
"""

import logging
from pathlib import Path

from google.cloud import storage
from iris.temp_buckets import REGION_TO_TMP_BUCKET, get_temp_bucket_path, region_from_metadata

logger = logging.getLogger(__name__)

# Re-export so existing ``from marin.utilities.gcs_utils import ...`` callers keep working.
__all__ = ["REGION_TO_TMP_BUCKET", "get_temp_bucket_path"]


def split_gcs_path(gs_uri: str) -> tuple[str, Path]:
    """Split a GCS URI of the form `gs://BUCKET/path/to/resource` into a tuple (BUCKET, Path(path/to/resource))."""
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI `{gs_uri}`; expected URI of form `gs://BUCKET/path/to/resource`")

    # Split on `/` --> return (BUCKET, Path(".")) if single element, otherwise (BUCKET, Path(path/to/resource))
    maybe_bucket_path = gs_uri[len("gs://") :].split("/", 1)
    if len(maybe_bucket_path) == 1:
        return maybe_bucket_path[0], Path(".")

    return maybe_bucket_path[0], Path(maybe_bucket_path[1])


def get_vm_region() -> str:
    """Get the current VM's region via GCP metadata, or raise ``ValueError``.

    Thin wrapper around ``iris.temp_buckets.region_from_metadata`` that
    preserves the ``ValueError``-raising contract expected by callers
    such as the region-check validation in training.
    """
    region = region_from_metadata()
    if region is not None:
        return region
    raise ValueError("Failed to determine VM region from GCP metadata")


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
