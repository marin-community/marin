"""
gcs_utils.py

Helpful functions for manipulating Google Cloud Storage (path manipulation, parsing, etc.).
"""

from pathlib import Path

import requests
from google.cloud import storage


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
    response = requests.get(metadata_url, headers=headers)
    if response.status_code == 200:
        # The response contains the full zone (e.g., projects/<project-id>/zones/<zone>)
        zone = response.text.split("/")[-1]
        # Remove the last part to get the region (e.g., us-central1-a -> us-central1)
        region = "-".join(zone.split("-")[:-1])
        return region
    else:
        raise Exception(f"Failed to get VM region: {response.text}")


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
