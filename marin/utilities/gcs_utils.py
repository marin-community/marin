"""
gcs_utils.py

Helpful functions for manipulating Google Cloud Storage (path manipulation, parsing, etc.).
"""

from pathlib import Path


def split_gcs_path(gs_uri: str) -> tuple[str, Path]:
    """Split a GCS URI of the form `gs://BUCKET/path/to/resource` into a tuple (BUCKET, Path(path/to/resource))."""
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI `{gs_uri}`; expected URI of form `gs://BUCKET/path/to/resource`")

    # Split on `/` --> return (BUCKET, Path(".")) if single element, otherwise (BUCKET, Path(path/to/resource))
    maybe_bucket_path = gs_uri[len("gs://") :].split("/", 1)
    if len(maybe_bucket_path) == 1:
        return maybe_bucket_path[0], Path(".")

    return maybe_bucket_path[0], Path(maybe_bucket_path[1])
