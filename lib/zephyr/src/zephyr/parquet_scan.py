# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Polars Parquet scans addressed correctly for S3-compatible object stores."""

import os
from urllib.parse import urlparse

import polars as pl
from rigging.filesystem.s3_compat import needs_virtual_host_addressing


def storage_options_for_path(path: str) -> dict[str, str] | None:
    """Return Polars options for an S3-compatible path, if endpoint qualification is needed."""
    endpoint = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
    if not path.startswith("s3://") or not endpoint or not needs_virtual_host_addressing(endpoint):
        return None

    bucket = urlparse(path).netloc
    parsed_endpoint = urlparse(endpoint)
    hostname = parsed_endpoint.hostname or ""
    if not hostname.startswith(f"{bucket}."):
        endpoint = parsed_endpoint._replace(netloc=f"{bucket}.{parsed_endpoint.netloc}").geturl()

    return {
        "aws_endpoint_url": endpoint,
        "aws_virtual_hosted_style_request": "true",
    }


def scan_parquet(path: str) -> pl.LazyFrame:
    """Scan a Parquet file, qualifying the endpoint where the store demands it.

    Polars' Rust object_store client reads credentials and region from the
    environment, but unlike fsspec it cannot infer CoreWeave's virtual-host
    requirement. In virtual-host mode object_store uses the endpoint verbatim,
    so the bucket must be part of the endpoint host; a path-style request
    returns HTTP 400. Every Zephyr Parquet read that goes through Polars
    (scatter chunks and external-sort spill runs alike) must use this.
    """
    storage_options = storage_options_for_path(path)
    if storage_options is None:
        return pl.scan_parquet(path)
    return pl.scan_parquet(path, storage_options=storage_options)
