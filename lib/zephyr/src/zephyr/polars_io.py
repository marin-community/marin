# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Polars readers that address S3-compatible object stores the way Zephyr writes them.

Zephyr writes every Parquet file it later reads back — scatter chunks and
external-sort spill runs — through rigging's fsspec layer, which knows that
CoreWeave object storage rejects path-style requests. Polars reads go through
its own Rust ``object_store`` client instead, which reads credentials and
region from the environment but cannot infer that requirement. Routing all
Zephyr scans through :func:`scan_parquet` keeps the read side addressed the
same way as the write side.
"""

import os
from urllib.parse import urlparse

import polars as pl
from rigging.filesystem.s3_compat import needs_virtual_host_addressing


def scan_parquet(path: str) -> pl.LazyFrame:
    """Scan a Parquet file with the addressing required by CoreWeave object storage."""
    endpoint = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
    if not path.startswith("s3://") or not endpoint or not needs_virtual_host_addressing(endpoint):
        return pl.scan_parquet(path)

    bucket = urlparse(path).netloc
    parsed_endpoint = urlparse(endpoint)
    hostname = parsed_endpoint.hostname or ""
    if not hostname.startswith(f"{bucket}."):
        endpoint = parsed_endpoint._replace(netloc=f"{bucket}.{parsed_endpoint.netloc}").geturl()

    # In virtual-host mode object_store uses the endpoint verbatim, so the
    # bucket must be part of the endpoint host.
    return pl.scan_parquet(
        path,
        storage_options={
            "aws_endpoint_url": endpoint,
            "aws_virtual_hosted_style_request": "true",
        },
    )
