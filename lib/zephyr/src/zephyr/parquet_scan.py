# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Polars Parquet scans addressed correctly for S3-compatible object stores."""

import os
from urllib.parse import urlparse

import polars as pl
from rigging.filesystem.s3_compat import needs_virtual_host_addressing, s3_credential_options


def scan_parquet(path: str, *, schema: pl.Schema | None = None) -> pl.LazyFrame:
    """Scan a Parquet file, qualifying the endpoint where the store demands it.

    Polars' Rust object_store client receives credentials resolved by Rigging
    and reads the region from the environment, but unlike fsspec it cannot
    infer CoreWeave's virtual-host requirement. In virtual-host mode
    object_store uses the endpoint verbatim, so the bucket must be part of
    the endpoint host; a path-style request returns HTTP 400. Every Zephyr
    Parquet read that goes through Polars (scatter chunks and external-sort
    spill runs alike) must use this.
    """
    if not path.startswith("s3://"):
        return pl.scan_parquet(path, schema=schema)

    credentials = s3_credential_options()
    storage_options = {
        native: credentials[option]
        for option, native in (
            ("key", "aws_access_key_id"),
            ("secret", "aws_secret_access_key"),
            ("token", "aws_session_token"),
        )
        if option in credentials
    }
    endpoint = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
    if endpoint and needs_virtual_host_addressing(endpoint):
        bucket = urlparse(path).netloc
        parsed_endpoint = urlparse(endpoint)
        hostname = parsed_endpoint.hostname or ""
        if not hostname.startswith(f"{bucket}."):
            endpoint = parsed_endpoint._replace(netloc=f"{bucket}.{parsed_endpoint.netloc}").geturl()
        storage_options.update(aws_endpoint_url=endpoint, aws_virtual_hosted_style_request="true")

    return pl.scan_parquet(path, schema=schema, storage_options=storage_options or None)
