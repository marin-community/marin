# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Polars Parquet scans addressed correctly for S3-compatible object stores."""

import os
from urllib.parse import urlparse

import polars as pl
from rigging.filesystem.s3_compat import needs_virtual_host_addressing, polars_s3_storage_options


def scan_parquet(path: str, *, schema: pl.Schema | None = None) -> pl.LazyFrame:
    """Scan a Parquet file, qualified with the object-store settings Polars cannot infer.

    Polars' Rust object_store client reads neither the fsspec configuration nor the
    file-delivered storage keys Iris task pods carry, so every credential and endpoint
    setting it needs must arrive through ``storage_options``: the declared store's
    static credentials (otherwise its environment chain falls through to instance
    metadata, which CoreWeave answers 404), and — where the store rejects path-style
    requests with HTTP 400 — the bucket-qualified virtual-host endpoint. In
    virtual-host mode object_store uses the endpoint verbatim, so the bucket must be
    part of the endpoint host. Every Zephyr Parquet read that goes through Polars
    (scatter chunks and external-sort spill runs alike) must use this.
    """
    options = polars_s3_storage_options(path)
    endpoint = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
    if path.startswith("s3://") and endpoint and needs_virtual_host_addressing(endpoint):
        bucket = urlparse(path).netloc
        parsed_endpoint = urlparse(endpoint)
        hostname = parsed_endpoint.hostname or ""
        if not hostname.startswith(f"{bucket}."):
            endpoint = parsed_endpoint._replace(netloc=f"{bucket}.{parsed_endpoint.netloc}").geturl()
        options |= {
            "aws_endpoint_url": endpoint,
            "aws_virtual_hosted_style_request": "true",
        }
    return pl.scan_parquet(path, schema=schema, storage_options=options)
