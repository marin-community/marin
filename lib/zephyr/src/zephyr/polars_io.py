# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Polars cloud reads, addressed the way the rest of the pipeline addresses the store.

Zephyr writes shuffle chunks and spill files through fsspec, which rigging configures for the
ambient object store -- endpoint, virtual-host addressing, credentials (the ``FSSPEC_S3`` block,
see ``rigging.filesystem.s3_compat``). ``pl.scan_parquet`` reads them back through the Rust
object_store crate instead, which knows none of that: handed a bare ``s3://`` URI and a custom
endpoint it issues path-style requests, and CoreWeave's gateways reject path-style outright with
a bare 400 on the first HEAD (``VIRTUAL_HOST_ONLY_S3_DOMAINS``). Every polars scan of a path this
package wrote must go through :func:`scan_parquet_chunk` so both halves address the store the
same way. ``zephyr.writers`` solves the identical problem for pyarrow's native S3 filesystem.

Virtual-hosted addressing needs one extra step s3fs does implicitly: object_store treats an
explicit endpoint as the literal host, so the bucket must already be folded into it
(``http://{bucket}.cwlota.com``). Setting only the flag strips the bucket from the request path
without adding it to the host, and the gateway 400s that just the same -- both shapes verified
against cwlota on cw-us-east-08a.
"""

import os
from urllib.parse import urlparse

import fsspec
import polars as pl
from rigging.filesystem.s3_compat import needs_virtual_host_addressing


def scan_storage_options(path: str) -> dict[str, str] | None:
    """object_store settings mirroring the ambient S3 config, or ``None``.

    ``None`` for non-S3 paths and where no custom endpoint is configured (plain AWS: polars'
    own environment defaults work there). The endpoint comes from the environment first
    (``AWS_ENDPOINT_URL_S3`` / ``AWS_ENDPOINT_URL``, which cluster tasks always carry), falling
    back to the fsspec config for processes configured off-cluster. Credentials are passed
    through when the fsspec config or environment holds them; inside a cluster the node-local
    gateway accepts unsigned requests and tasks carry no key material at all.
    """
    if not path.startswith("s3://"):
        return None
    conf = fsspec.config.conf.get("s3") or {}
    client_kwargs = conf.get("client_kwargs") or {}
    endpoint = (
        os.environ.get("AWS_ENDPOINT_URL_S3")
        or os.environ.get("AWS_ENDPOINT_URL")
        or conf.get("endpoint_url")
        or client_kwargs.get("endpoint_url")
    )
    if not endpoint:
        return None
    options = {"aws_endpoint_url": endpoint}
    parsed = urlparse(endpoint)
    if parsed.scheme == "http":
        # The LOTA endpoint is plain http; object_store refuses non-TLS endpoints unless told.
        options["aws_allow_http"] = "true"
    region = client_kwargs.get("region_name") or os.environ.get("AWS_REGION")
    if region:
        options["aws_region"] = region
    addressing = ((conf.get("config_kwargs") or {}).get("s3") or {}).get("addressing_style")
    if addressing == "virtual" or needs_virtual_host_addressing(endpoint):
        bucket = path.removeprefix("s3://").split("/", 1)[0]
        options["aws_virtual_hosted_style_request"] = "true"
        if not (parsed.hostname or "").startswith(f"{bucket}."):
            options["aws_endpoint_url"] = f"{parsed.scheme}://{bucket}.{parsed.netloc}"
    key = conf.get("key") or os.environ.get("AWS_ACCESS_KEY_ID")
    secret = conf.get("secret") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    if key and secret:
        options["aws_access_key_id"] = key
        options["aws_secret_access_key"] = secret
    return options


def scan_parquet_chunk(path: str) -> pl.LazyFrame:
    """Open one parquet file lazily, addressed the same way it was written."""
    options = scan_storage_options(path)
    if options is None:
        return pl.scan_parquet(path)
    return pl.scan_parquet(path, storage_options=options)
