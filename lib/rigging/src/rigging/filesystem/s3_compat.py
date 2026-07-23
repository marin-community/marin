# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fsspec/boto environment setup for S3-compatible object stores.

CoreWeave AI Object Storage (and R2) speak the S3 protocol but need three things plain AWS
environment variables cannot fully express together: a custom endpoint, virtual-hosted
addressing, and region-less signing. Their connections also need finite request bounds so a
wedged object-store call can fail into the task retry path. This module owns that setup: it
writes the standard ``AWS_*`` variables plus the ``FSSPEC_S3`` config block that
:mod:`rigging.filesystem.factory` and every plain ``fsspec`` caller read, then flushes fsspec's
caches so the settings take.

Processes inside a cluster usually arrive with ``FSSPEC_S3`` already exported by the runtime;
every function here is a no-op in that case.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import urlparse

import fsspec
import s3fs

# CoreWeave AI Object Storage, as seen from outside the cluster. Pods inside CoreWeave use the
# LOTA endpoint from their cluster config instead and never need this default.
CW_ENDPOINT_URL = "https://cwobject.com"

# Endpoint domains that reject path-style requests outright.
VIRTUAL_HOST_ONLY_S3_DOMAINS = ("cwobject.com", "cwlota.com")

_S3_CONNECT_TIMEOUT = 30
_S3_READ_TIMEOUT = 120
_S3_RETRY_MAX_ATTEMPTS = 5


def s3_request_bounds_config_kwargs() -> dict[str, Any]:
    """Return finite botocore timeouts and retries for fsspec S3 filesystems.

    s3fs/aiobotocore otherwise permit upload requests to wait indefinitely when a
    connection wedges. Finite bounds turn that stall into an error handled by the
    task retry path (#6487).
    """
    return {
        "connect_timeout": _S3_CONNECT_TIMEOUT,
        "read_timeout": _S3_READ_TIMEOUT,
        "retries": {"max_attempts": _S3_RETRY_MAX_ATTEMPTS, "mode": "standard"},
    }


def needs_virtual_host_addressing(endpoint_url: str) -> bool:
    """True when *endpoint_url* is served by a domain that only accepts virtual-hosted requests."""
    hostname = urlparse(endpoint_url).hostname or ""
    return any(hostname == domain or hostname.endswith("." + domain) for domain in VIRTUAL_HOST_ONLY_S3_DOMAINS)


def fsspec_s3_conf(endpoint: str) -> dict:
    """The ``FSSPEC_S3`` config block for *endpoint*: virtual-hosted addressing where the domain
    demands it, region-less ("auto") signing, and finite request bounds.

    Non-AWS S3-compatible endpoints (R2, CoreWeave Object Storage) don't honor the AWS region
    scheme; signing with the wrong region surfaces as 400 Bad Request. "auto" tells boto3 to skip
    region validation and let the endpoint route the request itself.
    """
    config_kwargs = s3_request_bounds_config_kwargs()
    if needs_virtual_host_addressing(endpoint):
        config_kwargs["s3"] = {"addressing_style": "virtual"}
    return {
        "endpoint_url": endpoint,
        "client_kwargs": {"region_name": "auto"},
        "config_kwargs": config_kwargs,
    }


def configure_fsspec_s3(endpoint: str, key: str | None = None, secret: str | None = None) -> None:
    """Point ``s3://`` access (fsspec and boto alike) at an S3-compatible *endpoint*.

    Sets the ``AWS_*`` credential/endpoint variables and the ``FSSPEC_S3`` config block, never
    overwriting values already present, then flushes fsspec's config and s3fs's instance cache so
    already-imported modules pick the settings up.
    """
    if key and secret:
        os.environ.setdefault("AWS_ACCESS_KEY_ID", key)
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", secret)

    os.environ.setdefault("AWS_ENDPOINT_URL", endpoint)
    os.environ.setdefault("AWS_REGION", "auto")
    os.environ.setdefault("AWS_DEFAULT_REGION", "auto")

    if "FSSPEC_S3" not in os.environ:
        os.environ["FSSPEC_S3"] = json.dumps(fsspec_s3_conf(endpoint))

    # Flush fsspec/s3fs cached instances so they pick up the new config.
    fsspec.config.set_conf_env(fsspec.config.conf)
    s3fs.S3FileSystem.clear_instance_cache()


def configure_coreweave_s3() -> None:
    """Configure ``s3://`` access to CoreWeave object storage from ambient ``CW_KEY_*`` credentials.

    For processes outside a CoreWeave cluster (a dev box, a dashboard) that read or write CW
    buckets. No-op when the keys are absent or an ``FSSPEC_S3`` block is already exported (a CW
    pod's runtime config wins).
    """
    key, secret = os.environ.get("CW_KEY_ID"), os.environ.get("CW_KEY_SECRET")
    if not key or not secret:
        return
    configure_fsspec_s3(os.environ.get("CW_S3_ENDPOINT", CW_ENDPOINT_URL), key=key, secret=secret)
