# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bucket-routed filesystems: one process, every Marin backend.

:mod:`rigging.filesystem.factory` resolves a URL against the process-wide fsspec
configuration, which holds exactly one S3 endpoint at a time. That is right for a
task pinned to one cluster and wrong for anything that spans backends — a browser
listing GCS and CoreWeave side by side, or a copy from CoreWeave to GCS.

:func:`filesystem_for` routes instead on the bucket's declared backend
(``config/*.yaml`` via :func:`rigging.filesystem.data_buckets`), building each
S3-compatible filesystem with explicit endpoint, signing region, and credentials
rather than reading or writing environment variables. GCS and unregistered buckets
fall through to the guarded factory, so cross-region metering and ``mirror://``
still apply where they already did.
"""

import logging
from typing import Any

import s3fs

from rigging.filesystem.cluster_config import BucketSpec, StoreType, data_buckets
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.s3_compat import (
    credentials_hint,
    fsspec_s3_conf,
    s3_credentials,
    s3_endpoint,
)
from rigging.filesystem.storage_path import StoragePath

logger = logging.getLogger(__name__)


class MissingCredentials(Exception):
    """A bucket's backend has no usable credentials in the environment."""


def filesystem_for(url: str) -> tuple[Any, str]:
    """Return ``(fs, path)`` for *url*, routed by the bucket's declared backend.

    An ``s3://`` URL naming a bucket declared as R2 or CoreWeave gets a dedicated
    :class:`s3fs.S3FileSystem` carrying that backend's endpoint, addressing style,
    signing region, and credentials; instances are shared through s3fs's own
    kwargs-keyed cache. Everything else — GCS, local paths, ``mirror://``, and
    ``s3://`` buckets no config declares — goes through the guarded
    :func:`rigging.filesystem.url_to_fs`, which reads the ambient environment.

    Raises:
        MissingCredentials: if the routed backend has no credentials configured.
    """
    parsed = StoragePath(url)
    if parsed.scheme != "s3":
        return url_to_fs(url)

    spec = data_buckets().get(parsed.bucket)
    if spec is None or spec.store not in (StoreType.R2, StoreType.COREWEAVE):
        return url_to_fs(url)

    return _s3_filesystem(spec), _s3_path(parsed)


def _s3_path(parsed: StoragePath) -> str:
    """``bucket/key`` — the protocol-stripped form s3fs expects."""
    return f"{parsed.bucket}/{parsed.key}" if parsed.key else parsed.bucket


def _s3_filesystem(spec: BucketSpec) -> s3fs.S3FileSystem:
    """Build (or reuse) the S3 filesystem serving *spec*'s backend.

    The signing region is the bucket's own for CoreWeave, whose endpoint routes on it, and
    ``"auto"`` for R2, which ignores the AWS region scheme entirely.
    """
    credentials = s3_credentials(spec.store)
    if credentials is None:
        raise MissingCredentials(f"no credentials for {spec.store} bucket {spec.name!r}: {credentials_hint(spec.store)}")
    key, secret = credentials

    endpoint = s3_endpoint(spec.store)
    conf = fsspec_s3_conf(endpoint)
    return s3fs.S3FileSystem(
        key=key,
        secret=secret,
        endpoint_url=endpoint,
        client_kwargs={"region_name": spec.signing_region or "auto"},
        config_kwargs=conf["config_kwargs"],
    )
