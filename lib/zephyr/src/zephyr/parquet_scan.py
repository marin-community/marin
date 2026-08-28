# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DataFusion Parquet scans configured for Zephyr's object stores."""

import os
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Protocol
from urllib.parse import urlparse

import pyarrow as pa
from datafusion import DataFrame, RuntimeEnvBuilder, SessionConfig, SessionContext, col
from datafusion.object_store import AmazonS3, GoogleCloud
from rigging.filesystem.s3_compat import needs_virtual_host_addressing

_DATAFUSION_BATCH_SIZE = 10_000
_DEFAULT_SORT_SPILL_RESERVATION_BYTES = 10 * 1024 * 1024
_S3_VIRTUAL_HOST_ENV = "AWS_VIRTUAL_HOSTED_STYLE_REQUEST"
_S3_ENVIRONMENT_LOCK = threading.Lock()
_HTTP_ENDPOINT_PREFIX = "http://"


class ObjectStoreRegistry(Protocol):
    """The SessionContext surface needed to register remote stores."""

    def register_object_store(self, scheme: str, store: AmazonS3 | GoogleCloud, host: str | None = None) -> None: ...


def datafusion_context(
    *,
    memory_limit_bytes: int | None = None,
    target_partitions: int | None = None,
) -> SessionContext:
    """Create a DataFusion context with optional memory limit and no disk spill."""
    if target_partitions is None:
        target_partitions = max(1, pa.cpu_count())
    if target_partitions <= 0:
        raise ValueError(f"target_partitions must be positive, got {target_partitions}")
    config = SessionConfig().with_batch_size(_DATAFUSION_BATCH_SIZE).with_target_partitions(target_partitions)
    runtime = RuntimeEnvBuilder().with_disk_manager_disabled()
    if memory_limit_bytes is not None:
        sort_spill_reservation_bytes = min(
            _DEFAULT_SORT_SPILL_RESERVATION_BYTES,
            max(1, memory_limit_bytes // 100),
        )
        config = config.set(
            "datafusion.execution.sort_spill_reservation_bytes",
            str(sort_spill_reservation_bytes),
        )
        runtime = runtime.with_greedy_memory_pool(max(1, memory_limit_bytes))
    return SessionContext(config, runtime)


@contextmanager
def _temporary_environment(overrides: dict[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _qualified_s3_endpoint(bucket: str, endpoint: str) -> str:
    parsed = urlparse(endpoint)
    hostname = parsed.hostname or ""
    if hostname.startswith(f"{bucket}."):
        return endpoint
    return parsed._replace(netloc=f"{bucket}.{parsed.netloc}").geturl()


def _s3_store(bucket: str) -> AmazonS3:
    endpoint = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
    if endpoint is None or not needs_virtual_host_addressing(endpoint):
        return AmazonS3(
            bucket_name=bucket,
            endpoint=endpoint,
            allow_http=bool(endpoint and endpoint.startswith(_HTTP_ENDPOINT_PREFIX)),
        )

    qualified_endpoint = _qualified_s3_endpoint(bucket, endpoint)
    # DataFusion's Python binding does not expose object_store's virtual-hosted
    # option. The Rust builder does read this standard AWS variable, so scope it
    # to construction and restore the caller's environment immediately after.
    with _S3_ENVIRONMENT_LOCK, _temporary_environment({_S3_VIRTUAL_HOST_ENV: "true"}):
        return AmazonS3(
            bucket_name=bucket,
            endpoint=qualified_endpoint,
            allow_http=qualified_endpoint.startswith(_HTTP_ENDPOINT_PREFIX),
        )


def register_object_stores(context: ObjectStoreRegistry, paths: Iterable[str]) -> None:
    """Register each GCS or S3 bucket referenced by ``paths`` once."""
    locations = {(parsed.scheme, parsed.netloc) for path in paths if (parsed := urlparse(path)).netloc}
    for scheme, bucket in sorted(locations):
        if scheme == "gs":
            context.register_object_store("gs://", GoogleCloud(bucket_name=bucket), bucket)
        elif scheme == "s3":
            context.register_object_store("s3://", _s3_store(bucket), bucket)


def scan_parquet(
    context: SessionContext,
    path: str,
    *,
    schema: pa.Schema | None = None,
    sorted_by: tuple[str, ...] = (),
) -> DataFrame:
    """Scan Parquet with optional on-disk ordering for merge optimization."""
    file_sort_order = [[col(name).sort() for name in sorted_by]] if sorted_by else None
    return context.read_parquet(path, schema=schema, file_sort_order=file_sort_order)
