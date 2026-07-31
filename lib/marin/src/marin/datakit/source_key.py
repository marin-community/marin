# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stable source identity for Datakit artifacts."""

from typing import Annotated

from pydantic import BeforeValidator, PlainSerializer, ValidationInfo
from rigging.filesystem import StoragePath, StoreType, data_buckets, marin_prefix, prefix_join

from marin.execution.artifact import ARTIFACT_LOAD_CONTEXT_KEY


def _marin_data_prefixes() -> list[StoragePath]:
    prefixes = {marin_prefix()}
    for bucket in data_buckets().values():
        if bucket.store == StoreType.GCS:
            prefixes.add(f"gs://{bucket.name}")
        else:
            prefixes.add(f"s3://{bucket.name}/marin")
    return sorted((StoragePath(prefix) for prefix in prefixes), key=lambda prefix: len(prefix.segments), reverse=True)


def datakit_source_key(source_path: str) -> str:
    """Remove ``MARIN_PREFIX`` from a materialized Datakit source path.

    The function recognizes every configured Marin data bucket, not only the
    active region. It also removes a local ``MARIN_PREFIX``. Other local paths
    and non-object-store URLs stay unchanged.
    """
    path = StoragePath(source_path)
    for prefix in _marin_data_prefixes():
        try:
            relative = path.relative_to(prefix)
        except ValueError:
            continue
        if not relative:
            raise ValueError(f"Datakit source path must be below a Marin data prefix: {source_path!r}")
        return relative
    if path.scheme in ("gs", "s3"):
        raise ValueError(f"Datakit source path is not under a configured Marin data prefix: {source_path!r}")
    return str(path)


def datakit_source_path(source_path_or_key: str) -> str:
    """Resolve a relative Datakit source key against the active ``MARIN_PREFIX``."""
    if not source_path_or_key:
        return source_path_or_key

    path = StoragePath(source_path_or_key)
    if path.scheme or source_path_or_key.startswith("/"):
        return str(path)
    return prefix_join(marin_prefix(), source_path_or_key)


def datakit_artifact_path(source_path: str) -> str:
    """Remove the active ``MARIN_PREFIX`` from a serialized Datakit path."""
    if not source_path:
        return source_path

    path = StoragePath(source_path)
    prefix = StoragePath(marin_prefix())
    try:
        relative = path.relative_to(prefix)
    except ValueError:
        return str(path)
    if not relative:
        raise ValueError(f"Datakit artifact path must be below MARIN_PREFIX: {source_path!r}")
    return relative


def _resolve_datakit_artifact_path(value: str, info: ValidationInfo) -> str:
    if info.context and info.context.get(ARTIFACT_LOAD_CONTEXT_KEY):
        return datakit_source_path(value)
    return value


DatakitArtifactPath = Annotated[
    str,
    BeforeValidator(_resolve_datakit_artifact_path),
    PlainSerializer(datakit_artifact_path, return_type=str, when_used="json"),
]
"""A Datakit path that omits the active ``MARIN_PREFIX`` in artifact payloads."""
