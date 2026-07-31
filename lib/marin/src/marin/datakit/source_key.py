# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stable source identity for Datakit artifacts."""

from rigging.filesystem import StoragePath, StoreType, data_buckets, marin_prefix


def _marin_data_prefixes() -> list[StoragePath]:
    prefixes = {marin_prefix()}
    for bucket in data_buckets().values():
        scheme = "gs" if bucket.store == StoreType.GCS else "s3"
        prefixes.add(f"{scheme}://{bucket.name}/marin")
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
