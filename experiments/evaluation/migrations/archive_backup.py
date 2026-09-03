# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Region-local 30-day destinations for eval data a migration is about to replace.

Anything a migration removes from a run's results tree is copied here first. The bucket carries a
lifecycle rule, so the copies expire on their own; the window is there to make a bad migration
recoverable, not to hold a second permanent copy.
"""

from __future__ import annotations

import hashlib

from rigging.filesystem.cluster_config import (
    get_bucket_location,
    load_cluster_config,
    marin_temp_bucket,
    use_data_config,
)
from rigging.filesystem.storage_path import StoragePath, prefix_join

LEGACY_SAMPLES_KIND = "eval-archive-legacy"
SUPERSEDED_SAMPLES_KIND = "eval-archive-superseded"
TEMP_ARCHIVE_TTL_DAYS = 30


def run_backup_prefix(results_path: str, kind: str) -> str:
    """Return a collision-resistant, region-local 30-day prefix holding ``kind`` for one run."""
    results = StoragePath(results_path)
    run_name = results.parent.name if results.name == "results" else results.name
    identity = hashlib.sha256(str(results).encode()).hexdigest()[:12]
    prefix = f"{kind}/{run_name}-{identity}"
    if results.scheme == "gs":
        region = get_bucket_location(results.bucket)
        config = load_cluster_config("marin")
        bucket = config.region_buckets.get(region)
        if bucket is None:
            raise ValueError(f"no region-local GCS temp bucket is configured for region {region!r}")
        # Iris clusters can bind a storage config without GCS region buckets. Use the canonical Marin
        # data config explicitly so the helper cannot fall back to a non-lifecycle ``tmp/`` prefix.
        with use_data_config(config):
            destination = marin_temp_bucket(
                TEMP_ARCHIVE_TTL_DAYS,
                prefix=prefix,
                source_prefix=f"gs://{bucket.name}",
            )
    else:
        destination = marin_temp_bucket(
            TEMP_ARCHIVE_TTL_DAYS,
            prefix=prefix,
            source_prefix=results_path,
        )
    if StoragePath(destination).scheme != results.scheme:
        raise ValueError(f"could not resolve region-local temp storage for {results_path!r}: got {destination!r}")
    return destination


def legacy_archive_prefix(results_path: str) -> str:
    """Where one run's superseded ``samples_*.parquet`` files go once migrated."""
    return run_backup_prefix(results_path, LEGACY_SAMPLES_KIND)


def superseded_samples_prefix(results_path: str, schema_version: int) -> str:
    """Where one run's samples table goes before a contract change replaces it.

    Keyed by the version being replaced, so a run carried across two contract changes keeps a
    snapshot of each rather than overwriting the first.
    """
    return prefix_join(run_backup_prefix(results_path, SUPERSEDED_SAMPLES_KIND), f"v{schema_version}")
