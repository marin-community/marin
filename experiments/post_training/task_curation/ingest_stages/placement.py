# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Require an Iris worker beside the canonical Atlas output bucket."""

from pathlib import Path

from iris.cluster.client.job_info import get_job_info
from rigging.filesystem.cluster_config import StoreType, load_cluster_config_from_dirs
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.atlas_deployment import region_deployment


def regional_worker(cluster: str, artifact_root: str, output_root: str) -> str:
    """Refuse local, unknown-region and cross-region work before dataset I/O."""
    info = get_job_info()
    if info is None or info.worker_region is None:
        raise ValueError("ingest dataset operations require a region-identified Iris worker")
    deployment = region_deployment(cluster)
    prefix = StoragePath(deployment.region_prefix)
    root = StoragePath(artifact_root)
    output = StoragePath(output_root)
    if prefix.scheme != "s3" or any(part in (".", "..") for path in (root, output) for part in path.segments):
        raise ValueError("ingest requires a canonical regional S3 artifact root")
    root.relative_to(prefix)
    output.relative_to(root)
    # Read only checked-in storage policy, excluding per-user cluster overrides.
    repo_root = Path(__file__).resolve().parents[4]
    storage = load_cluster_config_from_dirs(StoreType.COREWEAVE.value, [repo_root / "config"])
    bucket = storage.region_buckets[deployment.region_label]
    if bucket.name != prefix.bucket or bucket.store is not StoreType.COREWEAVE:
        raise ValueError("Atlas deployment and canonical bucket placement disagree")
    if info.worker_region.casefold() != deployment.region_label.casefold():
        raise ValueError("ingest worker is outside the target bucket region")
    return info.worker_region
