# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve a complete immutable raw snapshot before training decontamination."""

from dataclasses import replace
from uuid import uuid4

from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.task_curation.backend_census_manifest import seal_manifest
from experiments.post_training.task_curation.ingest_stages.graph import StageJob
from experiments.post_training.task_curation.ingest_stages.upstream import checked_inventory, download_inventory
from experiments.post_training.task_curation.raw_source_publish import publish_raw_source
from experiments.post_training.task_curation.remediation_publish import RawSourceArchive
from experiments.post_training.task_curation.source import RlDataSource


def raw_archive(job: StageJob, original: RlDataSource, root: str) -> RawSourceArchive:
    """Reuse the licensed, in-region downloader for every file in the original pin."""
    path = StoragePath(root) / "raw-source-archive.json"
    if path.exists():
        return RawSourceArchive.model_validate_json(path.read_bytes())
    policy = job.execution.policy
    full_policy = policy.model_copy(update={"upstream": policy.upstream.model_copy(update={"path": "."})})
    full_job = replace(
        job,
        execution=job.execution.model_copy(update={"policy": full_policy}),
        output_root=prefix_join(root, "raw-upstream"),
    )
    inventory, reference = checked_inventory(full_job)
    files = download_inventory(inventory, prefix_join(full_job.output_root, "files"))
    snapshot = {
        "source_id": policy.source_id,
        "dataset_id": policy.upstream.repository,
        "revision": policy.upstream.revision,
        "data_restriction": original.info.data_restriction.value,
        "artifact": reference.uri,
        "source_files": sorted(files),
        "files": [{"path": name, **value.model_dump(mode="json")} for name, value in sorted(files.items())],
    }
    assert original.info.license_receipt is not None
    archive = publish_raw_source(
        snapshot,
        original.info.license_receipt,
        version=uuid4().hex,
        evidence_root=prefix_join(root, f"raw-publication-attempts/{uuid4().hex}"),
        use=policy.intended_use[0].value,
    )
    seal_manifest(archive.model_dump(mode="json"), str(path))
    return archive
