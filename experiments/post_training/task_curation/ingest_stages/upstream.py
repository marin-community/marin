# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream pinned HF payloads in-region and bind every staged file to its upstream digest."""

import hashlib
from enum import StrEnum
from pathlib import PurePosixPath

from huggingface_hub import HfApi
from marin.datakit.download.huggingface import (
    HF_DATASET_REPO_TYPE_PREFIX,
    DownloadConfig,
    DownloadPlan,
    finish_download,
    plan_download,
    stream_file_to_fsspec,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.evaluation.corpus import file_sha256

from experiments.post_training.task_curation.backend_census_manifest import seal_manifest
from experiments.post_training.task_curation.ingest_stages.graph import StageJob, parent_receipt
from experiments.post_training.task_curation.ingest_stages.ingest_config import UpstreamPin
from experiments.post_training.task_curation.ingest_stages.licensing import LicenseAssessment, LicenseDisposition
from experiments.post_training.task_curation.ingest_stages.placement import regional_worker
from experiments.post_training.task_curation.ingest_stages.receipts import SHA256, IngestStage
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref
from infra.marina.apps.rl_data_catalog.server.source_identity import SourceId


class SourceDigest(StrEnum):
    LFS_SHA256 = "lfs-sha256"
    GIT_BLOB_SHA1 = "git-blob-sha1"


class UpstreamFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    size: int = Field(ge=0)
    algorithm: SourceDigest
    digest: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")

    @field_validator("path")
    @classmethod
    def canonical_path(cls, value: str) -> str:
        return checked_relative_path(value)

    @model_validator(mode="after")
    def content_pin(self) -> "UpstreamFile":
        expected = 64 if self.algorithm is SourceDigest.LFS_SHA256 else 40
        if len(self.digest) != expected:
            raise ValueError("HF content digest has the wrong length for its storage format")
        return self


class UpstreamInventory(BaseModel):
    """Metadata obtained at the exact upstream commit before any payload transfer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_sha256: str = Field(pattern=SHA256)
    source_id: SourceId
    upstream: UpstreamPin
    private: bool
    files: tuple[UpstreamFile, ...] = Field(min_length=1)


class StagedUpstream(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    inventory: ArtifactRef
    worker_region: str
    files: dict[str, ArtifactRef] = Field(min_length=1)


def checked_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or value in ("", ".")
        or ".." in path.parts
        or "\\" in value
        or any(character in value for character in "*?[]")
    ):
        raise ValueError("upstream metadata contains a non-canonical file path")
    return value


def upstream_inventory(job: StageJob) -> UpstreamInventory:
    """Freeze the selected file/folder tree using native HF sibling metadata."""
    policy = job.execution.policy
    upstream = policy.upstream
    info = HfApi().dataset_info(upstream.repository, revision=upstream.revision, files_metadata=True)
    if info.sha != upstream.revision:
        raise ValueError("HF metadata does not resolve to the configured immutable pin")
    selected = []
    for sibling in info.siblings or ():
        name = sibling.rfilename
        if not (upstream.path == "." or name == upstream.path or name.startswith(upstream.path + "/")):
            continue
        if sibling.size is None:
            raise ValueError("HF metadata omitted the selected file size")
        if sibling.lfs is not None:
            if sibling.lfs.size != sibling.size:
                raise ValueError("HF file size disagrees with its LFS metadata")
            algorithm, digest = SourceDigest.LFS_SHA256, sibling.lfs.sha256
        elif sibling.blob_id is not None:
            algorithm, digest = SourceDigest.GIT_BLOB_SHA1, sibling.blob_id
        else:
            raise ValueError("HF metadata omitted the selected file content pin")
        selected.append(UpstreamFile(path=name, size=sibling.size, algorithm=algorithm, digest=digest))
    names = [file.path for file in selected]
    if len(set(names)) != len(names) or (upstream.path in names and len(names) > 1):
        raise ValueError("HF metadata ambiguously selects a file and folder or duplicates files")
    return UpstreamInventory(
        input_sha256=job.inputs.digest(),
        source_id=policy.source_id,
        upstream=upstream,
        private=info.private,
        files=tuple(sorted(selected, key=lambda file: file.path)),
    )


def checked_inventory(job: StageJob) -> tuple[UpstreamInventory, ArtifactRef]:
    """Reuse only the selected tree frozen for these exact stage inputs."""
    path = prefix_join(job.output_root, "inventory.json")
    target = StoragePath(path)
    inventory = (
        UpstreamInventory.model_validate_json(target.read_bytes()) if target.exists() else upstream_inventory(job)
    )
    if (
        inventory.input_sha256 != job.inputs.digest()
        or inventory.source_id != job.execution.policy.source_id
        or inventory.upstream != job.execution.policy.upstream
    ):
        raise ValueError("cached upstream inventory belongs to another ingest execution")
    return inventory, seal_manifest(inventory.model_dump(mode="json"), path)


def checked_download_plan(inventory: UpstreamInventory, output: str) -> DownloadPlan:
    """Refuse mutable source URLs or a transfer listing different from the pinned tree."""
    upstream = inventory.upstream
    plan = plan_download(
        DownloadConfig(
            hf_dataset_id=upstream.repository,
            revision=upstream.revision,
            hf_urls_glob=[file.path for file in inventory.files],
            gcs_output_path=output,
        )
    )
    org, repository = upstream.repository.split("/")
    expected = {
        str(StoragePath(output) / file.path): (
            str(
                StoragePath(
                    scheme="hf",
                    netloc=HF_DATASET_REPO_TYPE_PREFIX,
                    segments=(org, f"{repository}@{upstream.revision}", *PurePosixPath(file.path).parts),
                )
            ),
            file.size,
        )
        for file in inventory.files
    }
    actual = {task.destination_path: (task.source_url, task.expected_size) for task in plan.tasks}
    if len(actual) != len(plan.tasks) or actual != expected or plan.output_path != output:
        raise ValueError("download plan differs from the immutable upstream inventory")
    return plan


def verified_file(file: UpstreamFile, path: StoragePath) -> ArtifactRef:
    """Check the upstream pin even when the downloader reused a same-size object."""
    sha256 = file_sha256(path)
    if file.algorithm is SourceDigest.LFS_SHA256:
        measured = sha256
    else:
        digest = hashlib.sha1(f"blob {file.size}\0".encode(), usedforsecurity=False)
        with path.open("rb") as stream:
            measured = hashlib.file_digest(stream, lambda: digest).hexdigest()
    if measured != file.digest:
        raise ValueError("staged payload differs from its pinned upstream content digest")
    return ArtifactRef(uri=str(path), sha256=sha256)


def stage_upstream(job: StageJob) -> dict[str, ArtifactRef]:
    """Acquire only licensed inputs on an Iris worker in the output bucket's region."""
    region = regional_worker(job.execution.policy.cluster, job.execution.artifact_root, job.output_root)
    license_receipt = parent_receipt(job.execution, IngestStage.LICENSE, job.inputs.dependencies[IngestStage.LICENSE])
    reference = job.inputs.artifacts["license/assessment"]
    if reference != license_receipt.outputs["assessment"]:
        raise ValueError("acquisition license evidence differs from its graph dependency")
    assessment = LicenseAssessment.model_validate(read_ref(reference))
    if (
        assessment.source_id != job.execution.policy.source_id
        or assessment.input_sha256 != license_receipt.inputs.digest()
        or assessment.disposition is not LicenseDisposition.PERMIT
        or assessment.receipt is None
    ):
        raise ValueError("upstream acquisition requires a permitted source-bound license assessment")
    inventory, reference = checked_inventory(job)
    root = prefix_join(job.output_root, "raw")
    files = download_inventory(inventory, root)
    staged = StagedUpstream(inventory=reference, worker_region=region, files=files)
    manifest = seal_manifest(staged.model_dump(mode="json"), prefix_join(job.output_root, "staged.json"))
    return {"inventory": reference, "staged": manifest, **{f"raw/{name}": ref for name, ref in files.items()}}


def download_inventory(inventory: UpstreamInventory, root: str) -> dict[str, ArtifactRef]:
    """Stream and verify the already frozen file map in its worker region."""
    plan = checked_download_plan(inventory, root)
    files = {}
    pins = {file.path: file for file in inventory.files}
    for task in plan.tasks:
        stream_file_to_fsspec(task)
        path = StoragePath(task.destination_path)
        relative = path.relative_to(StoragePath(root))
        files[relative] = verified_file(pins[relative], path)
    finish_download(plan)
    return files
