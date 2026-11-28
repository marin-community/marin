# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage evidence bound to source policy, landed code and upstream receipts."""

from enum import StrEnum

from marin.execution.artifact import Artifact
from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.backend_census_manifest import seal_manifest
from experiments.post_training.task_curation.ingest_stages.ingest_config import GIT_REVISION
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref
from infra.marina.apps.rl_data_catalog.server.source_identity import SourceId

SHA256 = r"^[0-9a-f]{64}$"


class IngestStage(StrEnum):
    CLAIM = "claim"
    LICENSE = "license"
    STAGE_UPSTREAM = "stage-upstream"
    CONVERT = "convert"
    DEDUP = "dedup"
    CENSUS = "census"
    FINGERPRINT_OVERLAP = "fingerprint-overlap"
    BACKEND_ORACLE = "backend-oracle"
    LEXICAL = "lexical"
    SAMPLE_QC = "sample-qc"
    DECLARE = "declare"


class StageInputs(BaseModel):
    """The complete identity against which a cached stage must be checked."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: IngestStage
    source_id: SourceId
    config_sha256: str = Field(pattern=SHA256)
    source_revision: str = Field(pattern=GIT_REVISION)
    worker_argv_sha256: str = Field(pattern=SHA256)
    dependencies: dict[IngestStage, ArtifactRef]
    artifacts: dict[str, ArtifactRef]

    def digest(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))


class StageReceipt(BaseModel):
    """Completed evidence only; incomplete stages never receive a resume receipt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    inputs: StageInputs
    outputs: dict[str, ArtifactRef] = Field(min_length=1)


class StageArtifact(Artifact):
    """The lazy graph stores the receipt's digest alongside the stage result."""

    receipt: ArtifactRef


def verify_evidence(references: list[ArtifactRef]) -> None:
    """Stream checksums without materializing evidence; call in the data's region."""
    for reference in references:
        if file_sha256(StoragePath(reference.uri)) != reference.sha256:
            raise ValueError(f"ingest evidence digest mismatch: {reference.uri}")


def verify_receipt_evidence(receipt: StageReceipt) -> None:
    verify_evidence(
        [*receipt.inputs.dependencies.values(), *receipt.inputs.artifacts.values(), *receipt.outputs.values()]
    )


def seal_receipt(output_root: str, receipt: StageReceipt) -> ArtifactRef:
    """Seal a completed stage at its input identity; changed results refuse reseal."""
    verify_receipt_evidence(receipt)
    path = prefix_join(output_root, f"{receipt.inputs.stage.value}/{receipt.inputs.digest()}/receipt.json")
    return seal_manifest(receipt.model_dump(mode="json"), path)


def load_receipt(reference: ArtifactRef, expected: StageInputs) -> StageReceipt:
    """Reuse only the graph's SHA-bound receipt for the exact current inputs."""
    receipt = StageReceipt.model_validate(read_ref(reference))
    if receipt.inputs != expected:
        raise ValueError("ingest receipt inputs changed; the stage must run for the new identity")
    verify_receipt_evidence(receipt)
    return receipt
