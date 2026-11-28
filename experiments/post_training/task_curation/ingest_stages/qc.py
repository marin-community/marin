# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze SAMPLE inputs in the source region and bind informational reviewer outcomes."""

from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from finestore.rl.curation import ActorCensusInput
from iris.cluster.config import load_config
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.admission_inventory import inventory_population
from experiments.post_training.task_curation.backend_census_manifest import seal_manifest
from experiments.post_training.task_curation.full_qc import SAMPLE_REVIEWER_VERSION, local_reviewer
from experiments.post_training.task_curation.ingest_stages.graph import StageJob, parent_receipt
from experiments.post_training.task_curation.ingest_stages.placement import regional_worker
from experiments.post_training.task_curation.ingest_stages.receipts import IngestStage
from experiments.post_training.task_curation.judges import PinnedLocalJudgeEndpoint, require_review_serving_root
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref
from experiments.post_training.task_curation.sample_qc_artifacts import SAMPLE_REVIEWER_SHAS
from experiments.post_training.task_curation.sample_qc_binding import SampleQCReceipt
from experiments.post_training.task_curation.sample_qc_receipts import sample_qc_observations

QC_CLUSTER = "cw-us-east-08a"
MAX_NO_VERDICT_FRACTION = 0.05


def require_sample_capacity(repo_root: Path) -> None:
    """Refuse a QC root without the pinned reviewer's declared GPU flavor."""
    require_review_serving_root(QC_CLUSTER)
    endpoint = local_reviewer(SAMPLE_REVIEWER_VERSION).endpoint
    assert isinstance(endpoint, PinnedLocalJudgeEndpoint)
    accelerator = endpoint.local.accelerator
    config = load_config(repo_root / "lib/iris/config" / f"{QC_CLUSTER}.yaml")
    for group in config.scale_groups.values():
        resource = group.resources
        if (
            resource is not None
            and group.max_slices > 0
            and (
                resource.device_variant == accelerator.gpu_type
                and resource.device_count * (group.num_vms or 1) >= accelerator.gpu_count
            )
        ):
            return
    raise ValueError("The QC root lacks the pinned reviewer's required accelerators")


class QCState(StrEnum):
    UNAVAILABLE = "unavailable"
    REVIEWED = "reviewed"


class QCRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input_sha256: str
    source_id: str
    source_revision: str
    queue: ArtifactRef
    output_root: str


class QCOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_sha256: str
    job_id: str
    state: QCState
    sample_count: int = Field(gt=0)
    no_verdict_count: int = Field(ge=0)
    sample_passed: bool
    review: ArtifactRef | None

    @model_validator(mode="after")
    def measured_outcome(self) -> "QCOutcome":
        if self.no_verdict_count > self.sample_count:
            raise ValueError("QC no-verdict count exceeds its attempted panel")
        if self.state is QCState.UNAVAILABLE:
            if self.review is not None or self.no_verdict_count != self.sample_count or self.sample_passed:
                raise ValueError("Unavailable QC cannot carry a review or passing verdict")
        elif self.review is None or self.sample_passed != (
            self.no_verdict_count <= MAX_NO_VERDICT_FRACTION * self.sample_count
        ):
            raise ValueError("QC verdict differs from the measured no-verdict fraction")
        return self


def prepare_sample(job: StageJob) -> ArtifactRef:
    """Prepare the remote review request without creating a serving job."""
    regional_worker(job.execution.policy.cluster, job.execution.artifact_root, job.output_root)
    parent_receipt(job.execution, IngestStage.LEXICAL, job.inputs.dependencies[IngestStage.LEXICAL])
    request_path = prefix_join(job.output_root, "request.json")
    existing = StoragePath(request_path)
    if existing.exists():
        request = QCRequest.model_validate_json(existing.read_bytes())
        if request.input_sha256 != job.inputs.digest():
            raise ValueError("SAMPLE request belongs to another population")
        read_ref(request.queue)
        return seal_manifest(request.model_dump(mode="json"), request_path)
    population = SourceTextPopulation.model_validate(read_ref(job.inputs.artifacts["dedup/population"]))
    binding = ActorCensusInput.model_validate(
        read_ref(job.inputs.artifacts["fingerprint-overlap/declaration-input"])["actor_census_input"]
    )
    attempt = prefix_join(job.output_root, f"attempts/{uuid4().hex}")
    panel = inventory_population(population, binding, prefix_join(attempt, "inventory"), None)
    queue = seal_manifest({"panels": [panel.model_dump(mode="json")]}, prefix_join(attempt, "queue-envelope.json"))
    # The maintained launcher consumes a JSON array, whose bytes are separately
    # pinned; the envelope preserves that array as an immutable graph output.
    request = QCRequest(
        input_sha256=job.inputs.digest(),
        source_id=population.source_id,
        source_revision=job.execution.source_revision,
        queue=queue,
        output_root=prefix_join(attempt, "review"),
    )
    return seal_manifest(request.model_dump(mode="json"), request_path)


def sample_outcome(request_ref: ArtifactRef, job_id: str, exit_code: int) -> QCOutcome:
    """Preserve infrastructure failure as no verdict; QC never gates admission."""
    request = QCRequest.model_validate(read_ref(request_ref))
    panel = read_ref(request.queue)["panels"][0]
    if exit_code:
        return QCOutcome(
            request_sha256=request_ref.sha256,
            job_id=job_id,
            state="unavailable",
            sample_count=panel["sample_size"],
            no_verdict_count=panel["sample_size"],
            sample_passed=False,
            review=None,
        )
    root = prefix_join(request.output_root, canonical_sha256(request.source_id)[:20])
    observed = sample_qc_observations(root)
    result = observed.review
    if (
        result.panel.source_id != request.source_id
        or result.panel.panel_sha256 != panel["panel_sha256"]
        or canonical_sha256({key: value for key, value in result.reviewer.items() if key != "mode"})
        != SAMPLE_REVIEWER_SHAS[SAMPLE_REVIEWER_VERSION]
    ):
        raise ValueError("SAMPLE result differs from its frozen panel or pinned reviewer")
    counts = result.result
    no_verdict = counts.unavailable_count + counts.uncertain_count
    sample_count = counts.sample_count
    if sample_count is None or sample_count != panel["sample_size"]:
        raise ValueError("SAMPLE result omits requested review attempts")
    review = seal_manifest(observed.model_dump(mode="json"), prefix_join(request.output_root, "review.json"))
    return QCOutcome(
        request_sha256=request_ref.sha256,
        job_id=job_id,
        state="reviewed",
        sample_count=sample_count,
        no_verdict_count=no_verdict,
        sample_passed=no_verdict <= MAX_NO_VERDICT_FRACTION * sample_count,
        review=review,
    )


def sample_qc_stage(job: StageJob) -> dict[str, ArtifactRef]:
    """Consume the host-orchestrated review without moving full source data to its GPU region."""
    request_ref = prepare_sample(job)
    path = prefix_join(job.output_root, "outcome.json")
    result = QCOutcome.model_validate_json(StoragePath(path).read_bytes())
    if result.request_sha256 != request_ref.sha256:
        raise ValueError("SAMPLE outcome belongs to another request")
    if result.review is not None:
        observed = SampleQCReceipt.model_validate(read_ref(result.review))
        queue = read_ref(QCRequest.model_validate(read_ref(request_ref)).queue)["panels"][0]
        review = observed.review
        no_verdict = review.result.unavailable_count + review.result.uncertain_count
        if (
            review.panel.panel_sha256 != queue["panel_sha256"]
            or result.sample_count != review.attempted_count
            or result.no_verdict_count != no_verdict
            or review.panel.source_id != job.execution.policy.source_id
            or canonical_sha256({key: value for key, value in review.reviewer.items() if key != "mode"})
            != SAMPLE_REVIEWER_SHAS[SAMPLE_REVIEWER_VERSION]
        ):
            raise ValueError("SAMPLE outcome differs from its frozen panel or verified observations")
    return {"request": request_ref, "outcome": seal_manifest(result.model_dump(mode="json"), path)}
