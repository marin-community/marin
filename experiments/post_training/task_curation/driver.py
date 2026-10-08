# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan or execute the RL data catalog inside a single Iris driver job."""

import json
import os
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

import click
from fray.types import ResourceConfig
from iris.cluster.client.job_info import get_job_info
from marin.execution.artifact import Artifact
from marin.execution.fingerprint import canonical_json, fingerprint_hash
from marin.execution.lazy import ArtifactStep
from marin.inference.openai_batch import OpenAIBatchClient
from marin.inference.openai_chat import OpenAIChatClient
from rigging.filesystem.storage_path import StoragePath
from shellbox.backends.gvisor.machine import GvisorMachineFactory
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import DockerImage, MachineFactory, MachineSpec, NetworkPolicy
from taskcompendium.pipeline.chat_requests import MAX_DIRECT_CONCURRENT_REQUESTS
from taskcompendium.pipeline.controls import GradingMachines
from taskcompendium.pipeline.models import FilterPolicy
from taskcompendium.pipeline.review import BatchReviewer, ChatReviewer, Reviewer
from taskcompendium.pipeline.source_processing import SourcePipelineConfig, SourceProcessingMode
from taskcompendium.pipeline.source_quality import SourceQualityPolicy
from taskcompendium.pipeline.source_verification import SourceVerificationPolicy
from taskcompendium.pipeline.stages import AuditExecution, ReviewConfig, ReviewMode

from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB, GLM_BULK_TOKEN_ENV, GLM_MODEL, resolve_glm_base_url
from experiments.post_training.task_curation.campaign import (
    ADMITTING_SAMPLE_STATUSES,
    CampaignPool,
    CampaignRuntime,
    SourceOutcome,
    campaign_identity,
    campaign_plan,
    require_matching_sample,
    run_campaign,
)
from experiments.post_training.task_curation.pipeline import RlDataArtifact, RlDataPipeline, source_step
from experiments.post_training.task_curation.sources import all_pipelines

REVIEW_REQUEST_TIMEOUT = 60
IRIS_SCHEDULING_TIMEOUT = 600
IRIS_JOB_TTL = 1800


class VerificationBackend(StrEnum):
    IRIS = "iris"
    GVISOR = "gvisor"


def machines_identity(backend: VerificationBackend, worker_image: str, controller: bool) -> dict[str, Any]:
    """The settings every backend's grading machines share; the worker image carries the grading code."""
    return {
        "backend": backend.value,
        "worker_image": worker_image,
        "network": NetworkPolicy.DENY.value,
        "controller": controller,
    }


@dataclass(frozen=True)
class IrisMachines:
    """Schedule each grader image as a task on the Iris controller at ``controller_url``."""

    worker_image: str
    controller_url: str

    def identity(self) -> dict[str, Any]:
        return machines_identity(VerificationBackend.IRIS, self.worker_image, controller=True)

    def machine(self, image: str, memory_mb: int) -> tuple[MachineFactory, MachineSpec]:
        factory = IrisMachineFactory(
            controller_url=self.controller_url,
            scheduling_timeout=IRIS_SCHEDULING_TIMEOUT,
            job_ttl=IRIS_JOB_TTL,
            secret_env=None,
        )
        return factory, MachineSpec(RegistryImage(image), network=NetworkPolicy.DENY, memory_mb=memory_mb)


@dataclass(frozen=True)
class GvisorMachines:
    """Run each grader image under gVisor on the local Docker daemon."""

    worker_image: str

    def identity(self) -> dict[str, Any]:
        return machines_identity(VerificationBackend.GVISOR, self.worker_image, controller=False)

    def machine(self, image: str, memory_mb: int) -> tuple[MachineFactory, MachineSpec]:
        return GvisorMachineFactory(), MachineSpec(DockerImage(image), network=NetworkPolicy.DENY, memory_mb=memory_mb)


def campaign_machines(backend: VerificationBackend, worker_image: str, controller_url: str | None) -> GradingMachines:
    """Fresh, network-denied grading machines on ``backend`` for every source in a campaign."""
    if backend == VerificationBackend.IRIS:
        if controller_url is None:
            raise ValueError("Iris verification requires a controller URL")
        return IrisMachines(worker_image, controller_url)
    return GvisorMachines(worker_image)


def job_controller_url() -> str | None:
    """The controller of the Iris job this process runs in, or ``None`` outside an Iris job."""
    info = get_job_info()
    return info.controller_address if info is not None else None


def _controller_url(backend: VerificationBackend, controller_url: str | None) -> str | None:
    """``controller_url``, or for Iris verification without one, the controller of this process's job."""
    if backend != VerificationBackend.IRIS or controller_url is not None:
        return controller_url
    job_url = job_controller_url()
    if job_url is None:
        raise click.UsageError("--verification-backend iris outside an Iris job requires --controller-url")
    return job_url


def _selected_pipelines(sources: tuple[str, ...]) -> dict[str, RlDataPipeline]:
    """The named catalog sources in catalog order, or the whole catalog when none is named."""
    catalog = all_pipelines()
    unknown = set(sources) - catalog.keys()
    if unknown:
        raise click.UsageError(f"Unknown source: {', '.join(sorted(unknown))}")
    return {name: pipeline for name, pipeline in catalog.items() if not sources or name in sources}


def _reviewer(review: ReviewConfig, base_url: str, *, review_cache: str, review_concurrency: int) -> Reviewer:
    token = os.environ[GLM_BULK_TOKEN_ENV]
    if review.mode == ReviewMode.CHAT:
        return ChatReviewer(
            OpenAIChatClient(base_url, token, timeout=REVIEW_REQUEST_TIMEOUT),
            review.model,
            review.model_revision,
            query_cache_root=review_cache,
            max_concurrent=review_concurrency,
            max_batch_bytes=review.max_batch_bytes,
        )
    return BatchReviewer(
        OpenAIBatchClient(base_url, token, timeout=REVIEW_REQUEST_TIMEOUT, request_attempts=3),
        review.model,
        review.model_revision,
        query_cache_root=review_cache,
        max_batch_bytes=review.max_batch_bytes,
    )


def _pipeline_config(
    mode: SourceProcessingMode,
    review: ReviewConfig,
    reviewer: Reviewer | None,
    machines: GradingMachines,
    *,
    seed: int,
    verification_sample_size: int,
    max_workers: int,
    worker_resources: ResourceConfig,
    normalized_shards: int,
) -> SourcePipelineConfig:
    return SourcePipelineConfig(
        mode=mode,
        quality_policy=SourceQualityPolicy(sample_size=100, seed=seed),
        verification_policy=SourceVerificationPolicy(verification_sample_size, seed, 1, 0.95),
        review=review,
        execution=AuditExecution(
            max_workers=max_workers,
            review_batch_size=64,
            reviewer=reviewer,
            worker_resources=worker_resources,
            # Iris isolates each shard in a process. Admit one review process per
            # worker so its request semaphore enforces the worker's provider cap.
            review_task_resources=worker_resources,
        ),
        filter_policy=FilterPolicy(),
        normalized_shards=normalized_shards,
        machines=machines,
    )


def _adopted_sample(
    pipeline: RlDataPipeline,
    sample_step: ArtifactStep[RlDataArtifact],
    outcome: SourceOutcome,
    *,
    sample_report: str,
    sample_identity: str,
) -> ArtifactStep[Artifact] | None:
    """The admitted sample output whose control trials a full run of ``pipeline`` reuses."""
    if outcome.status not in ADMITTING_SAMPLE_STATUSES:
        return None
    provenance = {
        "campaign_report": sample_report,
        "sample_identity": sample_identity,
        "sample_source": sample_step.name,
        "sample_fingerprint": sample_step.fingerprint(),
        "sample_path": outcome.path,
    }
    return ArtifactStep.adopt(
        f"task-curation/sample/{pipeline.name}-{fingerprint_hash(canonical_json(provenance))[:16]}",
        sample_step.version,
        source=outcome.path,
        kind=Artifact,
        config=provenance,
    )


def _full_steps(
    pipelines: dict[str, RlDataPipeline],
    sample_steps: list[ArtifactStep[RlDataArtifact]],
    config: SourcePipelineConfig,
    runtime: CampaignRuntime,
    *,
    sample_report: str,
    sample_identity: str,
) -> tuple[list[ArtifactStep[RlDataArtifact]], dict[str, SourceOutcome]]:
    """Full-mode source steps, each reusing its admitted sample output, and each step's sample outcome."""
    sampled = require_matching_sample(json.loads(StoragePath(sample_report).read_text()), sample_identity, sample_steps)
    steps = [
        source_step(
            pipeline,
            config,
            runtime,
            previous=_adopted_sample(
                pipeline, sample_step, outcome, sample_report=sample_report, sample_identity=sample_identity
            ),
        )
        for pipeline, sample_step, outcome in zip(pipelines.values(), sample_steps, sampled, strict=True)
    ]
    return steps, {step.name: outcome for step, outcome in zip(steps, sampled, strict=True)}


@click.command(help=__doc__)
@click.option("--model", default=GLM_MODEL, show_default=True)
@click.option("--model-revision", required=True)
@click.option("--base-url", help="OpenAI-compatible review endpoint; defaults to the relay job's endpoint.")
@click.option("--relay-job", default=DEFAULT_GLM_RELAY_JOB, show_default=True, help="Iris GLM relay job to resolve.")
@click.option("--review-cache", required=True)
@click.option("--review-mode", type=click.Choice(["batch", "chat"]), required=True)
@click.option(
    "--review-concurrency",
    type=click.IntRange(min=1, max=MAX_DIRECT_CONCURRENT_REQUESTS),
    default=MAX_DIRECT_CONCURRENT_REQUESTS,
    show_default=True,
)
@click.option("--mode", type=click.Choice(["sample", "full"]), default="sample", show_default=True)
@click.option("--max-workers", type=click.IntRange(min=1), required=True)
@click.option("--coordinator-memory", required=True, help="Explicit RAM budget for the shared coordinator, e.g. 16g.")
@click.option("--normalized-shards", type=click.IntRange(min=1), required=True)
@click.option("--concurrent-sources", type=click.IntRange(min=10), default=10, show_default=True)
@click.option("--worker-image", required=True, help="Zephyr worker image; it carries the grading code.")
@click.option(
    "--verification-backend",
    type=click.Choice([backend.value for backend in VerificationBackend]),
    default=VerificationBackend.IRIS.value,
    show_default=True,
)
@click.option(
    "--controller-url",
    help="Iris controller for --verification-backend iris; inside an Iris job, defaults to the job's controller.",
)
@click.option("--seed", type=int, default=0)
@click.option("--verification-sample-size", type=click.IntRange(min=1), default=20)
@click.option("--report-path", required=True)
@click.option("--sample-report", help="Terminal sample campaign report required before executing full mode.")
@click.option("--source", "sources", multiple=True, help="Catalog source name to execute; repeat to select multiple.")
@click.option("--run", "do_run", is_flag=True)
def main(
    model: str,
    model_revision: str,
    base_url: str | None,
    relay_job: str,
    review_cache: str,
    review_mode: str,
    review_concurrency: int,
    mode: str,
    max_workers: int,
    coordinator_memory: str,
    normalized_shards: int,
    concurrent_sources: int,
    worker_image: str,
    verification_backend: str,
    controller_url: str | None,
    seed: int,
    verification_sample_size: int,
    report_path: str,
    sample_report: str | None,
    sources: tuple[str, ...],
    do_run: bool,
) -> None:
    backend = VerificationBackend(verification_backend)
    controller_url = _controller_url(backend, controller_url)
    pipelines = _selected_pipelines(sources)
    if do_run and mode == "full" and sample_report is None:
        raise click.UsageError("Full execution requires --sample-report")
    review = ReviewConfig(model=model, model_revision=model_revision, mode=ReviewMode(review_mode))
    reviewer = None
    if do_run:
        reviewer = _reviewer(
            review,
            base_url if base_url is not None else resolve_glm_base_url(relay_job),
            review_cache=review_cache,
            review_concurrency=review_concurrency,
        )
    worker_resources = ResourceConfig(cpu=2, ram="8g", image=worker_image)
    config = _pipeline_config(
        SourceProcessingMode(mode),
        review,
        reviewer,
        campaign_machines(backend, worker_image, controller_url),
        seed=seed,
        verification_sample_size=verification_sample_size,
        max_workers=max_workers,
        worker_resources=worker_resources,
        normalized_shards=normalized_shards,
    )
    runtime = CampaignRuntime()
    steps = [source_step(pipeline, config, runtime) for pipeline in pipelines.values()]
    sample_steps = (
        steps
        if mode == "sample"
        else [
            source_step(pipeline, replace(config, mode=SourceProcessingMode.SAMPLE), runtime)
            for pipeline in pipelines.values()
        ]
    )
    sample_identity = campaign_identity(sample_steps, worker_image)
    pool = CampaignPool(
        max_workers,
        concurrent_sources,
        worker_resources=worker_resources,
        coordinator_resources=ResourceConfig(cpu=1, ram=coordinator_memory, preemptible=False),
    )
    if not do_run:
        click.echo(json.dumps(campaign_plan(steps, pool), indent=2))
        return
    sample_outcomes = None
    if mode == "full":
        assert sample_report is not None
        steps, sample_outcomes = _full_steps(
            pipelines, sample_steps, config, runtime, sample_report=sample_report, sample_identity=sample_identity
        )
    run_campaign(
        steps,
        runtime=runtime,
        pool=pool,
        report_path=report_path,
        sample_identity=sample_identity,
        mode=mode,
        sample_outcomes=sample_outcomes,
    )


if __name__ == "__main__":
    main()
