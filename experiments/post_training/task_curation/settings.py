# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Concrete configuration for the reusable source processor."""

import os
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

import click
from fray.types import ResourceConfig
from iris.cluster.client.job_info import get_job_info
from marin.inference.openai_batch import OpenAIBatchClient
from marin.inference.openai_chat import OpenAIChatClient
from shellbox.backends.gvisor.machine import GvisorMachineFactory
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import Backend, DockerImage, MachineFactory, MachineSpec, NetworkPolicy
from taskcompendium.models import EnvironmentRequirements, require_resolved_environment
from taskcompendium.pipeline.chat_requests import MAX_DIRECT_CONCURRENT_REQUESTS
from taskcompendium.pipeline.controls import GradingMachines
from taskcompendium.pipeline.models import Controls, FilterPolicy
from taskcompendium.pipeline.review import BatchReviewer, ChatReviewer, Reviewer
from taskcompendium.pipeline.source_processing import SourcePipelineConfig, SourceProcessingMode
from taskcompendium.pipeline.source_quality import SourceQualityPolicy
from taskcompendium.pipeline.source_verification import SourceVerificationPolicy
from taskcompendium.pipeline.stages import AuditExecution, ReviewConfig, ReviewMode
from taskcompendium.runtime.local import LocalGraderMachines

from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB, GLM_BULK_TOKEN_ENV, resolve_glm_base_url

REVIEW_REQUEST_TIMEOUT = 60
IRIS_SCHEDULING_TIMEOUT = 600
IRIS_MACHINE_CPUS = 4
IRIS_JOB_TTL = 1800


class VerificationBackend(StrEnum):
    IRIS = "iris"
    GVISOR = "gvisor"


def machines_identity(backend: VerificationBackend, worker_image: str, controller: bool) -> dict[str, Any]:
    """The settings every backend's grading machines share; the worker image carries the grading code."""
    return {
        "backend": backend.value,
        "worker_image": worker_image,
        # Local graders run in bubblewrap sandboxes on the worker; recorded so a backend change reverifies.
        "local_backend": "bubblewrap",
        "network": NetworkPolicy.DENY.value,
        "controller": controller,
    }


@dataclass(frozen=True)
class IrisMachines:
    """Schedule each sandbox image as a task on the Iris controller at ``controller_url``."""

    worker_image: str
    controller_url: str | None

    def identity(self) -> dict[str, Any]:
        return machines_identity(VerificationBackend.IRIS, self.worker_image, controller=True)

    def machine(self, environment: EnvironmentRequirements, memory_mb: int) -> tuple[MachineFactory, MachineSpec]:
        image = _sandbox_image(environment)
        factory = IrisMachineFactory(
            controller_url=_controller_url(self.controller_url),
            scheduling_timeout=IRIS_SCHEDULING_TIMEOUT,
            job_ttl=IRIS_JOB_TTL,
            secret_env=None,
        )
        # Kueue packs each pod onto the fullest node that still fits it; a larger CPU request fills a
        # node after fewer machines, so grading spreads across nodes instead of queueing behind one.
        return factory, MachineSpec(
            RegistryImage(image), network=NetworkPolicy.DENY, memory_mb=memory_mb, cpus=IRIS_MACHINE_CPUS
        )


@dataclass(frozen=True)
class GvisorMachines:
    """Run each sandbox image under gVisor on the local Docker daemon."""

    worker_image: str

    def identity(self) -> dict[str, Any]:
        return machines_identity(VerificationBackend.GVISOR, self.worker_image, controller=False)

    def machine(self, environment: EnvironmentRequirements, memory_mb: int) -> tuple[MachineFactory, MachineSpec]:
        image = _sandbox_image(environment)
        return GvisorMachineFactory(), MachineSpec(DockerImage(image), network=NetworkPolicy.DENY, memory_mb=memory_mb)


def _sandbox_image(environment: EnvironmentRequirements) -> str:
    require_resolved_environment(environment)
    if environment.docker_image is None:
        raise ValueError("A sandbox machine requires an environment with a digest-pinned image")
    return environment.docker_image


@dataclass(frozen=True)
class CampaignMachines:
    """Grading machines for a campaign: local environments grade in the worker, images on ``sandbox``."""

    sandbox: IrisMachines | GvisorMachines
    local: LocalGraderMachines

    def identity(self) -> dict[str, Any]:
        return {**self.sandbox.identity(), "local": self.local.identity()}

    def machine(self, environment: EnvironmentRequirements, memory_mb: int) -> tuple[MachineFactory, MachineSpec]:
        require_resolved_environment(environment)
        if Backend.LOCAL in environment.compatible_backends:
            return self.local.machine(environment, memory_mb)
        return self.sandbox.machine(environment, memory_mb)


def campaign_machines(backend: VerificationBackend, worker_image: str, controller_url: str | None) -> GradingMachines:
    """Grading machines for every source in a campaign: sandbox graders on ``backend``, local graders in the worker."""
    if backend == VerificationBackend.IRIS:
        return CampaignMachines(IrisMachines(worker_image, controller_url), LocalGraderMachines())
    return CampaignMachines(GvisorMachines(worker_image), LocalGraderMachines())


def job_controller_url() -> str | None:
    """The controller of the Iris job this process runs in, or ``None`` outside an Iris job."""
    info = get_job_info()
    return info.controller_address if info is not None else None


def _controller_url(controller_url: str | None) -> str:
    """The explicit Iris controller, or the controller of this process's job."""
    if controller_url is not None:
        return controller_url
    job_url = job_controller_url()
    if job_url is None:
        raise click.UsageError("--verification-backend iris outside an Iris job requires --controller-url")
    return job_url


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
    machines: GradingMachines | None,
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


@dataclass(frozen=True)
class RecipeSettings:
    """Concrete processor configuration; services are bound only during execution."""

    config: SourcePipelineConfig | None = None
    review: ReviewConfig | None = None
    review_cache: str | None = None
    base_url: str | None = None
    relay_job: str = DEFAULT_GLM_RELAY_JOB
    review_concurrency: int = MAX_DIRECT_CONCURRENT_REQUESTS
    normalized_shards: int | None = None
    verification_backend: VerificationBackend = VerificationBackend.IRIS
    controller_url: str | None = None
    seed: int = 0
    verification_sample_size: int = 20
    execution: AuditExecution = field(default_factory=AuditExecution)

    def source_config(self, mode: SourceProcessingMode, controls: Controls | None) -> SourcePipelineConfig:
        if self.config is not None:
            return replace(self.config, mode=mode)
        if self.review is None:
            raise click.UsageError("Recipe review configuration requires --model-revision and --review-mode")
        if self.normalized_shards is None:
            raise click.UsageError("Recipe conversion requires --normalized-shards")
        if self.execution.worker_resources is None:
            raise ValueError("Recipe execution requires worker resources")
        return _pipeline_config(
            mode,
            self.review,
            None,
            (
                campaign_machines(self.verification_backend, self.execution.worker_resources.image, self.controller_url)
                if controls is not None
                else None
            ),
            seed=self.seed,
            verification_sample_size=self.verification_sample_size,
            max_workers=self.execution.max_workers,
            worker_resources=self.execution.worker_resources,
            normalized_shards=self.normalized_shards,
        )

    def execution_config(self, config: SourcePipelineConfig, rubric: str | None) -> SourcePipelineConfig:
        if rubric is None or config.execution.reviewer is not None:
            return config
        if self.review_cache is None:
            raise click.UsageError("Recipe review requires --review-cache")
        reviewer = _reviewer(
            config.review,
            self.base_url if self.base_url is not None else resolve_glm_base_url(self.relay_job),
            review_cache=self.review_cache,
            review_concurrency=self.review_concurrency,
        )
        return replace(config, execution=replace(config.execution, reviewer=reviewer))
