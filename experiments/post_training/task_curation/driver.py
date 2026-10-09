# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan RL curation, then run QUICK locally or reviewed SAMPLE/FULL campaigns on Iris."""

import json
import os
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
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
from taskcompendium.pipeline.models import FilterPolicy
from taskcompendium.pipeline.review import BatchReviewer, ChatReviewer, Reviewer
from taskcompendium.pipeline.source_processing import SourcePipelineConfig, SourceProcessingMode
from taskcompendium.pipeline.source_quality import SourceQualityPolicy
from taskcompendium.pipeline.source_verification import SourceVerificationPolicy
from taskcompendium.pipeline.stages import AuditExecution, ReviewConfig, ReviewMode
from taskcompendium.runtime.local import LocalGraderMachines

from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB, GLM_BULK_TOKEN_ENV, GLM_MODEL, resolve_glm_base_url
from experiments.post_training.task_curation.binding import pipeline_step
from experiments.post_training.task_curation.campaign import CampaignPool, CampaignRuntime, campaign_plan, run_campaign
from experiments.post_training.task_curation.local import run_local_sources
<<<<<<< HEAD
from experiments.post_training.task_curation.pipeline import RlDataPipeline, download_identity, source_step
from experiments.post_training.task_curation.sources import all_pipelines
||||||| parent of a6ae499b25 ([rl-data] Invoke dataset-owned curation pipelines)
from experiments.post_training.task_curation.pipeline import download_identity, source_step
from experiments.post_training.task_curation.sources import selected_pipelines
=======
from experiments.post_training.task_curation.pipeline import RlDataPipeline, download_identity
from experiments.post_training.task_curation.source import RlDataSource
from experiments.post_training.task_curation.sources import selected_sources
>>>>>>> a6ae499b25 ([rl-data] Invoke dataset-owned curation pipelines)

REVIEW_REQUEST_TIMEOUT = 60
IRIS_SCHEDULING_TIMEOUT = 600
IRIS_MACHINE_CPUS = 4
IRIS_JOB_TTL = 1800
QUICK_MAX_WORKERS = 4


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
    controller_url: str

    def identity(self) -> dict[str, Any]:
        return machines_identity(VerificationBackend.IRIS, self.worker_image, controller=True)

    def machine(self, environment: EnvironmentRequirements, memory_mb: int) -> tuple[MachineFactory, MachineSpec]:
        image = _sandbox_image(environment)
        factory = IrisMachineFactory(
            controller_url=self.controller_url,
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
        if controller_url is None:
            raise ValueError("Iris verification requires a controller URL")
        return CampaignMachines(IrisMachines(worker_image, controller_url), LocalGraderMachines())
    return CampaignMachines(GvisorMachines(worker_image), LocalGraderMachines())


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


def local_source_plan(source: RlDataSource) -> dict[str, Any]:
    """Describe local invocation without staging inputs or inspecting custom ingestion."""
    pipeline = source.pipeline
    if isinstance(pipeline, RlDataPipeline):
        return {
            "name": source.name,
            "source": download_identity(pipeline.source),
            "inputs": {name: download_identity(upstream) for name, upstream in pipeline.inputs.items()},
        }
    return {"name": source.name, "dataset": asdict(source.dataset) if source.dataset is not None else None}


@click.command(help=__doc__)
@click.option("--model", default=GLM_MODEL, show_default=True)
@click.option("--model-revision", help="Required for SAMPLE/FULL.")
@click.option("--base-url", help="OpenAI-compatible review endpoint; defaults to the relay job's endpoint.")
@click.option("--relay-job", default=DEFAULT_GLM_RELAY_JOB, show_default=True, help="Iris GLM relay job to resolve.")
@click.option("--review-cache", help="Required for SAMPLE/FULL.")
@click.option("--review-mode", type=click.Choice(["batch", "chat"]), help="Required for SAMPLE/FULL.")
@click.option(
    "--review-concurrency",
    type=click.IntRange(min=1, max=MAX_DIRECT_CONCURRENT_REQUESTS),
    default=MAX_DIRECT_CONCURRENT_REQUESTS,
    show_default=True,
)
@click.option("--mode", type=click.Choice([mode.value for mode in SourceProcessingMode]), required=True)
@click.option("--max-workers", type=click.IntRange(min=1), help="Required for SAMPLE/FULL; defaults to 4 for QUICK.")
@click.option("--coordinator-memory", help="Required for SAMPLE/FULL: RAM for the shared coordinator, e.g. 16g.")
@click.option("--normalized-shards", type=click.IntRange(min=1), help="Required for SAMPLE/FULL.")
@click.option("--concurrent-sources", type=click.IntRange(min=10), default=10, show_default=True)
@click.option("--worker-image", help="Required for SAMPLE/FULL: Zephyr worker image carrying the grading code.")
@click.option(
    "--container-profile",
    default="CONTAINER_PROFILE_PRIVILEGED",
    show_default=True,
    help="Iris container profile of the Zephyr workers; local graders need a privileged pod to build sandboxes.",
)
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
@click.option("--report-path", help="Required for SAMPLE/FULL; QUICK writes campaign.json under --output-root.")
@click.option("--source", "sources", multiple=True, help="Catalog source name to execute; repeat to select multiple.")
@click.option(
    "--input-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="QUICK: staged primary inputs instead of downloading the declared pinned source.",
)
@click.option(
    "--input-file",
    "local_files",
    type=(str, click.Path(exists=True, dir_okay=False, path_type=Path)),
    multiple=True,
    help="QUICK: local primary file under its declared logical filename; repeat as needed.",
)
@click.option(
    "--input",
    "auxiliary",
    type=(str, click.Path(exists=True, file_okay=False)),
    multiple=True,
    help="QUICK: override a named auxiliary input with a local directory.",
)
@click.option("--output-root", type=click.Path(file_okay=False, path_type=Path), help="Required for QUICK.")
@click.option(
    "--download-cache",
    type=click.Path(file_okay=False, path_type=Path),
    help="QUICK: pinned download cache; defaults to ~/.cache/marin.",
)
@click.option("--run", "do_run", is_flag=True)
def main(
    model: str,
    model_revision: str | None,
    base_url: str | None,
    relay_job: str,
    review_cache: str | None,
    review_mode: str | None,
    review_concurrency: int,
    mode: str,
    max_workers: int | None,
    coordinator_memory: str | None,
    normalized_shards: int | None,
    concurrent_sources: int,
    worker_image: str | None,
    container_profile: str,
    verification_backend: str,
    controller_url: str | None,
    seed: int,
    verification_sample_size: int,
    report_path: str | None,
    sources: tuple[str, ...],
    input_root: Path | None,
    local_files: tuple[tuple[str, Path], ...],
    auxiliary: tuple[tuple[str, str], ...],
    output_root: Path | None,
    download_cache: Path | None,
    do_run: bool,
) -> None:
<<<<<<< HEAD
    pipelines = _selected_pipelines(sources)
||||||| parent of a6ae499b25 ([rl-data] Invoke dataset-owned curation pipelines)
    try:
        pipelines = selected_pipelines(sources)
    except ValueError as error:
        raise click.UsageError(str(error)) from error
=======
    try:
        selected = selected_sources(sources)
    except ValueError as error:
        raise click.UsageError(str(error)) from error
>>>>>>> a6ae499b25 ([rl-data] Invoke dataset-owned curation pipelines)
    processing_mode = SourceProcessingMode(mode)
    if processing_mode == SourceProcessingMode.QUICK:
        if not sources:
            raise click.UsageError("QUICK requires at least one --source")
        if output_root is None:
            raise click.UsageError("QUICK requires --output-root")
        if input_root is not None and local_files:
            raise click.UsageError("Choose either --input-root or --input-file")
        selected = {name: selected[name] for name in dict.fromkeys(sources)}
        inputs = {name: str(Path(path).resolve()) for name, path in auxiliary}
        download_cache = download_cache if download_cache is not None else Path.home() / ".cache/marin"
        max_workers = max_workers if max_workers is not None else QUICK_MAX_WORKERS
        if not do_run:
            plan = {
                "mode": processing_mode,
                "sources": [
                    {
<<<<<<< HEAD
                        "name": pipeline.name,
                        "source": download_identity(pipeline.source),
                        "inputs": {name: download_identity(source) for name, source in pipeline.inputs.items()},
                    }
                    for pipeline in pipelines.values()
                ],
                "input_root": str(input_root) if input_root is not None else None,
                "input_files": {name: str(path) for name, path in local_files},
                "inputs": inputs,
                "output_root": str(output_root),
                "download_cache": str(download_cache),
                "max_workers": max_workers,
            }
            click.echo(json.dumps(plan, indent=2))
||||||| parent of a6ae499b25 ([rl-data] Invoke dataset-owned curation pipelines)
                        "mode": processing_mode,
                        "sources": [
                            {
                                "name": pipeline.name,
                                "source": download_identity(pipeline.source),
                                "inputs": {name: download_identity(source) for name, source in pipeline.inputs.items()},
                            }
                            for pipeline in pipelines.values()
                        ],
                        "input_root": str(input_root) if input_root is not None else None,
                        "input_files": {name: str(path) for name, path in local_files},
                        "inputs": inputs,
                        "output_root": str(output_root),
                        "download_cache": str(download_cache),
                        "max_workers": max_workers,
                    },
                    indent=2,
                )
            )
=======
                        "mode": processing_mode,
                        "sources": [local_source_plan(source) for source in selected.values()],
                        "input_root": str(input_root) if input_root is not None else None,
                        "input_files": {name: str(path) for name, path in local_files},
                        "inputs": inputs,
                        "output_root": str(output_root),
                        "download_cache": str(download_cache),
                        "max_workers": max_workers,
                    },
                    indent=2,
                )
            )
>>>>>>> a6ae499b25 ([rl-data] Invoke dataset-owned curation pipelines)
            return
        run_local_sources(
            selected,
            input_root,
            output_root,
            inputs=inputs,
            max_workers=max_workers,
            download_cache=download_cache,
            source_files_override=dict(local_files),
        )
        return
    if input_root is not None or local_files or auxiliary or output_root is not None or download_cache is not None:
        raise click.UsageError("Local input, output, and download-cache options require --mode quick")
    if max_workers is None:
        raise click.UsageError("SAMPLE/FULL requires --max-workers")
    if coordinator_memory is None:
        raise click.UsageError("SAMPLE/FULL requires --coordinator-memory")
    if worker_image is None:
        raise click.UsageError("SAMPLE/FULL requires --worker-image")
    if report_path is None:
        raise click.UsageError("SAMPLE/FULL requires --report-path")
    worker_resources = ResourceConfig(cpu=2, ram="8g", image=worker_image, container_profile=container_profile)
    standard = [source.pipeline for source in selected.values() if isinstance(source.pipeline, RlDataPipeline)]
    config = None
    if standard:
        if model_revision is None:
            raise click.UsageError("Standard SAMPLE/FULL requires --model-revision")
        if review_cache is None:
            raise click.UsageError("Standard SAMPLE/FULL requires --review-cache")
        if review_mode is None:
            raise click.UsageError("Standard SAMPLE/FULL requires --review-mode")
        if normalized_shards is None:
            raise click.UsageError("Standard SAMPLE/FULL requires --normalized-shards")
        review = ReviewConfig(model=model, model_revision=model_revision, mode=ReviewMode(review_mode))
        reviewer = None
        if do_run and any(pipeline.rubric is not None for pipeline in standard):
            reviewer = _reviewer(
                review,
                base_url if base_url is not None else resolve_glm_base_url(relay_job),
                review_cache=review_cache,
                review_concurrency=review_concurrency,
            )
        machines = None
        if any(pipeline.controls is not None for pipeline in standard):
            backend = VerificationBackend(verification_backend)
            controller_url = _controller_url(backend, controller_url)
            machines = campaign_machines(backend, worker_image, controller_url)
        config = _pipeline_config(
            processing_mode,
            review,
            reviewer,
            machines,
            seed=seed,
            verification_sample_size=verification_sample_size,
            max_workers=max_workers,
            worker_resources=worker_resources,
            normalized_shards=normalized_shards,
        )
    runtime = CampaignRuntime()
    steps = [
        pipeline_step(source, mode=processing_mode, standard_config=config, campaign=runtime)
        for source in selected.values()
    ]
    pool = CampaignPool(
        max_workers,
        concurrent_sources,
        worker_resources=worker_resources,
        coordinator_resources=ResourceConfig(cpu=1, ram=coordinator_memory, preemptible=False),
    )
    if not do_run:
        click.echo(json.dumps(campaign_plan(steps, pool), indent=2))
        return
    run_campaign(steps, runtime=runtime, pool=pool, report_path=report_path, mode=processing_mode)


if __name__ == "__main__":
    main()
