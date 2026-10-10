# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dataset declarations and the artifact steps that ingest them.

A ``CurationRecipe`` names one pinned source, the converter that turns each row into a
``TaskSpec`` with its grader fixed, the agent's environment, what its graders' environment must
provide, an optional review rubric and optional grader controls. ``process_rows`` constructs a graph
for ``taskcompendium.pipeline.source_processing.run_source_pipeline``. SAMPLE/FULL cache a
``data/rl/<name>-<hash>`` artifact; QUICK converts at the explicit local output path.
"""

import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, cast

import click
import requests
from fray.types import ResourceConfig
from iris.cluster.client.job_info import get_job_info
from marin.datakit.download.huggingface import (
    DownloadConfig,
    finish_download,
    plan_download,
    stream_file_to_fsspec,
)
from marin.execution.artifact import Artifact
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep, StepContext
from marin.inference.openai_batch import OpenAIBatchClient
from marin.inference.openai_chat import OpenAIChatClient
from rigging.filesystem.storage_path import StoragePath
from shellbox.backends.gvisor.machine import GvisorMachineFactory
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import Backend, DockerImage, MachineFactory, MachineSpec, NetworkPolicy
from taskcompendium.convert.environment import IMAGE_BACKENDS
from taskcompendium.models import EnvironmentRequirements, require_resolved_environment
from taskcompendium.pipeline.controls import GradingMachines, controls_identity
from taskcompendium.pipeline.fingerprints import callable_identity, callable_module, recipe_code_identity
from taskcompendium.pipeline.inputs import ConversionContext, FileParts, SourceFileOverride, SourceFiles, SourceFormat
from taskcompendium.pipeline.models import (
    RESOURCE_BUDGET_BYTES,
    Controls,
    Converter,
    FilterPolicy,
    IntendedUse,
    ReviewRubric,
    SourceRecipe,
    SourceStatus,
)
from taskcompendium.pipeline.review import BatchReviewer, ChatReviewer, Reviewer
from taskcompendium.pipeline.source_processing import (
    SOURCE_PIPELINE_REVISION,
    ConversionResult,
    SourcePipelineConfig,
    SourcePipelineResult,
    SourceProcessingMode,
    run_source_pipeline,
)
from taskcompendium.pipeline.source_quality import SOURCE_QUALITY_REVISION, SourceQualityPolicy
from taskcompendium.pipeline.source_verification import SOURCE_VERIFICATION_REVISION, SourceVerificationPolicy
from taskcompendium.pipeline.sources import conversion_shards, source_files_identity
from taskcompendium.pipeline.stages import AuditExecution, ReviewConfig, ReviewMode
from taskcompendium.runtime.local import LocalGraderMachines, context_paths
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.runners import SubprocessRunner

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, resolve_glm_base_url
from experiments.post_training.task_curation.campaign import CampaignArtifact, CampaignRuntime, PipelineResult
from experiments.post_training.task_curation.config import ImageGraderExecution, PipelineOptions, RecipeSettings
from experiments.post_training.task_curation.environment import Environment, Placement, placement
from experiments.post_training.task_curation.images.build import (
    EnvironmentArtifact,
    built_environment,
    environment_artifact,
)
from experiments.post_training.task_curation.source import RlDataSource

PIPELINE_VERSION = "2026.10.07.1"
URL_CHUNK_BYTES = 1024 * 1024
URL_TIMEOUT = 60
REVIEW_REQUEST_TIMEOUT = 60
IRIS_MACHINE_CPUS = 4
IRIS_JOB_TTL = 1800

type RowSelector = Callable[[dict[str, Any], ConversionContext], bool]
type RowDecoder = Callable[[dict[str, Any], ConversionContext], dict[str, Any]]
type FileReader = Callable[[StoragePath, ConversionContext], Iterator[dict[str, Any]]]


@dataclass(frozen=True)
class HfSource:
    """Files from a Hugging Face dataset repository at a pinned revision.

    ``select`` drops rows before raw sampling (for example, one component of a blend). ``decode``
    rewrites a selected row before conversion (for example, unpacking an archive). ``read`` replaces
    the format reader for files that need a custom parser; ``parts`` replaces it for a file several
    workers read in parts, such as a slow generator. Each receives the ``ConversionContext``: the
    staged auxiliary inputs and the grader's environment.
    """

    repo: str
    revision: str
    files: tuple[str, ...]
    format: SourceFormat
    select: RowSelector | None = None
    decode: RowDecoder | None = None
    read: FileReader | None = None
    parts: FileParts | None = None

    @property
    def name(self) -> str:
        return self.repo

    @property
    def url(self) -> str:
        return f"https://huggingface.co/datasets/{self.repo}/tree/{self.revision}"


@dataclass(frozen=True)
class UrlSource:
    """One file fetched from ``url`` and checked against ``sha256``, staged as ``filename``."""

    url: str
    sha256: str
    filename: str
    format: SourceFormat
    select: RowSelector | None = None
    decode: RowDecoder | None = None
    read: FileReader | None = None
    parts: FileParts | None = None

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ValueError(f"URL source requires a lowercase SHA-256 digest: {self.url}")

    @property
    def name(self) -> str:
        return self.filename

    @property
    def revision(self) -> str:
        return self.sha256

    @property
    def files(self) -> tuple[str, ...]:
        return (self.filename,)


@dataclass(frozen=True)
class ShellSim:
    """No agent machine: conversation tasks, or shell tasks served by the simulated shell."""


def environment_requirements(
    environment: Environment, built: EnvironmentArtifact | None = None
) -> EnvironmentRequirements:
    """The requirements a task records for ``environment``; they name where the pipeline runs it.

    A declared image runs in a sandbox of that image, and an environment that needs apt packages the
    worker image lacks runs in a sandbox of the image built for it. Any other environment runs in the
    Zephyr worker, which builds a uv environment from the lock its artifact stores. ``built`` is that
    artifact, which every environment without a declared image requires.
    """
    where = placement(environment)
    if where == Placement.IMAGE:
        assert environment.image is not None
        return EnvironmentRequirements(docker_image=environment.image, compatible_backends=IMAGE_BACKENDS)
    if built is None:
        raise ValueError("An environment without a declared image runs from its built artifact")
    if where == Placement.BUILT_IMAGE:
        assert built.image is not None
        return EnvironmentRequirements(docker_image=built.image, compatible_backends=IMAGE_BACKENDS)
    return EnvironmentRequirements(compatible_backends=(Backend.LOCAL,), packages_lock=built.lock_url)


def environment_record(environment: Environment, built: EnvironmentArtifact | None) -> dict[str, Any]:
    """What a source artifact's identity records of an environment: its image, or its build and built image."""
    if environment.image is not None:
        return {"image": environment.image}
    assert built is not None
    return {"identity": built.identity, "image": built.image}


@dataclass(frozen=True)
class CurationRecipe:
    """One RL data source and how its rows become tasks.

    ``name`` is the catalog key and artifact name. ``version`` is the converter revision; bump it
    when conversion changes in a way the hashed files do not capture. ``rubric=None`` skips model
    review and ``controls=None`` skips grader verification. ``inputs`` are auxiliary pinned files
    staged before conversion; the source callables and converter find them in ``context.inputs``.

    ``environment`` is the agent's: ``ShellSim()`` or an ``Environment`` naming its image, which the
    converter records on each task. ``grader`` is what the source's grader scripts need; the pipeline
    builds it, decides where it runs (``environment_requirements``) and passes the result to the
    converter as ``context.grader_environment``. ``ships`` are directories, such as
    ``datasets/<family>/scorers``, whose files the converter packages into tasks. The artifact
    identity hashes the converter module's directory, every file below ``ships`` and the grader's
    built environment. A task whose decoded resources exceed ``resource_budget_bytes`` is deferred as
    ``resources_over_budget``.
    """

    name: str
    source: HfSource | UrlSource
    convert: Converter
    version: str
    environment: Environment | ShellSim
    intended_use: IntendedUse
    rubric: str | None = None
    controls: Controls | None = None
    inputs: Mapping[str, HfSource | UrlSource] = field(default_factory=dict)
    grader: Environment | None = None
    ships: tuple[Path, ...] = ()
    resource_budget_bytes: int = RESOURCE_BUDGET_BYTES

    def __post_init__(self) -> None:
        missing = [str(path) for path in self.ships if not path.is_dir()]
        if missing:
            raise ValueError(f"{self.name} ships directories that do not exist: {missing}")
        # Converters record the agent's requirements on each task without a built artifact to consult.
        if isinstance(self.environment, Environment) and self.environment.image is None:
            raise ValueError(f"{self.name} must name the agent environment's image")

    @property
    def dataset(self) -> HfSource | UrlSource:
        return self.source

    @property
    def files(self) -> tuple[str, ...]:
        return self.source.files


def machines_identity(backend: ImageGraderExecution, worker_image: str, controller: bool) -> dict[str, Any]:
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
        return machines_identity(ImageGraderExecution.IRIS, self.worker_image, controller=True)

    def machine(self, environment: EnvironmentRequirements, memory_mb: int) -> tuple[MachineFactory, MachineSpec]:
        image = _sandbox_image(environment)
        factory = IrisMachineFactory(
            controller_url=_controller_url(self.controller_url),
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
        return machines_identity(ImageGraderExecution.GVISOR, self.worker_image, controller=False)

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


def campaign_machines(backend: ImageGraderExecution, worker_image: str, controller_url: str | None) -> GradingMachines:
    """Grading machines for every source in a campaign: sandbox graders on ``backend``, local graders in the worker."""
    if backend == ImageGraderExecution.IRIS:
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
    review: ReviewConfig | None,
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


def _source_config(
    settings: RecipeSettings, mode: SourceProcessingMode, controls: Controls | None, rubric: str | None
) -> SourcePipelineConfig:
    review = settings.config.review if settings.config is not None else settings.review
    if rubric is not None and review is None:
        raise click.UsageError("Recipe review configuration requires --model-revision and --review-mode")
    if settings.config is not None:
        return replace(settings.config, mode=mode)
    if settings.normalized_shards is None:
        raise click.UsageError("Recipe conversion requires --normalized-shards")
    if settings.execution.worker_resources is None:
        raise ValueError("Recipe execution requires worker resources")
    return _pipeline_config(
        mode,
        review,
        None,
        (
            campaign_machines(
                settings.verification_backend, settings.execution.worker_resources.image, settings.controller_url
            )
            if controls is not None
            else None
        ),
        seed=settings.seed,
        verification_sample_size=settings.verification_sample_size,
        max_workers=settings.execution.max_workers,
        worker_resources=settings.execution.worker_resources,
        normalized_shards=settings.normalized_shards,
    )


def _execution_config(
    settings: RecipeSettings, config: SourcePipelineConfig, rubric: str | None
) -> SourcePipelineConfig:
    if rubric is None or config.execution.reviewer is not None:
        return config
    if settings.review_cache is None:
        raise click.UsageError("Recipe review requires --review-cache")
    reviewer = _reviewer(
        cast(ReviewConfig, config.review),
        settings.base_url if settings.base_url is not None else resolve_glm_base_url(settings.relay_job),
        review_cache=settings.review_cache,
        review_concurrency=settings.review_concurrency,
    )
    return replace(config, execution=replace(config.execution, reviewer=reviewer))


def process_rows(source: RlDataSource[CurationRecipe], options: PipelineOptions) -> ArtifactStep[CampaignArtifact]:
    """Construct the reusable source processor's graph from this dataset's recipe."""
    recipe = source.config
    assert recipe is not None
    if options.mode == SourceProcessingMode.QUICK:
        return _quick_step(recipe, options)
    if options.recipe_settings is None:
        raise ValueError(f"Source processing requires recipe settings: {source.name}")
    config = _source_config(options.recipe_settings, options.mode, recipe.controls, recipe.rubric)
    if recipe.controls is None:
        config = replace(config, machines=None)
    return cast(ArtifactStep[CampaignArtifact], _reviewed_step(recipe, config, options))


def _pipeline_result(result: ConversionResult | SourcePipelineResult) -> PipelineResult:
    if isinstance(result, ConversionResult):
        return PipelineResult(
            SourceStatus.COMPLETED,
            {"normalize": result.normalized_path},
            {"manifest": result.manifest_path},
            ("normalize",),
        )
    if result.status == SourceStatus.INCOMPLETE:
        raise SourcePipelineIncomplete(f"Source pipeline is incomplete; retained evidence: {result.manifest_path}")
    review_manifest = json.loads((StoragePath(result.review_path) / "manifest.json").read_text())
    manifest = json.loads(StoragePath(result.manifest_path).read_text())
    telemetry = json.loads(StoragePath(manifest["telemetry"]).read_text())
    stages = ["download", "normalize"]
    if review_manifest["rubric"] is not None:
        stages.append("review")
    if any(phase["phase"] == "verification" for phase in telemetry["phases"]):
        stages.append("verify")
    stages.append("final")
    return PipelineResult(
        result.status,
        {"normalize": result.normalize_path, "final": result.final_path},
        {
            "download": result.download_path,
            "review": result.review_path,
            "verify": result.verify_path,
            "manifest": result.manifest_path,
        },
        tuple(stages),
    )


class RlDataArtifact(CampaignArtifact):
    manifest: dict[str, Any]


class SourcePipelineIncomplete(RuntimeError):
    """The quality gate could not decide or a control trial hit an infrastructure error; evidence is retained."""


def review_rubric(pipeline: CurationRecipe) -> ReviewRubric | None:
    """Split a rubric string into criteria at blank lines."""
    if pipeline.rubric is None:
        return None
    criteria = tuple(" ".join(part.split()) for part in pipeline.rubric.strip().split("\n\n"))
    if not all(criteria):
        raise ValueError(f"Rubric criteria must be nonempty: {pipeline.name}")
    return ReviewRubric(id=pipeline.name, version=pipeline.version, criteria=criteria)


def source_files(source: HfSource | UrlSource) -> SourceFiles:
    if isinstance(source, HfSource):
        return SourceFiles(
            source.repo,
            source.revision,
            source.files,
            source.format,
            source.select,
            source.decode,
            source.read,
            source.parts,
        )
    return SourceFiles(
        source.url,
        source.sha256,
        (source.filename,),
        source.format,
        source.select,
        source.decode,
        source.read,
        source.parts,
    )


def source_recipe(
    pipeline: CurationRecipe, inputs: Mapping[str, str], grader_environment: EnvironmentRequirements | None
) -> SourceRecipe:
    return SourceRecipe(
        name=pipeline.name,
        version=pipeline.version,
        source=source_files(pipeline.source),
        convert=pipeline.convert,
        rubric=review_rubric(pipeline),
        controls=pipeline.controls,
        intended_use=pipeline.intended_use,
        inputs=dict(inputs),
        grader_environment=grader_environment,
        resource_budget_bytes=pipeline.resource_budget_bytes,
    )


def _run_curation(
    pipeline: CurationRecipe,
    *,
    mode: SourceProcessingMode,
    context: ZephyrContext,
    source_input: str,
    output_path: str,
    inputs: Mapping[str, str],
    config: SourcePipelineConfig | None = None,
    grader_environment: EnvironmentRequirements | None = None,
    source_overrides: Mapping[str, SourceFileOverride] | None = None,
) -> ConversionResult | SourcePipelineResult:
    """Run a declared source against staged inputs in quick, sample, or full mode.

    QUICK records the declared grader lock without building or running it.
    SAMPLE and FULL use the invocation mode and the campaign's resolved settings.
    """
    missing = pipeline.inputs.keys() - inputs.keys()
    if missing:
        raise ValueError(f"Missing staged auxiliary inputs for {pipeline.name}: {sorted(missing)}")
    if mode == SourceProcessingMode.QUICK:
        if config is not None:
            raise ValueError("QUICK conversion does not take review or verification settings")
        if grader_environment is None and pipeline.grader is not None:
            environment = pipeline.grader
            if environment.image is not None:
                grader_environment = environment_requirements(environment)
            elif environment.lock is not None:
                grader_environment = EnvironmentRequirements(
                    compatible_backends=(Backend.LOCAL,), packages_lock=str(environment.lock.resolve())
                )
            else:
                raise ValueError(
                    f"Quick conversion of {pipeline.name} needs a declared grader lock or resolved environment"
                )
    return run_source_pipeline(
        source_recipe(pipeline, inputs, grader_environment),
        context,
        source_input,
        output_path,
        config,
        mode=mode,
        source_overrides=source_overrides,
        canonical_source=pipeline.name,
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def converter_identity(pipeline: CurationRecipe, grader: EnvironmentArtifact | None) -> dict[str, Any]:
    """The converter, every file it can package into a task, and the environment its graders run in.

    Files are the converter module's directory's ``*.py`` and everything below ``ships``, keyed
    by path relative to the module's directory. Library code the converter calls, such as
    ``taskcompendium.convert.script_grader``, is not hashed: as for every ``taskcompendium.convert``
    helper, a library change that alters tasks bumps the normalization stage revision or the
    declaration's ``version``.
    """
    module = sys.modules[callable_module(pipeline.convert)]
    assert module.__file__ is not None
    directory = Path(module.__file__).parent
    files = {*directory.glob("*.py"), *(path for root in pipeline.ships for path in context_paths(root))}
    return {
        "function": callable_identity(pipeline.convert),
        "files": {os.path.relpath(path, directory): _file_sha256(path) for path in sorted(files)},
        "grader": environment_record(pipeline.grader, grader) if pipeline.grader is not None else None,
    }


def download_identity(source: HfSource | UrlSource) -> dict[str, Any]:
    """The pinned bytes a source download stages; reader callables do not change them."""
    if isinstance(source, HfSource):
        return {"repo": source.repo, "revision": source.revision, "files": sorted(source.files)}
    return {"url": source.url, "sha256": source.sha256, "filename": source.filename}


def pipeline_identity(
    pipeline: CurationRecipe, config: SourcePipelineConfig, grader: EnvironmentArtifact | None
) -> dict[str, Any]:
    """Everything that can change a source artifact's contents; ``grader`` is the grader's built environment."""
    recipe = source_recipe(pipeline, {}, None)
    execution = config.execution
    return {
        "name": pipeline.name,
        "version": pipeline.version,
        "intended_use": pipeline.intended_use,
        "source": {**download_identity(pipeline.source), "files": source_files_identity(recipe.source)},
        "inputs": {name: download_identity(source) for name, source in sorted(pipeline.inputs.items())},
        "code": recipe_code_identity(recipe),
        "converter": converter_identity(pipeline, grader),
        "environment": (
            environment_record(pipeline.environment, None)
            if isinstance(pipeline.environment, Environment)
            else {"shellsim": True}
        ),
        "resource_budget_bytes": pipeline.resource_budget_bytes,
        "rubric": pipeline.rubric,
        "review": asdict(config.review) if pipeline.rubric is not None else None,
        "controls": controls_identity(pipeline.controls) if pipeline.controls is not None else None,
        "machines": (
            config.machines.identity() if pipeline.controls is not None and config.machines is not None else None
        ),
        "mode": config.mode,
        "review_batch_size": execution.review_batch_size,
        "review_input_bytes": execution.review_input_bytes,
        "worker_image": execution.worker_resources.image if execution.worker_resources else None,
        "normalized_shards": config.normalized_shards,
        "quality_policy": asdict(config.quality_policy),
        "verification_policy": asdict(config.verification_policy),
        "filter_policy": asdict(config.filter_policy),
        "revisions": {
            "procedure": SOURCE_PIPELINE_REVISION,
            "quality": SOURCE_QUALITY_REVISION,
            "verification": SOURCE_VERIFICATION_REVISION,
        },
    }


@dataclass(frozen=True)
class DownloadRequest:
    source: HfSource | UrlSource
    output_path: str


def download_source(request: DownloadRequest, *, campaign: CampaignRuntime) -> None:
    """Stage a source's pinned files; a URL download must match its declared digest."""
    source = request.source
    if isinstance(source, HfSource):
        download = DownloadConfig(
            hf_dataset_id=source.repo,
            revision=source.revision,
            hf_urls_glob=list(source.files),
            gcs_output_path=request.output_path,
            wait_for_completion=True,
        )
        plan = plan_download(download)
        campaign.context.execute(
            Dataset.from_list(list(plan.tasks))
            .map(stream_file_to_fsspec)
            .write_jsonl(plan.metrics_path, skip_existing=True)
        )
        finish_download(plan)
        return
    digest = hashlib.sha256()
    destination = StoragePath(request.output_path) / source.filename
    with requests.get(source.url, stream=True, timeout=URL_TIMEOUT) as response:
        response.raise_for_status()
        with destination.open("wb", auto_mkdir=True) as stream:
            for chunk in response.iter_content(URL_CHUNK_BYTES):
                digest.update(chunk)
                stream.write(chunk)
    if digest.hexdigest() != source.sha256:
        destination.rm()
        raise ValueError(f"{source.url} has SHA-256 {digest.hexdigest()}; expected {source.sha256}")


def _download_request(source: HfSource | UrlSource, ctx: StepContext) -> DownloadRequest:
    return DownloadRequest(source, ctx.output_path)


def download_step(source: HfSource | UrlSource, campaign: CampaignRuntime) -> ArtifactStep[Artifact]:
    """One shared artifact per distinct set of pinned bytes."""
    identity = hashlib.sha256(canonical_json(download_identity(source)).encode()).hexdigest()[:16]
    return ArtifactStep(
        name=f"task-curation/download/{identity}",
        version=PIPELINE_VERSION,
        artifact_type=Artifact,
        run=partial(download_source, campaign=campaign),
        build_config=partial(_download_request, source),
    )


def source_downloads(
    pipeline: CurationRecipe, campaign: CampaignRuntime
) -> tuple[ArtifactStep[Artifact], dict[str, ArtifactStep[Artifact]]]:
    """Pinned primary and auxiliary downloads shared by local and reviewed runs."""
    return download_step(pipeline.source, campaign), {
        name: download_step(source, campaign) for name, source in sorted(pipeline.inputs.items())
    }


@dataclass(frozen=True)
class SourceRun:
    identity: dict[str, Any]
    source_input: str
    inputs: dict[str, str]
    output_path: str
    grader_artifact: str | None


def _source_run(
    identity: dict[str, Any],
    downloaded: ArtifactStep[Artifact],
    inputs: Mapping[str, ArtifactStep[Artifact]],
    grader: ArtifactStep[EnvironmentArtifact] | None,
    ctx: StepContext,
) -> SourceRun:
    return SourceRun(
        identity=identity,
        source_input=ctx.artifact_path(downloaded),
        inputs={name: ctx.artifact_path(step) for name, step in inputs.items()},
        output_path=ctx.output_path,
        grader_artifact=ctx.artifact_path(grader) if grader is not None else None,
    )


def _run_source(
    pipeline: CurationRecipe,
    config: SourcePipelineConfig,
    options: PipelineOptions,
    run: SourceRun,
) -> RlDataArtifact:
    grader_environment = None
    if pipeline.grader is not None:
        built = EnvironmentArtifact.raw_load(run.grader_artifact) if run.grader_artifact is not None else None
        grader_environment = environment_requirements(pipeline.grader, built)
    assert options.recipe_settings is not None
    config = _execution_config(options.recipe_settings, config, pipeline.rubric)
    result = _run_curation(
        pipeline,
        mode=options.mode,
        context=options.runtime.context,
        source_input=run.source_input,
        output_path=run.output_path,
        inputs=run.inputs,
        config=config,
        grader_environment=grader_environment,
    )
    envelope = _pipeline_result(result)
    return RlDataArtifact(path=run.output_path, status=envelope.status, manifest=asdict(result), result=envelope)


def _reviewed_step(
    pipeline: CurationRecipe,
    config: SourcePipelineConfig,
    options: PipelineOptions,
) -> ArtifactStep[RlDataArtifact]:
    downloaded, inputs = source_downloads(pipeline, options.runtime)
    grader_step = None
    grader_built = None
    if pipeline.grader is not None and placement(pipeline.grader) != Placement.IMAGE:
        grader_built = built_environment(pipeline.grader)
        grader_step = environment_artifact(pipeline.grader)
    identity = pipeline_identity(pipeline, config, grader_built)
    digest = hashlib.sha256(canonical_json(identity).encode()).hexdigest()[:16]
    return ArtifactStep(
        name=f"data/rl/{pipeline.name}-{digest}",
        version=PIPELINE_VERSION,
        artifact_type=RlDataArtifact,
        run=partial(_run_source, pipeline, config, options),
        build_config=partial(_source_run, identity, downloaded, inputs, grader_step),
        deps=(downloaded, *inputs.values(), *((grader_step,) if grader_step is not None else ())),
    )


@dataclass(frozen=True)
class QuickRun:
    source_input: str
    inputs: dict[str, str]
    output_path: str


def _quick_run(
    options: PipelineOptions,
    downloaded: ArtifactStep[Artifact] | None,
    auxiliary: Mapping[str, ArtifactStep[Artifact]],
    ctx: StepContext,
) -> QuickRun:
    return QuickRun(
        source_input=ctx.artifact_path(downloaded) if downloaded is not None else options.inputs.root or ctx.output_path,
        inputs={**options.inputs.auxiliary, **{name: ctx.artifact_path(step) for name, step in auxiliary.items()}},
        output_path=ctx.output_path,
    )


@contextmanager
def _conversion_context(context: ZephyrContext, pipeline: CurationRecipe, run: QuickRun, overrides):
    shards = conversion_shards(run.source_input, source_files(pipeline.source), overrides=overrides)
    assert context.max_workers is not None
    if context.max_workers <= 1 or not any(shard.row_end is not None and shard.parts > 1 for shard in shards):
        yield context
        return
    # Split Parquet conversion benefits from processes; small conversions keep the local runner.
    with ZephyrContext(
        client=context.client,
        max_workers=context.max_workers,
        resources=context.resources,
        chunk_storage_prefix=str(StoragePath(run.output_path).parent / ".zephyr-process"),
        name="task-curation-quick-process",
        stage_runner_factory=SubprocessRunner,
    ) as process_context:
        yield process_context


def _run_quick_source(pipeline: CurationRecipe, options: PipelineOptions, run: QuickRun) -> CampaignArtifact:
    overrides = {}
    for logical, file in options.inputs.files.items():
        with file.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        overrides[logical] = SourceFileOverride(str(file.resolve()), checksum)
    with _conversion_context(options.runtime.context, pipeline, run, overrides) as context:
        result = _run_curation(
            pipeline,
            mode=options.mode,
            context=context,
            source_input=run.source_input,
            output_path=run.output_path,
            inputs=run.inputs,
            source_overrides=overrides,
        )
    envelope = _pipeline_result(result)
    return CampaignArtifact(path=run.output_path, status=envelope.status, result=envelope)


def _quick_step(pipeline: CurationRecipe, options: PipelineOptions) -> ArtifactStep[CampaignArtifact]:
    primary = None
    if options.inputs.root is None and not options.inputs.files:
        primary = download_step(pipeline.source, options.runtime)
    auxiliary = {
        name: download_step(upstream, options.runtime)
        for name, upstream in sorted(pipeline.inputs.items())
        if name not in options.inputs.auxiliary
    }
    return ArtifactStep(
        name=pipeline.name,
        version=PIPELINE_VERSION,
        artifact_type=CampaignArtifact,
        run=partial(_run_quick_source, pipeline, options),
        build_config=partial(_quick_run, options, primary, auxiliary),
        deps=(*((primary,) if primary is not None else ()), *auxiliary.values()),
    )
