# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dataset declarations and the artifact steps that ingest them.

An ``RlDataPipeline`` names one pinned source, the converter that turns each row into a
``TaskSpec`` with its grader fixed, the agent's environment, what its graders' environment must
provide, an optional review rubric and optional grader controls. ``source_step`` turns a declaration
into one cached ``data/rl/<name>-<hash>`` artifact produced by
``taskcompendium.pipeline.source_processing.run_source_pipeline``.
"""

import hashlib
import os
import re
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import requests
from marin.datakit.download.huggingface import (
    DownloadConfig,
    finish_download,
    plan_download,
    stream_file_to_fsspec,
)
from marin.execution.artifact import Artifact
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep, StepContext
from rigging.filesystem.storage_path import StoragePath
from shellbox.machine import Backend
from taskcompendium.convert.environment import IMAGE_BACKENDS
from taskcompendium.models import EnvironmentRequirements
from taskcompendium.pipeline.controls import controls_identity
from taskcompendium.pipeline.fingerprints import callable_identity, callable_module, recipe_code_identity
from taskcompendium.pipeline.inputs import ConversionContext, FileParts, SourceFiles, SourceFormat
from taskcompendium.pipeline.models import (
    RESOURCE_BUDGET_BYTES,
    Controls,
    Converter,
    IntendedUse,
    ReviewRubric,
    SourceRecipe,
    SourceStatus,
)
from taskcompendium.pipeline.source_processing import (
    SOURCE_PIPELINE_REVISION,
    SourcePipelineConfig,
    run_source_pipeline,
)
from taskcompendium.pipeline.source_quality import SOURCE_QUALITY_REVISION
from taskcompendium.pipeline.source_verification import SOURCE_VERIFICATION_REVISION
from taskcompendium.pipeline.sources import source_files_identity
from zephyr.dataset import Dataset

from experiments.post_training.task_curation.campaign import CampaignArtifact, CampaignRuntime
from experiments.post_training.task_curation.environment import Environment, Placement, placement
from experiments.post_training.task_curation.images.build import (
    EnvironmentArtifact,
    built_environment,
    context_paths,
    environment_artifact,
)

PIPELINE_VERSION = "2026.10.07.1"
URL_CHUNK_BYTES = 1024 * 1024
URL_TIMEOUT = 60

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
class RlDataPipeline:
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
    atlas_id: str | None = None
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


class RlDataArtifact(CampaignArtifact):
    manifest: dict[str, Any]


class SourcePipelineIncomplete(RuntimeError):
    """The quality gate could not decide or a control trial hit an infrastructure error; evidence is retained."""


def review_rubric(pipeline: RlDataPipeline) -> ReviewRubric | None:
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
    pipeline: RlDataPipeline, inputs: Mapping[str, str], grader_environment: EnvironmentRequirements | None
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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def converter_identity(pipeline: RlDataPipeline, grader: EnvironmentArtifact | None) -> dict[str, Any]:
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
    pipeline: RlDataPipeline, config: SourcePipelineConfig, grader: EnvironmentArtifact | None
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
    pipeline: RlDataPipeline, config: SourcePipelineConfig, run: SourceRun, *, campaign: CampaignRuntime
) -> RlDataArtifact:
    grader_environment = None
    if pipeline.grader is not None:
        built = EnvironmentArtifact.raw_load(run.grader_artifact) if run.grader_artifact is not None else None
        grader_environment = environment_requirements(pipeline.grader, built)
    result = run_source_pipeline(
        source_recipe(pipeline, run.inputs, grader_environment),
        campaign.context,
        run.source_input,
        run.output_path,
        config,
        canonical_source=pipeline.name,
    )
    if result.status == SourceStatus.INCOMPLETE:
        raise SourcePipelineIncomplete(f"Source pipeline is incomplete; retained evidence: {result.manifest_path}")
    return RlDataArtifact(path=run.output_path, status=result.status, manifest=asdict(result))


def source_step(
    pipeline: RlDataPipeline, config: SourcePipelineConfig, campaign: CampaignRuntime
) -> ArtifactStep[RlDataArtifact]:
    """The ``data/rl/<name>-<hash>`` artifact for one declaration.

    A ``grader`` without a declared image requires its environment's artifact to be built already; see
    ``images.build``.
    """
    downloaded = download_step(pipeline.source, campaign)
    inputs = {name: download_step(source, campaign) for name, source in sorted(pipeline.inputs.items())}
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
        run=partial(_run_source, pipeline, config, campaign=campaign),
        build_config=partial(_source_run, identity, downloaded, inputs, grader_step),
        deps=(downloaded, *inputs.values(), *((grader_step,) if grader_step is not None else ())),
    )
