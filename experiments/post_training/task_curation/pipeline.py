# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dataset declarations and the artifact steps that ingest them.

An ``RlDataPipeline`` names one pinned source, the converter that turns each row into a
``TaskSpec`` with its grader fixed, the agent environment, the image its sandboxed graders run in,
an optional review rubric and optional grader controls. ``source_step`` turns a declaration into
one cached ``data/rl/<name>-<hash>`` artifact produced by
``taskcompendium.pipeline.source_processing.run_source_pipeline``.
"""

import hashlib
import os
import re
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
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
from taskcompendium.convert.environment import IMAGE_BACKENDS, grading_environment, local_grading_environment
from taskcompendium.models import DOCKER_IMAGE_PATTERN, EnvironmentRequirements
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
    VERIFY_REPORT_PATH,
    SourcePipelineConfig,
    run_source_pipeline,
)
from taskcompendium.pipeline.source_quality import SOURCE_QUALITY_REVISION
from taskcompendium.pipeline.source_verification import SOURCE_VERIFICATION_REVISION
from taskcompendium.pipeline.sources import source_files_identity
from zephyr.dataset import Dataset

from experiments.post_training.task_curation.campaign import CampaignArtifact, CampaignRuntime
from experiments.post_training.task_curation.images.build import (
    ImageArtifact,
    built_image,
    context_paths,
    image_artifact,
)
from experiments.post_training.task_curation.images.recipes import GRADER, ImageRecipe

PIPELINE_VERSION = "2026.10.07.1"
URL_CHUNK_BYTES = 1024 * 1024
URL_TIMEOUT = 60
PINNED_IMAGE = re.compile(DOCKER_IMAGE_PATTERN)

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
    staged auxiliary inputs and the grader image environment.
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
class AgentImage:
    """A digest-pinned image the agent's machine starts from, run under gVisor or Docker."""

    reference: str

    def __post_init__(self) -> None:
        if PINNED_IMAGE.fullmatch(self.reference) is None or "@sha256:" not in self.reference:
            raise ValueError(f"Agent image must be pinned by digest: {self.reference}")

    def requirements(self) -> EnvironmentRequirements:
        return EnvironmentRequirements(docker_image=self.reference, compatible_backends=IMAGE_BACKENDS)


class GraderIsolation(StrEnum):
    """Where a source's grader scripts run.

    ``LOCAL`` runs them as locked-down subprocesses of the Zephyr worker, in a uv environment built
    from the recipe's locked requirements; the built image is not used. ``SANDBOX`` runs them in a
    fresh gVisor or Docker machine of the built image. Graders that execute model programs declare
    ``SANDBOX``; graders that only parse model text declare ``LOCAL``.
    """

    LOCAL = "local"
    SANDBOX = "sandbox"


@dataclass(frozen=True)
class GraderEnvironment:
    """The image recipe a source's grader scripts need and the isolation they run under."""

    image: ImageRecipe
    isolation: GraderIsolation

    def requirements(self, built_image: str) -> EnvironmentRequirements:
        """The grader environment conversion records, given the recipe's built digest."""
        if self.isolation == GraderIsolation.LOCAL:
            return local_grading_environment(built_image)
        return grading_environment(built_image)


LOCAL_GRADER = GraderEnvironment(GRADER, GraderIsolation.LOCAL)
"""Grader scripts that parse model text: they run in the worker with the grader recipe's packages."""
SANDBOX_GRADER = GraderEnvironment(GRADER, GraderIsolation.SANDBOX)
"""Grader scripts that execute model programs: they run in a fresh machine of the grader image."""


@dataclass(frozen=True)
class ShellSim:
    """No agent image: conversation tasks, or shell tasks served by the simulated shell."""

    def requirements(self) -> EnvironmentRequirements:
        return EnvironmentRequirements()


@dataclass(frozen=True)
class RlDataPipeline:
    """One RL data source and how its rows become tasks.

    ``name`` is the catalog key and artifact name. ``version`` is the converter revision; bump it
    when conversion changes in a way the hashed files do not capture. ``rubric=None`` skips model
    review and ``controls=None`` skips grader verification. ``inputs`` are auxiliary pinned files
    staged before conversion; the source callables and converter find them in ``context.inputs``.

    ``grader`` names the image recipe whose packages the source's grader scripts need and whether
    they run locally in the worker or in a sandbox of the built image; the converter reads the
    resulting environment from ``context.grader_environment``. ``ships`` are directories,
    such as ``datasets/<family>/scorers``, whose files the converter packages into tasks. The
    artifact identity hashes the converter module's directory, every file below ``ships`` and the
    grader image digest. A task whose decoded resources exceed ``resource_budget_bytes`` is
    deferred as ``resources_over_budget``.
    """

    name: str
    source: HfSource | UrlSource
    convert: Converter
    version: str
    environment: AgentImage | ShellSim
    intended_use: IntendedUse
    rubric: str | None = None
    controls: Controls | None = None
    inputs: Mapping[str, HfSource | UrlSource] = field(default_factory=dict)
    atlas_id: str | None = None
    grader: GraderEnvironment | None = None
    ships: tuple[Path, ...] = ()
    resource_budget_bytes: int = RESOURCE_BUDGET_BYTES

    def __post_init__(self) -> None:
        missing = [str(path) for path in self.ships if not path.is_dir()]
        if missing:
            raise ValueError(f"{self.name} ships directories that do not exist: {missing}")


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


def converter_identity(pipeline: RlDataPipeline, grader_image: str | None) -> dict[str, Any]:
    """The converter, every file it can package into a task, and the image its graders run in.

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
        "grader_image": grader_image,
        "grader_isolation": pipeline.grader.isolation.value if pipeline.grader is not None else None,
    }


def download_identity(source: HfSource | UrlSource) -> dict[str, Any]:
    """The pinned bytes a source download stages; reader callables do not change them."""
    if isinstance(source, HfSource):
        return {"repo": source.repo, "revision": source.revision, "files": sorted(source.files)}
    return {"url": source.url, "sha256": source.sha256, "filename": source.filename}


def pipeline_identity(
    pipeline: RlDataPipeline, config: SourcePipelineConfig, grader_image: str | None
) -> dict[str, Any]:
    """Everything that can change a source artifact's contents; ``grader_image`` is the built digest."""
    recipe = source_recipe(pipeline, {}, None)
    execution = config.execution
    return {
        "name": pipeline.name,
        "version": pipeline.version,
        "intended_use": pipeline.intended_use,
        "source": {**download_identity(pipeline.source), "files": source_files_identity(recipe.source)},
        "inputs": {name: download_identity(source) for name, source in sorted(pipeline.inputs.items())},
        "code": recipe_code_identity(recipe),
        "converter": converter_identity(pipeline, grader_image),
        "environment": (
            {"image": pipeline.environment.reference}
            if isinstance(pipeline.environment, AgentImage)
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
    previous_verification_report: str | None
    grader_image: str | None


def _source_run(
    identity: dict[str, Any],
    downloaded: ArtifactStep[Artifact],
    inputs: Mapping[str, ArtifactStep[Artifact]],
    previous: ArtifactStep[Artifact] | None,
    grader_image: str | None,
    ctx: StepContext,
) -> SourceRun:
    return SourceRun(
        identity=identity,
        source_input=ctx.artifact_path(downloaded),
        inputs={name: ctx.artifact_path(step) for name, step in inputs.items()},
        output_path=ctx.output_path,
        grader_image=grader_image,
        previous_verification_report=(
            str(StoragePath(ctx.artifact_path(previous)) / VERIFY_REPORT_PATH) if previous is not None else None
        ),
    )


def _run_source(
    pipeline: RlDataPipeline, config: SourcePipelineConfig, run: SourceRun, *, campaign: CampaignRuntime
) -> RlDataArtifact:
    grader_environment = None
    if run.grader_image is not None:
        assert pipeline.grader is not None
        grader_environment = pipeline.grader.requirements(run.grader_image)
    result = run_source_pipeline(
        source_recipe(pipeline, run.inputs, grader_environment),
        campaign.context,
        run.source_input,
        run.output_path,
        config,
        previous_verification_report=run.previous_verification_report,
        canonical_source=pipeline.name,
    )
    if result.status == SourceStatus.INCOMPLETE:
        raise SourcePipelineIncomplete(f"Source pipeline is incomplete; retained evidence: {result.manifest_path}")
    return RlDataArtifact(path=run.output_path, status=result.status, manifest=asdict(result))


def source_step(
    pipeline: RlDataPipeline,
    config: SourcePipelineConfig,
    campaign: CampaignRuntime,
    *,
    previous: ArtifactStep[Artifact] | None = None,
) -> ArtifactStep[RlDataArtifact]:
    """The ``data/rl/<name>-<hash>`` artifact for one declaration.

    ``previous`` is an earlier output of the same source whose control trials are reused. A
    declaration with a ``grader`` requires that image's artifact to be built already; see
    ``images.build``.
    """
    downloaded = download_step(pipeline.source, campaign)
    inputs = {name: download_step(source, campaign) for name, source in sorted(pipeline.inputs.items())}
    image_steps: tuple[ArtifactStep[ImageArtifact], ...] = ()
    grader_image = None
    if pipeline.grader is not None:
        grader_image = built_image(pipeline.grader.image).image
        image_steps = (image_artifact(pipeline.grader.image),)
    identity = pipeline_identity(pipeline, config, grader_image)
    identity["previous"] = (
        {"name": previous.name, "version": previous.version, "fingerprint": previous.fingerprint()}
        if previous is not None
        else None
    )
    digest = hashlib.sha256(canonical_json(identity).encode()).hexdigest()[:16]
    return ArtifactStep(
        name=f"data/rl/{pipeline.name}-{digest}",
        version=PIPELINE_VERSION,
        artifact_type=RlDataArtifact,
        run=partial(_run_source, pipeline, config, campaign=campaign),
        build_config=partial(_source_run, identity, downloaded, inputs, previous, grader_image),
        deps=(downloaded, *inputs.values(), *image_steps, *((previous,) if previous is not None else ())),
    )
