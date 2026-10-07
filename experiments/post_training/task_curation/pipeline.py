# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit Atlas source selections bound to whole-source curation artifacts."""

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from functools import partial
from typing import Any, Literal

from marin.execution.artifact import Artifact
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep, StepContext
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import SecretSpec
from taskcompendium.datasets.kto_components import TRAIN_FILE, KtoComponentRows
from taskcompendium.pipeline.fingerprints import recipe_code_identity
from taskcompendium.pipeline.inputs import HubDownload, UrlDownload
from taskcompendium.pipeline.models import CheckSuite, DatasetRecipe, HFSource, IntendedUse
from taskcompendium.pipeline.recorded_review import RecordedReviewer, load_recorded_reviews
from taskcompendium.pipeline.source_processing import (
    SOURCE_PIPELINE_REVISION,
    SourcePipelineConfig,
    SourceProcessingMode,
    answer_check_suite,
    run_source_pipeline,
)
from taskcompendium.pipeline.source_quality import SOURCE_QUALITY_REVISION
from taskcompendium.pipeline.source_verification import SOURCE_VERIFICATION_REVISION
from taskcompendium.pipeline.sources import source_files_identity

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.datasets.nemotron_ultra.inputs import bind_reference_paths
from experiments.post_training.task_curation.staging import DownloadInputs, download_inputs

PIPELINE_VERSION = "2026.10.06.2"


class AtlasStatus(StrEnum):
    AVAILABLE = "Available"
    EXCLUDED = "Excluded"


@dataclass(frozen=True)
class AtlasSource:
    id: str
    name: str
    origin: str
    family: str
    status: AtlasStatus
    exclusion_reason: str | None
    dataset_id: str
    dataset_revision: str | None
    archive_revision: str | None
    verifier_revision: str | None
    historical_disposition: str | None
    historical_contract_changed: bool | None

    @property
    def input_revision(self) -> str | None:
        """Keep archive pins distinct from TaskTrove upstream lineage."""
        return self.archive_revision if self.origin == "Task Trove" else self.dataset_revision


@dataclass(frozen=True)
class SourceRuntime:
    backend: Literal["local-gvisor", "iris-gvisor", "qemu"]
    image: str
    worker_image: str | None = None
    qemu_bundle: str | None = None


@dataclass(frozen=True)
class RecordedReviewInput:
    path: str
    sha256: str


@dataclass(frozen=True)
class SourceRuntimeConfig:
    images: Mapping[str, SourceRuntime]
    controller_url: str | None
    campaign: CampaignRuntime = field(default_factory=CampaignRuntime)
    source_inputs: Mapping[str, ArtifactStep[Artifact]] = field(default_factory=dict)
    verification_suites: Mapping[str, CheckSuite] = field(default_factory=dict)
    verifier_secret_env: Mapping[str, SecretSpec] = field(default_factory=dict)
    verification_inputs: Mapping[str, ArtifactStep[Artifact]] = field(default_factory=dict)
    recorded_reviews: Mapping[str, RecordedReviewInput] = field(default_factory=dict)


class RlDataArtifact(Artifact):
    status: str
    report: dict[str, Any]


class SourcePipelineIncomplete(RuntimeError):
    """The source retained its evidence but has incomplete review or verification."""


@dataclass(frozen=True)
class RlDataPipeline:
    source_key: str
    source: HFSource
    intended_use: IntendedUse
    atlas: AtlasSource | None
    recipe_builder: Callable[[SourceRuntimeConfig], DatasetRecipe]

    @property
    def hf_id(self) -> str:
        return self.source.dataset

    @property
    def revision(self) -> str:
        return self.source.revision

    @property
    def config(self) -> str:
        return self.source.config

    @property
    def split(self) -> str:
        return self.source.split

    @property
    def atlas_id(self) -> str | None:
        return self.atlas.id if self.atlas is not None else None

    @property
    def atlas_status(self) -> AtlasStatus | None:
        return self.atlas.status if self.atlas is not None else None

    @property
    def atlas_revision(self) -> str | None:
        return self.atlas.input_revision if self.atlas is not None else None

    @property
    def atlas_verifier_revision(self) -> str | None:
        return self.atlas.verifier_revision if self.atlas is not None else None

    def recipe(self, runtime: SourceRuntimeConfig) -> DatasetRecipe:
        return self.recipe_builder(runtime)

    def bind(self, config: SourcePipelineConfig, runtime: SourceRuntimeConfig) -> ArtifactStep[RlDataArtifact]:
        recipe = self.recipe(runtime)
        return _bind(self, recipe, config, runtime)


def _source_identity(
    recipe: DatasetRecipe, config: SourcePipelineConfig, suite: CheckSuite, runtime: SourceRuntime | None
) -> dict[str, Any]:
    return {
        "source": asdict(recipe.source),
        "recipe": recipe.name,
        "version": recipe.version,
        "intended_use": recipe.intended_use,
        "code": recipe_code_identity(recipe),
        "files": source_files_identity(recipe.inputs.files),
        "rubric": asdict(recipe.policy.rubric),
        "review": asdict(config.review),
        "mode": config.mode,
        "review_batch_size": config.execution.review_batch_size,
        "review_input_bytes": config.execution.review_input_bytes,
        "execution_image": config.execution.worker_resources.image if config.execution.worker_resources else None,
        "normalized_shards": config.normalized_shards,
        "quality_policy": asdict(config.quality_policy),
        "verification_policy": asdict(config.verification_policy),
        "filter_policy": asdict(config.filter_policy),
        "suite": {"id": suite.id, "revision": suite.revision, "parameters": dict(suite.parameters)},
        "runtime": asdict(runtime) if runtime else None,
        "procedure_revision": SOURCE_PIPELINE_REVISION,
        "quality_revision": SOURCE_QUALITY_REVISION,
        "verification_revision": SOURCE_VERIFICATION_REVISION,
    }


@dataclass(frozen=True)
class SourceBinding:
    identity: dict[str, Any]
    source_input: str
    output_path: str
    reference_paths: dict[str, str]
    verification_report_path: str | None = None
    recorded_review_path: str | None = None
    recorded_review_sha256: str | None = None


def _source_config(
    identity: dict[str, Any],
    downloaded: ArtifactStep[Artifact],
    references: Mapping[str, ArtifactStep[Artifact]],
    verification_input: ArtifactStep[Artifact] | None,
    recorded_review_input: ArtifactStep[Artifact] | None,
    recorded_review_sha256: str | None,
    ctx: StepContext,
) -> SourceBinding:
    return SourceBinding(
        identity=identity,
        source_input=ctx.artifact_path(downloaded),
        output_path=ctx.output_path,
        reference_paths={name: ctx.artifact_path(step) for name, step in references.items()},
        recorded_review_path=ctx.artifact_path(recorded_review_input) if recorded_review_input is not None else None,
        recorded_review_sha256=recorded_review_sha256,
        verification_report_path=(
            str(StoragePath(ctx.artifact_path(verification_input)) / "verification/report.json")
            if verification_input is not None
            else None
        ),
    )


def _run_source(
    recipe: DatasetRecipe,
    config: SourcePipelineConfig,
    suite: CheckSuite,
    binding: SourceBinding,
    *,
    canonical_source: str,
    campaign: CampaignRuntime,
) -> RlDataArtifact:
    if binding.recorded_review_path is not None:
        assert binding.recorded_review_sha256 is not None
        load_recorded_reviews(binding.recorded_review_path, binding.recorded_review_sha256)
        fallback = config.execution.reviewer
        if fallback is None:
            raise ValueError("Recorded review requires the original fallback reviewer transport")
        config = replace(
            config,
            execution=replace(
                config.execution,
                reviewer=RecordedReviewer(fallback, binding.recorded_review_path, binding.recorded_review_sha256),
            ),
        )
    files = recipe.inputs.files
    if binding.reference_paths:
        if isinstance(files.reader, KtoComponentRows):
            parent_path = str(StoragePath(binding.reference_paths["component-parent"]) / TRAIN_FILE)
            files = replace(files, reader=replace(files.reader, parent_path=parent_path))
        else:
            files = bind_reference_paths(files, binding.reference_paths)
    result = run_source_pipeline(
        recipe,
        campaign.context,
        binding.source_input,
        binding.output_path,
        files,
        config,
        suite,
        previous_verification_report=binding.verification_report_path,
        previous_sample_path=(
            str(StoragePath(binding.verification_report_path).parent.parent)
            if config.mode == SourceProcessingMode.NORMALIZE_ONLY and binding.verification_report_path is not None
            else None
        ),
        canonical_source=canonical_source,
    )
    if result.status == "incomplete":
        raise SourcePipelineIncomplete(f"Source pipeline is incomplete; retained evidence: {result.report_path}")
    report = asdict(result)
    return RlDataArtifact(path=binding.output_path, status=str(result.status), report=report)


def _download_config(declaration: HubDownload | UrlDownload, ctx: StepContext) -> DownloadInputs:
    return DownloadInputs((declaration,), ctx.output_path)


def _download_source(config: DownloadInputs, *, campaign: CampaignRuntime) -> None:
    download_inputs(config, context=campaign.context)


def _source_download(declaration: HubDownload | UrlDownload, campaign: CampaignRuntime) -> ArtifactStep[Artifact]:
    # The same source bytes can appear under different auxiliary subdirectories.
    acquisition = replace(declaration, subdirectory="") if isinstance(declaration, HubDownload) else declaration
    identity = hashlib.sha256(canonical_json(acquisition).encode()).hexdigest()[:16]
    return ArtifactStep(
        name=f"task-curation/download/{identity}",
        version=PIPELINE_VERSION,
        artifact_type=Artifact,
        run=partial(_download_source, campaign=campaign),
        build_config=partial(_download_config, acquisition),
    )


def _source_inputs(
    recipe: DatasetRecipe,
    adopted: ArtifactStep[Artifact] | None,
    campaign: CampaignRuntime,
) -> tuple[ArtifactStep[Artifact], dict[str, ArtifactStep[Artifact]]]:
    if adopted is not None:
        return adopted, {}
    primary = []
    references = {}
    for declaration in recipe.inputs.downloads:
        artifact = _source_download(declaration, campaign)
        if isinstance(declaration, HubDownload) and declaration.subdirectory:
            references[declaration.subdirectory] = artifact
        else:
            primary.append(artifact)
    if len(primary) != 1:
        raise ValueError(f"Source {recipe.name} requires one primary acquisition or an explicitly staged input artifact")
    return primary[0], references


def _bind(
    definition: RlDataPipeline, recipe: DatasetRecipe, config: SourcePipelineConfig, runtime: SourceRuntimeConfig
) -> ArtifactStep[RlDataArtifact]:
    name = definition.source_key
    assert name is not None
    environment = runtime.images.get(name)
    suite = runtime.verification_suites.get(name) or recipe.policy.check_suite or answer_check_suite()
    downloaded, references = _source_inputs(recipe, runtime.source_inputs.get(name), runtime.campaign)
    verification_input = runtime.verification_inputs.get(name)
    recorded_review = runtime.recorded_reviews.get(name)
    recorded_review_input = (
        ArtifactStep.adopt(
            f"task-curation/recorded-review/{name}-{recorded_review.sha256[:16]}",
            PIPELINE_VERSION,
            source=recorded_review.path,
            kind=Artifact,
            config=asdict(recorded_review),
        )
        if recorded_review is not None
        else None
    )
    identity = _source_identity(recipe, config, suite, environment)
    if recorded_review is not None:
        identity["recorded_review"] = asdict(recorded_review)
    identity["acquisition"] = recipe.inputs.downloads
    identity["input"] = {"name": downloaded.name, "version": downloaded.version, "fingerprint": downloaded.fingerprint()}
    identity["references"] = {
        name: {"name": step.name, "version": step.version, "fingerprint": step.fingerprint()}
        for name, step in references.items()
    }
    identity["verification_input"] = (
        {
            "name": verification_input.name,
            "version": verification_input.version,
            "fingerprint": verification_input.fingerprint(),
        }
        if verification_input is not None
        else None
    )
    return ArtifactStep(
        name=f"data/rl/{name}-{hashlib.sha256(canonical_json(identity).encode()).hexdigest()[:16]}",
        version=PIPELINE_VERSION,
        artifact_type=RlDataArtifact,
        run=partial(_run_source, recipe, config, suite, canonical_source=name, campaign=runtime.campaign),
        build_config=partial(
            _source_config,
            identity,
            downloaded,
            references,
            verification_input,
            recorded_review_input,
            recorded_review.sha256 if recorded_review is not None else None,
        ),
        deps=(
            downloaded,
            *references.values(),
            *((verification_input,) if verification_input is not None else ()),
            *((recorded_review_input,) if recorded_review_input is not None else ()),
        ),
    )
