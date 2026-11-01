# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind evaluation conversion and sidecar publication to the existing acquisition graph."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial

from marin.execution.lazy import ArtifactStep, StepContext
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation import EvaluationProtocol
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.evaluation import MANIFEST_FILENAME, ItemBuilder, publish_evaluation_corpus
from taskcompendium.pipeline.fingerprints import callable_identity
from taskcompendium.pipeline.models import SourceStatus
from taskcompendium.pipeline.sources import source_files_identity

from experiments.post_training.task_curation.campaign import CampaignArtifact, PipelineResult
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.environment import Placement, placement
from experiments.post_training.task_curation.images.build import (
    EnvironmentArtifact,
    built_environment,
    environment_artifact,
)
from experiments.post_training.task_curation.pipeline import (
    CurationRecipe,
    HfSource,
    RecipeSource,
    download_identity,
    environment_record,
    environment_requirements,
    input_file_overrides,
    input_overrides_identity,
    source_downloads,
    source_recipe,
    validate_input_identity,
)
from experiments.post_training.task_curation.source import RlDataSource

EVALUATION_PIPELINE_VERSION = "2026.10.28"


@dataclass(frozen=True)
class EvalDataSet:
    """One source conversion and the additional protocol that turns its tasks into an evaluation."""

    recipe: CurationRecipe
    protocol: EvaluationProtocol
    item_builder: ItemBuilder

    def __post_init__(self) -> None:
        for source in (self.recipe.source, *self.recipe.inputs.values()):
            if isinstance(source, HfSource) and re.fullmatch(r"[0-9a-f]{40}", source.revision) is None:
                raise ValueError(f"Evaluation source requires an immutable Hugging Face commit: {source.repo}")

    @property
    def name(self) -> str:
        return self.recipe.name

    @property
    def version(self) -> str:
        return self.recipe.version

    @property
    def dataset(self) -> RecipeSource:
        return self.recipe.source

    @property
    def files(self) -> tuple[str, ...]:
        return self.recipe.files


@dataclass(frozen=True)
class EvaluationRun:
    source_input: str
    inputs: dict[str, str]
    output_path: str
    grader_artifact: str | None


def _build_run(
    options: PipelineOptions,
    downloaded: ArtifactStep | None,
    auxiliary: Mapping[str, ArtifactStep],
    grader: ArtifactStep | None,
    context: StepContext,
) -> EvaluationRun:
    return EvaluationRun(
        context.artifact_path(downloaded) if downloaded is not None else options.inputs.root or context.output_path,
        {**options.inputs.auxiliary, **{name: context.artifact_path(step) for name, step in auxiliary.items()}},
        context.output_path,
        context.artifact_path(grader) if grader is not None else None,
    )


def _publish(dataset: EvalDataSet, options: PipelineOptions, identity: dict, run: EvaluationRun) -> CampaignArtifact:
    validate_input_identity(dataset.recipe, options.inputs, identity)
    grader_environment = None
    if dataset.recipe.grader is not None:
        built = EnvironmentArtifact.raw_load(run.grader_artifact) if run.grader_artifact is not None else None
        grader_environment = environment_requirements(dataset.recipe.grader, built)
    publish_evaluation_corpus(
        source_recipe(dataset.recipe, run.inputs, grader_environment),
        dataset.protocol,
        dataset.item_builder,
        options.runtime.context,
        run.source_input,
        run.output_path,
        source_overrides=input_file_overrides(options.inputs.files),
    )
    validated = False
    try:
        validate_input_identity(dataset.recipe, options.inputs, identity)
        validated = True
    finally:
        if not validated:
            (StoragePath(run.output_path) / MANIFEST_FILENAME).rm()
    return CampaignArtifact(
        path=run.output_path,
        status=SourceStatus.COMPLETED,
        result=PipelineResult(
            SourceStatus.COMPLETED,
            {
                "normalize": str(StoragePath(run.output_path) / "normalize"),
                "evaluation": str(StoragePath(run.output_path) / "evaluation"),
            },
            {"manifest": str(StoragePath(run.output_path) / MANIFEST_FILENAME)},
            ("normalize", "evaluation"),
        ),
    )


def curate(source: RlDataSource[EvalDataSet], options: PipelineOptions) -> ArtifactStep[CampaignArtifact]:
    """Publish the complete evaluation population without RL admission or model review."""
    dataset = source.config
    assert dataset is not None
    downloaded, auxiliary = source_downloads(dataset.recipe, options)
    grader = None
    built = None
    if dataset.recipe.grader is not None and placement(dataset.recipe.grader) != Placement.IMAGE:
        built = built_environment(dataset.recipe.grader)
        grader = environment_artifact(dataset.recipe.grader)
    recipe = source_recipe(dataset.recipe, {}, None)
    identity = {
        "source": {
            **download_identity(dataset.recipe.source),
            "reader": source_files_identity(recipe.source),
        },
        "inputs": {name: download_identity(value) for name, value in sorted(dataset.recipe.inputs.items())},
        "grader": environment_record(dataset.recipe.grader, built) if dataset.recipe.grader is not None else None,
        "recipe": dataset.recipe.version,
        "intended_use": dataset.recipe.intended_use,
        "resource_budget_bytes": dataset.recipe.resource_budget_bytes,
        "convert": callable_identity(dataset.recipe.convert),
        "item_builder": callable_identity(dataset.item_builder),
        "protocol": dataset.protocol.model_dump(mode="json"),
        "input_overrides": input_overrides_identity(dataset.recipe, options.inputs),
    }
    return ArtifactStep(
        name=f"data/eval/{dataset.name}-{canonical_sha256(identity)[:16]}",
        version=EVALUATION_PIPELINE_VERSION,
        artifact_type=CampaignArtifact,
        run=partial(_publish, dataset, options, identity),
        build_config=partial(_build_run, options, downloaded, auxiliary, grader),
        deps=(
            *((downloaded,) if downloaded is not None else ()),
            *auxiliary.values(),
            *((grader,) if grader is not None else ()),
        ),
    )
