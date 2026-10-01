# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build pinned source acquisition, Zephyr audit, filtering, and merged task artifacts.

The default prints the artifact plan. Add --run to build it using the configured
Marin storage prefix and a GLM batch endpoint.
"""

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

import click
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import OUT, ArtifactStep, StepContext, apply, lower, run
from marin.execution.remote import remote
from marin.inference.openai_batch import OpenAIBatchClient
from taskcompendium.pipeline.fingerprints import code_digest
from taskcompendium.pipeline.models import DatasetRecipe, FilterPolicy, SnapshotSource
from taskcompendium.pipeline.review import BatchReviewer
from taskcompendium.pipeline.zephyr import (
    AuditExecution,
    ReviewConfig,
    SourceAcquisition,
)
from taskcompendium.pipeline.zephyr import acquire_source as acquire_source_rows
from taskcompendium.pipeline.zephyr import (
    audit_source as audit_source_rows,
)
from taskcompendium.pipeline.zephyr import (
    concat_sources as concatenate_source_rows,
)
from taskcompendium.pipeline.zephyr import (
    filter_source as filter_source_rows,
)

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_MODEL
from experiments.post_training.task_curation_source_bindings import SOURCE_NAMES, source_recipe

PIPELINE_VERSION = "2026.10.01.1"
PIPELINE_PREFIX = "task-curation"


class ParquetView(StrEnum):
    AUDIT = "audit"
    ACCEPTED = "accepted"


@dataclass(frozen=True)
class SourceBinding:
    name: str
    version: str
    recipe: DatasetRecipe
    acquisition: SourceAcquisition
    review: ReviewConfig

    def __post_init__(self) -> None:
        if self.acquisition.source != self.recipe.source:
            raise ValueError("Acquisition and recipe must describe the same source")
        if isinstance(self.acquisition.source, SnapshotSource) and self.acquisition.sample_sha256 is None:
            raise ValueError("Snapshot acquisitions require a pinned sample SHA256")


@dataclass(frozen=True)
class SourceArtifacts:
    name: str
    acquired: ArtifactStep[Artifact]
    audited: ArtifactStep[Artifact]
    accepted: ArtifactStep[Artifact]


@dataclass(frozen=True)
class CurationWorkflow:
    sources: tuple[SourceArtifacts, ...]
    audit: ArtifactStep[Artifact]
    accepted: ArtifactStep[Artifact]


class RecipeModule(Protocol):
    recipe: DatasetRecipe


class SnapshotRecipeModule(Protocol):
    def recipe(self, snapshot: Path) -> DatasetRecipe: ...


def content_name(name: str, config: object) -> str:
    """Address config changes separately so fixed artifacts cannot mask changed inputs."""
    return f"{name}-{hashlib.sha256(canonical_json(config).encode()).hexdigest()[:16]}"


def recipe_identity(recipe: DatasetRecipe) -> dict[str, Any]:
    """Record explicit recipe revisions and parameters without serializing callables."""
    checks = recipe.check_suite
    return {
        "name": recipe.name,
        "version": recipe.version,
        "source": recipe.source,
        "rubric": recipe.rubric,
        "intended_use": recipe.intended_use,
        "check_suite": (
            {"id": checks.id, "revision": checks.revision, "parameters": checks.parameters}
            if checks is not None
            else None
        ),
        "code_digest": code_digest(recipe),
    }


def acquire_source(binding: SourceBinding, resources: ResourceConfig) -> ArtifactStep[Artifact]:
    """Acquire bounded source shards under the configured artifact storage prefix."""
    # Local snapshots must be uploaded by the driver before remote workers can consume them.
    worker = (
        acquire_source_rows
        if isinstance(binding.acquisition.source, SnapshotSource)
        else remote(acquire_source_rows, resources=resources, pip_packages=["./lib/taskcompendium[pipeline]"])
    )
    return apply(
        content_name(f"{PIPELINE_PREFIX}/{binding.name}/acquired", binding.acquisition),
        worker,
        version=binding.version,
        acquisition=binding.acquisition,
        output_path=OUT,
    )


def audit_source(
    binding: SourceBinding,
    acquired: ArtifactStep[Artifact],
    execution: AuditExecution,
    resources: ResourceConfig,
) -> ArtifactStep[Artifact]:
    """Normalize, check, and review the acquired shards through the Zephyr stage."""
    recipe_config = recipe_identity(binding.recipe)
    identity = {"recipe": recipe_config, "review": binding.review, "acquired": (acquired.name, acquired.version)}

    def build_config(ctx: StepContext) -> dict[str, Any]:
        return {
            "source_path": ctx.artifact_path(acquired),
            "output_path": ctx.output_path,
            "recipe": recipe_config if ctx.is_fingerprint else binding.recipe,
            "review": binding.review,
            "max_workers": ctx.runtime_arg("max_workers"),
            "review_batch_size": ctx.runtime_arg("review_batch_size"),
            "resources": ctx.runtime_arg("resources"),
        }

    def execute(config: dict[str, Any]) -> None:
        # Artifact sidecars persist build_config values; clients and credentials stay outside it.
        remote(
            audit_source_rows,
            resources=config["resources"],
            pip_packages=["./lib/taskcompendium[pipeline]"],
        )(
            source_path=config["source_path"],
            output_path=config["output_path"],
            recipe=config["recipe"],
            review=config["review"],
            execution=replace(
                execution, max_workers=config["max_workers"], review_batch_size=config["review_batch_size"]
            ),
        )

    return ArtifactStep(
        name=content_name(f"{PIPELINE_PREFIX}/{binding.name}/audited", identity),
        version=binding.version,
        artifact_type=Artifact,
        run=execute,
        build_config=build_config,
        deps=(acquired,),
        runtime_args={
            "max_workers": execution.max_workers,
            "review_batch_size": execution.review_batch_size,
            "resources": resources,
        },
    )


def filter_source(
    binding: SourceBinding,
    audited: ArtifactStep[Artifact],
    policy: FilterPolicy,
    resources: ResourceConfig,
) -> ArtifactStep[Artifact]:
    """Keep final decisions in an audit view beside the accepted task view."""
    return apply(
        content_name(
            f"{PIPELINE_PREFIX}/{binding.name}/filtered", {"audited": (audited.name, audited.version), "policy": policy}
        ),
        remote(filter_source_rows, resources=resources, pip_packages=["./lib/taskcompendium[pipeline]"]),
        version=binding.version,
        audit_path=audited,
        output_path=OUT,
        policy=policy,
    )


def concat_sources(
    sources: Sequence[SourceArtifacts], view: ParquetView, resources: ResourceConfig
) -> ArtifactStep[Artifact]:
    """Merge the selected source views with explicit source dependencies."""
    inputs = tuple(source.accepted for source in sources)
    return apply(
        content_name(
            f"{PIPELINE_PREFIX}/merged/{view.value}",
            {"sources": [(step.name, step.version) for step in inputs], "view": view},
        ),
        remote(concatenate_source_rows, resources=resources, pip_packages=["./lib/taskcompendium[pipeline]"]),
        version=PIPELINE_VERSION,
        input_paths=inputs,
        output_path=OUT,
        view=view.value,
    )


def build_workflow(
    bindings: Sequence[SourceBinding],
    *,
    execution: AuditExecution,
    resources: ResourceConfig,
    policy: FilterPolicy = FilterPolicy(),
) -> CurationWorkflow:
    """Build independent source branches followed by separate merged audit and accepted artifacts."""
    if not bindings or len({binding.name for binding in bindings}) != len(bindings):
        raise ValueError("Choose at least one source, with unique artifact names")
    sources = []
    for binding in bindings:
        acquired = acquire_source(binding, resources)
        audited = audit_source(binding, acquired, execution, resources)
        accepted = filter_source(binding, audited, policy, resources)
        sources.append(SourceArtifacts(binding.name, acquired, audited, accepted))
    return CurationWorkflow(
        tuple(sources),
        concat_sources(sources, ParquetView.AUDIT, resources),
        concat_sources(sources, ParquetView.ACCEPTED, resources),
    )


@click.command(help=__doc__)
@click.option("--sources-dir", type=click.Path(path_type=Path), help="Directory containing pinned per-source samples.")
@click.option("--source", "source_names", multiple=True, type=click.Choice(SOURCE_NAMES))
@click.option("--image", help="Immutable Docker image for selected coding-source grader controls.")
@click.option("--recipe", "recipe_modules", multiple=True, help="Module exporting a pinned DatasetRecipe.")
@click.option(
    "--snapshot-recipe",
    "snapshot_recipes",
    type=(str, click.Path(path_type=Path), str),
    multiple=True,
    metavar="MODULE PATH SHA256",
    help="Module exporting recipe(snapshot), with pinned local snapshot bytes.",
)
@click.option(
    "--sample-sha256",
    "sample_digests",
    type=(str, str),
    multiple=True,
    metavar="SOURCE SHA256",
    help="Pin a snapshot source exported by --recipe using its recipe name.",
)
@click.option("--version", default=PIPELINE_VERSION, show_default=True)
@click.option(
    "--limit",
    type=int,
    default=100,
    show_default=True,
    help="Row limit for module recipes; source directories use their manifest counts.",
)
@click.option("--model", default=GLM_MODEL, show_default=True)
@click.option("--model-revision", required=True)
@click.option("--max-tokens", type=int, default=4096, show_default=True)
@click.option("--prompt-budget", type=int, default=128000, show_default=True)
@click.option("--base-url", help="GLM batch endpoint, required with --run.")
@click.option("--max-workers", type=int, default=4, show_default=True)
@click.option("--review-batch-size", type=int, default=100, show_default=True)
@click.option("--cpu", type=int, default=4, show_default=True)
@click.option("--ram", default="16g", show_default=True)
@click.option("--max-concurrent", type=int, default=8, show_default=True)
@click.option("--run", "do_run", is_flag=True, help="Build both merged views; the default prints their plans.")
def main(
    sources_dir: Path | None,
    source_names: tuple[str, ...],
    image: str | None,
    recipe_modules: tuple[str, ...],
    snapshot_recipes: tuple[tuple[str, Path, str], ...],
    sample_digests: tuple[tuple[str, str], ...],
    version: str,
    limit: int,
    model: str,
    model_revision: str,
    max_tokens: int,
    prompt_budget: int,
    base_url: str | None,
    max_workers: int,
    review_batch_size: int,
    cpu: int,
    ram: str,
    max_concurrent: int,
    do_run: bool,
) -> None:
    if not recipe_modules and not snapshot_recipes and sources_dir is None:
        raise click.UsageError("Choose --sources-dir, --recipe, or --snapshot-recipe")
    if source_names and sources_dir is None:
        raise click.UsageError("--source requires --sources-dir")
    review = ReviewConfig(model=model, model_revision=model_revision, prompt_budget=prompt_budget, max_tokens=max_tokens)
    reviewer = None
    if do_run:
        if base_url is None:
            raise click.UsageError("--base-url is required with --run")
        reviewer = BatchReviewer(
            OpenAIBatchClient(base_url, os.environ[GLM_BULK_TOKEN_ENV]),
            model,
            model_revision,
            max_tokens=max_tokens,
            max_prompt_characters=prompt_budget,
        )
    bindings = []
    digests = dict(sample_digests)
    for module_name in recipe_modules:
        recipe = cast(RecipeModule, import_module(module_name)).recipe
        bindings.append(
            SourceBinding(
                recipe.name,
                version,
                recipe,
                SourceAcquisition(recipe.source, limit, digests.get(recipe.name)),
                review,
            )
        )
    if set(digests) - {binding.name for binding in bindings}:
        raise click.UsageError("--sample-sha256 must name a --recipe source")
    for module_name, snapshot, digest in snapshot_recipes:
        recipe = cast(SnapshotRecipeModule, import_module(module_name)).recipe(snapshot)
        bindings.append(
            SourceBinding(recipe.name, version, recipe, SourceAcquisition(recipe.source, limit, digest), review)
        )
    if sources_dir is not None:
        for name in source_names or SOURCE_NAMES:
            directory = sources_dir / name
            manifest = json.loads((directory / "sample-manifest.json").read_text())
            recipe = source_recipe(name, directory / "sample.jsonl", image)
            bindings.append(
                SourceBinding(
                    name,
                    version,
                    recipe,
                    SourceAcquisition(recipe.source, manifest["sample_rows"], manifest["snapshot_sha256"]),
                    review,
                )
            )
    workflow = build_workflow(
        bindings,
        execution=AuditExecution(max_workers=max_workers, review_batch_size=review_batch_size, reviewer=reviewer),
        resources=ResourceConfig.with_cpu(cpu=cpu, ram=ram),
    )
    if do_run:
        run(workflow.audit, workflow.accepted, max_concurrent=max_concurrent)
        return
    click.echo(lower(workflow.audit))
    click.echo(lower(workflow.accepted))


if __name__ == "__main__":
    main()
