# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build pinned source acquisition, Zephyr audit, filtering, and merged task artifacts.

The default prints the artifact plan. Add --run to build it using the configured
Marin storage prefix and a GLM batch endpoint.
"""

import hashlib
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import click
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import OUT, ArtifactStep, StepContext, apply, lower, run
from marin.execution.remote import remote
from marin.inference.openai_batch import OpenAIBatchClient
from pydantic import TypeAdapter
from taskcompendium.pipeline.fingerprints import recipe_code_identity
from taskcompendium.pipeline.models import DatasetRecipe, EnvironmentInventory, FilterPolicy, ReviewRubric
from taskcompendium.pipeline.review import DEFAULT_PROMPT_CHARACTERS, BatchReviewer
from taskcompendium.pipeline.rewriting import BatchRewriter, rewrite_audit_source
from taskcompendium.pipeline.sources import SourceFiles, source_files_identity
from taskcompendium.pipeline.zephyr import (
    AuditExecution,
    ReviewConfig,
)
from taskcompendium.pipeline.zephyr import (
    audit_source as audit_source_rows,
)
from taskcompendium.pipeline.zephyr import canonicalize_sources as canonicalize_source_rows
from taskcompendium.pipeline.zephyr import (
    concat_sources as concatenate_source_rows,
)
from taskcompendium.pipeline.zephyr import (
    filter_source as filter_source_rows,
)

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_MODEL
from experiments.post_training.task_curation_downloads import source_download
from experiments.post_training.task_curation_source_bindings import SOURCE_NAMES, source_recipe

PIPELINE_VERSION = "2026.10.02.1"
AUDIT_REVISION = "staged-source-v1"
PIPELINE_PREFIX = "task-curation"
WORKER_PACKAGE = "./lib/taskcompendium[pipeline]"


@dataclass(frozen=True)
class RewriteSelection:
    task_ids: tuple[str, ...]
    rubric: ReviewRubric
    max_tokens: int = 8192
    prompt_budget: int = 64000


@dataclass(frozen=True)
class SourceBinding:
    name: str
    version: str
    recipe: DatasetRecipe
    downloaded: ArtifactStep[Artifact]
    files: SourceFiles
    review: ReviewConfig
    limit: int | None
    rewrite: RewriteSelection | None = None


@dataclass(frozen=True)
class SourceArtifacts:
    name: str
    downloaded: ArtifactStep[Artifact]
    audited: ArtifactStep[Artifact]
    accepted: ArtifactStep[Artifact]


@dataclass(frozen=True)
class CurationWorkflow:
    sources: tuple[SourceArtifacts, ...]
    canonical: ArtifactStep[Artifact]


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
        "audit_revision": AUDIT_REVISION,
        "implementation": recipe_code_identity(recipe),
    }


def audit_source(
    binding: SourceBinding,
    downloaded: ArtifactStep[Artifact],
    execution: AuditExecution,
    resources: ResourceConfig,
) -> ArtifactStep[Artifact]:
    """Normalize, check, and review the acquired shards through the Zephyr stage."""
    recipe_config = recipe_identity(binding.recipe)
    identity = {
        "recipe": recipe_config,
        "review": binding.review,
        "downloaded": (downloaded.name, downloaded.version),
        "files": source_files_identity(binding.files),
        "limit": binding.limit,
    }

    def build_config(ctx: StepContext) -> dict[str, Any]:
        return {
            "source_path": ctx.artifact_path(downloaded),
            "output_path": ctx.output_path,
            "recipe": recipe_config if ctx.is_fingerprint else binding.recipe,
            "files": source_files_identity(binding.files) if ctx.is_fingerprint else binding.files,
            "limit": binding.limit,
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
            pip_packages=[WORKER_PACKAGE],
        )(
            source_path=config["source_path"],
            output_path=config["output_path"],
            recipe=config["recipe"],
            review=config["review"],
            files=config["files"],
            limit=config["limit"],
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
        deps=(downloaded,),
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
        remote(filter_source_rows, resources=resources, pip_packages=[WORKER_PACKAGE]),
        version=binding.version,
        audit_path=audited,
        output_path=OUT,
        policy=policy,
    )


def rewrite_source(
    binding: SourceBinding,
    filtered: ArtifactStep[Artifact],
    policy: FilterPolicy,
    execution: AuditExecution,
    rewriter: BatchRewriter | None,
    resources: ResourceConfig,
) -> ArtifactStep[Artifact]:
    """Recheck selected instruction repairs and preserve the original audit evidence."""
    assert binding.rewrite is not None
    identity = {
        "input": (filtered.name, filtered.version),
        "selection": binding.rewrite,
        "recipe": recipe_identity(binding.recipe),
        "policy": policy,
        "model": binding.review.model,
        "model_revision": binding.review.model_revision,
        "rewrite_revision": "instruction-edits-v1",
    }

    def config(ctx: StepContext) -> dict[str, Any]:
        return {
            "source_path": ctx.artifact_path(filtered),
            "output_path": ctx.output_path,
            "identity": identity,
            "recipe": recipe_identity(binding.recipe) if ctx.is_fingerprint else binding.recipe,
            "resources": ctx.runtime_arg("resources"),
        }

    def execute(values: dict[str, Any]) -> None:
        if execution.reviewer is None or rewriter is None:
            raise ValueError("Rewrite execution requires both a rewriter and a reviewer")
        assert binding.rewrite is not None
        remote(rewrite_audit_source, resources=values["resources"], pip_packages=[WORKER_PACKAGE])(
            source_path=values["source_path"],
            output_path=values["output_path"],
            recipe=values["recipe"],
            policy=policy,
            rewrite_rubric=binding.rewrite.rubric,
            rewriter=replace(
                rewriter,
                model=binding.review.model,
                model_revision=binding.review.model_revision,
                max_tokens=binding.rewrite.max_tokens,
                max_prompt_characters=binding.rewrite.prompt_budget,
            ),
            reviewer=execution.reviewer,
            selected_task_ids=binding.rewrite.task_ids,
        )

    return ArtifactStep(
        name=content_name(f"{PIPELINE_PREFIX}/{binding.name}/rewritten", identity),
        version=binding.version,
        artifact_type=Artifact,
        run=execute,
        build_config=config,
        deps=(filtered,),
        runtime_args={"resources": resources},
    )


def concat_sources(sources: Sequence[SourceArtifacts], resources: ResourceConfig) -> ArtifactStep[Artifact]:
    """Merge the selected source views with explicit source dependencies."""
    inputs = tuple(source.accepted for source in sources)
    return apply(
        content_name(
            f"{PIPELINE_PREFIX}/merged-audit",
            {"sources": [(step.name, step.version) for step in inputs], "view": "audit"},
        ),
        remote(concatenate_source_rows, resources=resources, pip_packages=[WORKER_PACKAGE]),
        version=PIPELINE_VERSION,
        input_paths=inputs,
        output_path=OUT,
        view="audit",
    )


def build_workflow(
    bindings: Sequence[SourceBinding],
    *,
    execution: AuditExecution,
    resources: ResourceConfig,
    policy: FilterPolicy = FilterPolicy(),
    rewriter: BatchRewriter | None = None,
) -> CurationWorkflow:
    """Build source branches and one canonical artifact containing all output views."""
    if not bindings or len({binding.name for binding in bindings}) != len(bindings):
        raise ValueError("Choose at least one source, with unique artifact names")
    sources = []
    for binding in bindings:
        audited = audit_source(binding, binding.downloaded, execution, resources)
        accepted = filter_source(binding, audited, policy, resources)
        if binding.rewrite is not None:
            accepted = rewrite_source(binding, accepted, policy, execution, rewriter, resources)
        sources.append(SourceArtifacts(binding.name, binding.downloaded, audited, accepted))
    merged = concat_sources(sources, resources)
    canonical = apply(
        content_name(f"{PIPELINE_PREFIX}/canonical", {"merged": (merged.name, merged.version), "policy": "exact-v1"}),
        remote(canonicalize_source_rows, resources=resources, pip_packages=[WORKER_PACKAGE]),
        version=PIPELINE_VERSION,
        merged_path=merged,
        output_path=OUT,
    )
    return CurationWorkflow(tuple(sources), canonical)


@click.command(help=__doc__)
@click.option("--source", "source_names", multiple=True, required=True, type=click.Choice(SOURCE_NAMES))
@click.option("--image", help="Immutable Docker image for selected coding-source grader controls.")
@click.option("--version", default=PIPELINE_VERSION, show_default=True)
@click.option(
    "--limit",
    type=int,
    default=100,
    show_default=True,
    help="At most this many input records per source, across all staged files.",
)
@click.option("--model", default=GLM_MODEL, show_default=True)
@click.option("--all-rows", is_flag=True, help="Process every selected source record instead of applying --limit.")
@click.option("--model-revision", required=True)
@click.option("--max-tokens", type=int, default=4096, show_default=True)
@click.option("--prompt-budget", type=int, default=DEFAULT_PROMPT_CHARACTERS, show_default=True)
@click.option("--base-url", help="GLM batch endpoint, required with --run.")
@click.option("--review-cache", help="Stable FineStore query cache location, shared across catalog versions.")
@click.option(
    "--environment-inventory",
    "environment_inventories",
    type=(str, click.Path(exists=True, path_type=Path)),
    multiple=True,
    metavar="SOURCE JSON",
    help="Attach a scoped Shellbox or source-manifest file inventory to this source's review rubric.",
)
@click.option(
    "--rewrite-plan",
    "rewrite_plans",
    type=(str, click.Path(exists=True, path_type=Path)),
    multiple=True,
    metavar="SOURCE JSON",
    help="Select task_ids and a separate instruction-repair rubric for a source.",
)
@click.option("--max-workers", type=int, default=4, show_default=True)
@click.option("--review-batch-size", type=int, default=100, show_default=True)
@click.option("--cpu", type=int, default=4, show_default=True)
@click.option("--ram", default="16g", show_default=True)
@click.option("--max-concurrent", type=int, default=8, show_default=True)
@click.option("--run", "do_run", is_flag=True, help="Build the canonical views; the default prints the artifact plan.")
def main(
    source_names: tuple[str, ...],
    image: str | None,
    version: str,
    limit: int,
    all_rows: bool,
    model: str,
    model_revision: str,
    max_tokens: int,
    prompt_budget: int,
    base_url: str | None,
    review_cache: str | None,
    environment_inventories: tuple[tuple[str, Path], ...],
    rewrite_plans: tuple[tuple[str, Path], ...],
    max_workers: int,
    review_batch_size: int,
    cpu: int,
    ram: str,
    max_concurrent: int,
    do_run: bool,
) -> None:
    if limit < 1:
        raise click.UsageError("--limit must be positive")
    row_limit = None if all_rows else limit
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
            query_cache_root=review_cache,
        )
    resources = ResourceConfig.with_cpu(cpu=cpu, ram=ram)
    bindings = []
    for name in source_names:
        recipe = source_recipe(name, image)
        downloaded, files = source_download(name, resources, recipe)
        bindings.append(SourceBinding(name, version, recipe, downloaded, files, review, row_limit))
    inventories = {
        name: TypeAdapter(EnvironmentInventory).validate_json(path.read_bytes())
        for name, path in environment_inventories
    }
    if set(inventories) - {binding.name for binding in bindings}:
        raise click.UsageError("--environment-inventory must name a selected source")
    bindings = [
        (
            replace(
                binding,
                recipe=replace(
                    binding.recipe,
                    rubric=replace(binding.recipe.rubric, environment_inventory=inventories[binding.name]),
                ),
            )
            if binding.name in inventories
            else binding
        )
        for binding in bindings
    ]
    plans = {name: TypeAdapter(RewriteSelection).validate_json(path.read_bytes()) for name, path in rewrite_plans}
    if set(plans) - set(source_names):
        raise click.UsageError("--rewrite-plan must name a selected source")
    bindings = [replace(binding, rewrite=plans.get(binding.name)) for binding in bindings]
    rewriter = None
    if plans and reviewer is not None:
        rewriter = BatchRewriter(reviewer.client, model, model_revision)
    workflow = build_workflow(
        bindings,
        execution=AuditExecution(max_workers=max_workers, review_batch_size=review_batch_size, reviewer=reviewer),
        resources=resources,
        rewriter=rewriter,
    )
    if do_run:
        run(workflow.canonical, max_concurrent=max_concurrent)
        return
    click.echo(lower(workflow.canonical))


if __name__ == "__main__":
    main()
