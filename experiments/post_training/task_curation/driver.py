# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan RL curation, then run source pipelines locally or on Iris."""

import json
import os
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import cast

import click
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from iris.cluster.client.job_info import get_job_info
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import StoragePath
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.machine import Backend, MachineFactory
from taskcompendium.pipeline.assessment import AssessmentSampling
from taskcompendium.pipeline.assessment_review import ASSESSMENT_RUBRIC, AssessmentModel
from taskcompendium.pipeline.chat_requests import MAX_DIRECT_CONCURRENT_REQUESTS
from taskcompendium.pipeline.query_cache import compact_query_cache
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from taskcompendium.pipeline.source_quality import MAX_SOURCE_QUALITY_SAMPLE_SIZE, SOURCE_QUALITY_SAMPLE_SIZE
from taskcompendium.pipeline.stages import AuditExecution, ReviewConfig, ReviewMode

from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB, GLM_BULK_TOKEN_ENV, GLM_MODEL
from experiments.post_training.task_curation.assessment import (
    AssessmentSettings,
    assessment_paths,
    assessment_pipeline,
)
from experiments.post_training.task_curation.campaign import (
    CampaignFailed,
    CampaignPool,
    CampaignRuntime,
    campaign_plan,
    run_campaign,
)
from experiments.post_training.task_curation.catalog_snapshot import publish_catalog
from experiments.post_training.task_curation.config import (
    InputOverrides,
    PipelineOptions,
    RecipeSettings,
)
from experiments.post_training.task_curation.curation_catalog import CATALOG_DIRECTORY
from experiments.post_training.task_curation.local import local_artifact_prefix, run_local_steps
from experiments.post_training.task_curation.pipeline import CampaignMachines
from experiments.post_training.task_curation.source import CurationPipeline, InputPreparation, RlDataSource
from experiments.post_training.task_curation.sources import all_sources, runnable_sources

LOCAL_MAX_WORKERS = 4
LOCAL_MEMORY = "4g"
LOCAL_NORMALIZED_SHARDS = 4
WORKER_CONTAINER_PROFILE = "CONTAINER_PROFILE_PRIVILEGED"
IRIS_JOB_TTL = 1800


def _selected_sources(sources: tuple[str, ...]) -> dict[str, RlDataSource]:
    """Named runnable sources, or sources with automatic input preparation by default."""
    catalog = runnable_sources()
    unknown = set(sources) - catalog.keys()
    if unknown:
        raise click.UsageError(f"Unknown source: {', '.join(sorted(unknown))}")
    return {
        name: source
        for name, source in catalog.items()
        if (name in sources if sources else source.preparation == InputPreparation.AUTOMATIC)
    }


def _image_factory(backend: Backend | None, controller_url: str | None) -> MachineFactory | None:
    if backend is None:
        return None
    if backend == Backend.GVISOR:
        if controller_url is None:
            job = get_job_info()
            controller_url = job.controller_address if job is not None else None
        if controller_url is None:
            raise click.UsageError("--image-backend gvisor requires --controller-url outside an Iris job")
        return IrisMachineFactory(controller_url=controller_url, job_ttl=IRIS_JOB_TTL, secret_env=None)
    if backend != Backend.DAYTONA:
        raise ValueError(f"Unsupported image backend: {backend}")
    # Daytona is an optional Shellbox dependency; local and Iris campaigns do not import its SDK.
    try:
        from shellbox.backends.daytona.machine import DaytonaMachineFactory  # noqa: PLC0415
    except ImportError as error:
        raise click.UsageError("--image-backend daytona requires the Shellbox daytona extra") from error
    return DaytonaMachineFactory()


def _publish_campaign(report_path: str, selected: dict[str, RlDataSource]) -> None:
    path = StoragePath(report_path)
    report = json.loads(path.read_text())
    # Selected recipes may supply declarations absent from the default inventory.
    sources = {**all_sources(), **{source.info.id: source for source in selected.values()}}
    publish_catalog(str(path.parent / CATALOG_DIRECTORY), report["run_id"], sources.values())


@click.command(help=__doc__)
@click.option("--model", default=GLM_MODEL, show_default=True)
@click.option("--model-revision", help="Required when a recipe uses model review.")
@click.option("--base-url", help="OpenAI-compatible review endpoint; defaults to the relay job's endpoint.")
@click.option("--relay-job", default=DEFAULT_GLM_RELAY_JOB, show_default=True, help="Iris GLM relay job to resolve.")
@click.option("--review-cache", help="Required when a recipe uses model review.")
@click.option("--review-mode", type=click.Choice(["batch", "chat"]), help="Required when a recipe uses model review.")
@click.option(
    "--review-concurrency",
    type=click.IntRange(min=1, max=MAX_DIRECT_CONCURRENT_REQUESTS),
    default=MAX_DIRECT_CONCURRENT_REQUESTS,
    show_default=True,
)
@click.option("--mode", type=click.Choice([mode.value for mode in SourceProcessingMode]), required=True)
@click.option("--max-workers", type=click.IntRange(min=1), help="Local default: 4; required for Iris execution.")
@click.option("--coordinator-memory", help="Iris coordinator RAM, e.g. 16g; required for Iris execution.")
@click.option("--normalized-shards", type=click.IntRange(min=1), help="Local default: 4; required for Iris recipes.")
@click.option("--concurrent-sources", type=click.IntRange(min=1), default=10, show_default=True)
@click.option("--worker-image", help="Zephyr worker image; required for Iris execution.")
@click.option(
    "--container-profile",
    default=None,
    help="Iris worker profile; defaults to CONTAINER_PROFILE_PRIVILEGED for in-pod sandbox graders.",
)
@click.option(
    "--image-backend",
    type=click.Choice([Backend.GVISOR.value, Backend.DAYTONA.value]),
    help="Image controls: gvisor via Iris or daytona. Omit for local and simulator controls only.",
)
@click.option(
    "--controller-url",
    help="Controller for --image-backend gvisor; defaults to the current Iris job's controller.",
)
@click.option(
    "--simulator-catalog",
    help="Portable software catalog URL or path; use an immutable snapshot for cached campaign results.",
)
@click.option("--seed", type=int, default=0)
@click.option(
    "--quality-sample-size",
    type=click.IntRange(min=1, max=MAX_SOURCE_QUALITY_SAMPLE_SIZE),
    default=SOURCE_QUALITY_SAMPLE_SIZE,
    show_default=True,
)
@click.option("--verification-sample-size", type=click.IntRange(min=1), default=20)
@click.option(
    "--curation-attempt",
    type=click.IntRange(min=0),
    default=0,
    help="New curation key for retrying incomplete controls; successful review requests remain cached.",
)
@click.option("--assess/--skip-assessment", default=None, help="Automatic for FULL; opt in for SAMPLE. QUICK skips it.")
@click.option(
    "--assessment-sample-size", type=click.IntRange(min=1), default=AssessmentSampling.admitted, show_default=True
)
@click.option(
    "--excluded-sample-size", type=click.IntRange(min=0), default=AssessmentSampling.excluded, show_default=True
)
@click.option(
    "--assessment-rubric", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Extra rubric criteria."
)
@click.option("--assessment-model-revision", help="Grading model revision; defaults to the curation review revision.")
@click.option("--report-path", help="SAMPLE/FULL report; defaults under --cache-dir. QUICK writes under --output-root.")
@click.option(
    "--source",
    "sources",
    multiple=True,
    help="Catalog source to execute; repeat to select multiple. Defaults to sources with automatic input preparation.",
)
@click.option(
    "--input-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Local runs: staged primary inputs instead of downloading the pinned source.",
)
@click.option(
    "--source-input-root",
    "source_input_roots",
    type=(str, click.Path(exists=True, file_okay=False, path_type=Path)),
    multiple=True,
    help="Local runs: primary input directory for one selected source (NAME PATH); repeat for different sources.",
)
@click.option(
    "--input-file",
    "local_files",
    type=(str, click.Path(exists=True, dir_okay=False, path_type=Path)),
    multiple=True,
    help="Local runs: primary file under its declared logical filename; repeat as needed.",
)
@click.option(
    "--input",
    "auxiliary",
    type=(str, click.Path(exists=True, file_okay=False)),
    multiple=True,
    help="Local runs: override a named auxiliary input with a local directory.",
)
@click.option(
    "--output-root",
    type=click.Path(file_okay=False, path_type=Path),
    help="Required for QUICK outputs.",
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Local artifact cache. Enables local SAMPLE/FULL execution; QUICK defaults to ~/.cache/marin.",
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
    container_profile: str | None,
    image_backend: str | None,
    controller_url: str | None,
    simulator_catalog: str | None,
    seed: int,
    quality_sample_size: int,
    verification_sample_size: int,
    curation_attempt: int,
    assess: bool | None,
    assessment_sample_size: int,
    excluded_sample_size: int,
    assessment_rubric: Path | None,
    assessment_model_revision: str | None,
    report_path: str | None,
    sources: tuple[str, ...],
    input_root: Path | None,
    source_input_roots: tuple[tuple[str, Path], ...],
    local_files: tuple[tuple[str, Path], ...],
    auxiliary: tuple[tuple[str, str], ...],
    output_root: Path | None,
    cache_dir: Path | None,
    do_run: bool,
) -> None:
    processing_mode = SourceProcessingMode(mode)
    selected = _selected_sources(sources)
    runtime = CampaignRuntime()
    assess = assess if assess is not None else processing_mode == SourceProcessingMode.FULL
    if curation_attempt and processing_mode == SourceProcessingMode.QUICK:
        raise click.UsageError("--curation-attempt requires a SAMPLE/FULL curation run")
    if cache_dir is not None:
        cache_dir = cache_dir.expanduser().resolve()
    if input_root is not None and local_files:
        raise click.UsageError("Choose either --input-root or --input-file")
    inputs = InputOverrides(
        str(input_root.resolve()) if input_root is not None else None,
        {name: file.resolve() for name, file in local_files},
        {name: str(Path(path).resolve()) for name, path in auxiliary},
    )
    scoped_roots = {name: str(path.resolve()) for name, path in source_input_roots}
    if len(scoped_roots) != len(source_input_roots):
        raise click.UsageError("Repeat --source-input-root only for different sources")
    unknown_roots = scoped_roots.keys() - selected.keys()
    if unknown_roots:
        raise click.UsageError(f"Input roots require selected sources: {', '.join(sorted(unknown_roots))}")
    source_inputs = {
        name: replace(inputs, root=scoped_roots[name], files={}) if name in scoped_roots else inputs for name in selected
    }
    for name, source in selected.items():
        if source.preparation == InputPreparation.EXPLICIT and not (
            source_inputs[name].root is not None or source_inputs[name].files
        ):
            raise click.UsageError(
                f"{name} requires prepared inputs; use --source-input-root {name} PATH (see dataset README)"
            )
    if processing_mode == SourceProcessingMode.QUICK:
        if assess:
            raise click.UsageError("Assessment requires SAMPLE or FULL")
        if not sources:
            raise click.UsageError("QUICK requires at least one --source")
        if output_root is None:
            raise click.UsageError("QUICK requires --output-root")
        selected = {name: selected[name] for name in dict.fromkeys(sources)}
        cache_dir = cache_dir if cache_dir is not None else Path.home() / ".cache/marin"
        max_workers = max_workers if max_workers is not None else LOCAL_MAX_WORKERS
        output_root = output_root.resolve()
        options = PipelineOptions(processing_mode, runtime, inputs=inputs)
        steps = {
            name: cast(CurationPipeline, source.pipeline)(source, replace(options, inputs=source_inputs[name]))
            for name, source in selected.items()
        }
        if not do_run:
            plan = {
                "mode": processing_mode,
                "sources": [
                    {"name": name, "fingerprint": step.fingerprint(), "dependencies": [dep.name for dep in step.deps]}
                    for name, step in steps.items()
                ],
                "source_input_roots": scoped_roots,
                "input_root": inputs.root,
                "input_files": {name: str(file) for name, file in inputs.files.items()},
                "inputs": dict(inputs.auxiliary),
                "output_root": str(output_root),
                "cache_dir": str(cache_dir),
                "max_workers": max_workers,
            }
            click.echo(json.dumps(plan, indent=2))
            return
        run_local_steps(
            steps,
            output_root,
            runtime=runtime,
            max_workers=max_workers,
            cache_dir=cache_dir,
            source_ids={name: source.info.id for name, source in selected.items()},
        )
        return
    if output_root is not None:
        raise click.UsageError("--output-root requires --mode quick; use --report-path for SAMPLE/FULL")
    if cache_dir is None and (input_root is not None or local_files or auxiliary or scoped_roots):
        raise click.UsageError("Local input options require --cache-dir for SAMPLE/FULL")
    if cache_dir is not None and (
        worker_image is not None or container_profile is not None or coordinator_memory is not None
    ):
        raise click.UsageError(
            "Local execution does not accept --worker-image, --container-profile or --coordinator-memory"
        )
    if max_workers is None and cache_dir is None:
        raise click.UsageError("SAMPLE/FULL requires --max-workers")
    if coordinator_memory is None and cache_dir is None:
        raise click.UsageError("SAMPLE/FULL requires --coordinator-memory")
    if worker_image is None and cache_dir is None:
        raise click.UsageError("SAMPLE/FULL requires --worker-image")
    if report_path is None and cache_dir is None:
        raise click.UsageError("SAMPLE/FULL requires --report-path")
    max_workers = max_workers if max_workers is not None else LOCAL_MAX_WORKERS
    worker_resources = ResourceConfig(
        cpu=1 if cache_dir is not None else 2,
        ram=LOCAL_MEMORY if cache_dir is not None else "8g",
        image=worker_image,
        container_profile=(container_profile or WORKER_CONTAINER_PROFILE) if cache_dir is None else None,
    )
    review = (
        ReviewConfig(model=model, model_revision=model_revision, mode=ReviewMode(review_mode))
        if model_revision is not None and review_mode is not None
        else None
    )
    recipe_settings = RecipeSettings(
        review=review,
        review_cache=review_cache or (str(cache_dir / "review-cache") if cache_dir else None),
        base_url=base_url,
        api_key=os.environ.get(GLM_BULK_TOKEN_ENV),
        relay_job=relay_job,
        review_concurrency=review_concurrency,
        normalized_shards=(
            normalized_shards if normalized_shards is not None else (LOCAL_NORMALIZED_SHARDS if cache_dir else None)
        ),
        machines=CampaignMachines(
            image_factory=_image_factory(Backend(image_backend) if image_backend is not None else None, controller_url),
            simulator_catalog=simulator_catalog,
        ),
        seed=seed,
        quality_sample_size=quality_sample_size,
        verification_sample_size=verification_sample_size,
        curation_attempt=curation_attempt,
        execution=AuditExecution(max_workers=max_workers, worker_resources=worker_resources),
    )
    options = PipelineOptions(processing_mode, runtime, inputs=inputs, recipe_settings=recipe_settings)
    assessment = None
    if assess:
        if review is None or recipe_settings.review_cache is None:
            raise click.UsageError(
                "Assessment requires --model-revision, --review-mode and a review cache; use --skip-assessment to skip"
            )
        rubric = ASSESSMENT_RUBRIC
        if assessment_rubric is not None:
            rubric += "\nAdditional criteria:\n" + assessment_rubric.read_text()
        assessment = AssessmentSettings(
            AssessmentSampling(assessment_sample_size, excluded_sample_size, seed),
            AssessmentModel(review.model, assessment_model_revision or review.model_revision),
            rubric,
        )
    with ExitStack() as scope:
        if cache_dir is not None:
            scope.enter_context(local_artifact_prefix(str(cache_dir)))
            if report_path is None:
                report_path = str(cache_dir / "campaigns/campaign.json")
        steps = [
            cast(CurationPipeline, source.pipeline)(source, replace(options, inputs=source_inputs[name]))
            for name, source in selected.items()
        ]
        assessments = {}
        if assessment is not None:
            stages = [
                assessment_pipeline(
                    step,
                    source=source.name,
                    settings=assessment,
                    runtime=runtime,
                    recipe_settings=recipe_settings,
                )
                for step, source in zip(steps, selected.values(), strict=True)
            ]
            steps = [stage.judgments for stage in stages]
            assessments = {stage.judgments.name: assessment_paths(stage) for stage in stages}
        pool = CampaignPool(
            max_workers,
            concurrent_sources,
            worker_resources=worker_resources,
            coordinator_resources=ResourceConfig(cpu=1, ram=coordinator_memory or LOCAL_MEMORY, preemptible=False),
            chunk_storage_prefix=(
                str(cache_dir / ".zephyr")
                if cache_dir is not None
                else str(StoragePath(marin_prefix()) / ".zephyr/rl-data-campaign")
            ),
        )
        if not do_run:
            plan = campaign_plan(steps, pool)
            plan["cache_dir"] = str(cache_dir) if cache_dir else None
            plan["report_path"] = report_path
            plan["quality_policy"] = {"sample_size": quality_sample_size, "seed": seed}
            click.echo(json.dumps(plan, indent=2))
            return
        if cache_dir is not None:
            client = LocalClient()
            scope.callback(client.shutdown)
            scope.enter_context(set_current_client(client))
        assert report_path is not None
        if assessment is not None:
            assert recipe_settings.review_cache is not None
            compact_query_cache(recipe_settings.review_cache)
        try:
            run_campaign(
                steps,
                runtime=runtime,
                pool=pool,
                report_path=report_path,
                mode=processing_mode,
                source_ids={step.name: source.info.id for step, source in zip(steps, selected.values(), strict=True)},
                assessments=assessments,
            )
        except CampaignFailed:
            _publish_campaign(report_path, selected)
            raise
        else:
            _publish_campaign(report_path, selected)


if __name__ == "__main__":
    main()
