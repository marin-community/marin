# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan RL curation, then run QUICK locally or reviewed SAMPLE/FULL campaigns on Iris."""

import json
from pathlib import Path
from typing import cast

import click
from fray.types import ResourceConfig
from iris.cluster.client.job_info import get_job_info
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.machine import Backend, MachineFactory
from taskcompendium.pipeline.chat_requests import MAX_DIRECT_CONCURRENT_REQUESTS
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from taskcompendium.pipeline.stages import AuditExecution, ReviewConfig, ReviewMode

from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB, GLM_MODEL
from experiments.post_training.task_curation.campaign import CampaignPool, CampaignRuntime, campaign_plan, run_campaign
from experiments.post_training.task_curation.config import (
    InputOverrides,
    PipelineOptions,
    RecipeSettings,
)
from experiments.post_training.task_curation.local import run_local_steps
from experiments.post_training.task_curation.pipeline import CampaignMachines
from experiments.post_training.task_curation.source import CurationPipeline, RlDataSource
from experiments.post_training.task_curation.sources import runnable_sources

QUICK_MAX_WORKERS = 4
IRIS_JOB_TTL = 1800


def _selected_sources(sources: tuple[str, ...]) -> dict[str, RlDataSource]:
    """The named catalog sources in catalog order, or the whole catalog when none is named."""
    catalog = runnable_sources()
    unknown = set(sources) - catalog.keys()
    if unknown:
        raise click.UsageError(f"Unknown source: {', '.join(sorted(unknown))}")
    return {name: pipeline for name, pipeline in catalog.items() if not sources or name in sources}


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
@click.option("--max-workers", type=click.IntRange(min=1), help="Required for SAMPLE/FULL; defaults to 4 for QUICK.")
@click.option("--coordinator-memory", help="Required for SAMPLE/FULL: RAM for the shared coordinator, e.g. 16g.")
@click.option("--normalized-shards", type=click.IntRange(min=1), help="Required for recipe SAMPLE/FULL.")
@click.option("--concurrent-sources", type=click.IntRange(min=10), default=10, show_default=True)
@click.option("--worker-image", help="Required for SAMPLE/FULL: Zephyr worker image carrying the grading code.")
@click.option(
    "--container-profile",
    default="CONTAINER_PROFILE_PRIVILEGED",
    show_default=True,
    help="Iris container profile of the Zephyr workers; local graders need a privileged pod to build sandboxes.",
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
    image_backend: str | None,
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
    processing_mode = SourceProcessingMode(mode)
    selected = _selected_sources(sources)
    runtime = CampaignRuntime()
    if processing_mode == SourceProcessingMode.QUICK:
        if not sources:
            raise click.UsageError("QUICK requires at least one --source")
        if output_root is None:
            raise click.UsageError("QUICK requires --output-root")
        if input_root is not None and local_files:
            raise click.UsageError("Choose either --input-root or --input-file")
        selected = {name: selected[name] for name in dict.fromkeys(sources)}
        download_cache = download_cache if download_cache is not None else Path.home() / ".cache/marin"
        max_workers = max_workers if max_workers is not None else QUICK_MAX_WORKERS
        output_root = output_root.resolve()
        download_cache = download_cache.expanduser().resolve()
        inputs = InputOverrides(
            str(input_root.resolve()) if input_root is not None else None,
            {name: file.resolve() for name, file in local_files},
            {name: str(Path(path).resolve()) for name, path in auxiliary},
        )
        options = PipelineOptions(processing_mode, runtime, inputs=inputs)
        steps = {name: cast(CurationPipeline, source.pipeline)(source, options) for name, source in selected.items()}
        if not do_run:
            plan = {
                "mode": processing_mode,
                "sources": [
                    {"name": name, "fingerprint": step.fingerprint(), "dependencies": [dep.name for dep in step.deps]}
                    for name, step in steps.items()
                ],
                "input_root": inputs.root,
                "input_files": {name: str(file) for name, file in inputs.files.items()},
                "inputs": dict(inputs.auxiliary),
                "output_root": str(output_root),
                "download_cache": str(download_cache),
                "max_workers": max_workers,
            }
            click.echo(json.dumps(plan, indent=2))
            return
        run_local_steps(steps, output_root, runtime=runtime, max_workers=max_workers, download_cache=download_cache)
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
    review = (
        ReviewConfig(model=model, model_revision=model_revision, mode=ReviewMode(review_mode))
        if model_revision is not None and review_mode is not None
        else None
    )
    recipe_settings = RecipeSettings(
        review=review,
        review_cache=review_cache,
        base_url=base_url,
        relay_job=relay_job,
        review_concurrency=review_concurrency,
        normalized_shards=normalized_shards,
        machines=CampaignMachines(
            image_factory=_image_factory(Backend(image_backend) if image_backend is not None else None, controller_url)
        ),
        seed=seed,
        verification_sample_size=verification_sample_size,
        execution=AuditExecution(max_workers=max_workers, worker_resources=worker_resources),
    )
    options = PipelineOptions(processing_mode, runtime, recipe_settings=recipe_settings)
    steps = [cast(CurationPipeline, source.pipeline)(source, options) for source in selected.values()]
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
