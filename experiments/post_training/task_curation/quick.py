# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage pinned sources and convert them locally with Zephyr, without review or grader execution."""

import hashlib
import json
import logging
import time
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path

import click
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.execution.step_runner import StepRunner
from taskcompendium.pipeline.inputs import SourceFileOverride
from taskcompendium.pipeline.models import SourceStatus
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from taskcompendium.pipeline.sources import conversion_shards
from zephyr.context import ZephyrContext
from zephyr.runners import SubprocessRunner

from experiments.post_training.task_curation.campaign import (
    CampaignFailed,
    CampaignRuntime,
    CampaignStatus,
    OutcomeStatus,
    SourceOutcome,
    error_chain,
    write_campaign_report,
)
from experiments.post_training.task_curation.pipeline import (
    RlDataPipeline,
    run_curation,
    source_downloads,
    source_files,
)
from experiments.post_training.task_curation.sources import selected_pipelines

logger = logging.getLogger(__name__)


def stage_local_download(download: ArtifactStep[Artifact], cache_root: Path) -> str:
    """Download pinned source files into the local artifact cache, reusing successful downloads."""
    step = replace(download.lower(), output_path_prefix=str(cache_root))
    StepRunner().run([step], max_concurrent=1)
    return step.output_path


def stage_local_inputs(
    pipeline: RlDataPipeline,
    cache_root: Path,
    campaign: CampaignRuntime,
    *,
    source_input: str | None,
    inputs: Mapping[str, str],
) -> tuple[str, dict[str, str]]:
    """Stage declared pins unless a primary root or auxiliary override is supplied."""
    primary, auxiliary = source_downloads(pipeline, campaign)
    if source_input is None:
        source_input = stage_local_download(primary, cache_root)
    staged_inputs = dict(inputs)
    for name, download in auxiliary.items():
        if name not in staged_inputs:
            staged_inputs[name] = stage_local_download(download, cache_root)
    return source_input, staged_inputs


def run_local_sources(
    sources: Mapping[str, RlDataPipeline],
    input_root: Path | None,
    output_root: Path,
    *,
    inputs: Mapping[str, str],
    max_workers: int,
    download_cache: Path,
    source_files_override: Mapping[str, Path] | None = None,
) -> tuple[SourceOutcome, ...]:
    """Convert sources locally, recording failures while continuing the remaining sources."""
    if input_root is not None and source_files_override:
        raise ValueError("Choose either a staged input root or explicit local source files")
    source_overrides = {}
    for logical, file in (source_files_override or {}).items():
        file = file.resolve()
        with file.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        source_overrides[logical] = SourceFileOverride(str(file), checksum)
    input_root = input_root.resolve() if input_root is not None else None
    output_root, download_cache = output_root.resolve(), download_cache.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "campaign.json"
    outcomes = {name: SourceOutcome(name, str(output_root / name), OutcomeStatus.QUEUED) for name in sources}

    def report(status: CampaignStatus) -> None:
        write_campaign_report(
            str(report_path), status, mode=SourceProcessingMode.QUICK, outcomes=list(outcomes.values())
        )

    report(CampaignStatus.RUNNING)
    campaign = CampaignRuntime()
    client = LocalClient()
    with (
        set_current_client(client),
        ZephyrContext(
            client=client,
            max_workers=max_workers,
            resources=ResourceConfig(cpu=1, ram="4g"),
            chunk_storage_prefix=str(output_root / ".zephyr"),
            name="task-curation-quick",
        ) as context,
        campaign.activate(context),
        ExitStack() as process_pools,
    ):
        process_context: ZephyrContext | None = None
        for name, pipeline in sources.items():
            outcomes[name] = SourceOutcome(name, str(output_root / name), OutcomeStatus.RUNNING)
            report(CampaignStatus.RUNNING)
            try:
                started = time.monotonic()
                if source_overrides:
                    source_input = str(output_root)
                elif input_root is not None:
                    source_input = str(input_root)
                else:
                    source_input = None
                source_input, staged_inputs = stage_local_inputs(
                    pipeline, download_cache, campaign, source_input=source_input, inputs=inputs
                )
                logger.info("%s staging completed in %.2f seconds", name, time.monotonic() - started)
                shards = conversion_shards(source_input, source_files(pipeline.source), overrides=source_overrides)
                conversion_context = context
                if max_workers > 1 and any(shard.row_end is not None and shard.parts > 1 for shard in shards):
                    # Process startup dominates small conversions; reserve it for split Parquet files.
                    if process_context is None:
                        process_context = process_pools.enter_context(
                            ZephyrContext(
                                client=client,
                                max_workers=max_workers,
                                resources=context.resources,
                                chunk_storage_prefix=str(output_root / ".zephyr-process"),
                                name="task-curation-quick-process",
                                stage_runner_factory=SubprocessRunner,
                            )
                        )
                    conversion_context = process_context
                result = run_curation(
                    pipeline,
                    context=conversion_context,
                    source_input=source_input,
                    output_path=str(output_root / name),
                    inputs=staged_inputs,
                    source_overrides=source_overrides,
                )
            except Exception as error:
                outcomes[name] = SourceOutcome(name, str(output_root / name), OutcomeStatus.FAILED, error_chain(error))
                click.echo(json.dumps(asdict(outcomes[name])), err=True)
            else:
                outcomes[name] = SourceOutcome(name, str(output_root / name), SourceStatus.COMPLETED)
                click.echo(json.dumps({"source": name, **asdict(result)}))
            report(CampaignStatus.RUNNING)
    failed = any(outcome.status == OutcomeStatus.FAILED for outcome in outcomes.values())
    report(CampaignStatus.FAILED if failed else CampaignStatus.COMPLETED)
    if failed:
        raise CampaignFailed(f"Quick conversion failed for some sources; see {report_path}")
    return tuple(outcomes.values())


@click.command(help=__doc__)
@click.option("--source", "sources", multiple=True, required=True, help="Catalog key; repeat for several sources.")
@click.option(
    "--input-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Use staged primary inputs instead of downloading the declared pinned source.",
)
@click.option(
    "--input-file",
    "local_files",
    type=(str, click.Path(exists=True, dir_okay=False, path_type=Path)),
    multiple=True,
    help="Use a local file under its declared logical source filename; repeat for multiple files.",
)
@click.option("--output-root", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option(
    "--download-cache",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path.home() / ".cache" / "marin",
    show_default=True,
    help="Local cache of pinned primary and auxiliary downloads.",
)
@click.option("--input", "auxiliary", type=(str, click.Path(exists=True, file_okay=False)), multiple=True)
@click.option("--max-workers", type=click.IntRange(min=1), default=4, show_default=True)
def main(
    sources: tuple[str, ...],
    local_files: tuple[tuple[str, Path], ...],
    input_root: Path | None,
    output_root: Path,
    download_cache: Path,
    auxiliary: tuple[tuple[str, str], ...],
    max_workers: int,
) -> None:
    try:
        catalog = selected_pipelines(sources)
    except ValueError as error:
        raise click.UsageError(str(error)) from error
    inputs = {name: str(Path(path).resolve()) for name, path in auxiliary}
    run_local_sources(
        {name: catalog[name] for name in dict.fromkeys(sources)},
        input_root,
        output_root,
        inputs=inputs,
        max_workers=max_workers,
        download_cache=download_cache,
        source_files_override=dict(local_files),
    )


if __name__ == "__main__":
    main()
