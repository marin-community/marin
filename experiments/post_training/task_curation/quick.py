# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert staged sources locally with Zephyr, without model review or grader execution."""

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import click
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from taskcompendium.pipeline.models import SourceStatus
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.campaign import (
    CampaignFailed,
    CampaignStatus,
    OutcomeStatus,
    SourceOutcome,
    campaign_report,
    error_chain,
)
from experiments.post_training.task_curation.conversions import convert_source
from experiments.post_training.task_curation.source import RlDataSource
from experiments.post_training.task_curation.sources import all_sources


def run_local_sources(
    sources: Mapping[str, RlDataSource],
    input_root: Path,
    output_root: Path,
    *,
    inputs: Mapping[str, str],
    max_workers: int,
) -> tuple[SourceOutcome, ...]:
    """Convert sources on one local pool, recording failures while continuing the remaining sources."""
    input_root, output_root = input_root.resolve(), output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "campaign.json"
    outcomes = {name: SourceOutcome(name, str(output_root / name), OutcomeStatus.QUEUED) for name in sources}

    def report(status: CampaignStatus) -> None:
        report_path.write_text(
            json.dumps(campaign_report(status, mode="quick", outcomes=list(outcomes.values())), indent=2)
        )

    report(CampaignStatus.RUNNING)
    with ZephyrContext(
        client=LocalClient(),
        max_workers=max_workers,
        resources=ResourceConfig(cpu=1, ram="4g"),
        chunk_storage_prefix=str(output_root / ".zephyr"),
        name="task-curation-quick",
    ) as context:
        for name, source in sources.items():
            outcomes[name] = SourceOutcome(name, str(output_root / name), OutcomeStatus.RUNNING)
            report(CampaignStatus.RUNNING)
            try:
                result = convert_source(
                    source,
                    mode=SourceProcessingMode.QUICK,
                    context=context,
                    source_input=str(input_root),
                    output_path=str(output_root / name),
                    inputs=inputs,
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
@click.option("--input-root", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--output-root", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--input", "auxiliary", type=(str, click.Path(exists=True, file_okay=False)), multiple=True)
@click.option("--max-workers", type=click.IntRange(min=1), default=4, show_default=True)
def main(
    sources: tuple[str, ...],
    input_root: Path,
    output_root: Path,
    auxiliary: tuple[tuple[str, str], ...],
    max_workers: int,
) -> None:
    catalog = {source.name: source for source in all_sources().values() if source.pipeline is not None}
    unknown = set(sources) - catalog.keys()
    if unknown:
        raise click.UsageError(f"Unknown sources: {', '.join(sorted(unknown))}")
    inputs = {name: str(Path(path).resolve()) for name, path in auxiliary}
    run_local_sources(
        {name: catalog[name] for name in dict.fromkeys(sources)},
        input_root,
        output_root,
        inputs=inputs,
        max_workers=max_workers,
    )


if __name__ == "__main__":
    main()
