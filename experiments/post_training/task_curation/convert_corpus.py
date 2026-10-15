# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert complete pinned source populations through the existing Zephyr campaign."""

import json
from dataclasses import replace

import click
from fray.types import ResourceConfig
from marin.execution.lazy import ArtifactStep
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.source_processing import SourceProcessingMode

from experiments.post_training.task_curation.campaign import (
    CampaignArtifact,
    CampaignPool,
    CampaignRuntime,
    campaign_plan,
    run_campaign,
)
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.source import RlDataSource
from experiments.post_training.task_curation.source_metadata import source_declaration
from experiments.post_training.task_curation.sources import all_sources, runnable_sources

CORPUS_CONVERSION_VERSION = "2026.10.14.1"


def corpus_conversion_steps(
    sources: dict[str, RlDataSource], runtime: CampaignRuntime
) -> dict[str, ArtifactStep[CampaignArtifact]]:
    """Bind full conversion to pinned recipe and acquisition identities."""
    options = PipelineOptions(mode=SourceProcessingMode.QUICK, runtime=runtime)
    steps = {}
    for source in sources.values():
        assert source.pipeline is not None
        step = source.pipeline(source, options)
        declaration = source_declaration(source)
        assert declaration.pipeline is not None
        identity = canonical_sha256(
            {"pipeline": declaration.pipeline.model_dump(), "dependencies": step.fingerprint()}
        )
        steps[source.info.id] = replace(
            step,
            name=f"rl-semantics/converted/{source.name}-{identity[:16]}",
            version=CORPUS_CONVERSION_VERSION,
        )
    return steps


@click.command(help=__doc__)
@click.option("--source", multiple=True, help="Registry invocation name; omit to process every runnable source.")
@click.option("--output-root", required=True, help="Persistent region-local campaign and inventory output.")
@click.option("--max-workers", type=click.IntRange(min=1), required=True)
@click.option("--concurrent-sources", type=click.IntRange(min=1), default=4, show_default=True)
@click.option("--worker-cpu", type=click.FloatRange(min=0, min_open=True), default=2, show_default=True)
@click.option("--worker-memory", default="16g", show_default=True)
@click.option("--coordinator-memory", default="8g", show_default=True)
@click.option("--run", "do_run", is_flag=True)
def main(source, output_root, max_workers, concurrent_sources, worker_cpu, worker_memory, coordinator_memory, do_run):
    registry = runnable_sources()
    if unknown := set(source) - registry.keys():
        raise click.UsageError(f"Unknown sources: {sorted(unknown)}")
    selected = {name: value for name, value in registry.items() if not source or name in source}
    runtime = CampaignRuntime()
    steps = corpus_conversion_steps(selected, runtime)
    root = StoragePath(output_root)
    pool = CampaignPool(
        max_workers=max_workers,
        concurrent_sources=concurrent_sources,
        coordinator_resources=ResourceConfig(cpu=1, ram=coordinator_memory),
        worker_resources=ResourceConfig(cpu=worker_cpu, ram=worker_memory),
        chunk_storage_prefix=str(root / ".zephyr"),
    )
    click.echo(json.dumps(campaign_plan(list(steps.values()), pool), indent=2))
    if not do_run:
        return
    outcomes = run_campaign(
        list(steps.values()),
        pool=pool,
        runtime=runtime,
        report_path=str(root / "campaign.json"),
        source_ids={step.name: identifier for identifier, step in steps.items()},
        mode=SourceProcessingMode.QUICK,
    )
    by_name = {outcome.name: outcome for outcome in outcomes}
    converted = []
    for identifier, step in steps.items():
        path = by_name[step.name].path
        manifest = json.loads((StoragePath(path) / "manifest.json").read_text())
        if manifest["input_rows"] != manifest["converted_rows"] + sum(manifest["rejections"].values()):
            raise ValueError(f"Conversion does not account for all input rows: {identifier}")
        converted.append(
            {
                "source_id": identifier,
                "root": path,
                "source_key": canonical_sha256(manifest),
                "input_rows": manifest["input_rows"],
                "converted_rows": manifest["converted_rows"],
                "rejections": manifest["rejections"],
                "intended_use": manifest["intended_use"],
            }
        )
    inventory = [source_declaration(item).model_dump(mode="json") for item in all_sources().values()]
    result = {
        "version": CORPUS_CONVERSION_VERSION,
        "population": "converted",
        "sources": converted,
        "inventory": inventory,
        "campaign": str(root / "campaign.json"),
    }
    (root / "corpus.json").write_text(json.dumps(result, indent=2))
    click.echo(json.dumps({"sources": len(converted), "input_rows": sum(row["input_rows"] for row in converted)}))


if __name__ == "__main__":
    main()
