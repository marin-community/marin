# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a quality-head comparison from a frozen bundle."""

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options

from experiments.grug.fast_track.launch import ThroughputResult, build_h100_ladder_run
from experiments.grug.fast_track.quality_pipeline import (
    PinnedFile,
    QualityData,
    QualitySpec,
    QualityTrainingSource,
    SelectionMethod,
    build_quality_data,
)


@click.command()
@click.option("--bundle", required=True, help="Frozen quality bundle JSON path.")
@click.option("--bundle-sha256", required=True, help="SHA-256 of the bundle JSON bytes.")
@click.option("--run-id", required=True)
@click.option("--size", type=click.Choice(["d512", "d768", "d1024"]), default="d512", show_default=True)
@click.option(
    "--selection-method", type=click.Choice([item.value for item in SelectionMethod]), default="ridge", show_default=True
)
@click.option("--fraction", type=click.FloatRange(min=0, max=1, min_open=True), default=0.1, show_default=True)
@click.option("--regularization", type=click.FloatRange(min=0, min_open=True), default=0.01, show_default=True)
@click.option("--tie-seed", type=click.IntRange(min=0), default=0)
@click.option("--seed", type=click.IntRange(min=0), default=0)
@click.option("--data-seed", type=click.IntRange(min=0), default=0)
@click.option("--prepare-only", is_flag=True)
@build_options
def main(
    bundle: str,
    bundle_sha256: str,
    run_id: str,
    size: str,
    selection_method: str,
    fraction: float,
    regularization: float,
    tie_seed: int,
    seed: int,
    data_seed: int,
    prepare_only: bool,
) -> ArtifactStep[QualityData] | ArtifactStep[ThroughputResult]:
    spec = QualitySpec(
        PinnedFile(path=bundle, sha256=bundle_sha256),
        SelectionMethod(selection_method),
        fraction,
        regularization,
        tie_seed,
    )
    selection = build_quality_data(spec)
    if prepare_only:
        return selection
    return build_h100_ladder_run(
        run_id=run_id,
        size=size,
        dense=True,
        training_source=QualityTrainingSource(selection),
        seed=seed,
        data_seed=data_seed,
    )


if __name__ == "__main__":
    main()
