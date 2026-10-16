# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch a matched baseline run from a prepared hero sample artifact."""

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options

from experiments.grug.fast_track.contracts import adopt_artifact_path
from experiments.grug.fast_track.hero_sample import HeroTrainingSource, PreparedHeroSample
from experiments.grug.fast_track.launch import (
    MatchMode,
    ThroughputResult,
    build_h100_ladder_run,
    ladder_options,
)


@click.command()
@click.option("--baseline-artifact", required=True, help="PreparedHeroSample artifact path.")
@click.option("--run-id", required=True, help="Run identifier for artifact and W&B names.")
@ladder_options
@build_options
def main(
    baseline_artifact: str,
    run_id: str,
    size: str,
    dense: bool,
    match: str,
    num_steps: int | None,
    batch_size: int | None,
    seed: int,
    data_seed: int | None,
) -> ArtifactStep[ThroughputResult]:
    """Build a baseline run that consumes the named prepared hero sample."""
    return build_h100_ladder_run(
        run_id=run_id,
        size=size,
        dense=dense,
        match=MatchMode(match),
        num_steps=num_steps,
        batch_size=batch_size,
        seed=seed,
        data_seed=data_seed,
        training_source=HeroTrainingSource(adopt_artifact_path(baseline_artifact, kind=PreparedHeroSample)),
    )


if __name__ == "__main__":
    main()
