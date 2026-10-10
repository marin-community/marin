# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch a matched baseline run from a prepared hero sample artifact."""

import hashlib

import click
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name

from experiments.grug.fast_track.hero_sample import HeroTrainingSource, PreparedHeroSample
from experiments.grug.fast_track.launch import (
    H100_LADDER_SIZES,
    MatchMode,
    ThroughputResult,
    build_h100_ladder_run,
)


@click.command()
@click.option("--baseline-artifact", required=True, help="PreparedHeroSample artifact path.")
@click.option("--run-id", required=True, help="Run identifier for artifact and W&B names.")
@click.option("--size", type=click.Choice(H100_LADDER_SIZES), default="d512", show_default=True)
@click.option("--dense/--moe", default=True, show_default=True)
@click.option("--match", type=click.Choice([mode.value for mode in MatchMode]), default="data", show_default=True)
@click.option("--num-steps", type=click.IntRange(min=1), default=None)
@click.option("--batch-size", type=click.IntRange(min=1), default=None)
@click.option("--seed", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--data-seed", type=click.IntRange(min=0), default=0, show_default=True)
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
    data_seed: int,
) -> ArtifactStep[ThroughputResult]:
    """Build a baseline run that consumes the named prepared hero sample."""
    sample_digest = hashlib.sha256(baseline_artifact.encode()).hexdigest()[:20]
    artifact_name = f"fast-track/hero-sample/adopted/{sample_digest}"
    artifact_version = resolve_version(artifact_name, None)
    sample = ArtifactStep.adopt(
        user_namespaced_name(artifact_name, artifact_version),
        artifact_version,
        baseline_artifact,
        kind=PreparedHeroSample,
    )
    return build_h100_ladder_run(
        run_id=run_id,
        size=size,
        dense=dense,
        match=MatchMode(match),
        num_steps=num_steps,
        batch_size=batch_size,
        seed=seed,
        data_seed=data_seed,
        training_source=HeroTrainingSource(sample),
    )


if __name__ == "__main__":
    main()
