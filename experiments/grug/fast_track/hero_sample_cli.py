# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the bounded current hero-mixture sample."""

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options

from experiments.grug.fast_track.hero_sample import HERO_TARGET_TOKENS, hero_sample_step


@click.command()
@click.option(
    "--token-budget",
    type=click.IntRange(min=1, max=HERO_TARGET_TOKENS),
    default=HERO_TARGET_TOKENS,
    show_default=True,
    help="Maximum target-token budget for the frozen hero sample.",
)
@click.option("--data-seed", type=click.IntRange(min=0), default=0, show_default=True)
@build_options
def main(token_budget: int, data_seed: int) -> ArtifactStep:
    """Build an artifact step for the pinned hero sample."""
    return hero_sample_step(requested_tokens=token_budget, data_seed=data_seed)


if __name__ == "__main__":
    main()
