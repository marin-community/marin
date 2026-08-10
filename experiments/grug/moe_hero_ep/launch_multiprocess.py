# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Four-process-per-node fixed-all-to-all control matched to the MoK topology."""

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options

from experiments.grug.moe_hero_ep.launch import (
    DEFAULT_HERO_STEPS,
    HERO_EP_EXPERT_AXIS_SIZE,
    HeroThroughputResult,
    build_multiprocess_hero_run,
)


@click.command()
@click.option("--run-id", required=True, help="Run identifier for artifact and W&B names.")
@click.option(
    "--num-steps",
    type=click.IntRange(min=1),
    default=DEFAULT_HERO_STEPS,
    show_default=True,
    help="Number of training steps.",
)
@click.option(
    "--num-experts",
    type=click.IntRange(min=1),
    default=None,
    help=f"Override the routed expert count. Must be divisible by {HERO_EP_EXPERT_AXIS_SIZE}.",
)
@click.option("--num-experts-per-token", type=click.IntRange(min=1), default=None)
@click.option("--intermediate-dim", type=click.IntRange(min=1), default=None)
@click.option(
    "--capacity-factor",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    help="Override the fixed all-to-all capacity factor.",
)
@build_options
def main(
    run_id: str,
    num_steps: int,
    num_experts: int | None,
    num_experts_per_token: int | None,
    intermediate_dim: int | None,
    capacity_factor: float | None,
) -> ArtifactStep[HeroThroughputResult]:
    return build_multiprocess_hero_run(
        run_id=run_id,
        num_steps=num_steps,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        intermediate_dim=intermediate_dim,
        capacity_factor=capacity_factor,
    )


if __name__ == "__main__":
    main()
