# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-rack GB200 launcher for the dropless Mixture-of-Kittens comparison arm."""

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options

from experiments.grug.moe_hero_ep.launch import (
    DEFAULT_HERO_STEPS,
    HERO_EP_EXPERT_AXIS_SIZE,
    HERO_PROFILE_NUM_STEPS,
    HeroThroughputResult,
    build_mok_hero_run,
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
@click.option(
    "--num-experts-per-token",
    type=click.IntRange(min=1),
    default=None,
    help="Override routed top-k. MoK remains dropless at every top-k.",
)
@click.option(
    "--intermediate-dim",
    type=click.IntRange(min=1),
    default=None,
    help="Override the routed expert width.",
)
@click.option(
    "--profile-start-step",
    type=click.IntRange(min=1),
    default=None,
    help="Start an XProf capture at this training step; omitted disables profiling.",
)
@click.option(
    "--profile-num-steps",
    type=click.IntRange(min=1),
    default=HERO_PROFILE_NUM_STEPS,
    show_default=True,
    help="Number of training steps to capture.",
)
@click.option(
    "--mok-package",
    required=True,
    envvar="MOK_PACKAGE",
    help="Immutable cp312 Linux wheel URI for the worker architecture, built against torch 2.11+cu130.",
)
@click.option(
    "--mok-expert-placement",
    type=click.Choice(("contiguous", "r9_profile_hot_cold")),
    default="contiguous",
    show_default=True,
    help="Relabel the r9 expert bank at initialization so hot/cold pairs share an EP rank.",
)
@build_options
def main(
    run_id: str,
    num_steps: int,
    num_experts: int | None,
    num_experts_per_token: int | None,
    intermediate_dim: int | None,
    profile_start_step: int | None,
    profile_num_steps: int,
    mok_package: str,
    mok_expert_placement: str,
) -> ArtifactStep[HeroThroughputResult]:
    return build_mok_hero_run(
        run_id=run_id,
        num_steps=num_steps,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        intermediate_dim=intermediate_dim,
        mok_package=mok_package,
        mok_expert_placement=mok_expert_placement,
        profile_start_step=profile_start_step,
        profile_num_steps=profile_num_steps,
    )


if __name__ == "__main__":
    main()
