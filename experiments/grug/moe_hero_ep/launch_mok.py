# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-rack GB200 launcher for the dropless Mixture-of-Kittens comparison arm."""

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options

from experiments.grug.moe_hero_ep.launch import (
    DEFAULT_HERO_STEPS,
    HERO_EP_BATCH_SIZE,
    HERO_EP_EXPERT_AXIS_SIZE,
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
    "--latent-dim",
    type=click.IntRange(min=0),
    default=None,
    help=(
        "LatentMoE: run routed experts at this width, which is also the width MoK dispatches. "
        "Zero removes the projection and lets the fused call keep the shared experts; omitted "
        "keeps the hero width, which is the shape the pooled-wave arm runs."
    ),
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=HERO_EP_BATCH_SIZE,
    show_default=True,
    help="Global sequences per step.",
)
@click.option(
    "--schedule-steps",
    type=click.IntRange(min=1),
    default=None,
    help="Build the learning-rate schedule for this many steps instead of --num-steps.",
)
@click.option("--seed", type=int, default=0, show_default=True, help="Trainer seed.")
@click.option(
    "--profile-steps",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Number of training steps to capture with XProf. Zero disables profiling.",
)
@click.option(
    "--profile-start-step",
    type=click.IntRange(min=0),
    default=5,
    show_default=True,
    help="Training step the XProf capture starts on.",
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
    latent_dim: int | None,
    batch_size: int,
    schedule_steps: int | None,
    seed: int,
    profile_steps: int,
    profile_start_step: int,
    mok_package: str,
    mok_expert_placement: str,
) -> ArtifactStep[HeroThroughputResult]:
    return build_mok_hero_run(
        run_id=run_id,
        num_steps=num_steps,
        schedule_steps=schedule_steps,
        seed=seed,
        batch_size=batch_size,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        intermediate_dim=intermediate_dim,
        latent_dim=latent_dim,
        mok_package=mok_package,
        mok_expert_placement=mok_expert_placement,
        profile_steps=profile_steps,
        profile_start_step=profile_start_step,
    )


if __name__ == "__main__":
    main()
