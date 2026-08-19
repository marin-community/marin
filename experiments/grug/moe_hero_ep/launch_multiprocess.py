# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Four-process-per-node fixed-all-to-all control matched to the MoK topology.

This is the topology-matched control, not the pooled-wave arm's native configuration; `launch`
is. Run both and score MoK against the better of the two, per "Fairness controls" in the README.
The knob surface here deliberately mirrors `launch_mok` so the baseline is never the arm that
cannot be swept, seeded, or profiled.
"""

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options

from experiments.grug.moe_hero_ep.launch import (
    DEFAULT_HERO_STEPS,
    HERO_EP_BATCH_SIZE,
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
@click.option(
    "--latent-dim",
    type=click.IntRange(min=0),
    default=None,
    help="LatentMoE: run routed experts at this width. Zero removes the projection entirely.",
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
@build_options
def main(
    run_id: str,
    num_steps: int,
    num_experts: int | None,
    num_experts_per_token: int | None,
    intermediate_dim: int | None,
    capacity_factor: float | None,
    latent_dim: int | None,
    batch_size: int,
    schedule_steps: int | None,
    seed: int,
    profile_steps: int,
    profile_start_step: int,
) -> ArtifactStep[HeroThroughputResult]:
    return build_multiprocess_hero_run(
        run_id=run_id,
        num_steps=num_steps,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        intermediate_dim=intermediate_dim,
        capacity_factor=capacity_factor,
        latent_dim=latent_dim,
        batch_size=batch_size,
        schedule_steps=schedule_steps,
        seed=seed,
        profile_steps=profile_steps,
        profile_start_step=profile_start_step,
    )


if __name__ == "__main__":
    main()
